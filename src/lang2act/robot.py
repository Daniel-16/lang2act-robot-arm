"""Robot layer: IK, min-jerk motion, gripper control and contact checks.

This module is the only place that talks to actuators. Nothing above it
(skills, executor, planner) is allowed to write ``data.ctrl``.

Design notes
------------
* **IK runs on a robot-only model.** ``scene.build_scene(with_objects=False)``
  gives a model with the arm, table and cameras but no free-floating cubes, so
  the differential-IK solver can never "satisfy" a target by nudging a cube.
* **IK is solved offline, then tracked.** We solve for a joint configuration
  with mink, then drive the position actuators along a min-jerk interpolation
  from the current configuration to that one. The physics is never stepped
  inside the IK loop, which keeps motion deterministic given a seed.
* **Gravity compensation matters.** Under a pure position servo the arm sags
  6-8 mm below its target; ``scene.SceneConfig.gravity_compensation`` sets
  ``gravcomp=1`` on the arm bodies to remove that steady-state error.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import mink
import mujoco
import numpy as np

from .scene import (
    ARM_JOINTS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_QPOS,
    SceneConfig,
    SceneInfo,
    build_scene,
)

StepHook = Callable[[mujoco.MjModel, mujoco.MjData], None]

#: QP backend for mink. `daqp` ships with qpsolvers and needs no extra wheels.
IK_SOLVER = "daqp"


def gripper_down_quat(yaw: float = 0.0) -> np.ndarray:
    """Quaternion (w, x, y, z) for a tool frame pointing straight down.

    The hand's +z axis points out of the flange towards the fingers, and the
    fingers open along the hand's +/-y axis. ``Rz(yaw) @ Rx(pi)`` therefore puts
    the tool z axis along world -z and lets ``yaw`` align the jaws with a cube.
    """
    qx = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x
    qz = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, qx)
    return out


def min_jerk(s: float) -> float:
    """Min-jerk time scaling on ``s`` in [0, 1]: zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


@dataclass(frozen=True)
class IKSolution:
    """Result of an IK query."""

    qpos: np.ndarray
    ok: bool
    pos_error: float
    ori_error: float
    iterations: int

    @property
    def pos_error_mm(self) -> float:
        return self.pos_error * 1000.0


class Robot:
    """Everything the arm can be told to do, with no notion of a task."""

    #: IK is considered converged below these thresholds.
    POS_TOL = 1e-3  # 1 mm
    ORI_TOL = 2e-2  # ~1.1 deg

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
        ik_model: mujoco.MjModel,
        *,
        hooks: Sequence[StepHook] = (),
    ) -> None:
        self.model = model
        self.data = data
        self.info = info
        self.ik_model = ik_model
        self.hooks: list[StepHook] = list(hooks)

        self.arm_dofs = np.array(
            [model.joint(name).qposadr[0] for name in ARM_JOINTS], dtype=int
        )
        self.finger_dofs = np.array(
            [model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)], dtype=int
        )
        self.gripper_act = int(model.actuator("actuator8").id)

        self._left_body = int(model.body("left_finger").id)
        self._right_body = int(model.body("right_finger").id)

        # mink pieces, built once on the robot-only model.
        self._conf = mink.Configuration(ik_model)
        self._frame_task = mink.FrameTask(
            frame_name="tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.5,
            lm_damping=1.0,
        )
        self._posture_task = mink.PostureTask(ik_model, cost=1e-3)
        self._limits = [mink.ConfigurationLimit(ik_model)]

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls, cfg: SceneConfig | None = None, *, hooks: Sequence[StepHook] = ()
    ) -> "Robot":
        cfg = cfg or SceneConfig()
        model, data, info = build_scene(cfg, with_objects=True)
        ik_model, _, _ = build_scene(cfg, with_objects=False)
        return cls(model, data, info, ik_model, hooks=hooks)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    @property
    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_dofs].copy()

    @property
    def tcp_pos(self) -> np.ndarray:
        return np.array(self.data.site("tcp").xpos, dtype=float)

    @property
    def tcp_mat(self) -> np.ndarray:
        return np.array(self.data.site("tcp").xmat, dtype=float).reshape(3, 3)

    @property
    def gripper_width(self) -> float:
        """Distance between the finger pads, in metres (0.08 fully open)."""
        return float(self.data.qpos[self.finger_dofs].sum())

    @property
    def gripper_target(self) -> float:
        return float(self.data.ctrl[self.gripper_act])

    def body_pos(self, body: str) -> np.ndarray:
        return np.array(self.data.body(body).xpos, dtype=float)

    # ------------------------------------------------------------------ #
    # Contacts
    # ------------------------------------------------------------------ #

    def _contact_bodies(self) -> list[tuple[int, int]]:
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            out.append((b1, b2))
        return out

    def finger_contacts(self, body: str) -> tuple[bool, bool]:
        """``(left_touching, right_touching)`` for the given body."""
        target = int(self.model.body(body).id)
        left = right = False
        for b1, b2 in self._contact_bodies():
            pair = {b1, b2}
            if target not in pair:
                continue
            if self._left_body in pair:
                left = True
            if self._right_body in pair:
                right = True
        return left, right

    def is_grasping(self, body: str) -> bool:
        """Both pads in contact and the jaws not fully closed on thin air."""
        left, right = self.finger_contacts(body)
        return left and right and self.gripper_width > 0.01

    def touching(self, body_a: str, body_b: str) -> bool:
        a = int(self.model.body(body_a).id)
        b = int(self.model.body(body_b).id)
        return any({a, b} == {b1, b2} for b1, b2 in self._contact_bodies())

    # ------------------------------------------------------------------ #
    # Inverse kinematics
    # ------------------------------------------------------------------ #

    def solve_ik(
        self,
        pos: Sequence[float],
        *,
        yaw: float = 0.0,
        quat: Sequence[float] | None = None,
        max_iters: int = 300,
        dt: float = 0.02,
        seed_qpos: Sequence[float] | None = None,
    ) -> IKSolution:
        """Solve for an arm configuration putting the TCP at ``pos``.

        Returns an :class:`IKSolution`; check ``.ok`` before moving. ``ok=False``
        means the pose is unreachable from the current seed, which is what the
        ``move_to`` skill reports as ``unreachable``.
        """
        target_quat = np.asarray(quat, dtype=float) if quat is not None else gripper_down_quat(yaw)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(target_quat, dtype=float)), np.asarray(pos, dtype=float)
        )
        self._frame_task.set_target(target)

        q = np.zeros(self.ik_model.nq)
        start = self.arm_qpos if seed_qpos is None else np.asarray(seed_qpos, dtype=float)
        q[:7] = start
        q[7:9] = 0.04
        self._conf.update(q)
        self._posture_task.set_target(q)

        tasks = [self._frame_task, self._posture_task]
        pos_err = ori_err = float("inf")
        used = 0
        for used in range(1, max_iters + 1):
            vel = mink.solve_ik(
                self._conf, tasks, dt, IK_SOLVER, damping=1e-3, limits=self._limits
            )
            self._conf.integrate_inplace(vel, dt)
            err = self._frame_task.compute_error(self._conf)
            pos_err = float(np.linalg.norm(err[:3]))
            ori_err = float(np.linalg.norm(err[3:]))
            if pos_err < self.POS_TOL and ori_err < self.ORI_TOL:
                break

        ok = pos_err < self.POS_TOL and ori_err < self.ORI_TOL
        return IKSolution(
            qpos=self._conf.q[:7].copy(),
            ok=ok,
            pos_error=pos_err,
            ori_error=ori_err,
            iterations=used,
        )

    # ------------------------------------------------------------------ #
    # Motion
    # ------------------------------------------------------------------ #

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            for hook in self.hooks:
                hook(self.model, self.data)

    def settle(self, duration: float = 0.3) -> None:
        self.step(max(1, int(duration / self.model.opt.timestep)))

    def hold(self, duration: float) -> None:
        """Alias for :meth:`settle`, read as "keep the current command"."""
        self.settle(duration)

    def move_joints(self, q_target: Sequence[float], duration: float = 1.5) -> None:
        """Min-jerk interpolation of the arm position targets, then a short settle."""
        q0 = np.array(self.data.ctrl[:7], dtype=float)
        if not np.any(q0):  # first move of an episode: start from the measured pose
            q0 = self.arm_qpos
        q1 = np.asarray(q_target, dtype=float)
        n = max(2, int(duration / self.model.opt.timestep))
        for i in range(1, n + 1):
            self.data.ctrl[:7] = q0 + (q1 - q0) * min_jerk(i / n)
            self.step()
        self.settle(0.25)

    def move_tcp(
        self,
        pos: Sequence[float],
        *,
        yaw: float = 0.0,
        duration: float = 1.5,
        quat: Sequence[float] | None = None,
    ) -> IKSolution:
        """Solve IK for ``pos`` and drive there. Does nothing if IK fails."""
        sol = self.solve_ik(pos, yaw=yaw, quat=quat)
        if sol.ok:
            self.move_joints(sol.qpos, duration=duration)
        return sol

    def go_home(self, duration: float = 1.5) -> None:
        self.open_gripper()
        self.move_joints(HOME_QPOS, duration=duration)

    # ------------------------------------------------------------------ #
    # Gripper
    # ------------------------------------------------------------------ #

    def set_gripper(self, value: float, *, duration: float = 0.5) -> None:
        """Ramp the finger tendon target. 255 = open, 0 = closed."""
        start = float(self.data.ctrl[self.gripper_act])
        n = max(2, int(duration / self.model.opt.timestep))
        for i in range(1, n + 1):
            self.data.ctrl[self.gripper_act] = start + (value - start) * min_jerk(i / n)
            self.step()
        self.settle(0.2)

    def open_gripper(self, *, duration: float = 0.5) -> None:
        self.set_gripper(GRIPPER_OPEN, duration=duration)

    def close_gripper(self, *, duration: float = 0.7) -> None:
        self.set_gripper(GRIPPER_CLOSED, duration=duration)

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        for hook in self.hooks:
            closer = getattr(hook, "close", None)
            if closer is not None:
                closer()
        self.hooks.clear()


# --------------------------------------------------------------------------- #
# Step hooks: live viewer and MP4 recording
# --------------------------------------------------------------------------- #


class ViewerHook:
    """Passive MuJoCo viewer, synced at ~60 Hz and paced to wall-clock time."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, realtime: bool = True):
        import mujoco.viewer  # imported lazily: needs a display

        self.handle = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
        self.realtime = realtime
        self._every = max(1, int(round((1.0 / 60.0) / model.opt.timestep)))
        self._i = 0
        self._t0 = time.perf_counter()
        self._sim_t0 = float(data.time)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._i += 1
        if self._i % self._every:
            return
        if not self.handle.is_running():
            raise KeyboardInterrupt("viewer closed")
        self.handle.sync()
        if self.realtime:
            lag = (data.time - self._sim_t0) - (time.perf_counter() - self._t0)
            if lag > 0:
                time.sleep(lag)

    def close(self, timeout: float = 2.0) -> None:
        """Close the window and wait for the render thread to actually stop.

        The viewer renders on a daemon thread while mujoco registers
        ``glfw.terminate`` with ``atexit``. If the interpreter exits while that
        thread is still inside a GL call, teardown deadlocks and the process
        hangs after the episode has finished. Waiting for ``is_running()`` to go
        false closes that race.
        """
        try:
            self.handle.close()
            deadline = time.perf_counter() + timeout
            while self.handle.is_running() and time.perf_counter() < deadline:
                time.sleep(0.02)
        except Exception:  # pragma: no cover - viewer already gone
            pass


class VideoHook:
    """Render a fixed camera to an MP4 while the episode runs."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        path: str | Path,
        *,
        camera: str = "front",
        fps: int = 30,
        width: int = 960,
        height: int = 540,
    ):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.writer = imageio.get_writer(self.path, fps=fps, macro_block_size=None)
        self._every = max(1, int(round((1.0 / fps) / model.opt.timestep)))
        self._i = 0
        self.frames = 0

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._i += 1
        if self._i % self._every:
            return
        self.capture(model, data)

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.renderer.update_scene(data, camera=self.camera)
        self.writer.append_data(self.renderer.render())
        self.frames += 1

    def close(self) -> None:
        try:
            self.writer.close()
        finally:
            self.renderer.close()


def make_hooks(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    viewer: bool = False,
    video: str | Path | None = None,
    camera: str = "front",
    fps: int = 30,
) -> list[StepHook]:
    """Build the step hooks asked for on the command line."""
    hooks: list[StepHook] = []
    if viewer:
        hooks.append(ViewerHook(model, data))
    if video:
        hooks.append(VideoHook(model, data, video, camera=camera, fps=fps))
    return hooks


def iter_hook_of(hooks: Iterable[StepHook], kind: type) -> list[StepHook]:
    return [h for h in hooks if isinstance(h, kind)]
