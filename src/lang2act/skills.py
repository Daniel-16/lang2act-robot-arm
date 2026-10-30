"""The five skills the planner is allowed to compose.

Every skill returns a :class:`SkillResult` with ``ok`` and a *named* failure
reason, and every skill checks its own postcondition against simulator state.
The executor never trusts a step it has not verified, so a skill that returns
``ok=True`` has already proved its effect happened.

Failure reasons are a closed vocabulary -- the eval report groups by them:

``unknown_object``      the name is not in this scene
``not_graspable``       the target is a region (the bowl), not a cube
``unreachable``         IK found no configuration for the pose
``motion_error``        the arm tracked the IK solution badly (> 10 mm)
``already_holding``     grasp called while something is already held
``nothing_held``        lift/place/release called with an empty gripper
``grasp_slip``          the jaws closed but the object is not held
``object_dropped``      the object left the gripper mid-skill
``place_blocked``       the place pose is unreachable
``not_in_target``       the object settled outside the requested target
``object_stuck``        release opened the jaws but the object came with them
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .scene import (
    SceneInfo,
    is_inside_region,
    object_pos,
    object_yaw,
)
from .robot import Robot

#: Relation -> (dx, dy) offset from the reference object, in metres.
#: +y is the robot's left, +x is away from the robot base.
RELATIONS: dict[str, tuple[float, float]] = {
    "left_of": (0.0, 0.12),
    "right_of": (0.0, -0.12),
    "behind": (0.12, 0.0),
    "in_front_of": (-0.12, 0.0),
}


@dataclass
class SkillResult:
    """What a skill returns. ``reason`` is empty exactly when ``ok`` is True."""

    ok: bool
    reason: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # lets callers write `if result:`
        return self.ok

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "info": _jsonable(self.info)}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [round(float(v), 4) for v in obj.ravel()]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


class Skills:
    """Stateful skill set bound to one robot and one scene.

    The only state kept is *what is currently held* and *where the last place
    aimed*, because `release` has to verify the effect that `place` set up.
    """

    #: TCP height above a cube's centre when the jaws close on it.
    GRASP_Z = 0.005
    #: Height the TCP travels at when moving over the table.
    CARRY_Z = 0.22
    #: Tolerated tracking error between the IK target and the achieved TCP pose.
    TRACK_TOL = 0.010
    #: Horizontal tolerance for a relational place.
    PLACE_XY_TOL = 0.07
    #: Gap left between the carried object and the surface before releasing.
    DROP_GAP = 0.012
    #: Minimum clearance between a drop spot and anything already in the region.
    MIN_SPOT_GAP = 0.05

    def __init__(self, robot: Robot, info: SceneInfo | None = None) -> None:
        self.robot = robot
        self.info = info or robot.info
        self.held: str | None = None
        self._pending_target: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def model(self):
        return self.robot.model

    @property
    def data(self):
        return self.robot.data

    def _obj(self, name: str):
        return self.info.objects[name]

    def _pos(self, name: str) -> np.ndarray:
        return object_pos(self.model, self.data, self._obj(name).body)

    def _top_z(self, name: str) -> float:
        """Height of the "working surface" of an object: its centre, or a region's rim."""
        o = self._obj(name)
        p = self._pos(name)
        return float(p[2] + o.rim_z)

    def _goto(self, pos: np.ndarray, *, yaw: float = 0.0, duration: float = 1.5) -> SkillResult:
        """Move the TCP, mapping IK and tracking failures onto named reasons."""
        sol = self.robot.solve_ik(pos, yaw=yaw)
        if not sol.ok:
            return SkillResult(
                False,
                "unreachable",
                {"target": pos, "ik_error_mm": round(sol.pos_error_mm, 2)},
            )
        self.robot.move_joints(sol.qpos, duration=duration)
        err = float(np.linalg.norm(self.robot.tcp_pos - np.asarray(pos, float)))
        if err > self.TRACK_TOL:
            return SkillResult(False, "motion_error", {"target": pos, "error_mm": round(err * 1e3, 2)})
        return SkillResult(True, info={"target": pos, "error_mm": round(err * 1e3, 2)})

    def _grasp_yaw(self, name: str) -> float:
        """Align the jaws with the cube's faces."""
        return float(object_yaw(self.model, self.data, self._obj(name).body))

    # ------------------------------------------------------------------ #
    # Skill: move_to
    # ------------------------------------------------------------------ #

    def move_to(self, target: str, offset_z: float = 0.15, **_: Any) -> SkillResult:
        """Put the TCP ``offset_z`` above ``target``, gripper pointing down.

        Postcondition: the TCP is within 10 mm of the commanded pose.
        """
        if target not in self.info.objects:
            return SkillResult(False, "unknown_object", {"target": target})
        offset_z = float(np.clip(offset_z, 0.0, 0.5))
        pos = self._pos(target).copy()
        pos[2] = self._top_z(target) + offset_z
        yaw = self._grasp_yaw(target) if self._obj(target).graspable else 0.0
        res = self._goto(pos, yaw=yaw, duration=1.6)
        res.info["skill_target"] = target
        return res

    # ------------------------------------------------------------------ #
    # Skill: grasp
    # ------------------------------------------------------------------ #

    def grasp(self, target: str, **_: Any) -> SkillResult:
        """Descend onto ``target`` and close the jaws.

        Postcondition: both finger pads are in contact with the object and the
        jaws are not closed on thin air.
        """
        if target not in self.info.objects:
            return SkillResult(False, "unknown_object", {"target": target})
        if not self._obj(target).graspable:
            return SkillResult(False, "not_graspable", {"target": target})
        if self.held is not None:
            return SkillResult(False, "already_holding", {"held": self.held})

        body = self._obj(target).body
        self.robot.open_gripper(duration=0.4)

        yaw = self._grasp_yaw(target)
        above = self._pos(target).copy()
        above[2] += 0.10
        res = self._goto(above, yaw=yaw, duration=1.0)
        if not res.ok:
            return res

        down = self._pos(target).copy()
        down[2] += self.GRASP_Z
        # Slow descent: a fast one bumps the cube out of the jaws before they close.
        res = self._goto(down, yaw=yaw, duration=1.4)
        if not res.ok:
            return res

        self.robot.close_gripper(duration=0.7)
        if not self.robot.is_grasping(body):
            left, right = self.robot.finger_contacts(body)
            return SkillResult(
                False,
                "grasp_slip",
                {
                    "target": target,
                    "left_contact": left,
                    "right_contact": right,
                    "gripper_width": round(self.robot.gripper_width, 4),
                },
            )

        self.held = target
        self._pending_target = None
        return SkillResult(
            True, info={"target": target, "gripper_width": round(self.robot.gripper_width, 4)}
        )

    # ------------------------------------------------------------------ #
    # Skill: lift
    # ------------------------------------------------------------------ #

    def lift(self, height: float = 0.15, **_: Any) -> SkillResult:
        """Raise the held object by ``height`` metres.

        Postcondition: the object is clear of the table and still held.
        """
        if self.held is None:
            return SkillResult(False, "nothing_held")
        height = float(np.clip(height, 0.02, 0.40))
        body = self._obj(self.held).body

        pos = self.robot.tcp_pos.copy()
        pos[2] += height
        res = self._goto(pos, yaw=self._grasp_yaw(self.held), duration=1.2)
        if not res.ok:
            return res

        if not self.robot.is_grasping(body):
            self.held = None
            return SkillResult(False, "object_dropped", {"target": body})

        obj_z = float(self._pos(self.held)[2])
        clearance = obj_z - (self.info.table_z + self._obj(self.held).rest_z)
        if clearance < 0.03:
            return SkillResult(
                False, "object_dropped", {"target": body, "clearance_m": round(clearance, 4)}
            )
        return SkillResult(True, info={"height": height, "object_z": round(obj_z, 4)})

    # ------------------------------------------------------------------ #
    # Skill: place
    # ------------------------------------------------------------------ #

    def place(
        self,
        target: str | None = None,
        relation: str | None = None,
        reference: str | None = None,
        **_: Any,
    ) -> SkillResult:
        """Carry the held object over a target and lower it.

        Either ``target`` (an object or region name, e.g. ``bowl``) or
        ``relation`` + ``reference`` (e.g. ``left_of`` / ``green_cube``).

        Postcondition: the held object is horizontally inside the target region
        and still held. Whether it *settles* there is verified by `release`.
        """
        if self.held is None:
            return SkillResult(False, "nothing_held")

        goal, tol, kind = self._place_goal(target, relation, reference)
        if goal is None:
            return SkillResult(False, "unknown_object", {"target": target, "reference": reference})

        held_body = self._obj(self.held).body
        yaw = self._grasp_yaw(self.held)

        # 1. rise to carry height, 2. travel over the goal, 3. lower.
        up = self.robot.tcp_pos.copy()
        up[2] = max(up[2], self.CARRY_Z)
        res = self._goto(up, yaw=yaw, duration=1.0)
        if not res.ok:
            return res

        over = np.array([goal[0], goal[1], max(self.CARRY_Z, goal[2] + 0.10)])
        res = self._goto(over, yaw=yaw, duration=1.8)
        if not res.ok:
            return SkillResult(False, "place_blocked" if res.reason == "unreachable" else res.reason, res.info)

        res = self._goto(goal, yaw=yaw, duration=1.3)
        if not res.ok:
            return SkillResult(False, "place_blocked" if res.reason == "unreachable" else res.reason, res.info)

        if not self.robot.is_grasping(held_body):
            self.held = None
            return SkillResult(False, "object_dropped", {"target": held_body})

        obj_xy = self._pos(self.held)[:2]
        dist = float(np.linalg.norm(obj_xy - goal[:2]))
        if dist > tol:
            return SkillResult(
                False, "not_in_target", {"distance_m": round(dist, 4), "tolerance_m": tol}
            )

        self._pending_target = {
            "kind": kind,
            "target": target,
            "relation": relation,
            "reference": reference,
            "goal_xy": goal[:2].copy(),
            "tolerance": tol,
        }
        return SkillResult(
            True,
            info={
                "target": target or f"{relation}:{reference}",
                "goal": goal,
                "distance_m": round(dist, 4),
            },
        )

    def _place_goal(
        self, target: str | None, relation: str | None, reference: str | None
    ) -> tuple[np.ndarray | None, float, str]:
        """TCP pose to lower to, the horizontal tolerance, and the goal kind."""
        held = self._obj(self.held) if self.held else None
        held_half = held.radius if held else 0.02

        if target is not None:
            if target not in self.info.objects:
                return None, 0.0, "region"
            o = self._obj(target)
            p = self._pos(target)
            if not o.graspable:
                # A region such as the bowl. Aim at a spot that is not already
                # occupied, so "put all the cubes in the bowl" does not stack
                # them on top of each other and out over the rim.
                xy = self._free_spot(target, held_half)
                goal = np.array([xy[0], xy[1], p[2] + o.rest_z + held_half + self.DROP_GAP])
                return goal, max(o.radius - held_half, 0.02), "region"
            # stacking on another cube
            goal = np.array([p[0], p[1], p[2] + o.radius + held_half + self.GRASP_Z])
            return goal, 0.03, "on_object"

        if relation is not None and reference is not None:
            if relation not in RELATIONS or reference not in self.info.objects:
                return None, 0.0, "relational"
            dx, dy = RELATIONS[relation]
            p = self._pos(reference)
            goal = np.array(
                [p[0] + dx, p[1] + dy, self.info.table_z + held_half + self.GRASP_Z + self.DROP_GAP]
            )
            return goal, self.PLACE_XY_TOL, "relational"

        return None, 0.0, "region"


    def _free_spot(self, region: str, held_half: float) -> np.ndarray:
        """An (x, y) inside ``region`` that is clear of whatever is already in it.

        The region is divided into three slots on a small circle around its
        centre, far enough apart that three 4 cm cubes sit side by side rather
        than stacking out over the rim, and close enough in that the fingers
        clear the wall on the way down. The first slot at least ``2.2 * r`` from
        every occupant wins; if none is, the roomiest slot does.
        """
        reg = self._obj(region)
        centre = self._pos(region)[:2]
        margin = max(reg.radius - held_half - 0.015, 0.0)
        ring_r = min(margin, 2.0 * held_half)

        occupied = [
            self._pos(n)[:2]
            for n in self.info.cubes
            if n != self.held
            and is_inside_region(self.model, self.data, self.info, n, region)
        ]
        slots = [
            centre + ring_r * np.array([np.cos(a), np.sin(a)])
            for a in (np.pi / 2.0, np.pi * 7.0 / 6.0, np.pi * 11.0 / 6.0)
        ]
        if not occupied:
            return slots[0] if ring_r > 1e-3 else centre

        needed = 2.2 * held_half
        scored = [(min(float(np.linalg.norm(c - o)) for o in occupied), c) for c in slots]
        for gap, c in scored:
            if gap >= needed:
                return c
        return max(scored, key=lambda t: t[0])[1]

    # ------------------------------------------------------------------ #
    # Skill: release
    # ------------------------------------------------------------------ #

    def release(self, **_: Any) -> SkillResult:
        """Open the jaws, let the object settle, and retract.

        Postcondition: the object is out of the gripper, and -- when the last
        `place` named a target -- it settled inside that target.
        """
        if self.held is None:
            return SkillResult(False, "nothing_held")

        released = self.held
        body = self._obj(released).body
        self.robot.open_gripper(duration=0.4)
        self.robot.settle(0.4)

        retract = self.robot.tcp_pos.copy()
        retract[2] += 0.12
        self._goto(retract, yaw=self._grasp_yaw(released), duration=1.0)
        self.robot.settle(0.4)

        self.held = None
        left, right = self.robot.finger_contacts(body)
        if left and right:
            return SkillResult(False, "object_stuck", {"target": released})

        pending, self._pending_target = self._pending_target, None
        info: dict[str, Any] = {"target": released, "object_pos": self._pos(released)}
        if pending is None:
            return SkillResult(True, info=info)

        if pending["kind"] == "region" and pending["target"]:
            inside = is_inside_region(
                self.model, self.data, self.info, released, pending["target"]
            )
            info["region"] = pending["target"]
            if not inside:
                return SkillResult(False, "not_in_target", info)
            return SkillResult(True, info=info)

        dist = float(np.linalg.norm(self._pos(released)[:2] - pending["goal_xy"]))
        info["distance_m"] = round(dist, 4)
        if dist > pending["tolerance"] + 0.03:
            return SkillResult(False, "not_in_target", info)
        return SkillResult(True, info=info)

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #

    #: Name -> bound method. The planner may only emit these five names.
    @property
    def table(self) -> dict[str, Any]:
        return {
            "move_to": self.move_to,
            "grasp": self.grasp,
            "lift": self.lift,
            "place": self.place,
            "release": self.release,
        }

    def run(self, skill: str, **kwargs: Any) -> SkillResult:
        fn = self.table.get(skill)
        if fn is None:
            return SkillResult(False, "unknown_skill", {"skill": skill})
        return fn(**kwargs)
