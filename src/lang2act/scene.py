"""MjSpec scene builder: Panda arm, table, coloured cubes and a bowl.

The scene is built in Python with ``mujoco.MjSpec`` rather than hand-written XML.
Two reasons, both learned the hard way:

* ``<include>``-ing Menagerie's ``panda.xml`` from your own XML breaks its
  relative mesh paths; loading it as a spec and editing the spec does not.
* Per-seed randomisation of object poses is trivial on a spec and painful in XML.

Conventions used everywhere in this package:

* The table top is the plane ``z = 0``. The robot base sits on it at the origin.
* ``+x`` points away from the robot base (towards the viewer's "far side"),
  ``+y`` is the robot's **left**, ``+z`` is up.
* The TCP (tool centre point) is a site on the ``hand`` body, 0.1034 m out along
  the flange axis -- Franka's flange-to-fingertip distance. All IK targets it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# --------------------------------------------------------------------------- #
# Paths and model constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
PANDA_XML = REPO_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
ARM_BODIES = tuple(f"link{i}" for i in range(1, 8)) + ("hand", "left_finger", "right_finger")

#: Menagerie's `home` arm posture (7 arm joints), reused as our start pose.
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

#: Flange-to-TCP distance on the Franka hand, in metres.
TCP_OFFSET = 0.1034

#: `actuator8` drives the finger tendon: 255 = fully open, 0 = closed.
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

TABLE_Z = 0.0
FLOOR_Z = -0.75

CUBE_HALF = 0.02  # 4 cm cubes

COLOR_RGBA: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.12, 0.12, 1.0),
    "green": (0.10, 0.65, 0.20, 1.0),
    "blue": (0.15, 0.30, 0.85, 1.0),
    "yellow": (0.92, 0.80, 0.14, 1.0),
    "purple": (0.55, 0.20, 0.75, 1.0),
    "orange": (0.95, 0.50, 0.10, 1.0),
}

#: Words a command may use for a colour. The planner prompt lists the canonical
#: object names; these are only used by the offline mock planner and by tests.
COLOR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "red": ("red", "crimson", "scarlet"),
    "green": ("green", "emerald", "lime"),
    "blue": ("blue", "azure", "navy"),
    "yellow": ("yellow", "gold", "amber"),
    "purple": ("purple", "violet", "magenta"),
    "orange": ("orange", "amber"),
}


# --------------------------------------------------------------------------- #
# Scene description dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObjectInfo:
    """A manipulable object or a target region in the scene."""

    name: str
    kind: str  # "cube" | "bowl"
    color: str
    body: str
    geoms: tuple[str, ...]
    #: Cube half-extent, or bowl inner radius.
    radius: float
    #: Nominal resting height of the object's centre above the table.
    rest_z: float
    #: Only for regions: rim height above the table.
    rim_z: float = 0.0
    #: True when the object can be picked up.
    graspable: bool = True


@dataclass
class SceneInfo:
    """Everything the rest of the package needs to know about a built scene."""

    seed: int
    objects: dict[str, ObjectInfo]
    table_z: float = TABLE_Z
    bowl_name: str = "bowl"

    # -- convenience views -------------------------------------------------- #
    @property
    def cubes(self) -> list[str]:
        return [n for n, o in self.objects.items() if o.kind == "cube"]

    @property
    def regions(self) -> list[str]:
        return [n for n, o in self.objects.items() if not o.graspable]

    @property
    def names(self) -> list[str]:
        return list(self.objects)

    def color_of(self, name: str) -> str:
        return self.objects[name].color

    def by_color(self, color: str) -> list[str]:
        return [n for n, o in self.objects.items() if o.color == color and o.kind == "cube"]


@dataclass
class SceneConfig:
    """Knobs for :func:`build_scene`. Identical config + seed => identical scene."""

    seed: int = 0
    cube_colors: tuple[str, ...] = ("red", "green", "blue")
    randomize: bool = True
    #: Sampling window for cube centres, (lo, hi) in metres.
    cube_x_range: tuple[float, float] = (0.36, 0.58)
    cube_y_range: tuple[float, float] = (-0.28, 0.10)
    bowl_x_range: tuple[float, float] = (0.42, 0.54)
    bowl_y_range: tuple[float, float] = (0.22, 0.32)
    bowl_radius: float = 0.09
    bowl_rim: float = 0.05
    min_cube_separation: float = 0.10
    min_cube_bowl_clearance: float = 0.19
    timestep: float = 0.002
    gravity_compensation: bool = True
    #: Extra objects that exist in some scenes only, e.g. ("tray",). Unused in v1.
    extras: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Layout sampling
# --------------------------------------------------------------------------- #


def sample_layout(cfg: SceneConfig) -> dict[str, np.ndarray]:
    """Return ``{object_name: (x, y)}`` for the cubes and the bowl.

    Deterministic in ``cfg.seed``. Rejection-samples cube centres so that cubes
    are at least ``min_cube_separation`` apart and clear of the bowl.
    """
    rng = np.random.default_rng(cfg.seed)

    if not cfg.randomize:
        bowl = np.array([0.48, 0.27])
        xs = np.linspace(cfg.cube_x_range[0], cfg.cube_x_range[1], len(cfg.cube_colors))
        layout = {"bowl": bowl}
        for color, x in zip(cfg.cube_colors, xs):
            layout[f"{color}_cube"] = np.array([float(x), -0.12])
        return layout

    bowl = np.array(
        [rng.uniform(*cfg.bowl_x_range), rng.uniform(*cfg.bowl_y_range)], dtype=float
    )
    layout: dict[str, np.ndarray] = {"bowl": bowl}

    placed: list[np.ndarray] = []
    for color in cfg.cube_colors:
        for _ in range(500):
            p = np.array(
                [rng.uniform(*cfg.cube_x_range), rng.uniform(*cfg.cube_y_range)], dtype=float
            )
            if np.linalg.norm(p - bowl) < cfg.min_cube_bowl_clearance:
                continue
            if any(np.linalg.norm(p - q) < cfg.min_cube_separation for q in placed):
                continue
            placed.append(p)
            layout[f"{color}_cube"] = p
            break
        else:  # pragma: no cover - the window is wide enough that this never fires
            raise RuntimeError(f"could not place {color}_cube after 500 tries")
    return layout


# --------------------------------------------------------------------------- #
# Spec construction
# --------------------------------------------------------------------------- #


def _look_at(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``xyaxes`` for a camera at ``pos`` looking at ``target``."""
    z_cam = np.asarray(pos, float) - np.asarray(target, float)  # cameras look down -z
    z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(np.array([0.0, 0.0, 1.0]), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return np.concatenate([x_cam, y_cam])


def _add_visual_assets(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    mat = spec.add_material(name="groundplane", texuniform=True, texrepeat=[5, 5], reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    spec.add_material(name="table_mat", rgba=[0.72, 0.60, 0.45, 1.0], specular=0.2, shininess=0.1)
    spec.add_material(name="bowl_mat", rgba=[0.92, 0.92, 0.94, 1.0], specular=0.4, shininess=0.3)


def _add_table(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    world = spec.worldbody
    world.add_light(
        pos=[0.5, 0.0, 1.6],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.45, 0.45, 0.45],
        specular=[0.1, 0.1, 0.1],
    )
    # A soft fill from the camera side. Directional and shadowless on purpose:
    # a spot light here puts a hard bright patch on the table that reads as a
    # spill in recorded video.
    world.add_light(
        pos=[1.2, -0.8, 1.4],
        dir=[-0.6, 0.4, -0.8],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=False,
        diffuse=[0.25, 0.25, 0.25],
        specular=[0.05, 0.05, 0.05],
    )
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, FLOOR_Z],
        material="groundplane",
    )

    table = world.add_body(name="table", pos=[0.35, 0.0, 0.0])
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.50, 0.50, 0.02],
        pos=[0.0, 0.0, -0.02],
        material="table_mat",
        friction=[1.0, 0.03, 0.003],
        condim=3,
    )
    leg_h = (TABLE_Z - 0.04 - FLOOR_Z) / 2.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            table.add_geom(
                name=f"table_leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.025, 0.025, leg_h],
                pos=[sx * 0.45, sy * 0.45, -0.04 - leg_h],
                material="table_mat",
                contype=0,
                conaffinity=0,
            )


def _add_cameras(spec: mujoco.MjSpec) -> None:
    world = spec.worldbody
    focus = np.array([0.45, 0.02, 0.06])
    front_pos = np.array([1.45, 0.0, 0.62])
    world.add_camera(name="front", pos=front_pos, xyaxes=_look_at(front_pos, focus))
    side_pos = np.array([0.95, -0.95, 0.70])
    world.add_camera(name="side", pos=side_pos, xyaxes=_look_at(side_pos, focus))
    top_pos = np.array([0.45, 0.0, 1.25])
    world.add_camera(name="top", pos=top_pos, xyaxes=[1, 0, 0, 0, 1, 0])
    _ = top_pos


def _add_bowl(spec: mujoco.MjSpec, cfg: SceneConfig, xy: np.ndarray, segments: int = 24) -> ObjectInfo:
    """A static bowl: a disc base plus a ring of thin boxes for the wall.

    Static (welded to the world) on purpose -- in v1 the bowl is a *target
    region*, not something to be pushed around, and a fixed region makes the
    `place` postcondition a clean geometric test.
    """
    r, rim = cfg.bowl_radius, cfg.bowl_rim
    body = spec.worldbody.add_body(name="bowl", pos=[float(xy[0]), float(xy[1]), 0.0])
    geoms = ["bowl_base"]
    body.add_geom(
        name="bowl_base",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[r + 0.012, 0.005, 0.0],
        pos=[0.0, 0.0, 0.005],
        material="bowl_mat",
        friction=[1.0, 0.03, 0.003],
        condim=3,
    )
    seg_half = math.tan(math.pi / segments) * (r + 0.006) + 0.001
    for i in range(segments):
        ang = 2.0 * math.pi * i / segments
        name = f"bowl_wall_{i}"
        geoms.append(name)
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[seg_half, 0.006, rim / 2.0],
            pos=[(r + 0.006) * math.cos(ang), (r + 0.006) * math.sin(ang), rim / 2.0],
            euler=[0.0, 0.0, ang + math.pi / 2.0],
            material="bowl_mat",
            friction=[1.0, 0.03, 0.003],
            condim=3,
        )
    return ObjectInfo(
        name="bowl",
        kind="bowl",
        color="white",
        body="bowl",
        geoms=tuple(geoms),
        radius=r,
        rest_z=0.01,
        rim_z=rim,
        graspable=False,
    )


def _add_cube(spec: mujoco.MjSpec, color: str, xy: np.ndarray) -> ObjectInfo:
    name = f"{color}_cube"
    body = spec.worldbody.add_body(
        name=name, pos=[float(xy[0]), float(xy[1]), CUBE_HALF + 1e-4]
    )
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF] * 3,
        rgba=list(COLOR_RGBA[color]),
        mass=0.05,
        # High tangential friction so a closed gripper does not let the cube
        # creep out; solref softens the contact so the fingers do not bounce.
        friction=[1.2, 0.05, 0.005],
        condim=4,
        solref=[0.01, 1.0],
    )
    return ObjectInfo(
        name=name,
        kind="cube",
        color=color,
        body=name,
        geoms=(name,),
        radius=CUBE_HALF,
        rest_z=CUBE_HALF,
        graspable=True,
    )


def build_spec(
    cfg: SceneConfig | None = None, *, with_objects: bool = True
) -> tuple[mujoco.MjSpec, SceneInfo]:
    """Build the scene spec.

    ``with_objects=False`` returns the robot, table and cameras only. That model
    is what IK runs on, so the solver can never try to "reach" a target by
    sliding a free-floating cube.
    """
    cfg = cfg or SceneConfig()
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))

    # Menagerie's `home` keyframe no longer matches once we add free objects.
    for key in list(spec.keys):
        spec.delete(key)

    spec.option.timestep = cfg.timestep
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.ambient = [0.30, 0.30, 0.30]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.global_.azimuth = 150
    spec.visual.global_.elevation = -25
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    # Without gravity compensation the arm sags 6-8 mm below its IK target under
    # a pure position-servo controller; `02_ik_test.py` measures this.
    if cfg.gravity_compensation:
        for body_name in ARM_BODIES:
            spec.body(body_name).gravcomp = 1.0

    # The TCP site every IK target refers to.
    spec.body("hand").add_site(
        name="tcp", pos=[0.0, 0.0, TCP_OFFSET], size=[0.004] * 3, group=4
    )

    _add_visual_assets(spec)
    _add_table(spec, cfg)
    _add_cameras(spec)

    objects: dict[str, ObjectInfo] = {}
    if with_objects:
        layout = sample_layout(cfg)
        for color in cfg.cube_colors:
            name = f"{color}_cube"
            objects[name] = _add_cube(spec, color, layout[name])
        objects["bowl"] = _add_bowl(spec, cfg, layout["bowl"])

    info = SceneInfo(seed=cfg.seed, objects=objects)
    return spec, info


def _home_keyframe(spec: mujoco.MjSpec, info: SceneInfo) -> None:
    """Add a `home` keyframe sized for the compiled model (arm + every cube)."""
    qpos = list(HOME_QPOS) + [0.04, 0.04]
    for name in info.cubes:
        body = spec.body(info.objects[name].body)
        qpos += list(body.pos) + [1.0, 0.0, 0.0, 0.0]
    spec.add_key(
        name="home",
        qpos=np.array(qpos, dtype=float),
        ctrl=np.array(list(HOME_QPOS) + [GRIPPER_OPEN], dtype=float),
    )


def build_scene(
    cfg: SceneConfig | None = None, *, with_objects: bool = True
) -> tuple[mujoco.MjModel, mujoco.MjData, SceneInfo]:
    """Compile a scene and return ``(model, data, info)`` reset to the home pose."""
    cfg = cfg or SceneConfig()
    spec, info = build_spec(cfg, with_objects=with_objects)
    _home_keyframe(spec, info)
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_scene(model, data)
    return model, data, info


def reset_scene(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset to the `home` keyframe and settle the objects onto the table."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(int(0.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)


# --------------------------------------------------------------------------- #
# State queries used by skills, describe and the goal checks
# --------------------------------------------------------------------------- #


def object_pos(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> np.ndarray:
    """World position of a body's frame origin."""
    return np.array(data.body(body).xpos, dtype=float)


def object_quat(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> np.ndarray:
    return np.array(data.body(body).xquat, dtype=float)


def object_yaw(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> float:
    """Yaw of a body about +z, wrapped into [-pi/4, pi/4) for a 4-fold symmetric cube."""
    q = object_quat(model, data, body)
    yaw = math.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
    )
    quarter = math.pi / 2.0
    return (yaw + quarter / 2.0) % quarter - quarter / 2.0


def is_inside_region(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: SceneInfo,
    obj: str,
    region: str,
) -> bool:
    """True when ``obj``'s centre is within the region radius and below its rim."""
    reg = info.objects[region]
    o_pos = object_pos(model, data, info.objects[obj].body)
    r_pos = object_pos(model, data, reg.body)
    horizontal = float(np.linalg.norm(o_pos[:2] - r_pos[:2]))
    # bool(...) on purpose: numpy booleans are not JSON-serialisable and the
    # episode logs carry this value straight to disk.
    return bool(horizontal <= reg.radius and float(o_pos[2]) <= float(r_pos[2]) + reg.rim_z)


def is_on_table(model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo, obj: str) -> bool:
    o = info.objects[obj]
    z = float(object_pos(model, data, o.body)[2])
    return bool(abs(z - (info.table_z + o.rest_z)) < 0.01)
