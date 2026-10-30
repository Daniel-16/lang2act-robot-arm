"""Simulator state -> text, for the planner's prompt.

This is the perception layer stand-in for v1: object poses come from the
simulator rather than from a camera. Keeping it behind one function means
swapping in real vision later is a change to this module only -- and it means
the eval can attribute a failure to *perception* (the description was wrong)
separately from *planning* and *control*.
"""

from __future__ import annotations

import numpy as np

from .scene import SceneInfo, is_inside_region, object_pos
from .robot import Robot

FRAME_NOTE = (
    "Coordinates are in metres on the table surface (z = 0 is the table top). "
    "The robot base is at x=0, y=0. +x points away from the base, "
    "+y is to the robot's left, +z is up."
)


def object_state(robot: Robot, name: str, *, held: str | None = None) -> str:
    """One of ``held``, ``in the bowl``, ``on the table``."""
    info = robot.info
    if held == name:
        return "held by the gripper"
    for region in info.regions:
        if is_inside_region(robot.model, robot.data, info, name, region):
            return f"inside the {region}"
    z = float(object_pos(robot.model, robot.data, info.objects[name].body)[2])
    if z > info.table_z + info.objects[name].rest_z + 0.02:
        return "in the air"
    return "on the table"


def describe_scene(robot: Robot, *, held: str | None = None, include_frame: bool = True) -> str:
    """A compact, deterministic English description of the current scene."""
    info: SceneInfo = robot.info
    lines: list[str] = []
    if include_frame:
        lines.append(FRAME_NOTE)
    lines.append("Objects:")

    for name in info.cubes:
        o = info.objects[name]
        p = object_pos(robot.model, robot.data, o.body)
        lines.append(
            f"- {name}: a {o.color} cube {o.radius * 200:.0f} cm on a side, "
            f"at x={p[0]:+.3f}, y={p[1]:+.3f}, z={p[2]:+.3f}, "
            f"{object_state(robot, name, held=held)}"
        )
    for name in info.regions:
        o = info.objects[name]
        p = object_pos(robot.model, robot.data, o.body)
        contents = [
            c for c in info.cubes if is_inside_region(robot.model, robot.data, info, c, name)
        ]
        lines.append(
            f"- {name}: a {o.color} {o.kind}, inner radius {o.radius * 100:.0f} cm, "
            f"rim {o.rim_z * 100:.0f} cm high, at x={p[0]:+.3f}, y={p[1]:+.3f}. "
            f"Contains: {', '.join(contents) if contents else 'nothing'}"
        )

    tcp = robot.tcp_pos
    holding = held if held is not None else "nothing"
    lines.append(
        f"Gripper: holding {holding}, jaw opening {robot.gripper_width * 100:.1f} cm, "
        f"tool centre point at x={tcp[0]:+.3f}, y={tcp[1]:+.3f}, z={tcp[2]:+.3f}."
    )
    return "\n".join(lines)


def describe_objects(info: SceneInfo) -> str:
    """One line listing the legal object names, for the prompt's enum."""
    return ", ".join(info.names)


def pairwise_relations(robot: Robot) -> str:
    """Relational facts a planner would otherwise have to derive from coordinates.

    Supplying these explicitly is what lifts the "relational" eval category from
    coin-flip to reliable: LLMs are poor at comparing signed floats, and the
    comparison is cheap and exact to do here.
    """
    info = robot.info
    cubes = info.cubes
    if not cubes:
        return ""
    pos = {n: object_pos(robot.model, robot.data, info.objects[n].body) for n in info.names}
    lines = ["Spatial facts:"]

    for region in info.regions:
        ranked = sorted(cubes, key=lambda c: float(np.linalg.norm(pos[c][:2] - pos[region][:2])))
        dists = ", ".join(
            f"{c} {np.linalg.norm(pos[c][:2] - pos[region][:2]):.3f} m" for c in ranked
        )
        lines.append(f"- distance to {region}, nearest first: {dists}")

    left_to_right = sorted(cubes, key=lambda c: -float(pos[c][1]))
    lines.append(
        "- from the robot's left to its right (decreasing y): " + ", ".join(left_to_right)
    )
    far_to_near = sorted(cubes, key=lambda c: -float(pos[c][0]))
    lines.append("- from farthest to nearest the base (decreasing x): " + ", ".join(far_to_near))
    return "\n".join(lines)


def full_observation(robot: Robot, *, held: str | None = None) -> str:
    """The exact text handed to the planner."""
    parts = [describe_scene(robot, held=held), pairwise_relations(robot)]
    return "\n".join(p for p in parts if p)
