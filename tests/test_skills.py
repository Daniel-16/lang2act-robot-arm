"""Every skill's success path and its named failure paths (PRD F3)."""

from __future__ import annotations

import numpy as np
import pytest

from lang2act.robot import Robot
from lang2act.scene import is_inside_region
from lang2act.skills import RELATIONS, Skills


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_move_to_puts_the_tcp_above_the_target(skills: Skills, robot: Robot):
    result = skills.move_to("red_cube", offset_z=0.15)
    assert result.ok, result.reason
    cube = robot.body_pos("red_cube")
    assert np.linalg.norm(robot.tcp_pos[:2] - cube[:2]) < 0.01
    assert robot.tcp_pos[2] == pytest.approx(cube[2] + 0.15, abs=0.01)


def test_grasp_then_lift_then_place_then_release(skills: Skills, robot: Robot):
    assert skills.move_to("red_cube", offset_z=0.15).ok
    assert skills.grasp("red_cube").ok
    assert skills.held == "red_cube"
    assert robot.is_grasping("red_cube")

    assert skills.lift(height=0.15).ok
    assert robot.body_pos("red_cube")[2] > 0.10

    assert skills.place(target="bowl").ok
    assert skills.release().ok
    assert skills.held is None
    assert is_inside_region(robot.model, robot.data, robot.info, "red_cube", "bowl")


def test_relational_place_moves_the_cube_to_the_left(skills: Skills, robot: Robot):
    reference = robot.body_pos("green_cube").copy()
    assert skills.move_to("blue_cube", offset_z=0.15).ok
    assert skills.grasp("blue_cube").ok
    assert skills.lift(height=0.15).ok
    assert skills.place(relation="left_of", reference="green_cube").ok
    assert skills.release().ok
    moved = robot.body_pos("blue_cube")
    assert moved[1] > reference[1] + 0.05  # +y is the robot's left


def test_relations_are_opposite_pairs():
    assert RELATIONS["left_of"][1] == -RELATIONS["right_of"][1]
    assert RELATIONS["behind"][0] == -RELATIONS["in_front_of"][0]


# --------------------------------------------------------------------------- #
# Failure paths -- each returns a named reason, never an exception
# --------------------------------------------------------------------------- #


def test_move_to_unknown_object(skills: Skills):
    result = skills.move_to("purple_cube")
    assert not result.ok and result.reason == "unknown_object"


def test_grasp_rejects_a_region(skills: Skills):
    result = skills.grasp("bowl")
    assert not result.ok and result.reason == "not_graspable"


def test_grasp_while_holding(skills: Skills):
    assert skills.move_to("red_cube", offset_z=0.15).ok
    assert skills.grasp("red_cube").ok
    result = skills.grasp("green_cube")
    assert not result.ok and result.reason == "already_holding"


def test_lift_place_release_need_something_held(skills: Skills):
    assert skills.lift().reason == "nothing_held"
    assert skills.place(target="bowl").reason == "nothing_held"
    assert skills.release().reason == "nothing_held"


def test_an_unreachable_pose_fails_rather_than_approximating(skills: Skills):
    """Every skill goes through `_goto`, so this is the shared failure path."""
    result = skills._goto(np.array([1.6, 0.0, 0.2]))
    assert not result.ok and result.reason == "unreachable"


def test_place_outside_the_workspace_is_blocked(skills: Skills, robot: Robot):
    assert skills.move_to("red_cube", offset_z=0.15).ok
    assert skills.grasp("red_cube").ok
    assert skills.lift(height=0.15).ok
    # Pretend the reference sits far beyond the arm's reach.
    original = skills._pos
    skills._pos = lambda name: (
        np.array([1.5, 0.0, 0.02]) if name == "green_cube" else original(name)
    )
    try:
        result = skills.place(relation="left_of", reference="green_cube")
    finally:
        skills._pos = original
    assert not result.ok and result.reason == "place_blocked"


def test_grasp_reports_a_slip_when_the_jaws_close_on_nothing(
    skills: Skills, robot: Robot
):
    """Move the cube away between the approach and the squeeze."""
    original = skills._pos
    cube_pos = robot.body_pos("red_cube").copy()
    calls = {"n": 0}

    def shifted(name: str):
        # Report the true pose for the approach, then a pose 8 cm to the side,
        # which is exactly what a cube nudged by the fingers looks like.
        if name == "red_cube":
            calls["n"] += 1
            if calls["n"] > 2:
                return cube_pos + np.array([0.0, 0.08, 0.0])
        return original(name)

    skills._pos = shifted
    try:
        assert skills.move_to("red_cube", offset_z=0.15).ok
        result = skills.grasp("red_cube")
    finally:
        skills._pos = original
    assert not result.ok and result.reason == "grasp_slip"
    assert skills.held is None


def test_unknown_skill_name(skills: Skills):
    result = skills.run("teleport", target="red_cube")
    assert not result.ok and result.reason == "unknown_skill"


def test_skill_result_is_json_safe(skills: Skills):
    import json

    result = skills.move_to("red_cube", offset_z=0.15)
    json.dumps(result.as_dict())  # must not raise on numpy arrays
