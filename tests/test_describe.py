"""The scene describer: everything the planner is told must be true."""

from __future__ import annotations

import re

from lang2act.describe import describe_scene, full_observation, object_state, pairwise_relations
from lang2act.scene import object_pos
from lang2act.skills import Skills


def test_description_lists_every_object_with_its_colour(shared_robot):
    text = describe_scene(shared_robot)
    for name in shared_robot.info.names:
        assert name in text
    assert "red cube" in text and "bowl" in text
    assert "Contains: nothing" in text


def test_positions_in_the_text_match_the_simulator(shared_robot):
    text = describe_scene(shared_robot)
    for line in text.splitlines():
        m = re.match(r"- (\w+): .*x=([+-]\d+\.\d+), y=([+-]\d+\.\d+)", line)
        if not m:
            continue
        name, x, y = m.group(1), float(m.group(2)), float(m.group(3))
        pos = object_pos(shared_robot.model, shared_robot.data, shared_robot.info.objects[name].body)
        assert abs(pos[0] - x) < 1e-3 and abs(pos[1] - y) < 1e-3


def test_state_changes_once_a_cube_is_in_the_bowl(robot):
    skills = Skills(robot)
    assert object_state(robot, "red_cube") == "on the table"
    assert skills.move_to("red_cube", offset_z=0.15).ok
    assert skills.grasp("red_cube").ok
    assert object_state(robot, "red_cube", held="red_cube") == "held by the gripper"
    assert skills.lift(height=0.15).ok
    assert skills.place(target="bowl").ok
    assert skills.release().ok
    assert object_state(robot, "red_cube") == "inside the bowl"
    assert "Contains: red_cube" in describe_scene(robot)


def test_spatial_facts_are_ordered_consistently(shared_robot):
    text = pairwise_relations(shared_robot)
    left_to_right = re.search(r"decreasing y\): ([^\n]+)", text).group(1).split(", ")
    ys = [
        object_pos(shared_robot.model, shared_robot.data, shared_robot.info.objects[n].body)[1]
        for n in left_to_right
    ]
    assert ys == sorted(ys, reverse=True)


def test_full_observation_carries_the_frame_convention(shared_robot):
    text = full_observation(shared_robot)
    assert "+y is to the robot's left" in text
    assert "Spatial facts:" in text
