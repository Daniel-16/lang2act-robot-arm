"""Shared fixtures. Building a scene costs ~1 s, so sessions reuse one robot."""

from __future__ import annotations

import pytest

from lang2act.robot import Robot
from lang2act.scene import SceneConfig, reset_scene
from lang2act.skills import Skills

TEST_SEED = 2


@pytest.fixture(scope="session")
def cfg() -> SceneConfig:
    return SceneConfig(seed=TEST_SEED)


@pytest.fixture(scope="session")
def shared_robot(cfg: SceneConfig) -> Robot:
    robot = Robot.from_config(cfg)
    yield robot
    robot.close()


@pytest.fixture
def robot(shared_robot: Robot) -> Robot:
    """A robot reset to the home pose before every test."""
    reset_scene(shared_robot.model, shared_robot.data)
    shared_robot.data.ctrl[:7] = shared_robot.arm_qpos
    shared_robot.data.ctrl[shared_robot.gripper_act] = 255.0
    return shared_robot


@pytest.fixture
def skills(robot: Robot) -> Skills:
    return Skills(robot)


@pytest.fixture
def info(shared_robot: Robot):
    return shared_robot.info
