"""Phase 1 gate as a test: IK accuracy, reachability and tool orientation."""

from __future__ import annotations

import numpy as np
import pytest

from lang2act.robot import Robot, gripper_down_quat, min_jerk

#: The PRD's accuracy requirement (F2).
TOLERANCE_M = 0.005


def test_min_jerk_endpoints():
    assert min_jerk(0.0) == 0.0
    assert min_jerk(1.0) == pytest.approx(1.0)
    assert min_jerk(0.5) == pytest.approx(0.5)
    assert min_jerk(-1.0) == 0.0 and min_jerk(2.0) == pytest.approx(1.0)


def test_gripper_down_quat_points_the_tool_axis_down():
    import mujoco

    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, gripper_down_quat(0.0))
    assert np.allclose(mat.reshape(3, 3)[:, 2], [0, 0, -1], atol=1e-9)

    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, gripper_down_quat(np.pi / 2))
    assert np.allclose(mat.reshape(3, 3)[:, 2], [0, 0, -1], atol=1e-9)


@pytest.mark.parametrize(
    "target",
    [
        (0.45, 0.00, 0.10),
        (0.55, -0.20, 0.20),
        (0.38, 0.25, 0.30),
        (0.50, 0.30, 0.08),
    ],
)
def test_ik_solution_is_accurate(robot: Robot, target):
    sol = robot.solve_ik(target)
    assert sol.ok, f"IK did not converge: {sol}"
    assert sol.pos_error < TOLERANCE_M


def test_tracked_pose_is_within_tolerance(robot: Robot):
    """Solving is not enough -- the servo has to get there too (gravcomp)."""
    target = np.array([0.50, -0.10, 0.15])
    sol = robot.move_tcp(target)
    assert sol.ok
    assert np.linalg.norm(robot.tcp_pos - target) < TOLERANCE_M


def test_unreachable_target_is_reported_not_approximated(robot: Robot):
    sol = robot.solve_ik((1.8, 0.0, 0.2))
    assert not sol.ok
    assert sol.pos_error > TOLERANCE_M


def test_gripper_opens_and_closes(robot: Robot):
    robot.open_gripper()
    assert robot.gripper_width > 0.07
    robot.close_gripper()
    assert robot.gripper_width < 0.01
