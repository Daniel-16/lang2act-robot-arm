"""Executor behaviour: verification, retries, re-planning and the report."""

from __future__ import annotations

import json

from lang2act.executor import Executor, build_report, run_command
from lang2act.planner import MockBackend, Plan, Planner
from lang2act.skills import SkillResult


def _plan(**kwargs) -> Plan:
    return Plan.model_validate(
        {
            "status": "plan",
            "steps": [
                {"skill": "move_to", "target": "red_cube", "offset_z": 0.15},
                {"skill": "grasp", "target": "red_cube"},
                {"skill": "lift", "height": 0.15},
                {"skill": "place", "target": "bowl"},
                {"skill": "release"},
            ],
            **kwargs,
        }
    )


def test_executor_runs_and_verifies_every_step(robot, skills):
    records = Executor(robot, skills).run(_plan())
    assert len(records) == 5
    assert all(r.ok for r in records)
    assert records[-1].skill == "release"


def test_executor_stops_at_the_first_failure(robot, skills):
    plan = Plan.model_validate(
        {"status": "plan", "steps": [{"skill": "lift", "height": 0.1}, {"skill": "release"}]}
    )
    records = Executor(robot, skills, allow_replan=False).run(plan)
    assert len(records) == 1
    assert not records[0].ok and records[0].reason == "nothing_held"


def test_grasp_slip_is_retried_then_recovered(robot, skills):
    """The first squeeze slips; the re-approach fixes it (PRD F7)."""
    real_grasp = skills.grasp
    calls = {"n": 0}

    def flaky(target, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SkillResult(False, "grasp_slip", {"target": target})
        return real_grasp(target, **kwargs)

    skills.grasp = flaky
    records = Executor(robot, skills, allow_replan=False).run(_plan())
    assert calls["n"] == 2
    grasp_record = records[1]
    assert grasp_record.ok and grasp_record.retries == 1
    assert all(r.ok for r in records)


def test_retries_are_capped(robot, skills):
    skills.grasp = lambda target, **kw: SkillResult(False, "grasp_slip", {"target": target})
    executor = Executor(robot, skills, allow_replan=False, max_grasp_retries=2)
    records = executor.run(_plan())
    assert records[1].retries == 2
    assert not records[1].ok


def test_a_mid_plan_failure_triggers_exactly_one_replan(robot, skills, info):
    """Place fails once; the executor re-plans from the current state."""
    real_place = skills.place
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SkillResult(False, "place_blocked", {})
        return real_place(**kwargs)

    skills.place = flaky
    executor = Executor(robot, skills, planner=Planner(MockBackend(info)))
    records = executor.run(
        _plan(), command="put the red cube in the bowl", observation=""
    )
    assert executor.replanned
    assert any(r.attempt == 2 for r in records)


def test_final_state_reports_containment(robot, skills):
    executor = Executor(robot, skills, allow_replan=False)
    executor.run(_plan())
    state = executor.final_state()
    assert state["held"] is None
    assert state["objects"]["red_cube"]["in_bowl"] is True
    json.dumps(state)  # the episode log must serialise


# --------------------------------------------------------------------------- #
# Whole episodes
# --------------------------------------------------------------------------- #


def test_run_command_success_episode():
    result = run_command("put the red cube in the bowl", seed=4)
    assert result.outcome == "success" and result.ok
    assert result.final_state["objects"]["red_cube"]["in_bowl"] is True
    assert result.plan and len(result.plan["steps"]) == 5
    assert result.scene_before and result.scene_after
    assert "Outcome: success." in result.report
    json.dumps(result.to_dict(), default=str)


def test_ambiguous_command_asks_and_does_not_move():
    result = run_command("pick up the cube", seed=4)
    assert result.outcome == "clarify"
    assert result.question
    assert not result.robot_moved
    assert "ambiguous" in result.report


def test_infeasible_command_refuses_with_a_reason():
    result = run_command("put the purple cube in the bowl", seed=4)
    assert result.outcome == "infeasible"
    assert "purple" in result.failure_reason
    assert not result.robot_moved


def test_report_names_the_failing_step():
    from lang2act.executor import EpisodeResult, StepRecord

    result = EpisodeResult(
        command="x",
        seed=0,
        outcome="failure",
        failure_reason="grasp_slip",
        steps=[
            StepRecord(0, "move_to", {"target": "red_cube"}, True),
            StepRecord(1, "grasp", {"target": "red_cube"}, False, "grasp_slip", retries=2),
        ],
    )
    report = build_report(result)
    assert "failed at grasp" in report
    assert "the gripper closed but the cube was not held" in report
