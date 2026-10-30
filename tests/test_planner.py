"""Plan validation (PRD F5) and the offline mock planner."""

from __future__ import annotations

import json

import pytest

from lang2act.planner import (
    MAX_STEPS,
    MockBackend,
    Plan,
    PlanError,
    Planner,
    extract_json,
    make_backend,
    validate_plan,
)

GOOD_PLAN = {
    "status": "plan",
    "question": None,
    "reason": None,
    "steps": [
        {"skill": "move_to", "target": "red_cube", "offset_z": 0.15},
        {"skill": "grasp", "target": "red_cube"},
        {"skill": "lift", "height": 0.15},
        {"skill": "place", "target": "bowl"},
        {"skill": "release"},
    ],
}


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} -- hope that helps') == {
        "a": {"b": 2}
    }
    assert extract_json('{"a": "a } brace in a string"}') == {"a": "a } brace in a string"}


def test_extract_json_rejects_junk():
    with pytest.raises(PlanError):
        extract_json("I would rather not")
    with pytest.raises(PlanError):
        extract_json('{"a": 1')


# --------------------------------------------------------------------------- #
# Schema and scene binding
# --------------------------------------------------------------------------- #


def test_valid_plan_round_trips(info):
    plan = validate_plan(json.dumps(GOOD_PLAN), info)
    assert plan.status == "plan" and len(plan.steps) == 5
    assert plan.steps[0].args() == {"target": "red_cube", "offset_z": 0.15}
    assert plan.steps[4].args() == {}


def test_unknown_object_is_rejected(info):
    bad = json.loads(json.dumps(GOOD_PLAN))
    bad["steps"][0]["target"] = "purple_cube"
    with pytest.raises(PlanError, match="unknown object"):
        validate_plan(bad, info)


def test_unknown_skill_is_rejected(info):
    bad = {"status": "plan", "steps": [{"skill": "teleport", "target": "red_cube"}]}
    with pytest.raises(PlanError, match="validation failed"):
        validate_plan(bad, info)


def test_extra_fields_are_rejected(info):
    bad = json.loads(json.dumps(GOOD_PLAN))
    bad["steps"][0]["joint_angles"] = [0.1] * 7
    with pytest.raises(PlanError, match="validation failed"):
        validate_plan(bad, info)


def test_grasping_a_region_is_rejected(info):
    bad = {"status": "plan", "steps": [{"skill": "grasp", "target": "bowl"}]}
    with pytest.raises(PlanError, match="cannot be picked up"):
        validate_plan(bad, info)


def test_place_needs_exactly_one_form_of_target(info):
    both = {
        "status": "plan",
        "steps": [
            {"skill": "place", "target": "bowl", "relation": "left_of", "reference": "red_cube"}
        ],
    }
    with pytest.raises(PlanError, match="not both"):
        validate_plan(both, info)

    neither = {"status": "plan", "steps": [{"skill": "place"}]}
    with pytest.raises(PlanError, match="requires"):
        validate_plan(neither, info)


def test_out_of_range_numbers_are_rejected(info):
    bad = {"status": "plan", "steps": [{"skill": "lift", "height": 5.0}]}
    with pytest.raises(PlanError, match="validation failed"):
        validate_plan(bad, info)


def test_step_budget(info):
    bad = {"status": "plan", "steps": [{"skill": "release"}] * (MAX_STEPS + 1)}
    with pytest.raises(PlanError, match="too many steps"):
        validate_plan(bad, info)


def test_clarify_needs_a_question_and_no_steps(info):
    with pytest.raises(PlanError, match="requires a non-empty 'question'"):
        validate_plan({"status": "clarify", "steps": []}, info)
    with pytest.raises(PlanError, match="must not carry steps"):
        validate_plan(
            {"status": "clarify", "question": "which?", "steps": GOOD_PLAN["steps"]}, info
        )
    plan = validate_plan({"status": "clarify", "question": "which cube?"}, info)
    assert plan.status == "clarify" and not plan.is_executable


def test_infeasible_needs_a_reason(info):
    with pytest.raises(PlanError, match="requires a non-empty 'reason'"):
        validate_plan({"status": "infeasible"}, info)
    plan = validate_plan({"status": "infeasible", "reason": "no purple cube"}, info)
    assert plan.reason == "no purple cube"


def test_empty_plan_is_rejected(info):
    with pytest.raises(PlanError, match="at least one step"):
        validate_plan({"status": "plan", "steps": []}, info)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def test_make_backend_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("LANG2ACT_LLM", raising=False)
    assert make_backend().name == "mock"
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_backend("telepathy")


@pytest.mark.parametrize(
    "command, status, moves, lands",
    [
        ("put the red cube in the bowl", "plan", "red_cube", "bowl"),
        ("drop the crimson block into the dish", "plan", "red_cube", "bowl"),
        ("put the blue cube to the left of the green cube", "plan", "blue_cube", None),
        ("pick up the cube", "clarify", None, None),
        ("put the purple cube in the bowl", "infeasible", None, None),
    ],
)
def test_mock_planner_covers_the_eval_categories(
    shared_robot, info, command, status, moves, lands
):
    from lang2act.describe import full_observation

    planner = Planner(MockBackend(info))
    plan, attempts = planner.plan(command, full_observation(shared_robot), info)
    assert plan is not None and plan.status == status
    assert attempts and attempts[-1].error is None
    if moves:
        assert plan.steps[0].target == moves
        if lands:
            assert plan.steps[3].target == lands
        else:
            assert plan.steps[3].relation == "left_of"


def test_planner_repairs_an_invalid_plan_once(info):
    """A rejected plan is fed its own error and asked again -- exactly once."""

    class FlakyBackend(MockBackend):
        def __init__(self):
            super().__init__(info)
            self.calls = 0

        def complete(self, system, user):
            self.calls += 1
            if self.calls == 1:
                from lang2act.planner import LLMResponse

                return LLMResponse(
                    text=json.dumps(
                        {"status": "plan", "steps": [{"skill": "grasp", "target": "unicorn"}]}
                    ),
                    model="flaky",
                )
            return super().complete(system, user)

    backend = FlakyBackend()
    plan, attempts = Planner(backend).plan("put the red cube in the bowl", "", info)
    assert backend.calls == 2
    assert len(attempts) == 2
    assert attempts[0].error and "unknown object" in attempts[0].error
    assert plan is not None and plan.status == "plan"


def test_planner_gives_up_after_the_repair_budget(info):
    class AlwaysBadBackend(MockBackend):
        def complete(self, system, user):
            from lang2act.planner import LLMResponse

            return LLMResponse(text="not json at all", model="bad")

    plan, attempts = Planner(AlwaysBadBackend(info)).plan("put the red cube in the bowl", "", info)
    assert plan is None
    assert len(attempts) == 2 and all(a.error for a in attempts)
