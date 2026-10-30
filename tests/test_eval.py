"""The goal checks and the test set itself."""

from __future__ import annotations

import numpy as np
import pytest

from lang2act.eval import (
    CATEGORIES,
    DEFAULT_COMMANDS,
    EvalCase,
    check_goal,
    failure_layer,
    load_cases,
    render_markdown,
    resolve_object,
)
from lang2act.executor import EpisodeResult
from lang2act.scene import SceneConfig, sample_layout


# --------------------------------------------------------------------------- #
# The frozen set
# --------------------------------------------------------------------------- #


def test_the_test_set_loads_and_is_the_documented_shape():
    cases = load_cases(DEFAULT_COMMANDS)
    assert len(cases) == 30
    counts = {c: sum(1 for x in cases if x.category == c) for c in CATEGORIES}
    assert counts == {
        "simple": 8,
        "paraphrase": 6,
        "relational": 6,
        "multi": 4,
        "ambiguous": 3,
        "infeasible": 3,
    }
    assert all(c.command and c.goal.get("type") for c in cases)


def test_every_goal_type_is_one_the_checker_knows():
    known = {"in_region", "all_in_region", "relative", "clarify", "refuse"}
    assert {c.goal["type"] for c in load_cases(DEFAULT_COMMANDS)} <= known


# --------------------------------------------------------------------------- #
# Object specifiers
# --------------------------------------------------------------------------- #


def test_resolve_object_passes_literal_names_through():
    assert resolve_object("red_cube", 1) == "red_cube"


def test_nearest_and_farthest_are_resolved_against_the_start_layout():
    seed = 15
    layout = sample_layout(SceneConfig(seed=seed))
    cubes = {k: v for k, v in layout.items() if k.endswith("_cube")}
    ranked = sorted(cubes, key=lambda n: float(np.linalg.norm(cubes[n] - layout["bowl"])))
    assert resolve_object("nearest_to:bowl", seed) == ranked[0]
    assert resolve_object("farthest_from:bowl", seed) == ranked[-1]


def test_unknown_specifier_raises():
    with pytest.raises(ValueError):
        resolve_object("adjacent_to:bowl", 1)


# --------------------------------------------------------------------------- #
# Goal checks
# --------------------------------------------------------------------------- #


def _state(seed: int, **overrides) -> dict:
    """Final state matching the start layout, with the given cubes overridden."""
    layout = sample_layout(SceneConfig(seed=seed))
    objects = {
        name: {"pos": [float(p[0]), float(p[1]), 0.02], "in_bowl": False}
        for name, p in layout.items()
        if name.endswith("_cube")
    }
    for name, entry in overrides.items():
        objects[name] = entry
    return {"held": None, "objects": objects}


def _episode(seed: int, outcome: str, state: dict, **kwargs) -> EpisodeResult:
    return EpisodeResult(
        command="x", seed=seed, outcome=outcome, final_state=state, **kwargs
    )


def test_in_region_goal():
    case = EvalCase("t", "x", "simple", 1, {"type": "in_region", "object": "red_cube", "region": "bowl"})
    state = _state(1, red_cube={"pos": [0.5, 0.3, 0.03], "in_bowl": True})
    ok, _ = check_goal(case, 1, _episode(1, "success", state))
    assert ok

    state = _state(1)
    ok, detail = check_goal(case, 1, _episode(1, "success", state))
    assert not ok and "in_bowl=False" in detail


def test_in_region_goal_requires_a_successful_episode():
    case = EvalCase("t", "x", "simple", 1, {"type": "in_region", "object": "red_cube", "region": "bowl"})
    state = _state(1, red_cube={"pos": [0.5, 0.3, 0.03], "in_bowl": True})
    ok, detail = check_goal(case, 1, _episode(1, "failure", state))
    assert not ok and "outcome was failure" in detail


def test_all_in_region_goal():
    case = EvalCase("t", "x", "multi", 2, {"type": "all_in_region", "region": "bowl"})
    layout = sample_layout(SceneConfig(seed=2))
    full = {
        "held": None,
        "objects": {
            n: {"pos": [0.5, 0.3, 0.03], "in_bowl": True}
            for n in layout
            if n.endswith("_cube")
        },
    }
    ok, _ = check_goal(case, 2, _episode(2, "success", full))
    assert ok

    partial = {"held": None, "objects": dict(full["objects"])}
    partial["objects"]["blue_cube"] = {"pos": [0.4, 0.0, 0.02], "in_bowl": False}
    ok, detail = check_goal(case, 2, _episode(2, "success", partial))
    assert not ok and "blue_cube" in detail


@pytest.mark.parametrize(
    "relation, offset, expected",
    [
        ("left_of", (0.0, +0.12), True),
        ("left_of", (0.0, -0.12), False),
        ("left_of", (0.0, +0.01), False),  # inside the margin
        ("right_of", (0.0, -0.12), True),
        ("behind", (+0.12, 0.0), True),
        ("in_front_of", (-0.12, 0.0), True),
    ],
)
def test_relative_goal_checks_the_sign_and_the_margin(relation, offset, expected):
    case = EvalCase(
        "t", "x", "relational", 3,
        {"type": "relative", "object": "blue_cube", "relation": relation, "reference": "green_cube"},
    )
    state = {
        "held": None,
        "objects": {
            "green_cube": {"pos": [0.45, 0.0, 0.02], "in_bowl": False},
            "blue_cube": {"pos": [0.45 + offset[0], 0.0 + offset[1], 0.02], "in_bowl": False},
        },
    }
    ok, _ = check_goal(case, 3, _episode(3, "success", state))
    assert ok is expected


def test_clarify_goal_requires_a_question_and_a_still_scene():
    case = EvalCase("t", "x", "ambiguous", 5, {"type": "clarify"})
    ok, _ = check_goal(case, 5, _episode(5, "clarify", _state(5), question="which cube?"))
    assert ok

    ok, detail = check_goal(case, 5, _episode(5, "success", _state(5)))
    assert not ok and "expected a clarifying question" in detail

    moved = _state(5, red_cube={"pos": [0.5, 0.35, 0.03], "in_bowl": True})
    ok, detail = check_goal(case, 5, _episode(5, "clarify", moved, question="which?"))
    assert not ok and "scene changed" in detail


def test_refuse_goal_requires_a_refusal_and_a_still_scene():
    case = EvalCase("t", "x", "infeasible", 6, {"type": "refuse"})
    ok, _ = check_goal(
        case, 6, _episode(6, "infeasible", _state(6), failure_reason="no purple cube")
    )
    assert ok

    moved = _state(6, red_cube={"pos": [0.5, 0.35, 0.03], "in_bowl": True})
    ok, detail = check_goal(case, 6, _episode(6, "infeasible", moved))
    assert not ok and "scene changed" in detail


# --------------------------------------------------------------------------- #
# Layer attribution and reporting
# --------------------------------------------------------------------------- #


def test_layer_attribution():
    case = EvalCase("t", "x", "simple", 1, {"type": "in_region", "object": "red_cube", "region": "bowl"})
    assert failure_layer(case, _episode(1, "success", {}), True) == ""
    assert failure_layer(case, _episode(1, "invalid_plan", {}), False) == "planner"
    assert failure_layer(case, _episode(1, "clarify", {}), False) == "planner"
    assert (
        failure_layer(case, _episode(1, "failure", {}, failure_reason="grasp_slip"), False)
        == "controller"
    )
    assert (
        failure_layer(case, _episode(1, "failure", {}, failure_reason="unknown_object"), False)
        == "planner"
    )
    # Every step verified, goal still not met: the plan was wrong.
    assert failure_layer(case, _episode(1, "success", {}), False) == "planner"


def test_markdown_report_contains_the_headline_number():
    summary = {
        "run_id": "test",
        "backend": "mock",
        "model": "mock-rules-v1",
        "commands": 30,
        "seeds_per_command": 3,
        "episodes": 90,
        "passed": 81,
        "success_rate": 0.9,
        "plan_validity_rate": 1.0,
        "grasp_retries": 4,
        "retries_recovered": 3,
        "mean_wall_time_s": 1.1,
        "by_category": {"simple": {"n": 24, "pass": 24, "rate": 1.0}},
        "failures_by_layer": {"controller": 9},
        "failures_by_reason": {"grasp_slip": 9},
    }
    md = render_markdown(summary, [])
    assert "81/90 (90.0%)" in md
    assert "| controller | 9 |" in md


def test_degenerate_relational_goals_are_rejected():
    """A goal that resolves to its own reference is a broken case, not a failure."""
    from lang2act.eval import validate_cases

    good = EvalCase(
        "ok", "x", "relational", 34,
        {"type": "relative", "object": "farthest_from:bowl", "relation": "left_of",
         "reference": "red_cube"},
    )
    validate_cases([good], seeds=3)

    bad = EvalCase(
        "bad", "x", "relational", 21,
        {"type": "relative", "object": "farthest_from:bowl", "relation": "left_of",
         "reference": "red_cube"},
    )
    with pytest.raises(ValueError, match="degenerate goals"):
        validate_cases([bad], seeds=1)


def test_the_shipped_test_set_has_no_degenerate_goals():
    from lang2act.eval import validate_cases

    validate_cases(load_cases(DEFAULT_COMMANDS), seeds=3)
