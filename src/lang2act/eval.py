"""Evaluation runner: the frozen 30-command set, N seeds each, one table out.

    python -m lang2act.eval                  # full set, 3 seeds, mock planner
    python -m lang2act.eval --llm anthropic  # ... with a real model
    python -m lang2act.eval --category simple --seeds 1

Writes ``results/<run-id>/`` with one JSON log per episode, a machine-readable
``summary.json`` and the ``summary.md`` table that goes in the README.

The goal check is deliberately independent of the executor: an episode where
every step verified but the goal check fails is the most interesting row in the
table, because it means the *plan* was wrong, not the control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .executor import EpisodeResult, run_command
from .planner import Planner, make_backend

from .scene import SceneConfig, sample_layout

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMANDS = REPO_ROOT / "eval" / "commands.yaml"
DEFAULT_RESULTS = REPO_ROOT / "results"

CATEGORIES = ("simple", "paraphrase", "relational", "multi", "ambiguous", "infeasible")

#: Minimum displacement, in metres, for a relational goal to count as satisfied.
RELATION_MARGIN = 0.03

RELATION_AXIS = {
    # relation -> (axis index, sign the difference must have)
    "left_of": (1, +1),
    "right_of": (1, -1),
    "behind": (0, +1),
    "in_front_of": (0, -1),
}


# --------------------------------------------------------------------------- #
# Test-set loading
# --------------------------------------------------------------------------- #


@dataclass
class EvalCase:
    id: str
    command: str
    category: str
    seed: int
    goal: dict[str, Any]


def load_cases(path: Path = DEFAULT_COMMANDS) -> list[EvalCase]:
    raw = yaml.safe_load(Path(path).read_text())
    cases = [EvalCase(**entry) for entry in raw]
    ids = [c.id for c in cases]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate case ids in {path}: {duplicates}")
    unknown = sorted({c.category for c in cases} - set(CATEGORIES))
    if unknown:
        raise ValueError(f"unknown categories in {path}: {unknown}")
    return cases


def validate_cases(cases: list[EvalCase], seeds: int) -> None:
    """Reject test cases that are degenerate on any seed they will be run on.

    A relational goal whose resolved object *is* its reference ("put the red
    cube to the left of the red cube") can never be satisfied, and reads as a
    system failure in the results table. Catch it here, not in the write-up.
    """
    problems: list[str] = []
    for case in cases:
        if case.goal.get("type") != "relative":
            continue
        for k in range(seeds):
            seed = case.seed + k
            resolved = resolve_object(case.goal["object"], seed)
            if resolved == case.goal["reference"]:
                problems.append(
                    f"{case.id} (seed {seed}): goal object {case.goal['object']!r} "
                    f"resolves to {resolved}, which is also the reference"
                )
    if problems:
        raise ValueError("degenerate goals in the test set:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Goal checks
# --------------------------------------------------------------------------- #


def resolve_object(spec: str, seed: int) -> str:
    """Resolve ``nearest_to:bowl`` / ``farthest_from:bowl`` against the start layout.

    Scene generation is deterministic in the seed, so "the cube nearest the
    bowl" is well defined before the episode runs -- and must be, or the goal
    would move as the robot does.
    """
    if ":" not in spec:
        return spec
    kind, anchor = spec.split(":", 1)
    layout = sample_layout(SceneConfig(seed=seed))
    if anchor not in layout:
        raise ValueError(f"unknown anchor {anchor!r} in goal object {spec!r}")
    cubes = {k: v for k, v in layout.items() if k.endswith("_cube")}
    ranked = sorted(cubes, key=lambda n: float(np.linalg.norm(cubes[n] - layout[anchor])))
    if kind == "nearest_to":
        return ranked[0]
    if kind == "farthest_from":
        return ranked[-1]
    raise ValueError(f"unknown object specifier {spec!r}")


def check_goal(case: EvalCase, seed: int, result: EpisodeResult) -> tuple[bool, str]:
    """``(passed, detail)``. Reads only the episode's recorded final state."""
    goal = case.goal
    kind = goal["type"]
    objects = result.final_state.get("objects", {})

    if kind == "clarify":
        if result.outcome != "clarify":
            return False, f"expected a clarifying question, got {result.outcome}"
        if _anything_moved(seed, objects):
            return False, "a clarifying question was asked but the scene changed"
        return True, f'asked: "{result.question}"'

    if kind == "refuse":
        if result.outcome != "infeasible":
            return False, f"expected a refusal with a reason, got {result.outcome}"
        if _anything_moved(seed, objects):
            return False, "the command was refused but the scene changed"
        return True, f"refused: {result.failure_reason}"

    if result.outcome != "success":
        return False, f"episode outcome was {result.outcome}"

    if kind == "in_region":
        name, region = goal["object"], goal["region"]
        inside = bool(objects.get(name, {}).get(f"in_{region}", False))
        return inside, f"{name} in_{region}={inside}"

    if kind == "all_in_region":
        region = goal["region"]
        flags = {n: bool(e.get(f"in_{region}", False)) for n, e in objects.items()}
        missing = [n for n, v in flags.items() if not v]
        return not missing, ("all cubes in the bowl" if not missing else f"not in {region}: {missing}")

    if kind == "relative":
        name = resolve_object(goal["object"], seed)
        ref = goal["reference"]
        axis, sign = RELATION_AXIS[goal["relation"]]
        try:
            moved = np.array(objects[name]["pos"], dtype=float)
            anchor = np.array(objects[ref]["pos"], dtype=float)
        except KeyError:
            return False, f"missing final pose for {name} or {ref}"
        delta = sign * (moved[axis] - anchor[axis])
        detail = f"{name} vs {ref}: signed offset {delta:+.3f} m (need > {RELATION_MARGIN})"
        return bool(delta > RELATION_MARGIN), detail

    raise ValueError(f"unknown goal type {kind!r}")


def _anything_moved(seed: int, objects: dict[str, Any], tol: float = 0.02) -> bool:
    """Did any cube leave its starting position?

    Used to enforce "the robot did not move" on the ambiguous and infeasible
    categories -- a refusal that nudges a cube is not a refusal.
    """
    layout = sample_layout(SceneConfig(seed=seed))
    for name, entry in objects.items():
        start = layout.get(name)
        if start is None:
            continue
        pos = np.array(entry["pos"], dtype=float)[:2]
        if float(np.linalg.norm(pos - start)) > tol:
            return True
    return False


# --------------------------------------------------------------------------- #
# Layer attribution
# --------------------------------------------------------------------------- #


def failure_layer(case: EvalCase, result: EpisodeResult, goal_ok: bool) -> str:
    """planner | controller | perception | "" -- which layer got it wrong.

    *planner*     a schema-invalid plan, a plan that names the wrong object, or
                  a wrong status (executed an ambiguous command, refused a
                  feasible one).
    *controller*  a good plan the arm failed to carry out.
    *perception*  the scene description was wrong. In v1 this is 0 by
                  construction -- the describer and the goal check read the same
                  simulator state -- and only becomes meaningful with vision.
    """
    if goal_ok:
        return ""
    if result.outcome == "invalid_plan":
        return "planner"
    if result.outcome in ("clarify", "infeasible"):
        # Refusing or querying something the test set says is doable.
        return "planner"
    if result.outcome == "failure":
        from .executor import FAILURE_LAYER

        return FAILURE_LAYER.get(result.failure_reason, "controller")
    # Every step verified, yet the goal was not met: the plan was wrong.
    return "planner"


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeRow:
    case_id: str
    category: str
    command: str
    seed: int
    outcome: str
    goal_ok: bool
    goal_detail: str
    failure_reason: str
    layer: str
    retries: int
    steps: int
    wall_time_s: float
    tokens: dict[str, int] = field(default_factory=dict)
    log: str = ""


def run_eval(
    cases: list[EvalCase],
    *,
    seeds: int = 3,
    llm: str | None = None,
    out_dir: Path = DEFAULT_RESULTS,
    run_id: str | None = None,
    video_first_seed: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run every case on ``seeds`` seeds and write the report. Returns the summary."""
    validate_cases(cases, seeds)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / run_id
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    if video_first_seed:
        (run_dir / "videos").mkdir(exist_ok=True)

    planner = Planner(make_backend(llm))
    rows: list[EpisodeRow] = []
    t_start = time.perf_counter()

    for case in cases:
        for k in range(seeds):
            seed = case.seed + k
            video = (
                run_dir / "videos" / f"{case.id}_seed{seed}.mp4"
                if video_first_seed and k == 0
                else None
            )
            result = run_command(case.command, seed=seed, planner=planner, video=video)
            goal_ok, detail = check_goal(case, seed, result)
            layer = failure_layer(case, result, goal_ok)

            log_path = run_dir / "episodes" / f"{case.id}_seed{seed}.json"
            payload = result.to_dict()
            payload.update(
                {
                    "case_id": case.id,
                    "category": case.category,
                    "goal": case.goal,
                    "goal_ok": goal_ok,
                    "goal_detail": detail,
                    "failure_layer_attributed": layer,
                }
            )
            log_path.write_text(json.dumps(payload, indent=2, default=str))

            row = EpisodeRow(
                case_id=case.id,
                category=case.category,
                command=case.command,
                seed=seed,
                outcome=result.outcome,
                goal_ok=goal_ok,
                goal_detail=detail,
                failure_reason=result.failure_reason,
                layer=layer,
                retries=sum(s.retries for s in result.steps),
                steps=len(result.steps),
                wall_time_s=result.wall_time_s,
                tokens=result.tokens,
                log=str(log_path.relative_to(run_dir)),
            )
            rows.append(row)
            if not quiet:
                mark = "PASS" if goal_ok else "FAIL"
                print(
                    f"[{mark}] {case.id:10s} seed={seed:<3d} {case.category:11s} "
                    f"{result.outcome:12s} {detail}",
                    flush=True,
                )

    summary = summarise(rows, run_id=run_id, planner=planner, seeds=seeds)
    summary["wall_time_s"] = round(time.perf_counter() - t_start, 1)
    (run_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "episodes": [vars(r) for r in rows]}, indent=2)
    )
    (run_dir / "summary.md").write_text(render_markdown(summary, rows))
    summary["run_dir"] = str(run_dir)
    return summary


def summarise(
    rows: list[EpisodeRow], *, run_id: str, planner: Planner, seeds: int
) -> dict[str, Any]:
    total = len(rows)
    passed = sum(r.goal_ok for r in rows)
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "pass": 0})
    for r in rows:
        by_cat[r.category]["n"] += 1
        by_cat[r.category]["pass"] += int(r.goal_ok)

    layers = Counter(r.layer for r in rows if r.layer)
    reasons = Counter(r.failure_reason for r in rows if not r.goal_ok and r.failure_reason)
    plans_valid = sum(1 for r in rows if r.outcome != "invalid_plan")

    return {
        "run_id": run_id,
        "model": getattr(planner.backend, "model", "") or planner.backend.name,
        "backend": planner.backend.name,
        "seeds_per_command": seeds,
        "commands": len({r.case_id for r in rows}),
        "episodes": total,
        "passed": passed,
        "success_rate": round(passed / total, 4) if total else 0.0,
        "plan_validity_rate": round(plans_valid / total, 4) if total else 0.0,
        "by_category": {
            c: {
                "n": v["n"],
                "pass": v["pass"],
                "rate": round(v["pass"] / v["n"], 4) if v["n"] else 0.0,
            }
            for c, v in sorted(by_cat.items(), key=lambda kv: CATEGORIES.index(kv[0]))
        },
        "failures_by_layer": dict(layers),
        "failures_by_reason": dict(reasons),
        "grasp_retries": sum(r.retries for r in rows),
        "retries_recovered": sum(r.retries for r in rows if r.goal_ok),
        "mean_wall_time_s": round(sum(r.wall_time_s for r in rows) / total, 2) if total else 0.0,
        "tokens": {
            "input": sum(r.tokens.get("input_tokens", 0) for r in rows),
            "output": sum(r.tokens.get("output_tokens", 0) for r in rows),
        },
    }


def render_markdown(summary: dict[str, Any], rows: list[EpisodeRow]) -> str:
    lines = [
        f"# Lang2Act evaluation -- run `{summary['run_id']}`",
        "",
        f"- Planner: `{summary['backend']}` / `{summary['model']}`",
        f"- {summary['commands']} commands x {summary['seeds_per_command']} seeds "
        f"= {summary['episodes']} episodes",
        f"- **End-to-end success: {summary['passed']}/{summary['episodes']} "
        f"({summary['success_rate'] * 100:.1f}%)**",
        f"- Schema-valid plans: {summary['plan_validity_rate'] * 100:.1f}%",
        f"- Grasp retries used: {summary['grasp_retries']} "
        f"(episodes that still passed: {summary['retries_recovered']})",
        f"- Mean wall time per episode: {summary['mean_wall_time_s']} s",
        "",
        "## Success by category",
        "",
        "| Category | Episodes | Passed | Rate |",
        "| --- | --- | --- | --- |",
    ]
    for cat, v in summary["by_category"].items():
        lines.append(f"| {cat} | {v['n']} | {v['pass']} | {v['rate'] * 100:.1f}% |")

    lines += ["", "## Failure breakdown by layer", "", "| Layer | Failures |", "| --- | --- |"]
    if summary["failures_by_layer"]:
        for layer, n in sorted(summary["failures_by_layer"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {layer} | {n} |")
    else:
        lines.append("| (none) | 0 |")

    if summary["failures_by_reason"]:
        lines += ["", "## Failure reasons", "", "| Reason | Count |", "| --- | --- |"]
        for reason, n in sorted(summary["failures_by_reason"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{reason}` | {n} |")

    failed = [r for r in rows if not r.goal_ok]
    if failed:
        lines += [
            "",
            "## Failed episodes",
            "",
            "| Case | Seed | Command | Outcome | Why |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in failed:
            lines.append(
                f"| {r.case_id} | {r.seed} | {r.command} | {r.outcome} | {r.goal_detail} |"
            )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lang2act.eval",
        description="Run the frozen Lang2Act command set and write the results table.",
    )
    p.add_argument("--commands", type=Path, default=DEFAULT_COMMANDS, help="test-set YAML")
    p.add_argument("--seeds", type=int, default=3, help="seeds per command (default 3)")
    p.add_argument("--llm", default=None, help="provider[:model], e.g. anthropic:claude-opus-5-5")
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS, help="results directory")
    p.add_argument("--run-id", default=None, help="name the run folder")
    p.add_argument("--category", action="append", choices=CATEGORIES, help="only these categories")
    p.add_argument("--id", action="append", dest="ids", help="only these case ids")
    p.add_argument("--video", action="store_true", help="record an MP4 of the first seed")
    p.add_argument("--quiet", action="store_true", help="suppress the per-episode lines")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_cases(args.commands)
    if args.category:
        cases = [c for c in cases if c.category in args.category]
    if args.ids:
        cases = [c for c in cases if c.id in args.ids]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    summary = run_eval(
        cases,
        seeds=args.seeds,
        llm=args.llm,
        out_dir=args.out,
        run_id=args.run_id,
        video_first_seed=args.video,
        quiet=args.quiet,
    )
    print()
    print(Path(summary["run_dir"], "summary.md").read_text())
    print(f"Logs and table: {summary['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
