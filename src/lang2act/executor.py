"""Plan executor: run the steps, verify each one, retry, report.

The executor is the arbiter. It trusts nothing:

* the plan is validated before the first step runs,
* every step's postcondition is checked against simulator state,
* a slipped grasp is re-approached and retried (up to ``max_grasp_retries``),
* any other mid-plan failure triggers exactly one re-plan from current state,
* whatever happens, the episode log holds enough to replay it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .describe import full_observation
from .planner import Plan, PlanAttempt, Planner, Step
from .robot import Robot, VideoHook, make_hooks
from .scene import SceneConfig, is_inside_region
from .skills import SkillResult, Skills

#: Failure reasons a re-approach and another squeeze can fix.
RETRYABLE_GRASP = {"grasp_slip", "object_dropped"}

#: Which layer a named failure reason belongs to, for the eval breakdown.
FAILURE_LAYER = {
    "unknown_object": "planner",
    "not_graspable": "planner",
    "already_holding": "planner",
    "nothing_held": "planner",
    "unknown_skill": "planner",
    "unreachable": "controller",
    "motion_error": "controller",
    "grasp_slip": "controller",
    "object_dropped": "controller",
    "place_blocked": "controller",
    "not_in_target": "controller",
    "object_stuck": "controller",
    "invalid_plan": "planner",
}


@dataclass
class StepRecord:
    """One executed step and what the simulator said about it."""

    index: int
    skill: str
    args: dict[str, Any]
    ok: bool
    reason: str = ""
    retries: int = 0
    wall_time_s: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1  # 1 = original plan, 2 = after a re-plan


@dataclass
class EpisodeResult:
    """Everything about one command, written to ``results/.../episodes/*.json``."""

    command: str
    seed: int
    outcome: str  # success | failure | clarify | infeasible | invalid_plan
    report: str = ""
    failure_reason: str = ""
    failure_layer: str = ""
    question: str = ""
    plan: dict[str, Any] | None = None
    plan_attempts: list[dict[str, Any]] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    scene_before: str = ""
    scene_after: str = ""
    final_state: dict[str, Any] = field(default_factory=dict)
    replanned: bool = False
    robot_moved: bool = False
    wall_time_s: float = 0.0
    sim_time_s: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    model: str = ""
    video: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome == "success"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


class Executor:
    """Runs a validated plan against the simulator."""

    def __init__(
        self,
        robot: Robot,
        skills: Skills | None = None,
        *,
        planner: Planner | None = None,
        max_grasp_retries: int = 2,
        allow_replan: bool = True,
    ) -> None:
        self.robot = robot
        self.skills = skills or Skills(robot)
        self.planner = planner
        self.max_grasp_retries = max_grasp_retries
        self.allow_replan = allow_replan

    # ------------------------------------------------------------------ #
    # Running steps
    # ------------------------------------------------------------------ #

    def _run_step(self, step: Step, index: int, attempt: int) -> StepRecord:
        t0 = time.perf_counter()
        result: SkillResult = self.skills.run(step.skill, **step.args())
        retries = 0

        while (
            not result.ok
            and result.reason in RETRYABLE_GRASP
            and retries < self.max_grasp_retries
        ):
            retries += 1
            target = step.target or self.skills.held
            if target is None:
                break
            # Back off, open up, come down again. A slipped grasp is almost
            # always a nudged cube, so re-reading its pose is the whole fix.
            self.robot.open_gripper(duration=0.3)
            self.skills.held = None
            self.skills.move_to(target, offset_z=0.12)
            result = self.skills.run(step.skill, **step.args())

        return StepRecord(
            index=index,
            skill=step.skill,
            args=step.args(),
            ok=result.ok,
            reason=result.reason,
            retries=retries,
            wall_time_s=round(time.perf_counter() - t0, 3),
            info=result.as_dict()["info"],
            attempt=attempt,
        )

    def run(self, plan: Plan, *, observation: str = "", command: str = "") -> list[StepRecord]:
        """Execute ``plan``, verifying each step. Stops at the first hard failure."""
        records: list[StepRecord] = []
        self.replanned = False

        for i, step in enumerate(plan.steps):
            record = self._run_step(step, i, attempt=1)
            records.append(record)
            if record.ok:
                continue

            if not (self.allow_replan and self.planner and command):
                break

            # One re-plan from the state we are actually in.
            holding = (
                f"The gripper is still holding {self.skills.held}, so plan from there "
                f"(do not grasp again). "
                if self.skills.held
                else "The gripper is empty. "
            )
            note = (
                f"The previous plan failed at step {i} ({step.skill}) with "
                f"reason '{record.reason}'. {holding}"
                "Plan again from the scene as it is now."
            )
            new_plan, attempts = self.planner.plan(
                command, full_observation(self.robot, held=self.skills.held), self.robot.info,
                note=note,
            )
            self.replan_attempts = attempts
            self.replanned = True
            if new_plan is None or not new_plan.is_executable:
                break
            for j, step2 in enumerate(new_plan.steps):
                record2 = self._run_step(step2, j, attempt=2)
                records.append(record2)
                if not record2.ok:
                    return records
            return records

        return records

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def final_state(self) -> dict[str, Any]:
        info = self.robot.info
        state: dict[str, Any] = {"held": self.skills.held, "objects": {}}
        for name in info.cubes:
            pos = self.robot.body_pos(info.objects[name].body)
            entry = {"pos": [round(float(v), 4) for v in pos]}
            for region in info.regions:
                entry[f"in_{region}"] = is_inside_region(
                    self.robot.model, self.robot.data, info, name, region
                )
            state["objects"][name] = entry
        return state


# --------------------------------------------------------------------------- #
# Plain-English report
# --------------------------------------------------------------------------- #

HUMAN_REASON = {
    "unknown_object": "the plan named an object that is not in the scene",
    "not_graspable": "the plan tried to pick up something that is not a cube",
    "already_holding": "the plan tried to grasp while already holding something",
    "nothing_held": "the plan tried to move an object it was not holding",
    "unreachable": "the arm cannot reach that pose",
    "motion_error": "the arm did not reach the commanded pose accurately enough",
    "grasp_slip": "the gripper closed but the cube was not held",
    "object_dropped": "the cube came out of the gripper",
    "place_blocked": "the arm cannot reach the place pose",
    "not_in_target": "the cube did not end up in the target",
    "object_stuck": "the cube did not leave the gripper",
    "invalid_plan": "the planner did not produce a valid plan",
    "unknown_skill": "the plan used a skill that does not exist",
}


def build_report(result: EpisodeResult) -> str:
    """What the user reads: what was done, and success or the reason why not."""
    if result.outcome == "clarify":
        return f'Command is ambiguous, so nothing was moved. Question: "{result.question}"'
    if result.outcome == "infeasible":
        return f"Command cannot be carried out, so nothing was moved. Reason: {result.failure_reason}"
    if result.outcome == "invalid_plan":
        return f"No valid plan was produced, so nothing was moved. {result.failure_reason}"

    done = [s for s in result.steps if s.ok]
    lines: list[str] = []
    if done:
        lines.append(f"Carried out {len(done)} of {len(result.steps)} steps:")
        for s in done:
            lines.append(f"  - {s.skill}({_fmt_args(s.args)}) ok")
    retried = sum(s.retries for s in result.steps)
    if retried:
        lines.append(f"Recovered from {retried} grasp slip(s) by re-approaching.")
    if result.replanned:
        lines.append("Re-planned once from the state after the failure.")

    if result.outcome == "success":
        lines.append("Outcome: success.")
    else:
        bad = next((s for s in result.steps if not s.ok), None)
        why = HUMAN_REASON.get(result.failure_reason, result.failure_reason or "unknown")
        if bad is not None:
            lines.append(f"Outcome: failed at {bad.skill}({_fmt_args(bad.args)}) -- {why}.")
        else:
            lines.append(f"Outcome: failed -- {why}.")
    return "\n".join(lines)


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


# --------------------------------------------------------------------------- #
# One command, end to end
# --------------------------------------------------------------------------- #


def run_command(
    command: str,
    *,
    seed: int = 0,
    cfg: SceneConfig | None = None,
    planner: Planner | None = None,
    viewer: bool = False,
    video: str | Path | None = None,
    camera: str = "front",
    allow_replan: bool = True,
    max_grasp_retries: int = 2,
) -> EpisodeResult:
    """Plan and execute one natural-language command. Builds its own scene."""
    cfg = cfg or SceneConfig(seed=seed)
    if cfg.seed != seed:
        cfg = SceneConfig(**{**cfg.__dict__, "seed": seed})

    t0 = time.perf_counter()
    robot = Robot.from_config(cfg)
    robot.hooks = make_hooks(robot.model, robot.data, viewer=viewer, video=video, camera=camera)
    skills = Skills(robot)
    planner = planner or Planner()

    result = EpisodeResult(command=command, seed=cfg.seed, outcome="failure")
    try:
        for hook in robot.hooks:  # a first frame of the untouched scene
            if isinstance(hook, VideoHook):
                hook.capture(robot.model, robot.data)

        observation = full_observation(robot)
        result.scene_before = observation

        plan, attempts = planner.plan(command, observation, robot.info)
        result.plan_attempts = [vars(a) for a in attempts]
        result.model = attempts[-1].model if attempts else ""
        result.tokens = _sum_tokens(attempts)

        if plan is None:
            result.outcome = "invalid_plan"
            result.failure_reason = attempts[-1].error or "invalid_plan"
            result.failure_layer = "planner"
        elif plan.status == "clarify":
            result.outcome = "clarify"
            result.question = plan.question or ""
            result.plan = plan.model_dump()
        elif plan.status == "infeasible":
            result.outcome = "infeasible"
            result.failure_reason = plan.reason or ""
            result.plan = plan.model_dump()
        else:
            result.plan = plan.model_dump()
            executor = Executor(
                robot,
                skills,
                planner=planner,
                allow_replan=allow_replan,
                max_grasp_retries=max_grasp_retries,
            )
            result.steps = executor.run(plan, observation=observation, command=command)
            result.replanned = getattr(executor, "replanned", False)
            result.robot_moved = bool(result.steps)
            failed = next((s for s in result.steps if not s.ok), None)
            if failed is None:
                result.outcome = "success"
            else:
                result.outcome = "failure"
                result.failure_reason = failed.reason
                result.failure_layer = FAILURE_LAYER.get(failed.reason, "controller")
            result.final_state = executor.final_state()

        if not result.final_state:
            result.final_state = Executor(robot, skills).final_state()
        result.scene_after = full_observation(robot, held=skills.held)
        result.sim_time_s = round(float(robot.data.time), 3)
        result.video = str(video) if video else ""
        result.report = build_report(result)
    finally:
        robot.close()

    result.wall_time_s = round(time.perf_counter() - t0, 2)
    return result


def _sum_tokens(attempts: list[PlanAttempt]) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0}
    for a in attempts:
        for k in total:
            total[k] += int(a.usage.get(k, 0))
    return total
