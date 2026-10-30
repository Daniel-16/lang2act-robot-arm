"""Command-line entry points.

    lang2act view                      open the scene in the MuJoCo viewer
    lang2act ik-test                   measure TCP error on random reachable points
    lang2act pick                      the hard-coded skill sequence (no LLM)
    lang2act run "put the red ..."     plan and execute one command
    lang2act eval                      the frozen 30-command evaluation

Every subcommand takes ``--seed``; ``view``, ``pick`` and ``run`` also take
``--viewer`` to watch live and ``--video PATH`` to save an MP4 from the fixed
front camera.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .describe import full_observation
from .executor import run_command
from .planner import Planner, make_backend
from .robot import Robot, VideoHook, make_hooks
from .scene import SceneConfig, build_scene, is_inside_region
from .skills import Skills

#: The Phase 2 gate: the scripted pick, expressed as skill calls only.
PICK_SEQUENCE = (
    ("move_to", {"offset_z": 0.15}),
    ("grasp", {}),
    ("lift", {"height": 0.15}),
    ("place", {"target": "bowl"}),
    ("release", {}),
)


def _scene_config(args: argparse.Namespace) -> SceneConfig:
    return SceneConfig(seed=args.seed, randomize=not getattr(args, "fixed_layout", False))


# --------------------------------------------------------------------------- #
# view
# --------------------------------------------------------------------------- #


def cmd_view(args: argparse.Namespace) -> int:
    """Build the scene and let the physics run so the layout can be inspected."""
    model, data, info = build_scene(_scene_config(args))
    print(f"seed {args.seed}: {', '.join(info.names)}")
    print(f"model: nq={model.nq} nu={model.nu} bodies={model.nbody} timestep={model.opt.timestep}")

    robot = Robot(model, data, info, build_scene(_scene_config(args), with_objects=False)[0])
    robot.hooks = make_hooks(model, data, viewer=args.viewer, video=args.video)
    print(full_observation(robot))
    try:
        if args.viewer or args.video:
            robot.settle(args.seconds)
        else:
            print("(nothing to show: pass --viewer or --video)")
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
    if args.video:
        print(f"wrote {args.video}")
    return 0


# --------------------------------------------------------------------------- #
# ik-test
# --------------------------------------------------------------------------- #


def cmd_ik_test(args: argparse.Namespace) -> int:
    """Phase 1 gate: TCP within 5 mm of N random reachable points."""
    robot = Robot.from_config(_scene_config(args))
    robot.hooks = make_hooks(robot.model, robot.data, viewer=args.viewer, video=args.video)
    rng = np.random.default_rng(args.seed)

    rows = []
    try:
        for i in range(args.points):
            target = np.array(
                [rng.uniform(0.34, 0.62), rng.uniform(-0.32, 0.32), rng.uniform(0.06, 0.38)]
            )
            sol = robot.solve_ik(target)
            if not sol.ok:
                rows.append((i, target, sol.pos_error_mm, float("nan"), False))
                continue
            robot.move_joints(sol.qpos, duration=1.2)
            tracked = float(np.linalg.norm(robot.tcp_pos - target)) * 1000.0
            rows.append((i, target, sol.pos_error_mm, tracked, True))
    finally:
        robot.close()

    print(f"{'#':>3} {'target (x, y, z)':>26} {'ik err':>9} {'tracked':>9}")
    for i, target, ik_mm, tracked_mm, ok in rows:
        t = f"({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})"
        tracked = "unreachable" if not ok else f"{tracked_mm:7.2f}mm"
        print(f"{i:3d} {t:>26} {ik_mm:7.2f}mm {tracked:>9}")

    tracked = np.array([r[3] for r in rows if r[4]])
    unreachable = sum(1 for r in rows if not r[4])
    if tracked.size:
        print(
            f"\nreachable {tracked.size}/{len(rows)} | "
            f"mean {tracked.mean():.2f} mm | max {tracked.max():.2f} mm"
        )
    gate = tracked.size == len(rows) and bool((tracked < 5.0).all())
    print(f"gate (all points reachable and under 5 mm): {'PASS' if gate else 'FAIL'}")
    if unreachable:
        print(f"{unreachable} sampled point(s) were outside the workspace")
    return 0 if gate else 1


# --------------------------------------------------------------------------- #
# pick
# --------------------------------------------------------------------------- #


def cmd_pick(args: argparse.Namespace) -> int:
    """Phase 1/2 gate: put every cube in the bowl with no LLM in the loop."""
    robot = Robot.from_config(_scene_config(args))
    robot.hooks = make_hooks(robot.model, robot.data, viewer=args.viewer, video=args.video)
    skills = Skills(robot)
    cubes = [args.cube] if args.cube else robot.info.cubes
    ok_all = True

    try:
        for hook in robot.hooks:
            if isinstance(hook, VideoHook):
                hook.capture(robot.model, robot.data)
        for cube in cubes:
            print(f"\n{cube}:")
            failed = None
            for skill, kwargs in PICK_SEQUENCE:
                call = dict(kwargs)
                if skill in ("move_to", "grasp"):
                    call["target"] = cube
                result = skills.run(skill, **call)
                flag = "ok" if result.ok else f"FAILED ({result.reason})"
                print(f"  {skill:9s} {flag}")
                if not result.ok:
                    failed = result.reason
                    break
            inside = is_inside_region(robot.model, robot.data, robot.info, cube, "bowl")
            print(f"  -> in bowl: {inside}")
            ok_all &= failed is None and inside
        robot.go_home()
    except KeyboardInterrupt:
        return 130
    finally:
        robot.close()

    print(f"\nscripted pick, seed {args.seed}: {'PASS' if ok_all else 'FAIL'}")
    if args.video:
        print(f"wrote {args.video}")
    return 0 if ok_all else 1


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    """Plan and execute one natural-language command."""
    planner = Planner(make_backend(args.llm))
    result = run_command(
        args.command,
        seed=args.seed,
        planner=planner,
        viewer=args.viewer,
        video=args.video,
        allow_replan=not args.no_replan,
    )

    print(f"Command: {result.command}   (seed {result.seed}, planner {result.model})")
    print()
    print("Scene:")
    print(result.scene_before)
    print()
    if result.plan:
        print("Plan:", json.dumps(result.plan, separators=(",", ":")))
        print()
    print(result.report)
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2, default=str))
        print(f"\nepisode log: {args.json}")
    if args.video:
        print(f"video: {args.video}")
    return 0 if result.outcome in ("success", "clarify", "infeasible") else 1


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lang2act", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser, *, viz: bool = True) -> None:
        sp.add_argument("--seed", type=int, default=0, help="scene seed (default 0)")
        sp.add_argument(
            "--fixed-layout", action="store_true", help="use the non-randomised layout"
        )
        if viz:
            sp.add_argument("--viewer", action="store_true", help="watch in the MuJoCo viewer")
            sp.add_argument("--video", default=None, help="write an MP4 from the front camera")

    sp = sub.add_parser("view", help="open the scene")
    common(sp)
    sp.add_argument("--seconds", type=float, default=30.0, help="how long to run (default 30)")
    sp.set_defaults(func=cmd_view)

    sp = sub.add_parser("ik-test", help="measure IK accuracy")
    common(sp)
    sp.add_argument("--points", type=int, default=20, help="random targets (default 20)")
    sp.set_defaults(func=cmd_ik_test)

    sp = sub.add_parser("pick", help="scripted pick and place, no LLM")
    common(sp)
    sp.add_argument("--cube", default=None, help="only this cube, e.g. red_cube")
    sp.set_defaults(func=cmd_pick)

    sp = sub.add_parser("run", help="plan and execute one command")
    sp.add_argument("command", help='e.g. "put the red cube in the bowl"')
    common(sp)
    sp.add_argument("--llm", default=None, help="provider[:model] (default $LANG2ACT_LLM or mock)")
    sp.add_argument("--json", default=None, help="write the episode log here")
    sp.add_argument("--no-replan", action="store_true", help="disable the single re-plan")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("eval", help="run the frozen 30-command set")
    sp.set_defaults(func=None)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":  # hand the rest straight to the eval parser
        from .eval import main as eval_main

        return eval_main(argv[1:])
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
