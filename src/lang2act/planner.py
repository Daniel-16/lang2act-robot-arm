"""The LLM planner: natural language in, a validated JSON plan out.

The planner is the only component allowed to be wrong. It emits a plan over the
five fixed skills; it never emits joint angles, positions or new skill names. A
plan that does not validate against the schema below is rejected *before* the
arm moves, the error is fed back once, and a still-invalid plan is reported as a
planner failure.

Three statuses:

``plan``        a list of steps to execute.
``clarify``     the command is ambiguous; ``question`` is asked, nothing moves.
``infeasible``  the command cannot be carried out; ``reason`` says why.

Swapping models is a config change: set ``LANG2ACT_LLM`` to
``anthropic:claude-opus-5-5``, ``openai:gpt-...``, ``ollama:llama3.1`` or
``mock``, or pass ``--llm`` on the command line.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .scene import COLOR_SYNONYMS, SceneInfo
from .skills import RELATIONS

SkillName = Literal["move_to", "grasp", "lift", "place", "release"]
RelationName = Literal["left_of", "right_of", "behind", "in_front_of"]

MAX_STEPS = 24
DEFAULT_MODEL = "claude-opus-5-5"


# --------------------------------------------------------------------------- #
# Plan schema
# --------------------------------------------------------------------------- #


class Step(BaseModel):
    """One skill call. Unknown fields are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")

    skill: SkillName
    target: str | None = None
    offset_z: float | None = Field(default=None, ge=0.0, le=0.4)
    height: float | None = Field(default=None, ge=0.02, le=0.4)
    relation: RelationName | None = None
    reference: str | None = None

    def args(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`lang2act.skills.Skills.run`."""
        return {
            k: v
            for k, v in self.model_dump().items()
            if k != "skill" and v is not None
        }

    def describe(self) -> str:
        if self.skill == "move_to":
            return f"move above the {self.target}"
        if self.skill == "grasp":
            return f"close the gripper on the {self.target}"
        if self.skill == "lift":
            return f"lift it {(self.height or 0.15) * 100:.0f} cm"
        if self.skill == "place":
            if self.target:
                return f"carry it over the {self.target}"
            return f"carry it to the {str(self.relation).replace('_', ' ')} the {self.reference}"
        return "open the gripper"


class Plan(BaseModel):
    """The only shape the planner may return."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["plan", "clarify", "infeasible"]
    question: str | None = None
    reason: str | None = None
    steps: list[Step] = Field(default_factory=list)

    @property
    def is_executable(self) -> bool:
        return self.status == "plan"


class PlanError(ValueError):
    """Raised when a plan is unusable. The message is fed back to the LLM."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose or fences even when told not to; this is the one
    piece of leniency in the pipeline, and it is purely syntactic.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise PlanError("no JSON object found in the response")
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise PlanError(f"malformed JSON: {exc}") from exc
    raise PlanError("unterminated JSON object in the response")


def validate_plan(payload: str | dict[str, Any], info: SceneInfo) -> Plan:
    """Schema-check a plan and bind every name to this scene.

    Beyond the pydantic schema this enforces:

    * object names come from the current scene (the LLM cannot invent a cube),
    * ``grasp`` targets a graspable object, not a region,
    * ``place`` gets a target *or* a relation+reference, never both or neither,
    * step count stays inside ``MAX_STEPS``,
    * ``clarify`` carries a question and ``infeasible`` carries a reason.
    """
    data = extract_json(payload) if isinstance(payload, str) else payload
    try:
        plan = Plan.model_validate(data)
    except ValidationError as exc:
        raise PlanError(_short_validation_error(exc)) from exc

    names = set(info.names)
    graspable = {n for n, o in info.objects.items() if o.graspable}

    if plan.status == "clarify":
        if not (plan.question or "").strip():
            raise PlanError("status 'clarify' requires a non-empty 'question'")
        if plan.steps:
            raise PlanError("status 'clarify' must not carry steps")
        return plan

    if plan.status == "infeasible":
        if not (plan.reason or "").strip():
            raise PlanError("status 'infeasible' requires a non-empty 'reason'")
        if plan.steps:
            raise PlanError("status 'infeasible' must not carry steps")
        return plan

    if not plan.steps:
        raise PlanError("status 'plan' requires at least one step")
    if len(plan.steps) > MAX_STEPS:
        raise PlanError(f"too many steps: {len(plan.steps)} > {MAX_STEPS}")

    for i, step in enumerate(plan.steps):
        where = f"step {i} ({step.skill})"
        if step.skill in ("move_to", "grasp"):
            if not step.target:
                raise PlanError(f"{where} requires 'target'")
            if step.target not in names:
                raise PlanError(
                    f"{where}: unknown object {step.target!r}; "
                    f"this scene has {sorted(names)}"
                )
            if step.skill == "grasp" and step.target not in graspable:
                raise PlanError(f"{where}: {step.target!r} cannot be picked up")
        elif step.skill == "place":
            if step.target and (step.relation or step.reference):
                raise PlanError(f"{where}: give 'target' or 'relation'+'reference', not both")
            if step.target:
                if step.target not in names:
                    raise PlanError(
                        f"{where}: unknown object {step.target!r}; "
                        f"this scene has {sorted(names)}"
                    )
            elif step.relation and step.reference:
                if step.relation not in RELATIONS:
                    raise PlanError(
                        f"{where}: unknown relation {step.relation!r}; "
                        f"use one of {sorted(RELATIONS)}"
                    )
                if step.reference not in names:
                    raise PlanError(f"{where}: unknown reference {step.reference!r}")
            else:
                raise PlanError(f"{where} requires 'target' or 'relation'+'reference'")
        elif step.skill in ("lift", "release"):
            if step.target:
                raise PlanError(f"{where} takes no 'target'")

    return plan


def _short_validation_error(exc: ValidationError) -> str:
    bits = []
    for err in exc.errors()[:4]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        bits.append(f"{loc}: {err['msg']}")
    return "schema validation failed -- " + "; ".join(bits)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SKILL_REFERENCE = """\
move_to(target, offset_z)   put the gripper offset_z metres above target.
                            offset_z is 0.0-0.4; use 0.15 to approach.
grasp(target)               descend onto target and close the gripper.
                            target must be a cube. Fails if already holding.
lift(height)                raise the held object by height metres (0.02-0.4).
place(target)               carry the held object over target and lower it.
place(relation, reference)  carry it to a spot relative to another object.
                            relation is left_of, right_of, behind or in_front_of.
release()                   open the gripper and retract."""

SYSTEM_PROMPT = """\
You plan tabletop manipulation for a Franka Panda arm in simulation.

You decide WHAT to do. You never decide HOW: no joint angles, no coordinates,
no new skills. You may only compose the skills listed below, and you may only
name objects that the scene description lists.

Skills:
{skills}

A pick-and-place is always the same five steps in this order:
move_to(target, 0.15) -> grasp(target) -> lift(0.15) -> place(...) -> release().
Repeat that block once per object when several objects must be moved.

Reply with a single JSON object and nothing else:

{{"status": "plan"|"clarify"|"infeasible",
  "question": string or null,
  "reason": string or null,
  "steps": [ {{"skill": ..., "target": ..., "offset_z": ..., "height": ...,
               "relation": ..., "reference": ...}} ]}}

Rules:
- "plan": steps is non-empty, question and reason are null.
- "clarify": the command could mean more than one object in THIS scene. Ask one
  short question, leave steps empty. Do not guess.
- "infeasible": the command names something that is not in the scene, or asks
  for something the skills cannot do. Give a one-sentence reason, steps empty.
- Omit fields that do not apply to a skill rather than sending null for them.
- A colour word that matches an object in the scene is not ambiguous, even if
  the command words it differently ("crimson block" is the red cube)."""

FEW_SHOT = [
    (
        "put the red cube in the bowl",
        {
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
        },
    ),
    (
        "pick up the cube",
        {
            "status": "clarify",
            "question": "There are three cubes (red, green and blue). Which one should I pick up?",
            "reason": None,
            "steps": [],
        },
    ),
    (
        "put the purple cube in the bowl",
        {
            "status": "infeasible",
            "question": None,
            "reason": "There is no purple cube in the scene; the cubes are red, green and blue.",
            "steps": [],
        },
    ),
    (
        "move the blue cube to the left of the green one",
        {
            "status": "plan",
            "question": None,
            "reason": None,
            "steps": [
                {"skill": "move_to", "target": "blue_cube", "offset_z": 0.15},
                {"skill": "grasp", "target": "blue_cube"},
                {"skill": "lift", "height": 0.15},
                {"skill": "place", "relation": "left_of", "reference": "green_cube"},
                {"skill": "release"},
            ],
        },
    ),
]


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(skills=SKILL_REFERENCE)


def user_prompt(command: str, observation: str, info: SceneInfo, *, note: str = "") -> str:
    examples = "\n\n".join(
        f"Command: {cmd}\n{json.dumps(out)}" for cmd, out in FEW_SHOT
    )
    parts = [
        "Scene:",
        observation,
        "",
        f"Object names you may use: {', '.join(info.names)}",
        "",
        "Examples (from other scenes, for format only):",
        examples,
        "",
        f"Command: {command}",
    ]
    if note:
        parts += ["", f"Note: {note}"]
    parts.append("JSON:")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# LLM backends
# --------------------------------------------------------------------------- #


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False


class LLMBackend:
    """One interface, so the two-model comparison is a config change."""

    name = "base"
    model = ""

    def complete(self, system: str, user: str) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class AnthropicBackend(LLMBackend):
    """Claude via the official Anthropic SDK, with structured output."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 4096):
        import anthropic  # imported lazily so the package stays optional

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=Plan,
        )
        plan = response.parsed_output
        text = plan.model_dump_json() if plan is not None else _first_text(response)
        return LLMResponse(
            text=text,
            model=self.model,
            latency_s=time.perf_counter() - t0,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


def _first_text(response: Any) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


class OpenAIBackend(LLMBackend):
    """A second hosted provider, for the stretch-goal model comparison."""

    name = "openai"

    def __init__(self, model: str = "gpt-4.1", *, max_tokens: int = 4096):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI()

    def complete(self, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            latency_s=time.perf_counter() - t0,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0),
                "output_tokens": getattr(usage, "completion_tokens", 0),
            },
        )


class OllamaBackend(LLMBackend):
    """A local model over Ollama's HTTP API -- the no-API-access fallback."""

    name = "ollama"

    def __init__(self, model: str = "llama3.1", *, host: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def complete(self, system: str, user: str) -> LLMResponse:
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read())
        return LLMResponse(
            text=payload["message"]["content"],
            model=self.model,
            latency_s=time.perf_counter() - t0,
            usage={
                "input_tokens": payload.get("prompt_eval_count", 0),
                "output_tokens": payload.get("eval_count", 0),
            },
        )


class MockBackend(LLMBackend):
    """A deterministic, offline stand-in for the LLM.

    This is **not** the deliverable -- it is how the rest of the pipeline is
    tested and how the eval harness runs with no API key and no network. It
    pattern-matches the command instead of reasoning about it, so it is honest
    about one thing only: whether the executor, skills and goal checks work.
    Report numbers from a real model, and say which one.
    """

    name = "mock"

    CONTAINER_WORDS = ("bowl", "dish", "basin", "container", "bucket")
    ALL_WORDS = ("all", "every", "each", "both", "everything")
    CUBE_WORDS = ("cube", "block", "box", "brick", "one")

    def __init__(self, info: SceneInfo | None = None):
        self.model = "mock-rules-v1"
        self.info = info
        self.held: str | None = None

    def bind(self, info: SceneInfo) -> "MockBackend":
        self.info = info
        return self

    # -- helpers --------------------------------------------------------- #

    def _colors_in(self, text: str) -> list[str]:
        found = []
        for color, words in COLOR_SYNONYMS.items():
            if any(re.search(rf"\b{w}\b", text) for w in words):
                found.append(color)
        return found

    def _pick_block(
        self, cube: str, *, target: str | None, relation=None, reference=None, held: str | None = None
    ):
        place: dict[str, Any] = {"skill": "place"}
        if target:
            place["target"] = target
        else:
            place.update(relation=relation, reference=reference)
        if held == cube:
            # Already holding it: carry on from where the last plan stopped.
            return [place, {"skill": "release"}]
        return [
            {"skill": "move_to", "target": cube, "offset_z": 0.15},
            {"skill": "grasp", "target": cube},
            {"skill": "lift", "height": 0.15},
            place,
            {"skill": "release"},
        ]

    # -- the "model" ----------------------------------------------------- #

    def complete(self, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        info = self.info
        if info is None:
            raise RuntimeError("MockBackend needs bind(info) before use")
        command = user.rsplit("Command:", 1)[-1].split("JSON:")[0].strip().lower()
        out = self._decide(command, user.lower(), info)
        return LLMResponse(
            text=json.dumps(out), model=self.model, latency_s=time.perf_counter() - t0
        )

    def _decide(self, command: str, prompt: str, info: SceneInfo) -> dict[str, Any]:
        cubes = info.cubes
        self.held = self._held_object(prompt, info)
        present = {info.color_of(c): c for c in cubes}
        colors = self._colors_in(command)
        missing = [c for c in colors if c not in present]
        if missing:
            return {
                "status": "infeasible",
                "question": None,
                "reason": (
                    f"There is no {missing[0]} cube in the scene; "
                    f"the cubes are {', '.join(sorted(present))}."
                ),
                "steps": [],
            }

        container = next((w for w in self.CONTAINER_WORDS if w in command), None)
        to_bowl = container is not None and info.bowl_name in info.objects

        # "put all the cubes in the bowl"
        if any(re.search(rf"\b{w}\b", command) for w in self.ALL_WORDS) and to_bowl:
            steps: list[dict[str, Any]] = []
            for cube in cubes:
                steps += self._pick_block(cube, target=info.bowl_name, held=self.held)
            return {"status": "plan", "question": None, "reason": None, "steps": steps}

        # Where to put it, and the part of the command that says what to move.
        relation, reference, head = self._relation(command, present)
        subject = self._subject(head, prompt, present, exclude=reference)
        if subject is None and relation is None:
            subject = self._subject(command, prompt, present, exclude=None)

        if subject is not None:
            if relation and reference and reference != subject:
                return {
                    "status": "plan",
                    "question": None,
                    "reason": None,
                    "steps": self._pick_block(
                        subject, target=None, relation=relation, reference=reference,
                        held=self.held,
                    ),
                }
            if to_bowl:
                return {
                    "status": "plan",
                    "question": None,
                    "reason": None,
                    "steps": self._pick_block(subject, target=info.bowl_name, held=self.held),
                }
            return {
                "status": "infeasible",
                "question": None,
                "reason": "I can see what to pick up but not where to put it.",
                "steps": [],
            }

        if any(re.search(rf"\b{w}\b", command) for w in self.CUBE_WORDS) and len(cubes) > 1:
            listed = ", ".join(sorted(present))
            return {
                "status": "clarify",
                "question": f"There are {len(cubes)} cubes ({listed}). Which one do you mean?",
                "reason": None,
                "steps": [],
            }
        return {
            "status": "clarify",
            "question": "I could not tell which object you mean. Which one should I move?",
            "reason": None,
            "steps": [],
        }

    def _held_object(self, prompt: str, info: SceneInfo) -> str | None:
        """What the scene text says the gripper is holding, if anything."""
        match = re.search(r"gripper: holding (\w+)", prompt)
        if match and match.group(1) in info.objects:
            return match.group(1)
        return None

    def _subject(
        self,
        text: str,
        prompt: str,
        present: dict[str, str],
        *,
        exclude: str | None,
    ) -> str | None:
        """Which cube ``text`` refers to: by colour, or by a superlative."""
        for color in self._colors_in(text):
            if color in present and present[color] != exclude:
                return present[color]
        return self._superlative(text, prompt)

    def _relation(self, command: str, present: dict[str, str]):
        """``(relation, reference, head)`` -- head is the text before the relation."""
        patterns = {
            "left_of": r"(?:to the )?left of",
            "right_of": r"(?:to the )?right of",
            "in_front_of": r"in front of",
            "behind": r"behind",
        }
        for rel, pat in patterns.items():
            m = re.search(pat, command)
            if not m:
                continue
            tail = command[m.end() :]
            for color in self._colors_in(tail):
                if color in present:
                    return rel, present[color], command[: m.start()]
        return None, None, command

    def _superlative(self, command: str, prompt: str) -> str | None:
        """Resolve "the cube nearest/farthest from the bowl".

        A text-only stand-in cannot compute distances, but it does not have to:
        `describe.pairwise_relations` already ranked the cubes in the prompt, so
        this reads the ranking off the scene text -- which is exactly what an
        LLM would do with the same prompt.
        """
        want_near = re.search(r"\b(nearest|closest)\b", command) is not None
        want_far = re.search(r"\b(farthest|furthest)\b", command) is not None
        if not (want_near or want_far):
            return None
        ranking = re.search(r"distance to bowl, nearest first: ([^\n]+)", prompt)
        if not ranking:
            return None
        names = [part.strip().split()[0] for part in ranking.group(1).split(",")]
        if not names:
            return None
        return names[0] if want_near else names[-1]


class CachedBackend(LLMBackend):
    """Disk cache around any backend: keeps dev iteration free and repeatable."""

    def __init__(self, inner: LLMBackend, cache_dir: str | Path = "llm_cache"):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.model = inner.model
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def complete(self, system: str, user: str) -> LLMResponse:
        key = hashlib.sha256(f"{self.model}\n{system}\n{user}".encode()).hexdigest()[:32]
        path = self.dir / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            return LLMResponse(
                text=payload["text"], model=payload["model"], usage=payload.get("usage", {}),
                cached=True,
            )
        response = self.inner.complete(system, user)
        path.write_text(
            json.dumps({"text": response.text, "model": response.model, "usage": response.usage})
        )
        return response


def make_backend(spec: str | None = None, *, info: SceneInfo | None = None) -> LLMBackend:
    """Build a backend from ``provider[:model]``.

    Falls back to the ``LANG2ACT_LLM`` environment variable, then to ``mock``.
    Wrap any backend in a disk cache by setting ``LANG2ACT_LLM_CACHE=1``.
    """
    spec = spec or os.environ.get("LANG2ACT_LLM") or "mock"
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()

    backend: LLMBackend
    if provider in ("mock", "offline", "rules"):
        backend = MockBackend(info)
    elif provider in ("anthropic", "claude"):
        backend = AnthropicBackend(model or DEFAULT_MODEL)
    elif provider == "openai":
        backend = OpenAIBackend(model or "gpt-4.1")
    elif provider == "ollama":
        backend = OllamaBackend(model or "llama3.1")
    else:
        raise ValueError(
            f"unknown LLM provider {provider!r}; use mock, anthropic, openai or ollama"
        )

    if os.environ.get("LANG2ACT_LLM_CACHE") and not isinstance(backend, MockBackend):
        backend = CachedBackend(backend)
    return backend


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


@dataclass
class PlanAttempt:
    """What the planner did, kept verbatim for the episode log."""

    raw: str
    error: str | None
    model: str
    latency_s: float
    usage: dict[str, int] = field(default_factory=dict)


class Planner:
    """Turns a command plus a scene description into a validated plan."""

    def __init__(self, backend: LLMBackend | None = None, *, max_repairs: int = 1):
        self.backend = backend or make_backend()
        self.max_repairs = max_repairs

    def plan(
        self, command: str, observation: str, info: SceneInfo, *, note: str = ""
    ) -> tuple[Plan | None, list[PlanAttempt]]:
        """Return ``(plan, attempts)``. ``plan`` is None if nothing validated."""
        if isinstance(self.backend, MockBackend):
            self.backend.bind(info)

        system = system_prompt()
        attempts: list[PlanAttempt] = []
        extra = note

        for _ in range(self.max_repairs + 1):
            response = self.backend.complete(system, user_prompt(command, observation, info, note=extra))
            try:
                plan = validate_plan(response.text, info)
            except PlanError as exc:
                attempts.append(
                    PlanAttempt(response.text, str(exc), response.model, response.latency_s, response.usage)
                )
                extra = (
                    f"Your previous reply was rejected: {exc}. "
                    "Reply with corrected JSON only."
                )
                continue
            attempts.append(
                PlanAttempt(response.text, None, response.model, response.latency_s, response.usage)
            )
            return plan, attempts

        return None, attempts
