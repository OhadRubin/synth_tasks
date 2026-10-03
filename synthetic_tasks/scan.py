"""
- description:
    SCAN command-to-actions synthetic task family for the shadow-ADO
    learnability experiment. Probes systematic compositional
    interpretation: the model reads a short natural-language command
    built from a controlled grammar over four motion primitives
    (jump/walk/run/look), two directional modifiers (left/right), two
    repetition modifiers (twice/thrice), and two connectives (and,
    after) — and must emit the corresponding sequence of capitalized
    action tokens (JUMP, WALK, RUN, LOOK, TURN_LEFT, TURN_RIGHT).
    Grammar implemented (subset of Lake & Baroni 2018):
        C  -> P | P 'and' C | P 'after' C       (right-associative)
        P  -> V [DIR] [REP]
        V  -> 'jump' | 'walk' | 'run' | 'look'
        DIR-> 'left' | 'right'
        REP-> 'twice' | 'thrice'
    Semantics:
        emit(V)           = [ACTION(V)]
        emit(V DIR)       = [TURN_DIR, ACTION(V)]
        emit(V REP)       = emit(V) * n(REP)
        emit(V DIR REP)   = [TURN_DIR, ACTION(V)] * n(REP)
        emit(P 'and' C)   = emit(P) ++ emit(C)
        emit(P 'after' C) = emit(C) ++ emit(P)
    The `command_length` knob (short/medium/long) controls the
    target token count of the command (2-4 / 5-8 / 9-14 tokens
    respectively), realized by repeatedly extending a phrase chain
    until the total falls inside the bin window. domain_id is
    `scan_<command_length>`. Loss is over the answer region only via
    build_loss_mask.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("scan_command_to_actions", "scan_short", "train", 0)
    ex = get_task("scan_command_to_actions")().sample(
        rng, "train", {"command_length": "short"}
    )
- user_story:
    content: |
        SCAN is the canonical systematic-interpretation probe: solving
        it requires the model to parse a small grammar, apply
        modifier semantics in the correct order (direction wraps each
        repeated action; 'after' inverts the emission order of its
        operands), and concatenate sub-results. We adopt a controlled
        subset so the per-bin difficulty knob is the *command token
        length* rather than the original primitive-vs-composition
        split — this gives shadow ADO a monotone learnability axis
        comparable to other phase-2 generators while preserving the
        non-trivial compositional structure. Risks the design
        addresses: (1) ungrounded answers — an independently coded
        evaluator in the tests file re-parses the prompt to verify
        every target; (2) leakage between bins — the command-length
        knob is enforced as a hard range, not just a sampling bias;
        (3) byte-budget overflow — action counts are capped per
        sample so the worst-case long-bin example stays well under
        1024 bytes.
    was_generated_via_skill: false
"""

from __future__ import annotations

import random
from typing import Any

from synthetic_tasks.base import (
    Example,
    SyntheticTask,
    build_loss_mask,
    register_task,
)


_PRIMITIVES = {
    "jump": "JUMP",
    "walk": "WALK",
    "run": "RUN",
    "look": "LOOK",
}
_DIRECTIONS = {
    "left": "TURN_LEFT",
    "right": "TURN_RIGHT",
}
_REPETITIONS = {
    "twice": 2,
    "thrice": 3,
}
_CONNECTORS = ("and", "after")

_LENGTH_RANGES = {
    "short": (2, 4),
    "medium": (5, 8),
    "long": (9, 14),
}

_MAX_ACTIONS = 50
_MAX_SAMPLE_TRIES = 1000


def _sample_phrase(rng: random.Random) -> list[str]:
    tokens = [rng.choice(list(_PRIMITIVES))]
    if rng.random() < 0.5:
        tokens.append(rng.choice(list(_DIRECTIONS)))
    if rng.random() < 0.5:
        tokens.append(rng.choice(list(_REPETITIONS)))
    return tokens


def _phrase_actions(phrase: list[str]) -> list[str]:
    verb = phrase[0]
    if verb not in _PRIMITIVES:
        raise ValueError(f"phrase head {verb!r} is not a primitive")
    direction: str | None = None
    repetition: int = 1
    for tok in phrase[1:]:
        if tok in _DIRECTIONS:
            direction = tok
        elif tok in _REPETITIONS:
            repetition = _REPETITIONS[tok]
        else:
            raise ValueError(f"unknown modifier token {tok!r} in phrase {phrase!r}")
    unit: list[str] = []
    if direction is not None:
        unit.append(_DIRECTIONS[direction])
    unit.append(_PRIMITIVES[verb])
    return unit * repetition


def _evaluate(phrases: list[list[str]], connectors: list[str]) -> list[str]:
    actions = _phrase_actions(phrases[-1])
    for i in range(len(connectors) - 1, -1, -1):
        left = _phrase_actions(phrases[i])
        conn = connectors[i]
        if conn == "and":
            actions = left + actions
        elif conn == "after":
            actions = actions + left
        else:
            raise ValueError(f"unknown connector {conn!r}")
    return actions


def _serialize_command(phrases: list[list[str]], connectors: list[str]) -> str:
    out: list[str] = list(phrases[0])
    for i, conn in enumerate(connectors):
        out.append(conn)
        out.extend(phrases[i + 1])
    return " ".join(out)


def _sample_command(
    rng: random.Random, lo: int, hi: int
) -> tuple[list[list[str]], list[str]]:
    for _ in range(_MAX_SAMPLE_TRIES):
        phrases = [_sample_phrase(rng)]
        connectors: list[str] = []
        total = len(phrases[0])
        while total < lo:
            new_phrase = _sample_phrase(rng)
            phrases.append(new_phrase)
            connectors.append(rng.choice(_CONNECTORS))
            total += 1 + len(new_phrase)
        if lo <= total <= hi:
            actions = _evaluate(phrases, connectors)
            if len(actions) <= _MAX_ACTIONS:
                return phrases, connectors
    raise RuntimeError(
        f"failed to sample command with token count in [{lo}, {hi}] "
        f"after {_MAX_SAMPLE_TRIES} tries"
    )


@register_task
class ScanCommandToActionsTask(SyntheticTask):
    task_id = "scan_command_to_actions"

    def sample(
        self, rng: random.Random, split: str, difficulty: dict[str, Any]
    ) -> Example:
        if "command_length" not in difficulty:
            raise ValueError(
                f"scan difficulty must include 'command_length', got {difficulty!r}"
            )
        bin_name = difficulty["command_length"]
        if not isinstance(bin_name, str):
            raise ValueError(
                f"scan difficulty.command_length must be str, "
                f"got {type(bin_name).__name__}"
            )
        if bin_name not in _LENGTH_RANGES:
            raise ValueError(
                f"scan difficulty.command_length must be one of "
                f"{sorted(_LENGTH_RANGES)}, got {bin_name!r}"
            )
        lo, hi = _LENGTH_RANGES[bin_name]
        phrases, connectors = _sample_command(rng, lo, hi)
        command = _serialize_command(phrases, connectors)
        actions = _evaluate(phrases, connectors)
        answer = " ".join(actions)
        prompt = f"<task=scan_command_to_actions>\nInput: {command}\nAnswer:"
        target = f" {answer}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"scan_{bin_name}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "compositional_interpreter",
                "is_control": False,
                "difficulty": dict(difficulty),
            },
        )
