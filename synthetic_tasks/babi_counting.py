"""
- description:
    bAbI counting synthetic task (bAbI task 7) for the shadow-ADO
    learnability experiment. Group D / mutable_state_tracking macro
    family additive variant alongside babi_single_hop and
    babi_multi_hop. The model reads a short narrative containing K
    reserved "P picked up N <Q-pluralized>." facts (for a single
    target person P and target object class Q), plus distractor facts
    that mention neither P nor Q in take-events (and arbitrary
    person/location movement facts that do not bear on the count).
    Ground truth is the running sum of N across the K reserved facts.
    Difficulty knob is n_facts (story length). For n_facts=5, K is
    drawn from [2..3]; for n_facts>=20, K is drawn from [3..6]. The
    answer is a single integer token prefixed with one space, matching
    the babi family convention.

    Design mirrors babi_multi_hop's reserved-position pattern: K
    indices in [0, n_facts) are reserved for take-events for the
    (P, Q) pair, the remaining slots are filled with distractor
    take-events (other person + other object) or distractor movement
    events (any person + any location). This guarantees the answer is
    a deterministic function of the K reserved facts only and that no
    distractor can shift the count, so the runtime invariant check
    (sum_N == int(target)) is exact-by-construction.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("babi_counting", "babi_count_5facts", "train", 0)
    ex = get_task("babi_counting")().sample(
        rng, "train", {"n_facts": 5})
- user_story:
    content: |
        bAbI counting (task 7) is the explicit "may swap" alternate
        for babi_multi_hop in tasks.yaml and is additive here as the
        15th task family. It probes whether a small LM can maintain a
        running integer count across a short narrative while ignoring
        distractor events that mention different entities. The macro
        family mutable_state_tracking already covers location
        tracking (single_hop) and binding+location composition
        (multi_hop); babi_counting adds the third mutable-state
        sub-mechanism (running accumulator over a typed predicate),
        giving the shadow-ADO learnability curve a third independent
        sample within the same macro family. The generator guarantees
        that the answer is determined exactly by the K reserved take
        facts and that every distractor is provably irrelevant by
        construction (other-person AND other-object for take events,
        any person/location for movement events).
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


_PERSONS = ("Mary", "John", "Sandra", "Daniel")
_LOCATIONS = ("kitchen", "garden", "bedroom", "office", "hallway")
_OBJECTS = ("apple", "book", "ball", "key")


def _pluralize(obj: str) -> str:
    return obj + "s"


def _require_int(d: dict[str, Any], key: str, task_id: str) -> int:
    if key not in d:
        raise ValueError(f"{task_id} difficulty must include {key!r}, got {d!r}")
    v = d[key]
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(
            f"{task_id} difficulty.{key} must be int, got {type(v).__name__}"
        )
    return v


def _k_range(n_facts: int) -> tuple[int, int]:
    if n_facts >= 20:
        return 3, 6
    if n_facts >= 5:
        return 2, 3
    raise ValueError(
        f"babi_counting difficulty.n_facts must be >= 5 so the K range fits, "
        f"got {n_facts}"
    )


@register_task
class BabiCountingTask(SyntheticTask):
    task_id = "babi_counting"

    def sample(
        self,
        rng: random.Random,
        split: str,
        difficulty: dict[str, Any],
    ) -> Example:
        if split not in ("train", "val"):
            raise ValueError(
                f"babi_counting split must be 'train' or 'val', got {split!r}"
            )
        n_facts = _require_int(difficulty, "n_facts", self.task_id)
        k_min, k_max = _k_range(n_facts)
        if k_max >= n_facts:
            raise ValueError(
                f"babi_counting: K range [{k_min},{k_max}] must fit within "
                f"n_facts={n_facts}"
            )

        target_person = rng.choice(_PERSONS)
        target_object = rng.choice(_OBJECTS)
        target_plural = _pluralize(target_object)
        other_persons = [p for p in _PERSONS if p != target_person]
        other_objects = [o for o in _OBJECTS if o != target_object]

        k = rng.randint(k_min, k_max)
        reserved_positions = sorted(rng.sample(range(n_facts), k))

        fact_lines: list[str | None] = [None] * n_facts
        total = 0
        for pos in reserved_positions:
            n = rng.randint(1, 3)
            total += n
            fact_lines[pos] = (
                f"{target_person} picked up {n} {target_plural}."
            )

        for i in range(n_facts):
            if fact_lines[i] is not None:
                continue
            if rng.random() < 0.5:
                op = rng.choice(_PERSONS)
                loc = rng.choice(_LOCATIONS)
                fact_lines[i] = f"{op} went to the {loc}."
            else:
                op = rng.choice(other_persons)
                obj = rng.choice(other_objects)
                n = rng.randint(1, 3)
                fact_lines[i] = (
                    f"{op} picked up {n} {_pluralize(obj)}."
                )

        prompt = (
            "<task=babi_counting>\n"
            "Story:\n"
            + "\n".join(fact_lines)
            + f"\nQuestion: How many {target_plural} is {target_person} carrying?\n"
            + "Answer:"
        )
        target = f" {total}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"babi_count_{n_facts}facts",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "mutable_state_tracking",
                "operation_type": "count_accumulation",
                "solver_mechanism": "running_count",
                "is_control": False,
                "requires_state_tracking": True,
                "has_distractors": True,
                "difficulty": dict(difficulty),
            },
        )
