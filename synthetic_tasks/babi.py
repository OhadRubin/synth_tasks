"""
- description:
    bAbI synthetic task families for the shadow-ADO learnability experiment.
    Two task classes live here under the mutable_state_tracking macro
    family: babi_single_hop (entity location after a sequence of moves)
    and babi_multi_hop (object location requiring composition of a
    "took the X" fact with a subsequent "went to the Y" fact). Both share
    a small closed entity vocabulary so the target token set is tiny and
    the difficulty is governed by n_facts (and for multi-hop the implicit
    structural constraint that supporting_facts == 2). Domain ids encode
    the bin: babi_single_5facts, babi_single_20facts, babi_multi_20facts.
    Loss is applied only to the answer region via build_loss_mask.

    Design choices:
      - single_hop: pick a random move per fact. The queried person is
        sampled from the persons who actually moved in the story so the
        ground-truth answer is well-defined. Answer is that person's most
        recent location.
      - multi_hop: choose a target object Q, a target person P, and a
        target location L. Reserve two slots take_pos < went_pos in
        [0, n_facts) for "P took the Q." and "P went to the L."
        respectively. Fill remaining slots with distractor facts that
        never mention P and never mention Q so the two reserved facts
        are the only ones bearing on the answer (supporting_facts == 2,
        exactly). Distractors are split between other-person moves and
        other-person take-other-object facts.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("babi_single_hop", "babi_single_5facts", "train", 0)
    ex = get_task("babi_single_hop")().sample(rng, "train", {"n_facts": 5})

    rng = split_rng("babi_multi_hop", "babi_multi_20facts", "train", 0)
    ex = get_task("babi_multi_hop")().sample(
        rng, "train", {"n_facts": 20, "supporting_facts": 2})
- user_story:
    content: |
        bAbI-style tasks probe whether a small LM can track mutable
        entity state across a short narrative. Single-hop forces the
        model to keep an updated person->location table and read out one
        cell; multi-hop forces composition of a binding fact (person
        took object) with a movement fact (person went somewhere). Both
        live in the mutable_state_tracking macro family and are expected
        to show a distinct learnability profile from the
        retrieval_or_binding tasks like associative_recall: the loss
        plateau should be longer because the model has to maintain state
        across the prompt rather than do a single lookup. Shadow-ADO is
        expected to keep these domains learnable for an extended window,
        especially the multi-hop variant which has has_distractors=true
        and requires combining two facts. The generator guarantees a
        well-defined ground truth and (for multi-hop) the exact
        supporting_facts=2 structure so the test suite can mechanically
        verify that the answer is not derivable from any single fact.
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


def _require_int(d: dict[str, Any], key: str, task_id: str) -> int:
    if key not in d:
        raise ValueError(f"{task_id} difficulty must include {key!r}, got {d!r}")
    v = d[key]
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(
            f"{task_id} difficulty.{key} must be int, got {type(v).__name__}"
        )
    return v


@register_task
class BabiSingleHopTask(SyntheticTask):
    task_id = "babi_single_hop"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        n_facts = _require_int(difficulty, "n_facts", self.task_id)
        if n_facts < 1:
            raise ValueError(
                f"babi_single_hop difficulty.n_facts must be >= 1, got {n_facts}"
            )

        person_location: dict[str, str] = {}
        fact_lines: list[str] = []
        for _ in range(n_facts):
            p = rng.choice(_PERSONS)
            loc = rng.choice(_LOCATIONS)
            fact_lines.append(f"{p} went to the {loc}.")
            person_location[p] = loc

        moved_persons = sorted(person_location.keys())
        query_person = rng.choice(moved_persons)
        answer = person_location[query_person]

        prompt = (
            "<task=babi_single_hop>\n"
            "Story:\n"
            + "\n".join(fact_lines)
            + f"\nQuestion: Where is {query_person}?\n"
            + "Answer:"
        )
        target = f" {answer}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"babi_single_{n_facts}facts",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "mutable_state_tracking",
                "is_control": False,
                "difficulty": dict(difficulty),
            },
        )


@register_task
class BabiMultiHopTask(SyntheticTask):
    task_id = "babi_multi_hop"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        n_facts = _require_int(difficulty, "n_facts", self.task_id)
        supporting_facts = _require_int(difficulty, "supporting_facts", self.task_id)
        if n_facts < 2:
            raise ValueError(
                f"babi_multi_hop difficulty.n_facts must be >= 2, got {n_facts}"
            )
        if supporting_facts != 2:
            raise ValueError(
                f"babi_multi_hop only supports supporting_facts == 2 "
                f"(matches the took+went composition); got {supporting_facts}"
            )

        target_person = rng.choice(_PERSONS)
        target_object = rng.choice(_OBJECTS)
        target_location = rng.choice(_LOCATIONS)
        other_persons = [p for p in _PERSONS if p != target_person]
        other_objects = [o for o in _OBJECTS if o != target_object]

        positions = sorted(rng.sample(range(n_facts), 2))
        take_pos, went_pos = positions[0], positions[1]

        fact_lines: list[str | None] = [None] * n_facts
        fact_lines[take_pos] = f"{target_person} took the {target_object}."
        fact_lines[went_pos] = f"{target_person} went to the {target_location}."

        for i in range(n_facts):
            if fact_lines[i] is not None:
                continue
            op = rng.choice(other_persons)
            if rng.random() < 0.5:
                loc = rng.choice(_LOCATIONS)
                fact_lines[i] = f"{op} went to the {loc}."
            else:
                obj = rng.choice(other_objects)
                fact_lines[i] = f"{op} took the {obj}."

        prompt = (
            "<task=babi_multi_hop>\n"
            "Story:\n"
            + "\n".join(fact_lines)
            + f"\nQuestion: Where is the {target_object}?\n"
            + "Answer:"
        )
        target = f" {target_location}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"babi_multi_{n_facts}facts",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "mutable_state_tracking",
                "is_control": False,
                "difficulty": dict(difficulty),
                "has_distractors": True,
            },
        )
