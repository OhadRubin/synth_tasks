"""
- description:
    Majority task family for the shadow-ADO learnability experiment. The
    model receives a length-N binary sequence over {0, 1} and must emit
    the single token that occurs strictly more often. The `length`
    difficulty knob produces three bins (16, 64, 128) per tasks.yaml.
    domain_id is `majority_len_<length>`. Tie handling: sequences with
    an exact 0/1 split are resampled until a strict majority exists, so
    every emitted example has an unambiguous label. Loss is applied
    only to the answer region via build_loss_mask from base.py.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("majority", "majority_len_16", "train", 0)
    ex = get_task("majority")().sample(rng, "train", {"length": 16})
- user_story:
    content: |
        Majority is the canonical global-readout probe in the synthetic
        suite: the model must integrate evidence over the full input
        sequence before committing to a single-bit answer, which is
        qualitatively harder than exact transduction (copy / reverse)
        even though the output token space is tiny. Shadow-ADO is
        expected to show majority's short-length bin saturating quickly
        while the longer bins (64, 128) remain learnable for a longer
        window, mirroring the copy-family curve but at a lower ceiling
        because counting is harder than identity. Generator emits
        uniform iid Bernoulli(0.5) bits and rejects ties so every
        example has a strict majority label; this keeps the label
        distribution roughly balanced and the training signal
        unambiguous.
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


_ALPHABET = ("0", "1")


@register_task
class MajorityTask(SyntheticTask):
    task_id = "majority"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "length" not in difficulty:
            raise ValueError(f"majority difficulty must include 'length', got {difficulty!r}")
        length = difficulty["length"]
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError(f"majority difficulty.length must be int, got {type(length).__name__}")
        if length < 1:
            raise ValueError(f"majority difficulty.length must be >= 1, got {length}")

        while True:
            tokens = [rng.choice(_ALPHABET) for _ in range(length)]
            ones = tokens.count("1")
            zeros = length - ones
            if ones != zeros:
                break

        answer = "1" if ones > zeros else "0"
        sequence = " ".join(tokens)
        prompt = f"<task=majority>\nInput: {sequence}\nAnswer:"
        target = f" {answer}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"majority_len_{length}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "finite_state_or_global_readout",
                "is_control": False,
                "difficulty": dict(difficulty),
            },
        )
