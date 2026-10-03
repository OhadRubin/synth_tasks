"""
- description:
    Reverse-sequence task family for the shadow-ADO learnability experiment.
    The model must emit a length-N sequence of single-digit tokens in
    reverse order of the input. The `length` difficulty knob produces
    three bins (16, 64, 128) per tasks.yaml. domain_id is
    `reverse_len_<length>`. Loss is applied only to the answer region via
    build_loss_mask from base.py.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("reverse_sequence", "reverse_len_16", "train", 0)
    ex = get_task("reverse_sequence")().sample(rng, "train", {"length": 16})
- user_story:
    content: |
        Reverse-sequence is the positional-reversal counterpart to copy in
        the exact-transduction family: same input alphabet, same length
        sweep, but the model must invert the order. This isolates whether
        the model has acquired a positional-reversal mechanism rather than
        an identity-reproduction one. Shadow-ADO is expected to mark short
        reverse bins as saturated quickly while keeping the longer bins
        (64, 128) actively learnable longer, ideally with a steeper
        learnability decay than copy because reversal forces a
        non-trivial positional rearrangement. The generator emits
        sequences of single decimal digits drawn uniformly with
        replacement so any length-N input is equally likely; determinism
        is supplied by the seeded random.Random passed in.
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


_ALPHABET = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


@register_task
class ReverseSequenceTask(SyntheticTask):
    task_id = "reverse_sequence"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "length" not in difficulty:
            raise ValueError(
                f"reverse_sequence difficulty must include 'length', got {difficulty!r}"
            )
        length = difficulty["length"]
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError(
                f"reverse_sequence difficulty.length must be int, got {type(length).__name__}"
            )
        if length < 1:
            raise ValueError(
                f"reverse_sequence difficulty.length must be >= 1, got {length}"
            )

        tokens = [rng.choice(_ALPHABET) for _ in range(length)]
        input_sequence = " ".join(tokens)
        answer_sequence = " ".join(reversed(tokens))
        prompt = f"<task=reverse_sequence>\nInput: {input_sequence}\nAnswer:"
        target = f" {answer_sequence}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"reverse_len_{length}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "exact_transduction",
                "is_control": False,
                "difficulty": dict(difficulty),
            },
        )
