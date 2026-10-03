"""
- description:
    Copy task family for the shadow-ADO learnability experiment. The model
    must reproduce a length-N sequence of single-digit tokens verbatim. The
    `length` difficulty knob produces three bins (16, 64, 128) per
    tasks.yaml. domain_id is `copy_len_<length>`. Loss is applied only to
    the answer region via build_loss_mask from base.py.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("copy", "copy_len_16", "train", 0)
    ex = get_task("copy")().sample(rng, "train", {"length": 16})
- user_story:
    content: |
        The copy task is the simplest exact-transduction probe in the
        synthetic suite. Because the input and output are identical, the
        only thing the model has to learn is the identity-reproduction
        mechanism over a length-N sequence. Shadow-ADO is expected to mark
        short copy bins as saturated quickly while flagging the longer
        bins (64, 128) as actively learnable for a longer window — this
        is one of the clearest signals that ADO can separate easy from
        harder length-conditioned variants of the same operation. The
        generator emits sequences of single decimal digits drawn uniformly
        with replacement so that any length-N input is equally likely;
        determinism is supplied by the seeded random.Random passed in.
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
class CopyTask(SyntheticTask):
    task_id = "copy"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "length" not in difficulty:
            raise ValueError(f"copy difficulty must include 'length', got {difficulty!r}")
        length = difficulty["length"]
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError(f"copy difficulty.length must be int, got {type(length).__name__}")
        if length < 1:
            raise ValueError(f"copy difficulty.length must be >= 1, got {length}")

        tokens = [rng.choice(_ALPHABET) for _ in range(length)]
        sequence = " ".join(tokens)
        prompt = f"<task=copy>\nInput: {sequence}\nAnswer:"
        target = f" {sequence}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"copy_len_{length}",
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
