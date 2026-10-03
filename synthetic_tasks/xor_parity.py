"""
- description:
    XOR-parity task family for the shadow-ADO learnability experiment. The
    model receives a length-N binary sequence over {0, 1} and must emit
    the single-bit cumulative XOR (parity) of the input. The `length`
    difficulty knob produces three bins (16, 64, 128) per tasks.yaml.
    domain_id is `xor_parity_len_<length>`. Loss is applied only to the
    answer region via build_loss_mask from base.py.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("xor_parity", "xor_parity_len_16", "train", 0)
    ex = get_task("xor_parity")().sample(rng, "train", {"length": 16})
- user_story:
    content: |
        XOR-parity is the canonical global-readout probe that requires the
        model to integrate every single input bit before committing to a
        one-bit answer: flipping any single input bit must flip the label.
        Unlike majority, parity has no local-statistic shortcut, so it is
        expected to be the hardest of the binary-classification families
        in the suite. Shadow-ADO is expected to keep the longer parity
        bins (64, 128) flagged as actively learnable for a long window,
        with the short bin (16) saturating earlier; the contrast against
        majority at matched length isolates the cost of true global
        integration over local counting. Generator emits uniform iid
        Bernoulli(0.5) bits; parity is always defined (no tie edge case)
        and is roughly balanced across many samples.
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
class XorParityTask(SyntheticTask):
    task_id = "xor_parity"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "length" not in difficulty:
            raise ValueError(f"xor_parity difficulty must include 'length', got {difficulty!r}")
        length = difficulty["length"]
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError(
                f"xor_parity difficulty.length must be int, got {type(length).__name__}"
            )
        if length < 1:
            raise ValueError(f"xor_parity difficulty.length must be >= 1, got {length}")

        tokens = [rng.choice(_ALPHABET) for _ in range(length)]
        parity = 0
        for tok in tokens:
            parity ^= int(tok)
        answer = str(parity)
        sequence = " ".join(tokens)
        prompt = f"<task=xor_parity>\nInput: {sequence}\nAnswer:"
        target = f" {answer}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"xor_parity_len_{length}",
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
