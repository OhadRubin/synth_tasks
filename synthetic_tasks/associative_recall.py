"""
- description:
    Associative-recall task family for the shadow-ADO learnability
    experiment. The model is shown a set of N key=value pairs followed by
    a single key query, and must emit the value bound to the queried key.
    The `n_pairs` difficulty knob produces three bins (4, 16, 64) per
    tasks.yaml. domain_id is `associative_recall_pairs_<n_pairs>`. Loss
    is applied only to the answer region via build_loss_mask from base.py.

    Key alphabet extension: the pilot's highest bin requests 64 distinct
    keys, which exceeds the 26 lowercase letters a-z. Keys are drawn from
    a deterministic extended alphabet (a..z then aa, ab, ..., zz) so that
    bin n_pairs=64 uses 26 single-char keys plus the next 38 two-char
    keys. Keys remain unambiguous because they are always followed by
    `=<digit>` and separated from each other by spaces.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("associative_recall",
                    "associative_recall_pairs_16", "train", 0)
    ex = get_task("associative_recall")().sample(
        rng, "train", {"n_pairs": 16})
- user_story:
    content: |
        Associative recall probes the model's ability to bind variables to
        values inside a single context window and then retrieve one of
        those bindings on demand. Shadow-ADO is expected to mark
        n_pairs=4 as quickly saturated, while larger bins (16, 64)
        should remain actively learnable for a longer window because
        binding capacity scales with the number of distinct keys the
        model must keep distinguishable. The generator samples a set of
        N distinct keys from a deterministic extended alphabet, assigns
        each key a uniformly random single decimal digit value, then
        samples a query key uniformly from the bound keys. The target is
        the digit bound to the queried key; the loss mask covers only
        that target byte region.
    was_generated_via_skill: false
"""

from __future__ import annotations

import random
import string
from typing import Any

from synthetic_tasks.base import (
    Example,
    SyntheticTask,
    build_loss_mask,
    register_task,
)


_VALUE_ALPHABET = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


def _extended_key_alphabet(n: int) -> list[str]:
    if n < 1:
        raise ValueError(f"need at least 1 key, got {n}")
    letters = string.ascii_lowercase
    keys: list[str] = list(letters)
    if n <= len(keys):
        return keys[:n]
    for a in letters:
        for b in letters:
            keys.append(a + b)
            if len(keys) == n:
                return keys
    raise ValueError(
        f"extended key alphabet exhausted at {len(keys)} keys, requested {n}"
    )


@register_task
class AssociativeRecallTask(SyntheticTask):
    task_id = "associative_recall"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "n_pairs" not in difficulty:
            raise ValueError(
                f"associative_recall difficulty must include 'n_pairs', got {difficulty!r}"
            )
        n_pairs = difficulty["n_pairs"]
        if not isinstance(n_pairs, int) or isinstance(n_pairs, bool):
            raise ValueError(
                f"associative_recall difficulty.n_pairs must be int, got "
                f"{type(n_pairs).__name__}"
            )
        if n_pairs < 1:
            raise ValueError(
                f"associative_recall difficulty.n_pairs must be >= 1, got {n_pairs}"
            )

        key_pool = _extended_key_alphabet(n_pairs)
        keys = list(key_pool)
        rng.shuffle(keys)
        values = [rng.choice(_VALUE_ALPHABET) for _ in range(n_pairs)]
        bindings = dict(zip(keys, values))
        query_key = rng.choice(keys)
        answer = bindings[query_key]

        pairs_str = " ".join(f"{k}={v}" for k, v in zip(keys, values))
        prompt = (
            f"<task=associative_recall>\n"
            f"Pairs: {pairs_str}\n"
            f"Question: {query_key}?\n"
            f"Answer:"
        )
        target = f" {answer}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"associative_recall_pairs_{n_pairs}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "retrieval_or_binding",
                "is_control": False,
                "difficulty": dict(difficulty),
            },
        )
