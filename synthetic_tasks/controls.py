"""
- description:
    Mandatory control task families for the shadow-ADO learnability experiment.
    constant_label is a saturated control: every example has the same answer,
    so loss should drop immediately and ADO potential should vanish after
    saturation. random_label is a noisy control: the answer is independent of
    the input, drawn from a fixed label set, so loss should stay near
    log|Y| and ADO must NOT assign high learning potential just because loss
    is high. Together they pin the bottom and top of ADO's diagnostic range.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("constant_label", "constant_label", "train", 0)
    ex = get_task("constant_label")().sample(rng, "train", {"label": "1"})
- user_story:
    content: |
        When evaluating whether shadow ADO can distinguish learnable from
        useless domains, two minimal controls are mandatory. constant_label
        produces the same target for every input and verifies that ADO marks
        a domain as saturated once loss collapses to zero. random_label
        produces a label uncorrelated with the input and verifies that ADO
        does NOT mistake noise for learnable signal — the random-label
        control is the single best sanity check on the ADO implementation,
        because if ADO ranks it highly for a long time, then either the
        learning-curve fit, the loss normalization, or the per-target-token
        NLL is wrong. Both controls share the synthetic-task protocol so
        they pack into the same dataloader and ADO logger as any other
        family.
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


@register_task
class ConstantLabelTask(SyntheticTask):
    task_id = "constant_label"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        label = difficulty["label"]
        if not isinstance(label, str):
            raise ValueError(f"constant_label difficulty.label must be str, got {type(label).__name__}")
        token = rng.choice(["0", "1"])
        prompt = f"<task=constant_label>\nInput: {token}\nAnswer:"
        target = f" {label}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id="constant_label",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "control_or_noise",
                "is_control": True,
                "difficulty": dict(difficulty),
            },
        )


@register_task
class RandomLabelTask(SyntheticTask):
    task_id = "random_label"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        label_set = difficulty["label_set"]
        if not isinstance(label_set, (list, tuple)) or len(label_set) < 2:
            raise ValueError(
                f"random_label difficulty.label_set must be list/tuple of >=2 labels, "
                f"got {label_set!r}"
            )
        for lab in label_set:
            if not isinstance(lab, str):
                raise ValueError(f"random_label label_set entries must be str, got {type(lab).__name__}")
        token = rng.choice(["0", "1"])
        label = rng.choice(list(label_set))
        prompt = f"<task=random_label>\nInput: {token}\nAnswer:"
        target = f" {label}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id="random_label",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "control_or_noise",
                "is_control": True,
                "difficulty": dict(difficulty),
            },
        )
