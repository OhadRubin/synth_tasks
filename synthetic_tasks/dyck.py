"""
- description:
    Dyck balanced-parentheses task family for the shadow-ADO learnability
    experiment. Probes stack-state maintenance using two bracket types
    `( )` and `[ ]`. The model receives a VALID Dyck prefix (a sequence
    of opens/closes that never closes more than it has opened) whose
    maximum stack depth equals the configured `depth` knob, and must
    emit the sequence of closing brackets needed to balance the prefix
    (in correct LIFO order). domain_id is `dyck_depth_<depth>`. Loss is
    applied only over the answer region via build_loss_mask.

    Design choice: (b) closing-bracket prediction, NOT (a) binary
    classification. Rationale (documented per Lia's instruction): (b)
    forces the model to maintain and unwind a stack of bracket types,
    which is the spirit of macro_family `stack_or_hierarchy` and the
    `requires_stack=true` taxonomy flag. (a) only requires checking
    well-formedness, which collapses to a single bit and gives weak
    ADO signal compared to majority/xor_parity already in the suite.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("dyck_balanced_parentheses", "dyck_depth_4", "train", 0)
    ex = get_task("dyck_balanced_parentheses")().sample(
        rng, "train", {"depth": 4, "bracket_types": 2}
    )
- user_story:
    content: |
        The Dyck family is the canonical stack-or-hierarchy probe: to
        emit the correct closing sequence for a partially balanced
        prefix, the model must track which bracket type sits at each
        stack position and unwind them in last-in-first-out order. We
        deliberately constrain the generator so the maximum stack depth
        reached during the prefix equals the configured depth bin,
        which keeps the difficulty knob honest (depth-8 examples
        actually exercise a depth-8 stack at some point, not just a
        long shallow walk). Shadow-ADO is expected to show a clear
        depth-ordered phase: depth_2 saturates quickly, depth_4 takes
        longer, depth_8 remains learnable for a long window or plateaus
        below ceiling, mirroring the canonical Transformer-on-Dyck
        learnability story.
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


_OPENS = ("(", "[")
_CLOSES = (")", "]")
_PAIR = {"(": ")", "[": "]"}


@register_task
class DyckBalancedParenthesesTask(SyntheticTask):
    task_id = "dyck_balanced_parentheses"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "depth" not in difficulty:
            raise ValueError(f"dyck difficulty must include 'depth', got {difficulty!r}")
        if "bracket_types" not in difficulty:
            raise ValueError(f"dyck difficulty must include 'bracket_types', got {difficulty!r}")
        depth = difficulty["depth"]
        bracket_types = difficulty["bracket_types"]
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ValueError(f"dyck difficulty.depth must be int, got {type(depth).__name__}")
        if depth < 1:
            raise ValueError(f"dyck difficulty.depth must be >= 1, got {depth}")
        if not isinstance(bracket_types, int) or isinstance(bracket_types, bool):
            raise ValueError(
                f"dyck difficulty.bracket_types must be int, got {type(bracket_types).__name__}"
            )
        if bracket_types < 1 or bracket_types > len(_OPENS):
            raise ValueError(
                f"dyck difficulty.bracket_types must be in [1, {len(_OPENS)}], got {bracket_types}"
            )

        prefix_len = 4 * depth + 1
        opens = _OPENS[:bracket_types]

        while True:
            stack: list[str] = []
            tokens: list[str] = []
            max_seen = 0
            for _ in range(prefix_len):
                can_open = len(stack) < depth
                can_close = len(stack) > 0
                if can_open and can_close:
                    action = rng.choice(("open", "close"))
                elif can_open:
                    action = "open"
                else:
                    action = "close"

                if action == "open":
                    tok = rng.choice(opens)
                    tokens.append(tok)
                    stack.append(tok)
                else:
                    tokens.append(_PAIR[stack.pop()])
                if len(stack) > max_seen:
                    max_seen = len(stack)

            if max_seen == depth and len(stack) > 0:
                break

        answer_tokens = [_PAIR[c] for c in reversed(stack)]
        sequence = " ".join(tokens)
        answer = " ".join(answer_tokens)
        prompt = f"<task=dyck_balanced_parentheses>\nInput: {sequence}\nAnswer:"
        target = f" {answer}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"dyck_depth_{depth}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "stack_or_hierarchy",
                "is_control": False,
                "requires_stack": True,
                "difficulty": dict(difficulty),
            },
        )
