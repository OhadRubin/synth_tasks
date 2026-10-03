"""
- description:
    ListOps nested-prefix task family for the shadow-ADO learnability
    experiment. Probes compositional / recursive evaluation: the model
    sees a fully bracketed prefix expression built from four operators
    over single-digit operands and must emit the single-digit answer
    obtained by evaluating the expression. Operators are MAX (largest),
    MIN (smallest), MED (median, ties broken low via floor((n-1)/2)
    index of the sorted operand list), and SUM_MOD (sum modulo 10).
    Operands are digits 0-9; each subexpression has arity in [2, 3]
    (capped to keep depth-6 expressions inside the 1024-byte context
    budget). The maximum nesting depth reached is constrained to
    EXACTLY equal the configured `depth` bin, mirroring the dyck
    family's honest-knob discipline. domain_id is
    `listops_depth_<depth>`. Loss is applied only over the answer
    region via build_loss_mask.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("listops_nested_prefix", "listops_depth_4", "train", 0)
    ex = get_task("listops_nested_prefix")().sample(
        rng, "train", {"depth": 4}
    )
- user_story:
    content: |
        ListOps is the canonical compositional-interpreter probe: to
        produce the answer the model must parse a tree, recursively
        evaluate inner subexpressions, and combine results under the
        outer operator. We constrain the generator so the realized
        maximum nesting depth equals the configured depth knob, which
        keeps difficulty bins honest (depth-6 examples actually exercise
        a depth-6 stack of partial results, not just a wide shallow
        list). Shadow ADO is expected to show a clear depth-ordered
        learnability phase: depth_2 saturates first, depth_4 later,
        depth_6 remains learnable longest or plateaus below ceiling,
        contrasting with the stack-only Dyck signal because ListOps
        further requires the model to combine operand values under
        operator semantics, not merely match bracket types.
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


_OPS = ("MAX", "MIN", "MED", "SUM_MOD")
_MIN_ARITY = 2
_MAX_ARITY = 3


def _apply_op(op: str, args: list[int]) -> int:
    if op == "MAX":
        return max(args)
    if op == "MIN":
        return min(args)
    if op == "MED":
        s = sorted(args)
        return s[(len(s) - 1) // 2]
    if op == "SUM_MOD":
        return sum(args) % 10
    raise ValueError(f"unknown op {op!r}")


def _gen_expr(rng: random.Random, depth: int) -> tuple:
    op = rng.choice(_OPS)
    arity = rng.randint(_MIN_ARITY, _MAX_ARITY)
    children: list[Any] = []
    if depth == 1:
        for _ in range(arity):
            children.append(rng.randint(0, 9))
        return (op, children)
    special_idx = rng.randrange(arity)
    for i in range(arity):
        if i == special_idx:
            children.append(_gen_expr(rng, depth - 1))
        else:
            if depth - 1 < 1 or rng.random() < 0.75:
                children.append(rng.randint(0, 9))
            else:
                sub_d = rng.randint(1, depth - 1)
                children.append(_gen_expr(rng, sub_d))
    return (op, children)


def _expr_depth(node: Any) -> int:
    if isinstance(node, int):
        return 0
    _, children = node
    sub = [_expr_depth(c) for c in children]
    return 1 + max(sub)


def _evaluate(node: Any) -> int:
    if isinstance(node, int):
        return node
    op, children = node
    return _apply_op(op, [_evaluate(c) for c in children])


def _serialize(node: Any) -> str:
    if isinstance(node, int):
        return str(node)
    op, children = node
    inner = " ".join(_serialize(c) for c in children)
    return f"[ {op} {inner} ]"


@register_task
class ListOpsNestedPrefixTask(SyntheticTask):
    task_id = "listops_nested_prefix"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "depth" not in difficulty:
            raise ValueError(f"listops difficulty must include 'depth', got {difficulty!r}")
        depth = difficulty["depth"]
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ValueError(f"listops difficulty.depth must be int, got {type(depth).__name__}")
        if depth < 1:
            raise ValueError(f"listops difficulty.depth must be >= 1, got {depth}")

        while True:
            tree = _gen_expr(rng, depth)
            if _expr_depth(tree) == depth:
                break

        expression = _serialize(tree)
        answer = _evaluate(tree)
        prompt = f"<task=listops_nested_prefix>\nInput: {expression}\nAnswer:"
        target = f" {answer}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"listops_depth_{depth}",
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
