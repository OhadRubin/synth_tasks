"""
- description:
    Ordered Markov first-rewrite task family for the shadow-ADO learnability
    experiment. Each prompt gives an ordered list of string rewrite rules and
    one input string; the target is the result of applying exactly one
    first-applicable, leftmost-occurrence rewrite step, or the unchanged input
    when no rule applies. domain_id is
    `markov_first_rewrite_r<n_rules>_l<input_length>_p<lhs_len>`.
- paper_alignment:
    source_paper: "From Symbolic Tasks to Code Generation: Diversification Yields Better Task Performers"
    source_url: "https://arxiv.org/abs/2405.19787"
    note: >
        The task follows the paper's Markov-algorithm string-replacement setup,
        but uses an ordered rule list so examples test both instruction
        selection and exact substring replacement under fresh sampled rules.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("markov_first_rewrite",
                    "markov_first_rewrite_r4_l64_p3", "train", 0)
    ex = get_task("markov_first_rewrite")().sample(
        rng, "train", {
            "n_rules": 4, "input_length": 64, "lhs_len": 3,
            "rhs_min_len": 1, "rhs_max_len": 5, "no_op_prob": 0.25})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a symbolic instruction-following task where every
        example carries its own rewrite program, so a model must choose the
        first applicable rule, replace that rule's leftmost match, copy the
        untouched parts exactly, and recognize no-op cases when no rule matches.
        The generator samples fresh rules and strings per example while storing
        enough metadata for deterministic replay.
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


_ALPHABET = string.ascii_lowercase


def _sample_word(rng: random.Random, length: int) -> str:
    assert length >= 1, length
    return "".join(rng.choice(_ALPHABET) for _ in range(length))


def _first_rewrite(
        rules: list[dict[str, str]],
        input_string: str) -> tuple[str, int | None, int | None]:
    for index, rule in enumerate(rules):
        position = input_string.find(rule["lhs"])
        if position == -1:
            continue
        output = (
            input_string[:position]
            + rule["rhs"]
            + input_string[position + len(rule["lhs"]):]
        )
        return output, index, position
    return input_string, None, None


def _sample_rules(
        rng: random.Random,
        n_rules: int,
        lhs_len: int,
        rhs_min_len: int,
        rhs_max_len: int) -> list[dict[str, str]]:
    seen_lhs = set()
    rules = []
    while len(rules) < n_rules:
        lhs = _sample_word(rng, lhs_len)
        if lhs in seen_lhs:
            continue
        rhs = _sample_word(rng, rng.randint(rhs_min_len, rhs_max_len))
        if rhs == lhs:
            continue
        seen_lhs.add(lhs)
        rules.append({"lhs": lhs, "rhs": rhs})
    return rules


def _sample_no_op_input(
        rng: random.Random,
        rules: list[dict[str, str]],
        input_length: int) -> str:
    for _ in range(10000):
        candidate = _sample_word(rng, input_length)
        if all(rule["lhs"] not in candidate for rule in rules):
            return candidate
    raise RuntimeError((input_length, rules))


def _sample_applicable_input(
        rng: random.Random,
        rules: list[dict[str, str]],
        input_length: int,
        lhs_len: int) -> str:
    intended_index = rng.randrange(len(rules))
    intended_lhs = rules[intended_index]["lhs"]
    earlier_lhs = [rule["lhs"] for rule in rules[:intended_index]]
    for _ in range(10000):
        filler = _sample_word(rng, input_length - lhs_len)
        position = rng.randint(0, input_length - lhs_len)
        candidate = filler[:position] + intended_lhs + filler[position:]
        if any(lhs in candidate for lhs in earlier_lhs):
            continue
        output, selected_index, _ = _first_rewrite(rules, candidate)
        assert output != candidate, (rules, candidate)
        if selected_index == intended_index:
            return candidate
    raise RuntimeError((input_length, lhs_len, intended_index, rules))


@register_task
class MarkovFirstRewriteTask(SyntheticTask):
    task_id = "markov_first_rewrite"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        n_rules = difficulty["n_rules"]
        input_length = difficulty["input_length"]
        lhs_len = difficulty["lhs_len"]
        rhs_min_len = difficulty["rhs_min_len"]
        rhs_max_len = difficulty["rhs_max_len"]
        no_op_prob = difficulty["no_op_prob"]

        assert n_rules >= 1, n_rules
        assert input_length >= lhs_len, difficulty
        assert 1 <= rhs_min_len <= rhs_max_len, difficulty
        assert 0 <= no_op_prob <= 1, no_op_prob

        rules = _sample_rules(rng, n_rules, lhs_len, rhs_min_len, rhs_max_len)
        should_no_op = rng.random() < no_op_prob
        if should_no_op:
            input_string = _sample_no_op_input(rng, rules, input_length)
        else:
            input_string = _sample_applicable_input(rng, rules, input_length, lhs_len)

        output, selected_rule_index, selected_position = _first_rewrite(rules, input_string)
        no_op = selected_rule_index is None
        assert no_op == (output == input_string), (rules, input_string, output)
        assert no_op == should_no_op, (difficulty, rules, input_string, output)

        prompt_lines = ["<task=markov_first_rewrite>", "Rules:"]
        for index, rule in enumerate(rules, 1):
            prompt_lines.append(f"{index}. {rule['lhs']} -> {rule['rhs']}")
        prompt_lines.extend((f"Input: {input_string}", "Output:"))
        prompt = "\n".join(prompt_lines)
        target = f" {output}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"markov_first_rewrite_r{n_rules}_l{input_length}_p{lhs_len}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "algorithmic_string_rewriting",
                "is_control": False,
                "difficulty": dict(difficulty),
                "rules": [dict(rule) for rule in rules],
                "input": input_string,
                "output": output,
                "selected_rule_index": selected_rule_index,
                "selected_position": selected_position,
                "no_op": no_op,
            },
        )
