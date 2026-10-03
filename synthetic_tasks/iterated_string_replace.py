"""
- description:
    Ordered global string-replacement task family for the shadow-ADO
    learnability experiment. Each prompt gives an initial string and a short
    program of Python-style `str.replace(old, new)` operations; the target is
    exactly the final string after executing every operation in order.
    domain_id is `replace_steps_<num_steps>_len_<initial_len>`.
- paper_alignment:
    source_paper: "From Symbolic Tasks to Code Generation: Diversification Yields Better Task Performers"
    source_url: "https://arxiv.org/abs/2405.19787"
    note: >
        The task uses the same broad symbolic string-replacement setting as the
        Markov-algorithm family, but swaps first-applicable one-step rewriting
        for an ordered program of Python global replacements.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("iterated_string_replace",
                    "replace_steps_4_len_32", "train", 0)
    ex = get_task("iterated_string_replace")().sample(
        rng, "train", {
            "initial_len": 32, "num_steps": 4, "alphabet_size": 4,
            "pattern_len_min": 1, "pattern_len_max": 3,
            "replacement_len_min": 0, "replacement_len_max": 3,
            "active_replace_rate": 0.80, "noop_rate": 0.15,
            "cascade_rate": 0.20, "delete_rate": 0.10,
            "growth_bias": 0.25, "max_final_len": 72,
            "min_final_len": 1, "min_active_steps": 2,
            "min_distinct_states": 3, "allow_identity_examples": False,
            "require_deletion": False, "require_growth": False,
            "require_shrink": False, "require_cascade": False})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a narrow symbolic-execution task where the program is
        fully visible and the state is a mutable string, so a model must track
        ordered global replacements, non-overlapping Python replace semantics,
        no-op updates, deletion, growth, and cascades without needing a full
        Python interpreter. The generator stores every intermediate state so
        failures can be replayed from metadata.
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


def _sample_word(rng: random.Random, alphabet: list[str], length: int) -> str:
    assert length >= 0, length
    return "".join(rng.choice(alphabet) for _ in range(length))


def _sample_substring(
        rng: random.Random,
        value: str,
        min_len: int,
        max_len: int) -> str:
    capped_max_len = min(max_len, len(value))
    assert min_len <= capped_max_len, (value, min_len, max_len)
    length = rng.randint(min_len, capped_max_len)
    start = rng.randint(0, len(value) - length)
    return value[start:start + length]


def _sample_absent_word(
        rng: random.Random,
        alphabet: list[str],
        value: str,
        min_len: int,
        max_len: int) -> str:
    for length in range(max_len, min_len - 1, -1):
        for _ in range(512):
            candidate = _sample_word(rng, alphabet, length)
            if candidate not in value:
                return candidate
    raise RuntimeError((value, min_len, max_len, alphabet))


def _sample_replacement(
        rng: random.Random,
        alphabet: list[str],
        old: str,
        replacement_len_min: int,
        replacement_len_max: int,
        delete_rate: float,
        growth_bias: float) -> str:
    if rng.random() < delete_rate:
        return ""
    min_len = replacement_len_min
    growth_min_len = max(replacement_len_min, len(old) + 1)
    if rng.random() < growth_bias and growth_min_len <= replacement_len_max:
        min_len = growth_min_len
    length = rng.randint(min_len, replacement_len_max)
    return _sample_word(rng, alphabet, length)


def _eval_replace_program(
        initial: str,
        operations: list[dict[str, str]]) -> tuple[str, list[str]]:
    current = initial
    trace = [current]
    for operation in operations:
        current = current.replace(operation["old"], operation["new"])
        trace.append(current)
    return current, trace


def _has_cascade(trace: list[str], operations: list[dict[str, str]]) -> bool:
    for maker_index in range(len(operations)):
        before = trace[maker_index]
        after = trace[maker_index + 1]
        for later_index in range(maker_index + 1, len(operations)):
            old = operations[later_index]["old"]
            if old not in before and old in after and trace[later_index] != trace[later_index + 1]:
                return True
    return False


def _sample_program(
        rng: random.Random,
        initial: str,
        alphabet: list[str],
        num_steps: int,
        pattern_len_min: int,
        pattern_len_max: int,
        replacement_len_min: int,
        replacement_len_max: int,
        active_replace_rate: float,
        noop_rate: float,
        cascade_rate: float,
        delete_rate: float,
        growth_bias: float) -> list[dict[str, str]]:
    current = initial
    operations = []
    forced_old: str | None = None
    for step_index in range(num_steps):
        can_sample_active = len(current) >= pattern_len_min
        should_force_cascade = (
            forced_old is None
            and step_index + 1 < num_steps
            and can_sample_active
            and rng.random() < cascade_rate
        )
        if should_force_cascade:
            old = _sample_substring(rng, current, pattern_len_min, pattern_len_max)
            bridge_max_len = min(pattern_len_max, replacement_len_max)
            bridge_min_len = max(pattern_len_min, replacement_len_min)
            forced_old = _sample_absent_word(
                rng,
                alphabet,
                current,
                bridge_min_len,
                bridge_max_len,
            )
            new = forced_old
        elif forced_old is not None and forced_old in current:
            old = forced_old
            forced_old = None
            for _ in range(512):
                new = _sample_replacement(
                    rng,
                    alphabet,
                    old,
                    replacement_len_min,
                    replacement_len_max,
                    delete_rate,
                    growth_bias,
                )
                if new != old:
                    break
            else:
                raise RuntimeError((old, replacement_len_min, replacement_len_max))
        else:
            forced_old = None
            if rng.random() < noop_rate:
                old = _sample_absent_word(
                    rng,
                    alphabet,
                    current,
                    pattern_len_min,
                    pattern_len_max,
                )
            elif can_sample_active and rng.random() < active_replace_rate:
                old = _sample_substring(rng, current, pattern_len_min, pattern_len_max)
            else:
                old = _sample_word(
                    rng,
                    alphabet,
                    rng.randint(pattern_len_min, pattern_len_max),
                )
            new = _sample_replacement(
                rng,
                alphabet,
                old,
                replacement_len_min,
                replacement_len_max,
                delete_rate,
                growth_bias,
            )
        operations.append({"old": old, "new": new})
        current = current.replace(old, new)
    return operations


@register_task
class IteratedStringReplaceTask(SyntheticTask):
    task_id = "iterated_string_replace"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        initial_len = difficulty["initial_len"]
        num_steps = difficulty["num_steps"]
        alphabet_size = difficulty["alphabet_size"]
        pattern_len_min = difficulty["pattern_len_min"]
        pattern_len_max = difficulty["pattern_len_max"]
        replacement_len_min = difficulty["replacement_len_min"]
        replacement_len_max = difficulty["replacement_len_max"]
        active_replace_rate = difficulty["active_replace_rate"]
        noop_rate = difficulty["noop_rate"]
        cascade_rate = difficulty["cascade_rate"]
        delete_rate = difficulty["delete_rate"]
        growth_bias = difficulty["growth_bias"]
        max_final_len = difficulty["max_final_len"]
        min_final_len = difficulty["min_final_len"]
        min_active_steps = difficulty["min_active_steps"]
        min_distinct_states = difficulty["min_distinct_states"]
        allow_identity_examples = difficulty["allow_identity_examples"]
        require_deletion = difficulty["require_deletion"]
        require_growth = difficulty["require_growth"]
        require_shrink = difficulty["require_shrink"]
        require_cascade = difficulty["require_cascade"]

        assert initial_len >= 1, initial_len
        assert num_steps >= 1, num_steps
        assert 1 <= alphabet_size <= len(string.ascii_lowercase), alphabet_size
        assert 1 <= pattern_len_min <= pattern_len_max, difficulty
        assert 0 <= replacement_len_min <= replacement_len_max, difficulty
        assert min_final_len >= 1, min_final_len
        assert min_active_steps >= 1, min_active_steps
        assert min_distinct_states >= 2, min_distinct_states

        alphabet = list(string.ascii_lowercase[:alphabet_size])
        domain_id = f"replace_steps_{num_steps}_len_{initial_len}"
        for _ in range(1000):
            initial = _sample_word(rng, alphabet, initial_len)
            operations = _sample_program(
                rng,
                initial,
                alphabet,
                num_steps,
                pattern_len_min,
                pattern_len_max,
                replacement_len_min,
                replacement_len_max,
                active_replace_rate,
                noop_rate,
                cascade_rate,
                delete_rate,
                growth_bias,
            )
            final, trace = _eval_replace_program(initial, operations)
            active_steps = sum(
                1
                for before, after in zip(trace, trace[1:])
                if before != after
            )
            noop_steps = num_steps - active_steps
            has_deletion = any(operation["new"] == "" for operation in operations)
            has_growth = any(
                len(operation["new"]) > len(operation["old"])
                for operation in operations
            )
            has_shrink = any(
                len(operation["new"]) < len(operation["old"])
                for operation in operations
            )
            has_cascade = _has_cascade(trace, operations)
            if len(final) > max_final_len or len(final) < min_final_len:
                continue
            if active_steps < min_active_steps:
                continue
            if len(set(trace)) < min_distinct_states:
                continue
            if final == initial and not allow_identity_examples:
                continue
            if require_deletion and not has_deletion:
                continue
            if require_growth and not has_growth:
                continue
            if require_shrink and not has_shrink:
                continue
            if require_cascade and not has_cascade:
                continue
            break
        else:
            raise RuntimeError((self.task_id, difficulty))

        prompt = "\n".join([
            "<task=iterated_string_replace>",
            f"Start string: {initial}",
            "",
            "Operations:",
            *[
                f"replace(\"{operation['old']}\", \"{operation['new']}\")"
                for operation in operations
            ],
            "",
            "Final string:",
            "",
        ])
        target = final
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=domain_id,
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "mutable_state_tracking",
                "domain_id": domain_id,
                "difficulty_bin": domain_id,
                "control_status": "normal",
                "is_control": False,
                "difficulty": dict(difficulty),
                "initial_len": len(initial),
                "final_len": len(final),
                "num_steps": len(operations),
                "alphabet_size": alphabet_size,
                "active_steps": active_steps,
                "noop_steps": noop_steps,
                "has_deletion": has_deletion,
                "has_growth": has_growth,
                "has_shrink": has_shrink,
                "has_cascade": has_cascade,
                "requires_symbolic_execution": True,
                "requires_ordered_updates": True,
                "requires_mutable_state": True,
                "uses_global_replace": True,
                "has_program_trace": True,
                "initial": initial,
                "final": final,
                "ops": [dict(operation) for operation in operations],
                "trace": list(trace),
            },
        )
