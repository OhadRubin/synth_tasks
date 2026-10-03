"""
- description:
    Bounded tiny-Python execution prediction task family for the shadow-ADO
    learnability experiment. Each prompt shows a deterministic program from a
    restricted Python subset, and the target is the exact line printed by the
    final `print(...)` call. domain_id is
    `bounded_tiny_python_exec_v<n_vars>_u<n_updates>_d<loop_depth>_i<max_loop_iters>`.
- paper_alignment:
    source_paper: "SURGE: On the Potential of Large Language Models as General-Purpose Surrogate Code Executors"
    source_url: "https://arxiv.org/abs/2502.11167"
    note: >
        The task is an automatically sampled SURGE-style surrogate execution
        probe: models must predict the output of small bounded programs without
        relying on external execution, imports, input, or unbounded control flow.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("bounded_tiny_python_execution",
                    "bounded_tiny_python_exec_v4_u8_d1_i4", "train", 0)
    ex = get_task("bounded_tiny_python_execution")().sample(
        rng, "train", {
            "n_vars": 4, "n_updates": 8, "loop_depth": 1,
            "max_loop_iters": 4, "branch_prob": 0.25,
            "modulus": 10, "print_count": 3})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a compact surrogate code-execution domain where examples
        are safe to sample indefinitely, so learning curves can test sequential
        state tracking, modular arithmetic, bounded loops, branches, and exact
        output formatting. The generator builds a restricted program AST,
        renders it as Python, interprets the same AST directly, and masks loss
        only over the printed output.
    was_generated_via_skill: false
"""

from __future__ import annotations

import operator
import random
import string
from typing import Any

from synthetic_tasks.base import (
    Example,
    SyntheticTask,
    build_loss_mask,
    register_task,
)


_COMPARATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
_LOOP_VARS = ("i", "j", "k")
_PROMPT_HEADER = (
    "<task=bounded_tiny_python_execution>\n"
    "Given the following Python program, predict the line printed by the program.\n"
    "Program:"
)


def _variable_names(n_vars: int) -> list[str]:
    assert 1 <= n_vars <= 10, n_vars
    return list(string.ascii_lowercase[:n_vars])


def _assignment_expr(
        rng: random.Random,
        variables: list[str],
        loop_vars: list[str],
        target: str,
        modulus: int) -> dict[str, Any]:
    expr_kind = rng.choice(("add", "sub", "mul", "loop_add") if loop_vars else ("add", "sub", "mul"))
    if expr_kind == "add":
        y, z = rng.sample(variables, 2)
        return {"kind": "add", "y": y, "z": z, "const": rng.randrange(modulus)}
    if expr_kind == "sub":
        y, z = rng.sample(variables, 2)
        return {"kind": "sub", "y": y, "z": z, "const": rng.randrange(modulus)}
    if expr_kind == "mul":
        y, z = rng.sample(variables, 2)
        return {"kind": "mul", "y": y, "z": z, "const": rng.randrange(modulus)}
    if expr_kind == "loop_add":
        return {
            "kind": "loop_add",
            "x": target,
            "loop_var": rng.choice(loop_vars),
            "const": rng.randrange(modulus),
        }
    raise RuntimeError(expr_kind)


def _render_expr(expr: dict[str, Any], modulus: int) -> str:
    match expr["kind"]:
        case "add":
            return f"({expr['y']} + {expr['z']} + {expr['const']}) % {modulus}"
        case "sub":
            return f"({expr['y']} - {expr['z']} + {expr['const']}) % {modulus}"
        case "mul":
            return f"({expr['y']} * {expr['const']} + {expr['z']}) % {modulus}"
        case "loop_add":
            return f"({expr['x']} + {expr['loop_var']} + {expr['const']}) % {modulus}"
        case _:
            raise RuntimeError(expr)


def _eval_expr(expr: dict[str, Any], state: dict[str, int], modulus: int) -> int:
    match expr["kind"]:
        case "add":
            return (state[expr["y"]] + state[expr["z"]] + expr["const"]) % modulus
        case "sub":
            return (state[expr["y"]] - state[expr["z"]] + expr["const"]) % modulus
        case "mul":
            return (state[expr["y"]] * expr["const"] + state[expr["z"]]) % modulus
        case "loop_add":
            return (state[expr["x"]] + state[expr["loop_var"]] + expr["const"]) % modulus
        case _:
            raise RuntimeError(expr)


def _value_expr(
        rng: random.Random,
        variables: list[str],
        loop_vars: list[str],
        modulus: int) -> dict[str, Any]:
    value_kinds = ("var", "plus_const", "loop_plus") if loop_vars else ("var", "plus_const")
    value_kind = rng.choice(value_kinds)
    if value_kind == "var":
        return {"kind": "var", "var": rng.choice(variables)}
    if value_kind == "plus_const":
        return {
            "kind": "plus_const",
            "var": rng.choice(variables),
            "const": rng.randrange(modulus),
        }
    if value_kind == "loop_plus":
        return {
            "kind": "loop_plus",
            "var": rng.choice(variables),
            "loop_var": rng.choice(loop_vars),
            "const": rng.randrange(modulus),
        }
    raise RuntimeError(value_kind)


def _render_value_expr(expr: dict[str, Any], modulus: int) -> str:
    match expr["kind"]:
        case "var":
            return expr["var"]
        case "plus_const":
            return f"({expr['var']} + {expr['const']}) % {modulus}"
        case "loop_plus":
            return f"({expr['var']} + {expr['loop_var']} + {expr['const']}) % {modulus}"
        case _:
            raise RuntimeError(expr)


def _eval_value_expr(expr: dict[str, Any], state: dict[str, int], modulus: int) -> int:
    match expr["kind"]:
        case "var":
            return state[expr["var"]]
        case "plus_const":
            return (state[expr["var"]] + expr["const"]) % modulus
        case "loop_plus":
            return (state[expr["var"]] + state[expr["loop_var"]] + expr["const"]) % modulus
        case _:
            raise RuntimeError(expr)


def _condition(
        rng: random.Random,
        variables: list[str],
        loop_vars: list[str],
        modulus: int) -> dict[str, Any]:
    return {
        "left": _value_expr(rng, variables, loop_vars, modulus),
        "op": rng.choice(tuple(_COMPARATORS)),
        "right": _value_expr(rng, variables, loop_vars, modulus),
    }


def _render_condition(condition: dict[str, Any], modulus: int) -> str:
    return (
        f"{_render_value_expr(condition['left'], modulus)} "
        f"{condition['op']} "
        f"{_render_value_expr(condition['right'], modulus)}"
    )


def _eval_condition(condition: dict[str, Any], state: dict[str, int], modulus: int) -> bool:
    left = _eval_value_expr(condition["left"], state, modulus)
    right = _eval_value_expr(condition["right"], state, modulus)
    return bool(_COMPARATORS[condition["op"]](left, right))


def _assignment(
        rng: random.Random,
        variables: list[str],
        loop_vars: list[str],
        modulus: int) -> dict[str, Any]:
    target = rng.choice(variables)
    return {
        "kind": "assign",
        "target": target,
        "expr": _assignment_expr(rng, variables, loop_vars, target, modulus),
    }


def _statements(
        rng: random.Random,
        variables: list[str],
        loop_vars: list[str],
        assignment_budget: int,
        loop_depth: int,
        max_loop_iters: int,
        branch_prob: float,
        modulus: int) -> list[dict[str, Any]]:
    assert assignment_budget >= 1, assignment_budget
    statements = []
    remaining = assignment_budget
    while remaining > 0:
        if loop_depth > 0 and max_loop_iters >= 2 and remaining >= 2 and rng.random() < 0.35:
            loop_var = _LOOP_VARS[len(loop_vars)]
            body_budget = rng.randint(1, remaining - 1)
            statements.append({
                "kind": "for",
                "loop_var": loop_var,
                "iters": rng.randint(2, max_loop_iters),
                "body": _statements(
                    rng,
                    variables,
                    [*loop_vars, loop_var],
                    body_budget,
                    loop_depth - 1,
                    max_loop_iters,
                    branch_prob,
                    modulus,
                ),
            })
            remaining -= body_budget
        elif remaining >= 2 and rng.random() < branch_prob:
            statements.append({
                "kind": "if",
                "condition": _condition(rng, variables, loop_vars, modulus),
                "then": [_assignment(rng, variables, loop_vars, modulus)],
                "else": [_assignment(rng, variables, loop_vars, modulus)],
            })
            remaining -= 2
        else:
            statements.append(_assignment(rng, variables, loop_vars, modulus))
            remaining -= 1
    return statements


def _render_statement(statement: dict[str, Any], indent: int, modulus: int) -> list[str]:
    prefix = " " * indent
    match statement["kind"]:
        case "assign":
            return [
                f"{prefix}{statement['target']} = "
                f"{_render_expr(statement['expr'], modulus)}"
            ]
        case "if":
            lines = [f"{prefix}if {_render_condition(statement['condition'], modulus)}:"]
            for child in statement["then"]:
                lines.extend(_render_statement(child, indent + 4, modulus))
            lines.append(f"{prefix}else:")
            for child in statement["else"]:
                lines.extend(_render_statement(child, indent + 4, modulus))
            return lines
        case "for":
            lines = [f"{prefix}for {statement['loop_var']} in range({statement['iters']}):"]
            for child in statement["body"]:
                lines.extend(_render_statement(child, indent + 4, modulus))
            return lines
        case _:
            raise RuntimeError(statement)


def _execute_statement(statement: dict[str, Any], state: dict[str, int], modulus: int) -> None:
    match statement["kind"]:
        case "assign":
            state[statement["target"]] = _eval_expr(statement["expr"], state, modulus)
        case "if":
            branch = "then" if _eval_condition(statement["condition"], state, modulus) else "else"
            for child in statement[branch]:
                _execute_statement(child, state, modulus)
        case "for":
            loop_var = statement["loop_var"]
            had_previous = loop_var in state
            previous = state.get(loop_var)
            for value in range(statement["iters"]):
                state[loop_var] = value
                for child in statement["body"]:
                    _execute_statement(child, state, modulus)
            if had_previous:
                assert previous is not None, previous
                state[loop_var] = previous
            else:
                del state[loop_var]
        case _:
            raise RuntimeError(statement)


def _has_kind(statements: list[dict[str, Any]], kind: str) -> bool:
    for statement in statements:
        if statement["kind"] == kind:
            return True
        if statement["kind"] == "if" and (
                _has_kind(statement["then"], kind)
                or _has_kind(statement["else"], kind)):
            return True
        if statement["kind"] == "for" and _has_kind(statement["body"], kind):
            return True
    return False


def _render_program(
        initial_values: dict[str, int],
        statements: list[dict[str, Any]],
        printed_vars: list[str],
        modulus: int) -> list[str]:
    lines = [
        f"{var} = {value}"
        for var, value in initial_values.items()
    ]
    for statement in statements:
        lines.extend(_render_statement(statement, 0, modulus))
    lines.append(f"print({', '.join(printed_vars)})")
    return lines


@register_task
class BoundedTinyPythonExecutionTask(SyntheticTask):
    task_id = "bounded_tiny_python_execution"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        n_vars = difficulty["n_vars"]
        n_updates = difficulty["n_updates"]
        loop_depth = difficulty["loop_depth"]
        max_loop_iters = difficulty["max_loop_iters"]
        branch_prob = difficulty["branch_prob"]
        modulus = difficulty["modulus"]
        print_count = difficulty["print_count"]

        assert n_vars >= 2, n_vars
        assert n_updates >= 1, n_updates
        assert 0 <= loop_depth <= len(_LOOP_VARS), loop_depth
        assert max_loop_iters >= 0, max_loop_iters
        assert 0 <= branch_prob <= 1, branch_prob
        assert modulus >= 2, modulus
        assert 1 <= print_count <= n_vars, print_count

        variables = _variable_names(n_vars)
        initial_values = {
            var: rng.randrange(modulus)
            for var in variables
        }
        statements = _statements(
            rng,
            variables,
            [],
            n_updates,
            loop_depth,
            max_loop_iters,
            branch_prob,
            modulus,
        )
        printed_vars = variables[:print_count]

        final_state = dict(initial_values)
        for statement in statements:
            _execute_statement(statement, final_state, modulus)
        output = " ".join(str(final_state[var]) for var in printed_vars)

        program_lines = _render_program(initial_values, statements, printed_vars, modulus)
        prompt = "\n".join([
            _PROMPT_HEADER,
            *program_lines,
            "Output:",
        ])
        target = f" {output}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=(
                f"bounded_tiny_python_exec_v{n_vars}_u{n_updates}"
                f"_d{loop_depth}_i{max_loop_iters}"
            ),
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "surrogate_code_execution",
                "is_control": False,
                "difficulty": dict(difficulty),
                "variables": list(variables),
                "printed_vars": list(printed_vars),
                "initial_values": dict(initial_values),
                "final_values": {
                    var: final_state[var]
                    for var in variables
                },
                "program_lines": list(program_lines),
                "output": output,
                "has_loop": _has_kind(statements, "for"),
                "has_branch": _has_kind(statements, "if"),
            },
        )
