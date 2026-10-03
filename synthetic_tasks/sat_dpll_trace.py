"""
- description:
    Bounded SAT solving with canonical DPLL trace task family for the
    shadow-ADO learnability experiment. Each prompt serializes a small
    DIMACS-like 3-CNF formula, and the target is a deterministic reasoning
    trace containing decisions, unit propagations, backtracks, and a terminal
    `SAT` or `UNSAT` token. domain_id is `sat_dpll_p<n_vars>_c<n_clauses>`.
- paper_alignment:
    source_paper: "Can Transformers Reason Logically? A Study in SAT Solving"
    source_url: "https://arxiv.org/abs/2410.07432"
    note: >
        The task follows the paper's SAT-solving-with-CoT setup by requiring a
        model to emit the deductive DPLL reasoning path, not just the final SAT
        label, over formula sizes small enough for exact generation.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("sat_dpll_trace", "sat_dpll_p4_c17", "train", 0)
    ex = get_task("sat_dpll_trace")().sample(
        rng, "train", {"n_vars": 4, "n_clauses": 17, "max_trace_tokens": 80})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a formal logical-reasoning task whose target exposes
        the intermediate search procedure, so small LMs can be evaluated on
        clause evaluation, unit propagation, decision choice, and backtracking
        rather than only a binary satisfiability label. The generator samples
        bounded random 3-CNF formulas and emits one canonical DPLL trace with
        deterministic tie-breakers.
    was_generated_via_skill: false
"""

from __future__ import annotations

from itertools import product
from math import comb
import random
from typing import Any

from synthetic_tasks.base import (
    Example,
    SyntheticTask,
    build_loss_mask,
    register_task,
)


_LABELS = ("SAT", "UNSAT")


def _literal_value(literal: int, assignment: dict[int, bool]) -> bool | None:
    assert literal != 0, literal
    variable = abs(literal)
    if variable not in assignment:
        return None
    value = assignment[variable]
    return value if literal > 0 else not value


def _clause_satisfied(clause: tuple[int, ...], assignment: dict[int, bool]) -> bool:
    return any(_literal_value(literal, assignment) is True for literal in clause)


def _unit_propagate(
        clauses: list[tuple[int, ...]],
        assignment: dict[int, bool],
        trace: list[str]) -> bool:
    while True:
        assigned_unit = False
        for clause in clauses:
            if _clause_satisfied(clause, assignment):
                continue
            unresolved = [
                literal
                for literal in clause
                if _literal_value(literal, assignment) is not False
            ]
            if not unresolved:
                return False
            if len(unresolved) == 1:
                unit = unresolved[0]
                assert _literal_value(unit, assignment) is None, (unit, assignment)
                assignment[abs(unit)] = unit > 0
                trace.append(str(unit))
                assigned_unit = True
                break
        if not assigned_unit:
            return True


def _all_clauses_satisfied(
        clauses: list[tuple[int, ...]],
        assignment: dict[int, bool]) -> bool:
    return all(_clause_satisfied(clause, assignment) for clause in clauses)


def _decision_variable(
        clauses: list[tuple[int, ...]],
        assignment: dict[int, bool]) -> int:
    candidates = [
        abs(literal)
        for clause in clauses
        if not _clause_satisfied(clause, assignment)
        for literal in clause
        if abs(literal) not in assignment
    ]
    assert candidates, (clauses, assignment)
    return min(candidates)


def _restore_assignment(
        assignment: dict[int, bool],
        snapshot: dict[int, bool]) -> None:
    assignment.clear()
    assignment.update(snapshot)


def _dpll(
        clauses: list[tuple[int, ...]],
        assignment: dict[int, bool],
        trace: list[str]) -> bool:
    if not _unit_propagate(clauses, assignment, trace):
        return False
    if _all_clauses_satisfied(clauses, assignment):
        trace.append("SAT")
        return True

    variable = _decision_variable(clauses, assignment)
    snapshot = dict(assignment)

    trace.extend(("D", str(variable)))
    assignment[variable] = True
    if _dpll(clauses, assignment, trace):
        return True

    _restore_assignment(assignment, snapshot)
    trace.extend(("[BT]", "D", f"-{variable}"))
    assignment[variable] = False
    if _dpll(clauses, assignment, trace):
        return True

    _restore_assignment(assignment, snapshot)
    return False


def _dpll_trace(clauses: list[tuple[int, ...]]) -> list[str]:
    trace: list[str] = []
    assignment: dict[int, bool] = {}
    if not _dpll(clauses, assignment, trace):
        trace.append("UNSAT")
    assert trace[-1] in _LABELS, trace
    return trace


def _brute_force_label(clauses: list[tuple[int, ...]], n_vars: int) -> str:
    for values in product((False, True), repeat=n_vars):
        assignment = {variable: values[variable - 1] for variable in range(1, n_vars + 1)}
        if all(_clause_satisfied(clause, assignment) for clause in clauses):
            return "SAT"
    return "UNSAT"


def _sample_clause(rng: random.Random, n_vars: int) -> tuple[int, ...]:
    variables = sorted(rng.sample(range(1, n_vars + 1), 3))
    return tuple(
        variable if rng.randrange(2) else -variable
        for variable in variables
    )


def _sample_formula(
        rng: random.Random,
        n_vars: int,
        n_clauses: int) -> list[tuple[int, ...]]:
    assert n_clauses <= comb(n_vars, 3) * 8, (n_vars, n_clauses)
    clauses: list[tuple[int, ...]] = []
    seen = set()
    while len(clauses) < n_clauses:
        clause = _sample_clause(rng, n_vars)
        if clause in seen:
            continue
        seen.add(clause)
        clauses.append(clause)
    return clauses


def _serialize_formula(clauses: list[tuple[int, ...]]) -> str:
    return " ".join(
        str(token)
        for clause in clauses
        for token in (*clause, 0)
    )


@register_task
class SatDpllTraceTask(SyntheticTask):
    task_id = "sat_dpll_trace"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        n_vars = difficulty["n_vars"]
        n_clauses = difficulty["n_clauses"]
        max_trace_tokens = difficulty["max_trace_tokens"]

        assert n_vars >= 3, n_vars
        assert n_clauses >= 1, n_clauses
        assert max_trace_tokens >= 1, max_trace_tokens

        desired_label = rng.choice(_LABELS)
        for _ in range(5000):
            clauses = _sample_formula(rng, n_vars, n_clauses)
            trace_tokens = _dpll_trace(clauses)
            label = trace_tokens[-1]
            assert label == _brute_force_label(clauses, n_vars), (clauses, trace_tokens)
            if label != desired_label:
                continue
            if len(trace_tokens) > max_trace_tokens:
                continue

            dimacs_formula = _serialize_formula(clauses)
            prompt = (
                "<task=sat_dpll_trace>\n"
                f"Formula: {dimacs_formula}\n"
                "Trace:"
            )
            target = " " + " ".join(trace_tokens)
            full_text = prompt + target
            loss_mask = build_loss_mask(prompt, target)
            return Example(
                task_id=self.task_id,
                domain_id=f"sat_dpll_p{n_vars}_c{n_clauses}",
                prompt=prompt,
                target=target,
                full_text=full_text,
                loss_mask=loss_mask,
                metadata={
                    "task_id": self.task_id,
                    "macro_family": "formal_logical_reasoning_with_cot",
                    "is_control": False,
                    "difficulty": dict(difficulty),
                    "n_vars": n_vars,
                    "n_clauses": n_clauses,
                    "clauses": [list(clause) for clause in clauses],
                    "label": label,
                    "trace_tokens": list(trace_tokens),
                    "trace_length": len(trace_tokens),
                    "n_decisions": trace_tokens.count("D"),
                    "n_backtracks": trace_tokens.count("[BT]"),
                },
            )
        raise RuntimeError((self.task_id, difficulty, desired_label))
