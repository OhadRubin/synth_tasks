"""
- description:
    Randomly relabeled cyclic-group row-completion task family for the
    shadow-ADO learnability experiment. Each prompt shows all but one entry
    from a single row of a freshly relabeled cyclic-group operation table and
    asks for the missing output. Optional distractor facts come from a disjoint
    relabeled cyclic group. domain_id is
    `relabeled_cyclic_row_completion_n<order>_d<distractors>`.
- paper_alignment:
    source_paper: "In-Context Algebra"
    source_url: "https://arxiv.org/abs/2512.16902"
    note: >
        The task isolates the paper's in-context variable-token algebra setup:
        visible symbols have fresh meanings per example, and the answer follows
        from closure/cancellation over the local cyclic-group row.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("relabeled_cyclic_row_completion",
                    "relabeled_cyclic_row_completion_n8_d8", "train", 0)
    ex = get_task("relabeled_cyclic_row_completion")().sample(
        rng, "train", {"order": 8, "distractors": 8})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want an in-context algebra task where visible symbols have
        no stable global meaning, so a model must infer the missing group-table
        output from local row closure rather than memorizing token identities.
        The generator randomly relabels cyclic-group elements per example and
        can add disjoint distractor-group facts, making the answer the unique
        local row symbol not already emitted by the shown row.
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


_TOKEN_POOL = tuple(f"v{i:02d}" for i in range(100))


@register_task
class RelabeledCyclicRowCompletionTask(SyntheticTask):
    task_id = "relabeled_cyclic_row_completion"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        order = difficulty["order"]
        distractors = difficulty["distractors"]

        assert order >= 2, order
        assert distractors >= 0, distractors
        assert order * (2 if distractors > 0 else 1) <= len(_TOKEN_POOL)

        tokens = list(_TOKEN_POOL)
        rng.shuffle(tokens)

        main_tokens = tokens[:order]
        rng.shuffle(main_tokens)
        phi = {element: main_tokens[element] for element in range(order)}

        x = rng.randrange(order)
        y = rng.randrange(order)
        z = (x + y) % order

        support_facts = [
            f"{phi[x]} * {phi[r]} = {phi[(x + r) % order]}"
            for r in range(order)
            if r != y
        ]

        distractor_facts: list[str] = []
        if distractors > 0:
            distractor_tokens = tokens[order:order * 2]
            rng.shuffle(distractor_tokens)
            psi = {
                element: distractor_tokens[element]
                for element in range(order)
            }
            for _ in range(distractors):
                a = rng.randrange(order)
                b = rng.randrange(order)
                distractor_facts.append(
                    f"{psi[a]} * {psi[b]} = {psi[(a + b) % order]}"
                )

        facts = support_facts + distractor_facts
        rng.shuffle(facts)

        prompt = (
            "<task=relabeled_cyclic_row_completion>\n"
            f"Facts: {'; '.join(facts)}\n"
            f"Question: {phi[x]} * {phi[y]} =\n"
            "Answer:"
        )
        target = f" {phi[z]}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"relabeled_cyclic_row_completion_n{order}_d{distractors}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "in_context_algebra",
                "is_control": False,
                "difficulty": dict(difficulty),
                "operation": "cyclic_group_addition_mod_n",
                "solver_mechanism": "row_completion_by_closure_and_cancellation",
                "has_distractors": distractors > 0,
                "order": order,
                "distractors": distractors,
                "query_left": phi[x],
                "query_right": phi[y],
                "answer": phi[z],
                "main_tokens": list(main_tokens),
                "support_facts": list(support_facts),
                "distractor_facts": list(distractor_facts),
                "shuffled_facts": list(facts),
            },
        )
