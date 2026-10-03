"""
- description:
    Horn-rule forward-chaining task family for the shadow-ADO learnability
    experiment. Each prompt gives initial true atoms and shuffled implication
    rules, then asks whether a query atom is true after computing the least
    fixed point. domain_id is `horn_forward_h<hops>_d<distractors>`.
- paper_alignment:
    source_paper: "SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs"
    source_url: "https://arxiv.org/abs/2509.00930"
    note: >
        The task follows SATQuest's verifier-centered logical-reasoning spirit
        with a controllable Horn fragment: labels are objective, automatically
        checked by forward chaining, and difficulty separates proof depth from
        irrelevant rule distractors.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("horn_forward_chaining", "horn_forward_h3_d8", "train", 0)
    ex = get_task("horn_forward_chaining")().sample(
        rng, "train", {"hops": 3, "distractors": 8})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a logical reasoning probe with exact proof-depth control,
        so small LMs must perform multi-step state updates instead of matching a
        surface pattern. The generator builds a derivation chain, optionally
        breaks one required link for negative examples, adds shuffled harmless
        distractor rules, and verifies the answer through Horn closure.
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


_RULE_MEANING = (
    "Rule meaning: if all atoms on the left side of a rule are true, then the "
    "atom on the right side becomes true."
)


def _atom(prefix: str, index: int) -> str:
    assert index >= 0, index
    return f"{prefix}{index:02d}"


def _format_rule(rule: tuple[str, str, str]) -> str:
    left1, left2, right = rule
    return f"{left1} & {left2} -> {right}"


def _closure_depths(
        initial_facts: list[str],
        rules: list[tuple[str, str, str]]) -> dict[str, int]:
    depths = {atom: 0 for atom in initial_facts}
    changed = True
    while changed:
        changed = False
        for left1, left2, right in rules:
            if left1 in depths and left2 in depths and right not in depths:
                depths[right] = 1 + max(depths[left1], depths[left2])
                changed = True
    return depths


@register_task
class HornForwardChainingTask(SyntheticTask):
    task_id = "horn_forward_chaining"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        hops = difficulty["hops"]
        distractors = difficulty["distractors"]

        assert hops >= 1, hops
        assert distractors >= 0, distractors

        chain_atoms = [_atom("a", index) for index in range(hops + 1)]
        helper_atoms = [_atom("h", index) for index in range(hops)]
        query = chain_atoms[-1]
        label = "1" if rng.random() < 0.5 else "0"
        broken_link = None if label == "1" else rng.randint(1, hops)

        chain_rules: list[tuple[str, str, str]] = []
        for step in range(1, hops + 1):
            left1 = "b00" if step == broken_link else chain_atoms[step - 1]
            chain_rules.append((left1, helper_atoms[step - 1], chain_atoms[step]))

        n_distractor_facts = 0 if distractors == 0 else max(1, distractors // 4)
        distractor_initial_facts = [
            _atom("d", index)
            for index in range(n_distractor_facts)
        ]
        initial_facts = [chain_atoms[0], *helper_atoms, *distractor_initial_facts]

        distractor_rules: list[tuple[str, str, str]] = []
        known_distractor_atoms = list(distractor_initial_facts)
        for index in range(distractors):
            consequent = _atom("d", n_distractor_facts + index)
            body_pool = [*chain_atoms, *helper_atoms, *known_distractor_atoms]
            left1, left2 = rng.sample(body_pool, 2)
            distractor_rules.append((left1, left2, consequent))
            known_distractor_atoms.append(consequent)

        rules = chain_rules + distractor_rules
        rng.shuffle(rules)
        rng.shuffle(initial_facts)

        depths = _closure_depths(initial_facts, rules)
        if label == "1":
            assert depths[query] == hops, (depths, query, hops)
        else:
            assert query not in depths, (depths, query, broken_link)

        formatted_rules = [_format_rule(rule) for rule in rules]
        prompt = "\n".join([
            "<task=horn_forward_chaining>",
            _RULE_MEANING,
            f"Initial facts: {' '.join(initial_facts)}",
            "Rules:",
            *formatted_rules,
            f"Question: Is {query} true after applying all rules?",
            "Answer:",
        ])
        target = f" {label}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"horn_forward_h{hops}_d{distractors}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "logical_forward_chaining",
                "is_control": False,
                "requires_logical_closure": True,
                "requires_multi_step_reasoning": True,
                "difficulty": dict(difficulty),
                "label": label,
                "query": query,
                "initial_facts": list(initial_facts),
                "rules": list(formatted_rules),
                "support_chain": [_format_rule(rule) for rule in chain_rules],
                "distractor_rules": [_format_rule(rule) for rule in distractor_rules],
                "broken_link": broken_link,
                "closure": sorted(depths),
                "derivation_depths": dict(sorted(depths.items())),
            },
        )
