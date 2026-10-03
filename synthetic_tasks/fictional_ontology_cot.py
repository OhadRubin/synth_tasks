"""
- description:
    Fictional ontology chain-of-thought entailment task family for the
    shadow-ADO learnability experiment. Each prompt contains shuffled
    fictional `X is Y.` facts with one hidden support chain from an individual
    to a terminal property, plus distractors. The target is a canonical
    derivation over the ordered support chain followed by `True` or `False`.
    domain_id is `fictional_ontology_cot_h<hops>_d<distractors>`.
- paper_alignment:
    source_paper: "How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning"
    source_url: "https://arxiv.org/abs/2402.18312"
    note: >
        The task mirrors the paper's PrOntoQA-style fictional ontology setting:
        solving requires selecting relevant facts, composing the support chain,
        and emitting an explicit canonical reasoning trace.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("fictional_ontology_cot",
                    "fictional_ontology_cot_h3_d4", "train", 0)
    ex = get_task("fictional_ontology_cot")().sample(
        rng, "train", {"hops": 3, "distractors": 4})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a PrOntoQA-style synthetic reasoning domain whose
        facts use fictional names rather than real-world categories, so a
        small LM must compose local `is-a` facts in context. The generator
        shuffles the prompt facts, keeps distractors disconnected from the
        queried individual, and emits a canonical chain-of-thought target so
        learning curves can expose copying, fact selection, and induction
        steps rather than just final-label prediction.
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


_NAMES = (
    "Alex",
    "Blair",
    "Casey",
    "Drew",
    "Eli",
    "Fran",
    "Gale",
    "Mira",
    "Nia",
    "Omar",
    "Rex",
    "Zoya",
)
_PREFIXES = (
    "ba", "be", "bi", "bo", "bu",
    "da", "de", "di", "do", "du",
    "fa", "fe", "fi", "fo", "fu",
    "ga", "ge", "gi", "go", "gu",
    "la", "le", "li", "lo", "lu",
    "ma", "me", "mi", "mo", "mu",
    "na", "ne", "ni", "no", "nu",
)
_SUFFIXES = ("pus", "dax", "tor", "vim", "kel", "mip", "zan", "lor")
_TYPE_POOL = tuple(
    f"{prefix}{suffix}"
    for prefix in _PREFIXES
    for suffix in _SUFFIXES
)
_PROPERTIES = (
    "velish",
    "glurish",
    "mavish",
    "snedish",
    "prallish",
    "dornish",
    "zantish",
    "foppish",
    "kelvish",
    "lorpish",
    "brenish",
    "tazish",
    "neshish",
    "wuggish",
    "pimmish",
    "veskish",
)


@register_task
class FictionalOntologyCotTask(SyntheticTask):
    task_id = "fictional_ontology_cot"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        hops = difficulty["hops"]
        distractors = difficulty["distractors"]

        assert hops >= 2, hops
        assert distractors >= 0, distractors
        assert hops - 1 + distractors * 2 <= len(_TYPE_POOL)

        person = rng.choice(_NAMES)

        type_names = list(_TYPE_POOL)
        rng.shuffle(type_names)
        chain_classes = type_names[:hops - 1]
        distractor_subjects = type_names[hops - 1:hops - 1 + distractors]
        distractor_objects = type_names[hops - 1 + distractors:]

        properties = list(_PROPERTIES)
        rng.shuffle(properties)
        true_property, false_property = properties[:2]

        chain_nodes = [person] + chain_classes
        chain_objects = chain_classes + [true_property]
        support_edges = list(zip(chain_nodes, chain_objects))
        support_chain = [
            f"{subject} is {obj}."
            for subject, obj in support_edges
        ]

        label = "True" if rng.random() < 0.5 else "False"
        query_property = true_property if label == "True" else false_property

        distractor_facts: list[str] = []
        for index, subject in enumerate(distractor_subjects):
            if index == 0 and label == "False":
                obj = false_property
            elif rng.random() < 0.5:
                obj = rng.choice(distractor_objects)
            else:
                obj = rng.choice(_PROPERTIES)
            distractor_facts.append(f"{subject} is {obj}.")

        shuffled_facts = support_chain + distractor_facts
        rng.shuffle(shuffled_facts)

        reasoning_steps = [support_chain[0]]
        for subject, obj in support_edges[1:]:
            reasoning_steps.append(f"{subject} is {obj}.")
            reasoning_steps.append(f"{person} is {obj}.")

        prompt = (
            "<task=fictional_ontology_cot>\n"
            f"Facts: {' '.join(shuffled_facts)}\n"
            f"Question: True or false: {person} is {query_property}.\n"
            "Answer: Let us think step by step."
        )
        target = " " + " ".join(reasoning_steps + [label])
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"fictional_ontology_cot_h{hops}_d{distractors}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "relational_reasoning_with_cot",
                "is_control": False,
                "difficulty": dict(difficulty),
                "label": label,
                "person": person,
                "query_property": query_property,
                "true_property": true_property,
                "false_property": false_property,
                "support_chain": list(support_chain),
                "distractor_facts": list(distractor_facts),
                "shuffled_facts": list(shuffled_facts),
            },
        )
