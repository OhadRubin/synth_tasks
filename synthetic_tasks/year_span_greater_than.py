"""
- description:
    Year-span greater-than classification task family for the shadow-ADO
    learnability experiment. Each prompt contains a natural-language sentence
    with a start year and an end year in the same century; the target is `1`
    when the end year is later than the start year and `0` otherwise.
    domain_id is `year_span_gt_gap_<gap_min>_<gap_max>`.
- paper_alignment:
    source_paper: "How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model"
    source_url: "https://arxiv.org/abs/2305.00586"
    note: >
        The task turns the paper's year-span greater-than continuation setup
        into a deterministic binary classification probe with controlled
        suffix gaps and one unambiguous answer token.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("year_span_greater_than",
                    "year_span_gt_gap_1_3", "train", 0)
    ex = get_task("year_span_greater_than")().sample(
        rng, "train", {"gap_min": 1, "gap_max": 3})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a deterministic year-span comparison probe with one
        correct answer per example, so small LMs can be evaluated on the
        paper's greater-than behavior without open-ended next-token ambiguity.
        The generator keeps both years in the same century, balances positive
        and negative labels by construction, and varies the comparison gap so
        near-boundary cases can be separated from easier far-apart cases.
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


_NOUNS = (
    "war",
    "treaty",
    "dynasty",
    "expedition",
    "voyage",
    "project",
    "trial",
    "reign",
    "illness",
    "campaign",
    "journey",
    "siege",
    "marriage",
    "investigation",
    "construction",
    "relationship",
)
_CENTURY_PREFIXES = tuple(range(11, 19))


@register_task
class YearSpanGreaterThanTask(SyntheticTask):
    task_id = "year_span_greater_than"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        gap_min = difficulty["gap_min"]
        gap_max = difficulty["gap_max"]

        assert 1 <= gap_min <= gap_max <= 49, difficulty

        noun = rng.choice(_NOUNS)
        century_prefix = rng.choice(_CENTURY_PREFIXES)
        gap = rng.randint(gap_min, gap_max)
        start_suffix = rng.randint(gap, 99 - gap)
        label = str(rng.randint(0, 1))
        if label == "1":
            end_suffix = start_suffix + gap
        else:
            end_suffix = start_suffix - gap

        start_year = f"{century_prefix}{start_suffix:02d}"
        end_year = f"{century_prefix}{end_suffix:02d}"

        prompt = (
            "<task=year_span_greater_than>\n"
            "Rule: output 1 if the end year is later than the start year, "
            "otherwise output 0.\n"
            f"Sentence: The {noun} lasted from the year {start_year} "
            f"to the year {end_year}.\n"
            "Answer:"
        )
        target = f" {label}"
        assert target in (" 0", " 1"), target
        assert target == (" 1" if int(end_year) > int(start_year) else " 0")

        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"year_span_gt_gap_{gap_min}_{gap_max}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "numerical_comparison",
                "operation_type": "greater_than",
                "is_control": False,
                "difficulty": dict(difficulty),
                "noun": noun,
                "century_prefix": century_prefix,
                "start_year": start_year,
                "end_year": end_year,
                "start_suffix": start_suffix,
                "end_suffix": end_suffix,
                "gap": gap,
                "label": label,
            },
        )
