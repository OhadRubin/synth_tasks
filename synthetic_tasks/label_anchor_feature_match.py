"""
- description:
    In-context label-anchor matching task family for the shadow-ADO
    learnability experiment. Each prompt samples arbitrary label words and
    feature marker tokens, shows labeled demonstrations, then asks for the
    label attached to the marker token repeated in the query. domain_id is
    `label_anchor_c<n_classes>_s<shots_per_class>_l<item_length>`.
- paper_alignment:
    source_paper: "Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning"
    source_url: "https://arxiv.org/abs/2305.14160"
    note: >
        The task makes demonstration label words the arbitrary anchors that
        bind sampled feature markers to final predictions, matching the
        paper's label-token information-flow hypothesis while removing
        semantic label shortcuts.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("label_anchor_feature_match",
                    "label_anchor_c4_s1_l10", "train", 0)
    ex = get_task("label_anchor_feature_match")().sample(
        rng, "train", {"n_classes": 4, "shots_per_class": 1, "item_length": 10})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a clean synthetic probe where label tokens are
        arbitrary anchors rather than semantic categories, so a model must
        bind each sampled feature marker to the label word shown in the
        current prompt. The generator randomizes both marker tokens and label
        assignments per example, keeps filler tokens globally unique, and
        masks loss only over the answer label so learning curves measure the
        in-context binding behavior rather than prompt-template imitation.
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


_FEATURE_TOKENS = tuple(f"f{i:03d}" for i in range(1000))
_LABEL_POOL = ("A", "B", "C", "D", "E", "F", "G", "H")


@register_task
class LabelAnchorFeatureMatchTask(SyntheticTask):
    task_id = "label_anchor_feature_match"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        n_classes = difficulty["n_classes"]
        shots_per_class = difficulty["shots_per_class"]
        item_length = difficulty["item_length"]

        assert n_classes >= 1, n_classes
        assert shots_per_class >= 1, shots_per_class
        assert n_classes <= len(_LABEL_POOL), n_classes
        assert item_length >= 1, item_length
        n_prompt_items = n_classes * shots_per_class + 1
        assert n_classes + n_prompt_items * (item_length - 1) <= len(_FEATURE_TOKENS)

        labels = list(_LABEL_POOL)
        rng.shuffle(labels)
        labels = labels[:n_classes]

        feature_tokens = list(_FEATURE_TOKENS)
        rng.shuffle(feature_tokens)
        markers = feature_tokens[:n_classes]
        fillers = feature_tokens[n_classes:]
        label_by_marker = dict(zip(markers, labels))

        filler_index = 0
        demo_blocks: list[tuple[list[str], str]] = []
        for marker in markers:
            for _ in range(shots_per_class):
                sequence = [marker] + fillers[
                    filler_index:filler_index + item_length - 1
                ]
                filler_index += item_length - 1
                assert len(sequence) == item_length, sequence
                rng.shuffle(sequence)
                demo_blocks.append((sequence, label_by_marker[marker]))
        rng.shuffle(demo_blocks)

        query_marker = rng.choice(markers)
        query_sequence = [query_marker] + fillers[
            filler_index:filler_index + item_length - 1
        ]
        assert len(query_sequence) == item_length, query_sequence
        rng.shuffle(query_sequence)

        prompt_lines = ["<task=label_anchor_feature_match>"]
        for sequence, label in demo_blocks:
            prompt_lines.append(f"Input: {' '.join(sequence)}")
            prompt_lines.append(f"Label: {label}")
        prompt_lines.append(f"Query: {' '.join(query_sequence)}")
        prompt_lines.append("Label:")

        prompt = "\n".join(prompt_lines)
        target = f" {label_by_marker[query_marker]}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"label_anchor_c{n_classes}_s{shots_per_class}_l{item_length}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "in_context_binding",
                "is_control": False,
                "difficulty": dict(difficulty),
            },
        )
