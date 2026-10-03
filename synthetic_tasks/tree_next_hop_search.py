"""
- description:
    Shuffled tree next-hop search task family for the shadow-ADO learnability
    experiment. Each prompt serializes a complete rooted tree as shuffled edge
    triples, gives a start root and a goal leaf, and asks for the next vertex
    after the start on the unique path to the goal. domain_id is
    `tree_next_hop_b<branching_factor>_d<depth>`.
- paper_alignment:
    source_paper: "Transformers Struggle to Learn to Search"
    source_url: "https://arxiv.org/abs/2412.04703"
    note: >
        The task is a controlled next-hop search problem: the model sees a
        graph, start, and reachable goal, then must identify the first step on
        the path rather than rely on edge order or label heuristics.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("tree_next_hop_search",
                    "tree_next_hop_b2_d4", "train", 0)
    ex = get_task("tree_next_hop_search")().sample(
        rng, "train", {"depth": 4, "branching_factor": 2})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a compact graph-search synthetic task that still has
        a unique verifiable answer, so a small LM must inspect shuffled edges
        and discover which root child can reach the sampled goal leaf. The
        generator uses a complete hidden tree with random visible labels,
        making depth an honest multi-hop search knob while preventing label
        names or edge order from leaking the path.
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


_LABEL_CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits
_LABEL_POOL = tuple(f"{a}{b}" for a in _LABEL_CHARS for b in _LABEL_CHARS)


@register_task
class TreeNextHopSearchTask(SyntheticTask):
    task_id = "tree_next_hop_search"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        depth = difficulty["depth"]
        branching_factor = difficulty["branching_factor"]

        assert depth >= 1, depth
        assert branching_factor >= 1, branching_factor

        nodes_by_depth: list[list[tuple[int, ...]]] = [[()]]
        for _ in range(depth):
            nodes_by_depth.append([
                node + (child_index,)
                for node in nodes_by_depth[-1]
                for child_index in range(branching_factor)
            ])
        nodes = [
            node
            for level_nodes in nodes_by_depth
            for node in level_nodes
        ]
        assert len(nodes) <= len(_LABEL_POOL), len(nodes)

        goal_path = tuple(rng.randrange(branching_factor) for _ in range(depth))
        answer_path = (goal_path[0],)

        labels = list(_LABEL_POOL)
        rng.shuffle(labels)
        label_by_node = dict(zip(nodes, labels))

        edge_nodes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for parent in nodes[:-len(nodes_by_depth[-1])]:
            for child_index in range(branching_factor):
                child = parent + (child_index,)
                edge_nodes.append((parent, child))

        edge_triples = [
            f"E {label_by_node[parent]} {label_by_node[child]}"
            for parent, child in edge_nodes
        ]
        rng.shuffle(edge_triples)

        start_label = label_by_node[()]
        goal_label = label_by_node[goal_path]
        answer_label = label_by_node[answer_path]

        prompt = (
            "<task=tree_next_hop_search>\n"
            f"Edges: {' '.join(edge_triples)}\n"
            f"Question: start {start_label} goal {goal_label}\n"
            "Answer:"
        )
        target = f" {answer_label}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"tree_next_hop_b{branching_factor}_d{depth}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "search_or_planning",
                "is_control": False,
                "difficulty": dict(difficulty),
                "depth": depth,
                "branching_factor": branching_factor,
                "goal_path": list(goal_path),
                "answer_path": list(answer_path),
                "start_label": start_label,
                "goal_label": goal_label,
                "answer_label": answer_label,
                "shuffled_edge_triples": list(edge_triples),
            },
        )
