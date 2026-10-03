"""
- description:
    Greedy-trap gridworld planning task family for the shadow-ADO
    learnability experiment. Each prompt shows a fully observed grid with
    walls, open cells, a start, and a goal; the target is the unique shortest
    action sequence from start to goal. Dead-end branches include at least
    one locally attractive greedy trap. domain_id is
    `gridworld_detour_n<grid_size>_l<path_length>_b<n_branches>`.
- paper_alignment:
    source_paper: "Unlocking the Future: Exploring Look-Ahead Planning Mechanistic Interpretability in Large Language Models"
    source_url: "https://arxiv.org/abs/2406.16033"
    note: >
        The task captures the paper's look-ahead planning question in a compact
        fully observed domain: a greedy move can look locally better while the
        unique shortest plan requires anticipating later states.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("gridworld_detour_planning",
                    "gridworld_detour_n8_l16_b5", "train", 0)
    ex = get_task("gridworld_detour_planning")().sample(
        rng, "train", {
            "grid_size": 8, "path_length": 16, "n_branches": 5,
            "max_branch_length": 4, "require_greedy_trap": True})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a fully observed shortest-path planning domain where
        a locally tempting action can lead into a dead end, so model behavior
        can be probed for look-ahead rather than one-step greedy movement. The
        generator constructs a unique verified shortest path, carves tree-like
        dead-end branches, and emits the whole action plan as the answer-only
        target.
    was_generated_via_skill: false
"""

from __future__ import annotations

from collections import deque
import random
from typing import Any

from synthetic_tasks.base import (
    Example,
    SyntheticTask,
    build_loss_mask,
    register_task,
)


_ACTIONS = (
    ("U", (-1, 0)),
    ("D", (1, 0)),
    ("L", (0, -1)),
    ("R", (0, 1)),
)
_RULES = (
    "Rules: You are in a grid. # is a wall, . is open, S is the start, and G "
    "is the goal. You may move one cell at a time using U, D, L, or R. Do not "
    "enter walls. Output the unique shortest action sequence from S to G."
)


def _neighbors(cell: tuple[int, int], grid_size: int) -> list[tuple[str, tuple[int, int]]]:
    row, col = cell
    out = []
    for action, (dr, dc) in _ACTIONS:
        nxt = (row + dr, col + dc)
        if 0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size:
            out.append((action, nxt))
    return out


def _action_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for action, delta in _ACTIONS:
        if delta == (dr, dc):
            return action
    raise RuntimeError((a, b))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _open_neighbor_count(
        cell: tuple[int, int],
        open_cells: set[tuple[int, int]],
        grid_size: int) -> int:
    return sum(1 for _, nxt in _neighbors(cell, grid_size) if nxt in open_cells)


def _generate_backbone(
        rng: random.Random,
        grid_size: int,
        path_length: int) -> list[tuple[int, int]]:
    for _ in range(400):
        start = (rng.randrange(grid_size), rng.randrange(grid_size))
        path = [start]
        occupied = {start}

        def dfs() -> list[tuple[int, int]] | None:
            if len(path) == path_length + 1:
                return list(path)
            candidates = [nxt for _, nxt in _neighbors(path[-1], grid_size)]
            rng.shuffle(candidates)
            for candidate in candidates:
                if candidate in occupied:
                    continue
                if any(
                        _manhattan(candidate, previous) == 1
                        for previous in path[:-1]):
                    continue
                occupied.add(candidate)
                path.append(candidate)
                result = dfs()
                if result is not None:
                    return result
                path.pop()
                occupied.remove(candidate)
            return None

        result = dfs()
        if result is not None:
            return result
    raise RuntimeError((grid_size, path_length))


def _greedy_trap_candidates(
        path: list[tuple[int, int]],
        open_cells: set[tuple[int, int]],
        grid_size: int) -> list[tuple[int, tuple[int, int]]]:
    goal = path[-1]
    candidates = []
    for path_index, parent in enumerate(path[:-1]):
        correct_next = path[path_index + 1]
        for _, candidate in _neighbors(parent, grid_size):
            if candidate in open_cells:
                continue
            if _open_neighbor_count(candidate, open_cells, grid_size) != 1:
                continue
            if _manhattan(candidate, goal) < _manhattan(correct_next, goal):
                candidates.append((path_index, candidate))
    return candidates


def _extend_branch(
        rng: random.Random,
        start_cell: tuple[int, int],
        open_cells: set[tuple[int, int]],
        grid_size: int,
        max_branch_length: int) -> list[tuple[int, int]]:
    branch_length = rng.randint(1, max_branch_length)
    branch = [start_cell]
    open_cells.add(start_cell)
    endpoint = start_cell
    for _ in range(branch_length - 1):
        candidates = [
            nxt
            for _, nxt in _neighbors(endpoint, grid_size)
            if nxt not in open_cells
            and _open_neighbor_count(nxt, open_cells, grid_size) == 1
        ]
        if not candidates:
            break
        rng.shuffle(candidates)
        endpoint = candidates[0]
        branch.append(endpoint)
        open_cells.add(endpoint)
    return branch


def _carve_random_branch(
        rng: random.Random,
        backbone: list[tuple[int, int]],
        open_cells: set[tuple[int, int]],
        grid_size: int,
        max_branch_length: int) -> list[tuple[int, int]] | None:
    parent = rng.choice(backbone[:-1])
    candidates = [
        nxt
        for _, nxt in _neighbors(parent, grid_size)
        if nxt not in open_cells
        and _open_neighbor_count(nxt, open_cells, grid_size) == 1
    ]
    if not candidates:
        return None
    rng.shuffle(candidates)
    return _extend_branch(
        rng,
        candidates[0],
        open_cells,
        grid_size,
        max_branch_length,
    )


def _bfs_shortest_path(
        grid: list[list[str]],
        start: tuple[int, int],
        goal: tuple[int, int]) -> tuple[int, int, list[tuple[int, int]]]:
    queue = deque([start])
    dist = {start: 0}
    count = {start: 1}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while queue:
        cell = queue.popleft()
        for _, nxt in _neighbors(cell, len(grid)):
            if grid[nxt[0]][nxt[1]] == "#":
                continue
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                count[nxt] = count[cell]
                parent[nxt] = cell
                queue.append(nxt)
            elif dist[nxt] == dist[cell] + 1:
                count[nxt] = min(2, count[nxt] + count[cell])
    if goal not in dist:
        raise RuntimeError((start, goal))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return dist[goal], count[goal], path


def _trap_positions(
        backbone: list[tuple[int, int]],
        open_cells: set[tuple[int, int]],
        grid_size: int) -> list[dict[str, Any]]:
    backbone_set = set(backbone)
    goal = backbone[-1]
    traps = []
    for path_index, parent in enumerate(backbone[:-1]):
        correct_next = backbone[path_index + 1]
        for _, candidate in _neighbors(parent, grid_size):
            if candidate not in open_cells or candidate in backbone_set:
                continue
            if _manhattan(candidate, goal) < _manhattan(correct_next, goal):
                traps.append({
                    "path_index": path_index,
                    "correct_action": _action_between(parent, correct_next),
                    "trap_action": _action_between(parent, candidate),
                    "trap_cell": [candidate[0], candidate[1]],
                })
    return traps


@register_task
class GridworldDetourPlanningTask(SyntheticTask):
    task_id = "gridworld_detour_planning"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        grid_size = difficulty["grid_size"]
        path_length = difficulty["path_length"]
        n_branches = difficulty["n_branches"]
        max_branch_length = difficulty["max_branch_length"]
        require_greedy_trap = difficulty["require_greedy_trap"]

        assert grid_size >= 3, grid_size
        assert path_length >= 1, path_length
        assert n_branches >= 0, n_branches
        assert max_branch_length >= 1, max_branch_length
        assert not require_greedy_trap or n_branches >= 1, difficulty

        for _ in range(600):
            backbone = _generate_backbone(rng, grid_size, path_length)
            open_cells = set(backbone)
            branches: list[list[tuple[int, int]]] = []

            if require_greedy_trap:
                candidates = _greedy_trap_candidates(backbone, open_cells, grid_size)
                if not candidates:
                    continue
                _, trap_start = rng.choice(candidates)
                branches.append(_extend_branch(
                    rng,
                    trap_start,
                    open_cells,
                    grid_size,
                    max_branch_length,
                ))

            branch_attempts = 0
            while len(branches) < n_branches and branch_attempts < n_branches * 80:
                branch_attempts += 1
                branch = _carve_random_branch(
                    rng,
                    backbone,
                    open_cells,
                    grid_size,
                    max_branch_length,
                )
                if branch is not None:
                    branches.append(branch)
            if len(branches) != n_branches:
                continue

            grid = [["#" for _ in range(grid_size)] for _ in range(grid_size)]
            for row, col in open_cells:
                grid[row][col] = "."
            start = backbone[0]
            goal = backbone[-1]
            grid[start[0]][start[1]] = "S"
            grid[goal[0]][goal[1]] = "G"

            distance, shortest_path_count, gold_cells = _bfs_shortest_path(
                grid,
                start,
                goal,
            )
            greedy_trap_positions = _trap_positions(backbone, open_cells, grid_size)
            if distance != path_length:
                continue
            if shortest_path_count != 1:
                continue
            if require_greedy_trap and not greedy_trap_positions:
                continue

            gold_actions = [
                _action_between(a, b)
                for a, b in zip(gold_cells, gold_cells[1:])
            ]
            prompt = (
                "<task=gridworld_detour_planning>\n"
                f"{_RULES}\n"
                "Grid:\n"
                + "\n".join(" ".join(row) for row in grid)
                + "\nPlan:"
            )
            target = " " + " ".join(gold_actions)
            full_text = prompt + target
            loss_mask = build_loss_mask(prompt, target)
            return Example(
                task_id=self.task_id,
                domain_id=f"gridworld_detour_n{grid_size}_l{path_length}_b{n_branches}",
                prompt=prompt,
                target=target,
                full_text=full_text,
                loss_mask=loss_mask,
                metadata={
                    "task_id": self.task_id,
                    "macro_family": "lookahead_planning",
                    "is_control": False,
                    "requires_planning": True,
                    "requires_lookahead": True,
                    "difficulty": dict(difficulty),
                    "grid_size": grid_size,
                    "path_length": path_length,
                    "n_branches": n_branches,
                    "max_branch_length": max_branch_length,
                    "start": [start[0], start[1]],
                    "goal": [goal[0], goal[1]],
                    "gold_actions": list(gold_actions),
                    "gold_cells": [[row, col] for row, col in gold_cells],
                    "backbone_cells": [[row, col] for row, col in backbone],
                    "branch_cells": [
                        [[row, col] for row, col in branch]
                        for branch in branches
                    ],
                    "greedy_trap_positions": greedy_trap_positions,
                    "grid": ["".join(row) for row in grid],
                },
            )
        raise RuntimeError(difficulty)
