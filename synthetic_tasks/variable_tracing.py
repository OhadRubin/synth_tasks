"""
- description:
    Variable-tracing synthetic task family for the shadow-ADO learnability
    experiment. Each example is a short straight-line program of
    `name = <literal-or-name>` lines followed by a `Question: <var>?`
    query. The model must follow an alias chain of length `hops` back to
    a single decimal digit literal and emit it. Difficulty bins (hop_1,
    hop_2, hop_4) come from tasks.yaml under
    `mvp_extension.variable_tracing.bins`. domain_id is
    `variable_tracing_hop_<hops>`. Loss is applied only to the answer
    region via build_loss_mask from base.py.

    Design choices:
      - Variable names are single lowercase letters a..z. The hardest bin
        needs hops+1 chain vars plus distractors variables; hop_4 +
        distractors=8 uses 5 + 8 = 13 distinct names, well within 26.
      - Distractors are standalone literal assignments of the form
        `<var> = <digit>` only. They never alias other variables and
        therefore cannot accidentally extend or shadow the chain. This is
        the simplest interpretation of "off-chain noise"; using alias
        chains as distractors would risk forming alternate paths to the
        queried variable and complicate ground-truth verification.
      - "hops" counts alias edges. hops=N => one literal assignment
        plus N alias assignments => N+1 chain lines, N+1 chain vars.
      - The query always targets the END of the chain so chain length
        controls difficulty directly.
      - Lines are newline-separated. Chain lines retain dependency order
        (root literal first, then each alias), and distractor lines are
        interleaved into a random position before or after each chain
        line such that every variable is introduced before it is
        referenced (no forward references).
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("variable_tracing",
                    "variable_tracing_hop_4", "train", 0)
    ex = get_task("variable_tracing")().sample(
        rng, "train", {"hops": 4, "distractors": 8})
- user_story:
    content: |
        Variable tracing probes whether a small LM can follow a chain of
        single-step alias bindings inside one context window and return
        the literal at the chain's root. It sits in the
        retrieval_or_binding macro family alongside associative_recall
        but adds compositional path-following: each additional hop forces
        the model to chain another resolution step rather than do a
        single lookup. Shadow-ADO is expected to saturate hop_1 quickly,
        leave hop_2 actively learnable for a medium window, and keep
        hop_4 learnable longest. The generator builds an alias chain of
        exactly `hops` edges down to a digit literal, mixes in
        `distractors` standalone literal assignments using disjoint
        variable names, randomly orders the lines while preserving the
        "introduce-before-use" invariant, and emits the question line
        for the chain's tail. The target is the root digit and the loss
        mask covers only that one-byte answer region.
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


_DIGIT_ALPHABET = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
_VAR_ALPHABET = tuple(string.ascii_lowercase)


def _interleave_preserving_order(
    rng: random.Random,
    chain_lines: list[str],
    distractor_lines: list[str],
) -> list[str]:
    """Return a random interleaving where chain_lines keep relative order.

    Distractor lines have no dependencies (they are standalone literal
    assignments using variable names disjoint from the chain), so any
    interleaving that preserves the chain order is dependency-safe.
    """
    chain_idx = 0
    distractor_idx = 0
    out: list[str] = []
    total = len(chain_lines) + len(distractor_lines)
    for _ in range(total):
        remaining_chain = len(chain_lines) - chain_idx
        remaining_distractor = len(distractor_lines) - distractor_idx
        if remaining_chain == 0:
            out.append(distractor_lines[distractor_idx])
            distractor_idx += 1
            continue
        if remaining_distractor == 0:
            out.append(chain_lines[chain_idx])
            chain_idx += 1
            continue
        if rng.random() < remaining_chain / (remaining_chain + remaining_distractor):
            out.append(chain_lines[chain_idx])
            chain_idx += 1
        else:
            out.append(distractor_lines[distractor_idx])
            distractor_idx += 1
    return out


@register_task
class VariableTracingTask(SyntheticTask):
    task_id = "variable_tracing"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "hops" not in difficulty:
            raise ValueError(
                f"variable_tracing difficulty must include 'hops', got {difficulty!r}"
            )
        if "distractors" not in difficulty:
            raise ValueError(
                f"variable_tracing difficulty must include 'distractors', got {difficulty!r}"
            )
        hops = difficulty["hops"]
        distractors = difficulty["distractors"]
        if not isinstance(hops, int) or isinstance(hops, bool):
            raise ValueError(
                f"variable_tracing difficulty.hops must be int, got {type(hops).__name__}"
            )
        if not isinstance(distractors, int) or isinstance(distractors, bool):
            raise ValueError(
                f"variable_tracing difficulty.distractors must be int, got "
                f"{type(distractors).__name__}"
            )
        if hops < 1:
            raise ValueError(
                f"variable_tracing difficulty.hops must be >= 1, got {hops}"
            )
        if distractors < 0:
            raise ValueError(
                f"variable_tracing difficulty.distractors must be >= 0, got {distractors}"
            )

        n_chain_vars = hops + 1
        n_total_vars = n_chain_vars + distractors
        if n_total_vars > len(_VAR_ALPHABET):
            raise ValueError(
                f"variable_tracing needs {n_total_vars} distinct variable names "
                f"(hops+1+distractors), but alphabet only has {len(_VAR_ALPHABET)}"
            )

        names = list(_VAR_ALPHABET)
        rng.shuffle(names)
        chain_vars = names[:n_chain_vars]
        distractor_vars = names[n_chain_vars:n_total_vars]

        root_literal = rng.choice(_DIGIT_ALPHABET)
        chain_lines: list[str] = [f"{chain_vars[0]} = {root_literal}"]
        for i in range(1, n_chain_vars):
            chain_lines.append(f"{chain_vars[i]} = {chain_vars[i - 1]}")

        distractor_lines = [
            f"{v} = {rng.choice(_DIGIT_ALPHABET)}" for v in distractor_vars
        ]

        program_lines = _interleave_preserving_order(rng, chain_lines, distractor_lines)
        query_var = chain_vars[-1]
        answer = root_literal

        prompt = (
            "<task=variable_tracing>\n"
            + "\n".join(program_lines)
            + f"\nQuestion: {query_var}?\n"
            + "Answer:"
        )
        target = f" {answer}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"variable_tracing_hop_{hops}",
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "retrieval_or_binding",
                "is_control": False,
                "difficulty": dict(difficulty),
            },
        )
