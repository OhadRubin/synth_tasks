"""
- description:
    Passkey-retrieval task family for the shadow-ADO learnability
    experiment. The model is shown a Context line containing random
    single-digit filler with a single english sentence
    "The passkey is XXXXX." embedded at a random offset, then asked
    "What is the passkey?". The target is the 5-digit passkey. The
    `context_length` difficulty knob controls the target byte length
    of the Context region (haystack); bins are 128, 512, 1024.
    domain_id is `passkey_ctx_<context_length>`. Loss is applied only
    to the answer region via build_loss_mask from base.py.

    Design choices (documented):
      * The 5-digit passkey is drawn independently per-digit from 0-9
        with replacement (so e.g. "00000" is possible). Using
        replacement preserves a flat output distribution over the
        10**5 passkey space and avoids biasing the binding probe
        toward all-distinct strings.
      * `context_length` is interpreted as a TARGET byte length of the
        Context region (the haystack), per option (a) in the brief.
        Because the pilot model's context_length is 1024 BYTES total
        and the byte-level tokenizer is 1 byte = 1 token, the
        haystack target is capped so the full serialized example
        fits within 1024 bytes. The fixed-overhead bytes (task tag +
        Question + Answer + target) sum to ~80 bytes, so the haystack
        is capped at 1024 - overhead. For the 128 and 512 bins this
        cap never bites; for the 1024 bin the effective haystack
        length is the cap.
      * Filler tokens are single decimal digits joined by single
        spaces. They never contain the substring "passkey is", so
        the only literal binding the model can extract is the real
        embedded sentence.
      * The passkey sentence is placed at a uniformly random offset
        inside the haystack: a random split of the non-sentence
        bytes between "before" and "after" the sentence. This
        prevents the model from shortcutting to a fixed offset.
- usage:
    from synthetic_tasks.base import get_task, split_rng
    rng = split_rng("passkey_retrieval",
                    "passkey_ctx_512", "train", 0)
    ex = get_task("passkey_retrieval")().sample(
        rng, "train", {"context_length": 512})
- user_story:
    content: |
        Passkey-retrieval is the long-context needle-in-haystack probe
        in the synthetic pilot suite. Shadow-ADO is expected to mark
        ctx_128 as quickly saturated (the passkey sentence dominates
        the short haystack), while ctx_512 and ctx_1024 should remain
        actively learnable for a longer window because the model must
        attend across more filler bytes to locate the binding.
        Critically, the passkey sentence position is randomised
        inside the haystack so the model cannot shortcut to a fixed
        offset; the surrounding filler is random digits that cannot
        accidentally encode "The passkey is" via lexical collision.
        The generator emits a Context line of approximately
        `context_length` bytes (capped so the full example fits the
        pilot's 1024-byte budget), a fixed Question, and the 5-digit
        Answer; the loss mask covers only the answer bytes.
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


_PASSKEY_DIGITS = 5
_CONTEXT_BUDGET_BYTES = 1024
# Overhead = bytes contributed by everything except the haystack content
# itself. Computed from the literal template; recomputed below to stay
# in sync if the template changes.
_PROMPT_TEMPLATE_HEAD = "<task=passkey_retrieval>\nContext: "
_PROMPT_TEMPLATE_TAIL = "\nQuestion: What is the passkey?\nAnswer:"
_TARGET_PREFIX = " "
# target_len = len(" ") + _PASSKEY_DIGITS
_NON_HAYSTACK_BYTES = (
    len(_PROMPT_TEMPLATE_HEAD.encode("utf-8"))
    + len(_PROMPT_TEMPLATE_TAIL.encode("utf-8"))
    + len(_TARGET_PREFIX.encode("utf-8"))
    + _PASSKEY_DIGITS
)
_MAX_HAYSTACK_BYTES = _CONTEXT_BUDGET_BYTES - _NON_HAYSTACK_BYTES


def _random_digit_filler(rng: random.Random, n_bytes: int) -> str:
    if n_bytes <= 0:
        return ""
    tokens: list[str] = []
    total = 0
    while total < n_bytes:
        tokens.append(str(rng.randint(0, 9)))
        # each appended digit adds 1 byte; spaces add 1 byte between tokens
        total = len(tokens) + (len(tokens) - 1)
    s = " ".join(tokens)
    if len(s) > n_bytes:
        s = s[:n_bytes].rstrip(" ")
    return s


@register_task
class PasskeyRetrievalTask(SyntheticTask):
    task_id = "passkey_retrieval"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        if "context_length" not in difficulty:
            raise ValueError(
                f"passkey_retrieval difficulty must include 'context_length', "
                f"got {difficulty!r}"
            )
        context_length = difficulty["context_length"]
        if not isinstance(context_length, int) or isinstance(context_length, bool):
            raise ValueError(
                f"passkey_retrieval difficulty.context_length must be int, got "
                f"{type(context_length).__name__}"
            )
        if context_length < 1:
            raise ValueError(
                f"passkey_retrieval difficulty.context_length must be >= 1, got "
                f"{context_length}"
            )

        passkey = "".join(str(rng.randint(0, 9)) for _ in range(_PASSKEY_DIGITS))
        sentence = f"The passkey is {passkey}."

        target_haystack = min(context_length, _MAX_HAYSTACK_BYTES)
        sentence_len = len(sentence.encode("utf-8"))
        if target_haystack < sentence_len:
            # Bin too tight to even fit the sentence: emit just the sentence.
            haystack = sentence
        else:
            remaining = target_haystack - sentence_len
            # Reserve up to two bytes for the spaces separating filler from sentence.
            # If remaining < 2, drop the filler entirely.
            if remaining < 2:
                haystack = sentence
            else:
                payload = remaining - 2
                before_payload = rng.randint(0, payload)
                after_payload = payload - before_payload
                before = _random_digit_filler(rng, before_payload)
                after = _random_digit_filler(rng, after_payload)
                parts: list[str] = []
                if before:
                    parts.append(before)
                parts.append(sentence)
                if after:
                    parts.append(after)
                haystack = " ".join(parts)

        prompt = f"{_PROMPT_TEMPLATE_HEAD}{haystack}{_PROMPT_TEMPLATE_TAIL}"
        target = f"{_TARGET_PREFIX}{passkey}"
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        return Example(
            task_id=self.task_id,
            domain_id=f"passkey_ctx_{context_length}",
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
