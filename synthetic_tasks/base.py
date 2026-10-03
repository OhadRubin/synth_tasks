"""
- description:
    Base abstractions for the synthetic task suite. Defines the Example
    dataclass returned by every generator, the SyntheticTask abstract base
    with a `.sample(rng, split, difficulty) -> Example` contract, a registry
    decorator for task discovery by `task_id`, and helpers for byte-level
    loss-mask construction and seeded train/val RNG splitting.
- usage:
    from synthetic_tasks.base import SyntheticTask, register_task, build_loss_mask

    @register_task
    class MyTask(SyntheticTask):
        task_id = "my_task"
        def sample(self, rng, split, difficulty):
            prompt = "<task=my_task>\\nInput: ..."
            target = " 4 2"
            full_text = prompt + target
            loss_mask = build_loss_mask(prompt, target)
            return Example(
                task_id=self.task_id,
                domain_id=f"my_task_n_{difficulty['n']}",
                prompt=prompt, target=target, full_text=full_text,
                loss_mask=loss_mask,
                metadata={"task_id": self.task_id, "difficulty": difficulty},
            )
- user_story:
    content: |
        Researchers training small LMs on synthetic task mixtures need a uniform
        protocol so the dataloader, validator, and shadow-ADO logger stay
        tokenizer-agnostic and tolerate difficulty bins. The base layer pins
        the Example shape (prompt/target/full_text/loss_mask/metadata over
        bytes — matching the byte-level tokenizer config), the SyntheticTask
        abstract base with a single `.sample` method, and a decorator-based
        registry so the dataloader discovers every task family by id at
        import time. The byte-level loss-mask helper guarantees that any
        researcher building a new generator computes the answer-only mask the
        same way, so per-task ADO curves all measure task-solution learning
        rather than prompt-template learning.
    was_generated_via_skill: false
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import hashlib
import random


@dataclass
class Example:
    task_id: str
    domain_id: str
    prompt: str
    target: str
    full_text: str
    loss_mask: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.loss_mask) != len(self.full_text.encode("utf-8")):
            raise ValueError(
                f"loss_mask length {len(self.loss_mask)} does not match "
                f"byte length {len(self.full_text.encode('utf-8'))} of full_text "
                f"for domain {self.domain_id}"
            )
        if not self.full_text.startswith(self.prompt):
            raise ValueError(
                f"full_text must start with prompt for domain {self.domain_id}"
            )
        if not self.full_text.endswith(self.target):
            raise ValueError(
                f"full_text must end with target for domain {self.domain_id}"
            )


class SyntheticTask(ABC):
    task_id: str

    @abstractmethod
    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        ...


_REGISTRY: dict[str, type[SyntheticTask]] = {}


def register_task(cls: type[SyntheticTask]) -> type[SyntheticTask]:
    if not hasattr(cls, "task_id"):
        raise ValueError(f"{cls.__name__} must define task_id class attribute")
    task_id = cls.task_id
    if task_id in _REGISTRY:
        raise ValueError(f"task_id {task_id!r} already registered by {_REGISTRY[task_id].__name__}")
    _REGISTRY[task_id] = cls
    return cls


def get_task(task_id: str) -> type[SyntheticTask]:
    if task_id not in _REGISTRY:
        raise KeyError(f"task_id {task_id!r} not registered. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[task_id]


def all_task_ids() -> list[str]:
    return sorted(_REGISTRY)


def build_loss_mask(prompt: str, target: str) -> list[int]:
    prompt_bytes = len(prompt.encode("utf-8"))
    target_bytes = len(target.encode("utf-8"))
    return [0] * prompt_bytes + [1] * target_bytes


def split_rng(task_id: str, domain_id: str, split: str, sample_index: int) -> random.Random:
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    key = f"{task_id}|{domain_id}|{split}|{sample_index}".encode("utf-8")
    seed = int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16)
    return random.Random(seed)
