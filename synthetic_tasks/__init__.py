"""
- description:
    Synthetic task suite for the Shadow ADO learnability experiment.
    Re-exports the public surface (Example, SyntheticTask, registry helpers)
    from base, and triggers registration of every implemented task family by
    importing the per-family modules.
- usage:
    from synthetic_tasks import get_task, all_task_ids, split_rng
    cls = get_task("copy")
    task = cls()
    rng = split_rng("copy", "copy_len_16", "train", 0)
    example = task.sample(rng, "train", {"length": 16})
- user_story:
    content: |
        Researchers training small LMs on a fixed mixture of synthetic task
        families need one entry point that surfaces every implemented task by
        a stable id, so the training loop, the validation evaluator, and the
        shadow-ADO logger can route domains without knowing each task family's
        Python module name. Importing this package registers all known task
        families and exposes registry helpers plus the Example dataclass.
    was_generated_via_skill: false
"""

import importlib
import pkgutil

from synthetic_tasks.base import (
    Example,
    SyntheticTask,
    register_task,
    get_task,
    all_task_ids,
    build_loss_mask,
    split_rng,
)

# Auto-register every task family by walking this package directory.
# Each per-family module calls @register_task at import time, so importing
# them here populates the registry. base.py is excluded because it is
# already imported above for the public symbols and contains no @register_task.
_REGISTRY_SKIP = {"base"}
for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name in _REGISTRY_SKIP:
        continue
    importlib.import_module(f"{__name__}.{_module_info.name}")

__all__ = [
    "Example",
    "SyntheticTask",
    "register_task",
    "get_task",
    "all_task_ids",
    "build_loss_mask",
    "split_rng",
]
