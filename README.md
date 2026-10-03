# synth_tasks

Procedural generators for 27 synthetic task families, plus a taxonomy that describes each task along 10 axes.

This is the task suite behind *Conditional Transfer from Controlled Pretraining Mixtures to Code* (`writeup.pdf`). The paper mixes these tasks into a code pretraining corpus and asks two questions about each one: does the model learn the task from the task's own examples, and does adding the task to the mixture improve HumanEval after fine-tuning.

## Layout

```
synthetic_tasks/           one module per task family; base.py holds the shared protocol
task_taxonomy_by_factor/   62 YAML files, one per (axis, factor) pair, covering all 27 tasks
writeup.pdf                the paper
```

## Quick start

The generators use only the Python standard library.

```bash
uv run python - <<'PY'
from synthetic_tasks import get_task, split_rng, all_task_ids

print(all_task_ids())
rng = split_rng("variable_tracing", "variable_tracing_hop_2", "train", 0)
ex = get_task("variable_tracing")().sample(rng, "train", {"hops": 2, "distractors": 2})
print(ex.full_text)
PY
```

```
<task=variable_tracing>
z = 9
k = 4
a = z
d = a
j = 8
Question: d?
Answer: 9
```

## Task protocol

Every task family subclasses `SyntheticTask`, sets a `task_id`, and implements one method:

```python
sample(rng: random.Random, split: str, difficulty: dict) -> Example
```

`difficulty` holds the task's knobs (listed in the tables below). Every knob is required.

The returned `Example` has these fields:

| Field | Meaning |
|---|---|
| `task_id` | Task family, e.g. `copy` |
| `domain_id` | Task family plus difficulty bin, e.g. `copy_len_64`. Mixture weights and per-domain loss curves are keyed by this. |
| `prompt` | Input text. Its first line is `<task=<task_id>>`. |
| `target` | Answer text |
| `full_text` | `prompt + target` |
| `loss_mask` | One entry per UTF-8 byte of `full_text`: `0` over the prompt, `1` over the target, so training loss covers only the answer |
| `metadata` | Includes `task_id` and `difficulty`; the two control tasks also set `is_control: True` |

**Reproducible sampling.** `split_rng(task_id, domain_id, split, sample_index)` hashes its four arguments with BLAKE2b into a seed. Example `i` of a given domain and split is therefore identical on every machine, and `"train"` and `"val"` draw from different seeds.

**Registration.** `@register_task` adds a class to a registry keyed by `task_id`. Importing `synthetic_tasks` imports every module in the package, so a new task is picked up by adding a module that uses the decorator. `base.py` shows a minimal example in its docstring.

## Tasks

The paper splits the 27 tasks into two suites.

### Literature suite (15)

Machine-generated versions of standard probes from prior work, including two controls. `constant_label` always has the same answer, so its loss should collapse quickly. `random_label` draws its answer independently of the input, so its loss should stay near the entropy of the label set.

| `task_id` | Module | Difficulty knobs | `domain_id` |
|---|---|---|---|
| `copy` | `copy.py` | `length` | `copy_len_<length>` |
| `reverse_sequence` | `reverse_sequence.py` | `length` | `reverse_len_<length>` |
| `majority` | `majority.py` | `length` | `majority_len_<length>` |
| `xor_parity` | `xor_parity.py` | `length` | `xor_parity_len_<length>` |
| `associative_recall` | `associative_recall.py` | `n_pairs` | `associative_recall_pairs_<n_pairs>` |
| `dyck_balanced_parentheses` | `dyck.py` | `depth`, `bracket_types` | `dyck_depth_<depth>` |
| `listops_nested_prefix` | `listops.py` | `depth` | `listops_depth_<depth>` |
| `passkey_retrieval` | `passkey.py` | `context_length` | `passkey_ctx_<context_length>` |
| `variable_tracing` | `variable_tracing.py` | `hops`, `distractors` | `variable_tracing_hop_<hops>` |
| `scan_command_to_actions` | `scan.py` | `command_length` (`short` / `medium` / `long`) | `scan_<command_length>` |
| `babi_single_hop` | `babi.py` | `n_facts` | `babi_single_<n_facts>facts` |
| `babi_multi_hop` | `babi.py` | `n_facts`, `supporting_facts` | `babi_multi_<n_facts>facts` |
| `babi_counting` | `babi_counting.py` | `n_facts` | `babi_count_<n_facts>facts` |
| `constant_label` | `controls.py` | `label` | `constant_label` |
| `random_label` | `controls.py` | `label_set` | `random_label` |

### Curated suite (12)

Code-adjacent tasks: execution, search, planning, formal inference, string rewriting, ranking, and in-context binding. Each was produced by giving a language model the text of one paper and asking it to describe a task that can be sampled automatically; the prompt is in Appendix F of the writeup. Most modules name their source paper in a `paper_alignment` docstring field. Each curated task lives in `synthetic_tasks/<task_id>.py`.

| `task_id` | Difficulty knobs | `domain_id` |
|---|---|---|
| `bounded_tiny_python_execution` | `n_vars`, `n_updates`, `loop_depth`, `max_loop_iters`, `branch_prob`, `modulus`, `print_count` | `bounded_tiny_python_exec_v<n_vars>_u<n_updates>_d<loop_depth>_i<max_loop_iters>` |
| `fictional_ontology_cot` | `hops`, `distractors` | `fictional_ontology_cot_h<hops>_d<distractors>` |
| `gridworld_detour_planning` | `grid_size`, `path_length`, `n_branches`, `max_branch_length`, `require_greedy_trap` | `gridworld_detour_n<grid_size>_l<path_length>_b<n_branches>` |
| `horn_forward_chaining` | `hops`, `distractors` | `horn_forward_h<hops>_d<distractors>` |
| `iterated_string_replace` | `initial_len`, `num_steps`, and 19 more (full call in the module docstring) | `replace_steps_<num_steps>_len_<initial_len>` |
| `label_anchor_feature_match` | `n_classes`, `shots_per_class`, `item_length` | `label_anchor_c<n_classes>_s<shots_per_class>_l<item_length>` |
| `markov_first_rewrite` | `n_rules`, `input_length`, `lhs_len`, `rhs_min_len`, `rhs_max_len`, `no_op_prob` | `markov_first_rewrite_r<n_rules>_l<input_length>_p<lhs_len>` |
| `relabeled_cyclic_row_completion` | `order`, `distractors` | `relabeled_cyclic_row_completion_n<order>_d<distractors>` |
| `sat_dpll_trace` | `n_vars`, `n_clauses`, `max_trace_tokens` | `sat_dpll_p<n_vars>_c<n_clauses>` |
| `tfidf_rank_documents` | `n_docs`, `doc_len`, `vocab_size`, `query_terms`, `max_tf`, `allow_score_ties`, `require_integer_rarity` | `tfidf_rank_<n_docs>_docs_<query_terms>_terms` |
| `tree_next_hop_search` | `branching_factor`, `depth` | `tree_next_hop_b<branching_factor>_d<depth>` |
| `year_span_greater_than` | `gap_min`, `gap_max` | `year_span_gt_gap_<gap_min>_<gap_max>` |

## Taxonomy

`task_taxonomy_by_factor/<axis>__<factor>.yaml` describes one factor of one axis for every task:

```yaml
axis: target_operation
factor: operation_family
axis_frame: <what the whole axis covers>
factor_description: <what this factor measures>
tasks:
- task_id: copy
  value: copying
  summary: 'The mapping from input to answer is the identity: ...'
```

| Axis | Factors | Describes |
|---|---|---|
| `target_operation` | 6 | The computation that maps the input to the answer |
| `solver_mechanism` | 7 | The machinery an exact solver would use: memory, control flow, search |
| `latent_topology` | 6 | The structure of the instance before it is written as text |
| `input_realization` | 6 | How that structure is written as text: format, ordering, register |
| `output_interface` | 7 | The required answer form: cardinality, syntax, length, visible reasoning |
| `composition_and_interference_pattern` | 7 | How relevant pieces combine and how distractors interfere |
| `symbol_grounding_regime` | 5 | How tokens get their meaning within one example |
| `generator_mechanism` | 7 | How inputs and labels are produced and constrained |
| `generalization_regime` | 7 | How the training and evaluation distributions relate |
| `diagnostic_capability` | 5 | The model ability or failure mode the task exposes |

## Paper findings

The paper fixes 70% of the pretraining corpus as general Python and varies the remaining 30% across OpenCodeInstruct, the curated suite, and the literature suite.

- 14 of the 27 tasks show loss that responds to the task's own training budget: 10 of 12 curated tasks and 4 of 15 literature tasks.
- The best observed HumanEval pass@20 after fine-tuning is 22.6, when the 30% slice is 75% OpenCodeInstruct and 25% curated tasks. A slice of only curated tasks scores 15.9, below the 16.5 of a baseline with no synthetic data.
- ADO, a scheduler that reweights data by predicted loss reduction, cuts OpenCodeInstruct to 1.2–1.4% of the slice when allowed to move weight between sources, and scores 2.4 to 11.0 points below the matching fixed mixtures.
