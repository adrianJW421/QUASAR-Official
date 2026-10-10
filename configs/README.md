# Saved configurations and commands

For Tables 1 and 2, copy `reviewer_paths.example.json` to the repository root and use `quasar-reviewer --paths reviewer_paths.json`. `table12_sources.json` fixes dataset revisions, complete ordered members, frame manifests and fixed samples; `table12_profiles.json` fixes the five model roles, generation recipes, unrounded paper comparisons and per-cell blockers. The entry prepares raw inputs and runs the recorded generation/scoring modules. See [the reviewer guide](../README.md).


The six training JSON files retain the actual recorded SFT, all-task GRPO, Basic and trained-NoFlow arguments. Evaluation files retain saved arguments, including fields derived after checkpoint loading. Absolute asset locations are replaced by explicit `${PARAMETER}` values. Saved absent fields remain absent.

`commands.json` maps each collected role to its real packaged module and explicit argument list. It also lists fields not recorded historically and runtime-derived fields, such as the checkpoint-restored spatial-query setting. Source-version uncertainty is separate from current parser compatibility.

Create a JSON file mapping the required parameter names to your local files, then preview a command:

```bash
python scripts/run_recorded.py --role decoderaux_repaired_training --parameters my_paths.json
```

`--execute` runs that command. `parameters.json` lists the parameter names; dataset, cache, model and checkpoint files are obtained separately. Each role has its own output parameter.

`base_vsti_commands.json` separately records the two actual Base VSTI shards. Preview them with `--commands configs/base_vsti_commands.json --role base_vsti_shard0`. The packaged `quasar.base_vsti.inference` entry requires an explicit `--run-dir` containing the frozen `PROTOCOL.json`, `inputs/public.jsonl`, and existing acceptance record. The manifest rows have `sample_id`, `prompt`, `image_paths`, `cache_key`, and `feature_path`; each RGB feature file has `image_grid_thw`, `image_embeds`, and `deepstack_features`. Original questions and these derived feature files are not included. No QUASAR/Visual-only adapter is loaded by this Base entry.

`evaluation/base_vsi_run_config.json` retains M01's March 17/18 raw-video arguments, with only dataset/video roots parameterized. `base_vsi_commands.json` records its historical launch arguments and the CPU rescoring command. Its exact historical raw-video loader is unavailable; this generation record is not executable through `run_recorded.py`. For rescoring, `data/annotations/vsi_m01_5130_scoring.jsonl` supplies the licensed historical labels/options already aligned to package IDs. `experiments/main/base_vsi/id_alignment.tsv` retains the official ID correspondence. Final's expanded options and the Base VSTI cached-feature protocol do not replace M01's inputs. Model revision, actual sampled frames, full decoding settings and environment are unknown where unrecorded.

| Directory | Recipes |
| --- | --- |
| `sft/` | Four-token QUASAR and Visual-only SFT: selected main roles, full endpoints and data/auxiliary-target revision |
| `grpo/` | QUASAR and Visual-only all-task GRPO: SFT parent, trainable scope, reward, sampling, KL and update scale |
| `evaluation/` | Main VSI/VSTI evaluations for QUASAR, Visual-only, Basic and NoFlow, Base VSI/VSTI protocols, and the writeback (I01), numerical-reference (I05) and runtime (I06) interventions |
| `ablations/` | Basic and Basic-without-writeback training pair |
| `evidence/` | Saved training audits (data filter, first batch, reference schema, visual cache, run summary) |

Each configuration records the model role, source revision where known, data split and target version, prompt/frame/options, generation settings, reference construction and scorer. A command-line flag recorded before checkpoint loading may differ from the effective flag; both are recorded where available.

Data/model/output locations are caller-supplied paths. The QUASAR main SFT (step 44,544), its GRPO parent (step 35,840), the full SFT (step 53,365) and the Visual-only full parent (step 53,365) are separate parameters. Current parser defaults do not recover unrecorded historical fields. Recipes for numeric-only GRPO and the other configuration families in this table without files under `configs/` are not included.
