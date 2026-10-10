# Experiment records

`index.tsv` lists 38 experiment groups. Answer files and scores are included for eleven of them: M01–M06, A01/A02 and I01/I05/I06 (see the [experiment index](../docs/EXPERIMENT_INDEX.md)). M01 keeps the historical 8B raw-video answers, configuration and exact original summary; its unrecorded execution details are listed in [reproducibility](../docs/REPRODUCIBILITY.md). The other 27 rows are listed for reference without answer files. Each populated group has a manifest, original model text and unrounded summaries; shared configurations are referenced from `configs/`.

The inventory unit is an experiment family or record group. A row can contain several trained models or inference arms. Row count is not a number of independent runs or random seeds.

## Artifact groups

| Directory | Contents |
| --- | --- |
| `main/` | Main SFT/GRPO/Base comparisons, exact task scores, model-role mapping, and paired statistics |
| `ablations/` | Independently trained architecture/readout/coordinate configurations and their control definitions |
| `interventions/` | Within-checkpoint writeback, norm, reference/gate/router, geometry-oracle, and spatial-cue records |
| `transfer/` | Backbone transfer and external datasets, preserving comparator stages and protocol differences |
| `qualitative/` | Selected-case original inputs, questions, gold verification, all compared answers, and non-input illustrations |
| `runtime/` | Time samples, warm-up/timing boundary, hardware/environment, and memory reports |

## Per-record contents

Each populated experiment directory contains a `manifest.json` that names its files: original model answers (`*.predictions.jsonl`, with sample IDs, question types and per-item scores), unrounded summaries (`*.summary.json`) and `validation.json`, which lists the checks performed on those files. Shared configurations are in `configs/`. Original labels/options needed for rescoring are obtained as described in [data terms](../data/TERMS.md). The revised 5292 score overlay shares the 5598 original answer file. Five-task, nine-subtype and fixed-512 aggregates remain separate. The base VSTI run dates to September and is not a June source snapshot; Basic NoFlow retains the missing historical post-load effective flag.

Negative and mixed-result conditions are included.
