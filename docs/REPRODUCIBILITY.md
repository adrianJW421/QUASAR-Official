# Reproducibility scope

## What is included

The independent main-table run is summarized in [reproduction results](REPRODUCTION_RESULTS.md), separately from the paper values and the eleven archived experiment groups. Model, input and scoring code are the same as in that run; later edits changed only documentation, comments and user-facing messages.

Runnable source, saved arguments and original model answers are included for eleven experiment groups and 33 retained-answer conditions; no weights are included in the source. For M01, the historical Qwen3-VL-8B-Instruct run on raw video, the original answers, recorded configuration, official-to-package ID map and unrounded summary are included. Its 5,130 scores agree item by item with both the collected historical scorer and the packaged scorer. [Source versions](SOURCE_VERSIONS.md) describes how current code relates to the historical sources.

## Recorded training identity

| Item | Original four-token recipe |
| --- | --- |
| Main backbone | Qwen3-VL-8B-Instruct |
| SFT data | 426,916 spatial instructions |
| Full SFT schedule | One epoch; full endpoint 53,365 updates |
| Selected main SFT | QUASAR 44,544; Visual-only 46,592 |
| All-task GRPO SFT parents | QUASAR 35,840; Visual-only 53,365 |
| GRPO scale | 2,048 updates; 8,192 question groups; four answers per group |
| SFT optimizer settings | Learning rate $3\times10^{-5}$; microbatch 1; accumulation 8 |
| LoRA settings | Rank 16; scaling 32; dropout 0.05 |
| GRPO optimizer settings | Learning rate $10^{-6}$; accumulation 4; policy KL coefficient 0.02 |
| SFT trainable components | LoRA, spatial module/scale, summary embeddings, auxiliary readouts, regularization projection |
| GRPO trainable components | LoRA only; auxiliary/stability losses disabled, spatial forward path retained |

The released configurations are the saved recipes and invocations, not parser defaults. Each names its checkpoint role and data/auxiliary-target revision. Two runs with the same number of QA rows do not necessarily have the same target coverage or total training budget.

## Protocol identities

| Protocol ID | Set | Aggregate |
| --- | ---: | --- |
| `vsi_5130_8task` | 5,130 | Eight-task macro; numeric MRA and multiple-choice accuracy; nested direction subtypes |
| `vsti_6042_5task` | 6,042 historical | Five-task macro; relation subtypes pooled within tasks |
| `vsti_6042_9subtype` | 6,042 historical | Nine-subtype macro; fixed-model diagnostics |
| `vsti_5736_9subtype` | 5,736 revised | Revised labels/set; InternVL transfer |
| `vsti_5598_5task` | 5,598 parser-valid historical | Five-task macro; numerical-reference intervention |
| `vsti_5292_9subtype` | 5,292 revised intersection | Nine-subtype macro; distinct intersection |
| `vsti_512_accuracy` | 512 historical multiple-choice | Question accuracy; 127 base scenes |
| `external_shared` | Benchmark-specific | Numeric MRA, count exact match, choice accuracy; not each official scorer |
| `question_accuracy` | Benchmark-specific | Equal question weights; CV-Bench and MindCube |
| `cached_generation_cost` | 10 warm-ups + 100 timed per setting | Generation-call time and peak allocated memory |

A question count alone does not identify the protocol. Labels, options, subset membership, aggregation and parsing rules travel with the predictions. Revised VSTI scores are not interchangeable with the historical main scores.

M01 is the historical raw-video Qwen3-VL-8B-Instruct evaluation, with no project training recorded in its configuration. Its original overall score is 59.8247049665833%; the packaged scorer returns 59.82470496658328%, a floating-point difference below $10^{-10}$ percentage points. Relative-direction easy, medium and hard means receive equal weight, then the eight task means receive equal weight. The licensed historical questions, options and labels are included in `data/annotations/vsi_m01_5130_scoring.jsonl`, already aligned to package IDs. `experiments/main/base_vsi/id_alignment.tsv` maps the original official IDs, including their gaps, to the package IDs.

M01 uses raw video, while the trained-model evaluations use saved frame candidates. The option lists differ for 671 questions: 217 easy-direction, 378 medium-direction and 76 route-planning questions. These differences do not affect rescoring of the retained answers, but Base-to-trained gains on VSI are not a strictly matched-input comparison. The historical generation arguments in `configs/base_vsi_commands.json` are a record; the March 17 loader source and environment are unavailable, so they are not a packaged generation entry.

## Geometry, targets and answering references

The data release comprises the ordered 426,916 training QA rows and 17,651 validation QA rows, source-specific licenses, the GRPO occurrence lists and evaluation member versions. Training QA is downloaded separately; validation QA and the lists are included in the repository (see [data downloads](../data/DOWNLOAD_GUIDE.md#included-records-and-additional-inputs)). [Generation](../data/GENERATION.md) describes rebuilding inputs from original data and the gaps in sourced references. [Historical migration](../data/PREPARATION.md) is a separate maintainer route that preserves existing originals. Four training and three validation samples have been prepared and read back with identical geometry, reference, mask, saved-target and cached-tensor values. On those fixed samples the runtime also passed strict 8B loading, SFT/GRPO forward/backward calls and three normal/zero-writeback generation pairs; one preselected target-camera sample passed the complete RGB-to-answer chain. See [package checks](PACKAGE_CHECKS.md). The original training runs and benchmark tables were not rerun.

At answering time the model receives estimated observation geometry and an explicit reference type and pose. The full question goes to the MLLM, not to the spatial gate. Decoder auxiliary predictions, object annotations, gold answers and target-generation metadata are not inference inputs.

Training uses mixed precomputed geometry: CUT3R caches and depth/intrinsics/pose records. Object statistics come from a mixed object bank. Coordinate units are not guaranteed to be metres across sources. The inference CUT3R path is documented separately from the training-time sources.

The seven saved metric-target families occupy $1+3+3+3+3+3+1=17$ components. Saved targets take precedence over on-the-fly construction; replacing them with the current default builder would change the recipe.

The canonical BEV records use the first and third coordinate components. This does not imply gravity estimation or physical ground alignment across sources. A supplied reference label does not by itself show that a pose is correctly grounded, so references, numerical validity, geometric validity and target availability are recorded separately.

## Historical details not recorded

- The exact executing source and environment of the June SFT/GRPO runs. Scale construction, reductions, the KL implementation and branch-target rules are only partly tied to those runs.
- Source-specific mask rules, object-bank truncation and coordinate/unit transformations where production records are missing.
- For M01: the resolved model revision, sampled frames/fps/resolution, the March 17 loader and dependency versions, full decoding settings and dtype. Its answers, configuration and score chain are included.
- For Basic NoFlow: the training audit is included, but the effective evaluation flag after checkpoint loading was not logged.
- Checkpoint selection criteria and visual-cache identities across training stages where not recorded. SFT exposure differs between the main systems.
- A prefill/decode split for the historical timing runs; current generation code does not recover it.

## CPU checks

CPU tests exercise the NPZ/native-schema/reference/target readers, patch ordering, four queries, five branches, native and DeepStack writes, and Basic NoFlow. They use a small randomly initialized Qwen model, without historical checkpoints or generation. `--aggregate-retained-scores` recomputes only the aggregate of supplied per-item scores. Statistical sampling units remain experiment-specific; these checks are not new benchmark runs or training seeds.

For the revised 5,292-question intersection, use `--sample-ids experiments/interventions/numerical_pose/intersection5292.ids.txt` with the shared 5,598-answer file and revised external labels. The separate revised score overlay records the changed labels; applying the historical per-item scores to this subset would use the wrong protocol.
