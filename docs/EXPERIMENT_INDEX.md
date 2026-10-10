# Experiment index

## Inventory scope

This index organizes the experiments reported in the QUASAR paper. `experiments/index.tsv` has 38 rows, one per experiment family or record group (not per training run). IDs start with M (main results), A (ablations), I (fixed-model interventions), T (transfer and external tests), C (runtime cost) or Q (qualitative cases). Scores below are rounded paper values; included directories hold full-precision rescored results.

Answer files and scores are included for eleven groups: M01–M06, A01, A02, I01, I05 and I06. M01 is the historical Base VSI run on raw video, with its answers, recorded configuration and scoring chain. For the other 27 groups, `index.tsv` records the paper reference, protocol and status, but their answer files are not included. Literature-only rows, including published spatial-memory and scene-prior methods, remain cited results rather than locally reproduced models.

## Main results and model roles

| Public role | Training / endpoint | VSI: 5,130, eight-task macro | Historical VSTI: 6,042, five-task macro |
| --- | --- | ---: | ---: |
| Base | Released Qwen3-VL-8B-Instruct | 59.82 | 46.31 |
| Visual-only (SFT) | Selected SFT step 46,592 | 62.79 | 61.86 |
| QUASAR (SFT) | Selected four-token SFT step 44,544 | 64.86 | 65.11 |
| Visual-only (SFT+GRPO) | SFT step 53,365 → all-task GRPO 2,048 | 63.62 | 62.26 |
| QUASAR (SFT+GRPO) | SFT step 35,840 → all-task GRPO 2,048 | 65.36 | 66.19 |

The displayed SFT rows are not the parents of the displayed GRPO endpoints. Main comparisons share backbone, data recipe, and GRPO update count, but not total SFT exposure. Historical Base VSI uses raw video and different option lists for 671 questions; its original retained summary is 59.8247049665833%. Its responses, recorded configuration and ID map are packaged and CPU checked; resolved model revision, actual sampled frames and full historical environment remain unknown.

The numeric-only GRPO variant shares QUASAR's step-35,840 SFT parent. Its restriction applies to GRPO, not SFT. It belongs to the appendix, not the all-task main row.

## Structure and supervision controls

| Family | Included conditions | VSI / VSTI reference scores | Interpretation |
| --- | --- | --- | --- |
| Complete SFT structure family | Visual-only; Basic spatial; Basic without training-time writeback; Metric-slot A/B; Branch-gated without metric slots; four-token QUASAR A/B | 63.22/59.95; 62.64/63.84; 63.67/60.75; 64.25/66.29; 64.32/66.17; 65.16/65.52; 65.10/65.28; 65.08/66.11 | Separate dated training configurations, not successive deletions from final QUASAR |
| Readout placement | Summary-token readout; visual readout with tokens; visual readout without tokens; local-only decoder supervision | 64.58/66.25; 64.73/65.14; 65.18/65.28; 64.54/66.56 | Readout/SIGReg location, token insertion, and global decoder supervision have distinct contrasts |
| Decoder reference-coordinate supervision | On / off | 64.54/65.87; 64.84/64.87 | Two decoder loss weights change; spatial-module coordinate supervision remains |
| Joint no-summary-token configuration | Removes summary tokens, their readouts/objectives, and SIGReg | 65.67/66.33 | Bundled simplification, not a token-only effect |

Basic spatial has explicit reference inputs, four updates, and a single writeback path. It has no summary tokens, branch mixing, metric heads, or decoder auxiliary heads. Its separately trained no-writeback counterpart preserves the four basic auxiliary objectives. Matching saved recipes support a training-time writeback contrast inside Basic. A historical evaluation flag after checkpoint loading was not directly logged; this limit remains part of the record.

## Fixed-model interventions and diagnostics

| Family | Model / set | Required retained distinctions |
| --- | --- | --- |
| Writeback off | Full four-token SFT endpoint, step 53,365; VSI 5,130 / historical VSTI 6,042 | Normal 65.08/72.87 versus off 63.47/68.64; VSTI uses nine subtypes, not five main tasks |
| Norm-budget sweep | Same diagnostic SFT endpoint | Multiplier 0; near-zero budget; budgets 0.05, 0.15, 0.30, 0.50; each score retains its diagnostic aggregate |
| Reference type/path permutation | All-task GRPO endpoint | Changes type and corresponding pose path; matched normal is not the main-table normal |
| Uniform branch mixing | All-task GRPO endpoint | Changes weights while retaining the other inputs; separate from reference permutation |
| Numerical reference-pose intervention | All-task GRPO; 5,598 parser-valid historical VSTI questions | Fixed type; pose-conditioning off and three pose permutations; scene-cluster intervals |
| Fixed 512 writeback/pose controls | All-task GRPO; 512 questions, 127 base scenes | Own reference 415; writeback off 390; pose permutations 393/390/394; question accuracy, not task macro |
| Text-only type router | 11,172 VSI/VSTI questions | Predicts a type label, not an object anchor or numerical pose |
| Branch distribution | 19 question groups | Descriptive gate statistics |
| Geometry-source oracles | CUT3R versus VGGT, 344/549 rows and 274 held-out rows | Ground-truth masks and calibration; not a full QUASAR frontend replacement |
| Spatial-cue visualizations | Existing validation examples | Qualitative representational diagnostics |

Permutation seeds are perturbations of one checkpoint, not independent training seeds. The fixed-512 normal and writeback-off predictions were reused; only the three pose-permuted arms were newly generated. Inputs, donor mappings and prediction provenance record which arms were reused.

## Transfer, external tests, and cost

- Qwen3-VL-2B/4B: QUASAR versus Visual-only, historical five-task VSTI.
- Qwen2.5-VL-3B/7B: QUASAR versus the Metric-slot model, not Visual-only.
- SenseNova-SI-1.5-InternVL3-8B: paired SFT configurations; revised 5,736-question, nine-subtype VSTI.
- Plain InternVL3-8B: an absolute QUASAR SFT result without an endpoint paired comparator.
- SPAR-Bench (7,211), ViewSpatial (5,712), SPBench (1,328): shared evaluator, Base / Visual-only SFT / QUASAR GRPO / writeback off / fixed image-count rule. The fixed rule reuses existing predictions; it is not another trained model.
- CV-Bench (2,638): question accuracy; the writeback-off gain and negative full-model comparisons stay in the record.
- MindCube-Tiny (1,050): QUASAR SFT accuracy 413/1,050; no paired Base/Visual-only gain is inferred.
- Runtime: five internal model roles, 10 warm-ups and 100 timed examples. Cached-input generation calls exclude geometry extraction, loading, cache I/O, and scoring.

## Selected qualitative cases

The four main cases are apartment frame 18, office frame 32, VSTI `vsti_4268` / `scene0354_00`, and VSI `vsi_2983` / `scene0426_00`. The kitchen case is a supplement where Base also succeeds.

Each selected question has two retained answers per model across Base, Visual-only, trained QUASAR with all visual writeback disabled, and full QUASAR. They are selected illustrative examples. Illustrations and bounding boxes are explanatory assets, not model inputs.

The motion case uses the starting orientation at input slot 8, unlike the historical endpoint-reference convention. The distance case verifies closest points between annotated 3D boxes, not center distances. Neither should be substituted for a main-table prediction row.
