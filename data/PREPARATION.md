# Data generation and historical migration

This page is for maintainers who hold the original native records and caches. Other users should follow [GENERATION.md](GENERATION.md): original QA/media or measured depth/cameras → exact frames → new geometry → sourced reference → new features → answering. That route reads the QA files directly, keeps labels separate and does not need the original native records or historical feature caches. Computed features, geometry and tensor caches are not distributed.

The sections below describe strict migration of the historical originals, which is separate from reconstruction. Historical object/reference/target production and some semantic-reference conversions are incomplete.

## Data records

Download the training QA separately and place it at the path below; the other listed records are included. See [downloads](DOWNLOAD_GUIDE.md#included-records-and-additional-inputs).

| File | Content |
| --- | --- |
| `annotations/sft_426916_qa.jsonl` | Original ordered training questions/answers with source and per-record license |
| `annotations/validation_17651_qa.jsonl` | Saved validation questions/answers |
| `annotations/vsi_m01_5130_scoring.jsonl` | Historical Base VSI labels/options aligned to retained-answer IDs |
| `manifests/sft_426916_members.tsv` | Row order, IDs, reference and target masks, geometry route |
| `manifests/validation_17651_members.tsv` | Saved validation membership and masks |
| `manifests/grpo_*_8192_members.jsonl` | Actual processing order and question IDs from the saved GRPO runs |
| `manifests/grpo_*_source_rows.jsonl` | Original training row matched by logged ID, answer type, task type and target text |
| `manifests/DATASET_INDEX.tsv` | Evaluation versions and their member files |

The 5598/5736/5292 member lists and donor maps are in `../experiments/interventions/numerical_pose/`. Fixed-512 IDs and donor mappings are in `../experiments/interventions/reference_match_512/`. QA projections alone cannot be passed to a model. Reconstruction needs original media/depth/cameras and sourced references; training also needs documented target generation. Historical migration requires the original native inputs and caches.

## Migrate historical inputs

Install the package and supply the original native JSONL, cache roots, model and asset bindings. `bindings.json` is a list of `source_root` and `logical_root` pairs. Source roots must be actual absolute local roots; logical roots name directories inside the prepared data root. Every absolute record path must have a binding.

```bash
quasar-prepare-data \
  --input-jsonl "$NATIVE_JSONL" --output-root "$PREPARED_ROOT" \
  --bindings "$BINDINGS_JSON" --assets-mode copy \
  --historical-model-path "$ORIGINAL_MODEL_PATH" --model-path "$MODEL_PATH" \
  --historical-scannet-frame-root "$ORIGINAL_SCANNET_FRAMES" \
  --scannet-frame-root "$PREPARED_SCANNET_FRAMES" \
  --visual-cache-root "$HISTORICAL_PRIMARY_CACHE" \
  --visual-cache-root "$HISTORICAL_EXTRA_CACHE" \
  --max-visual-frames 32 --max-pixels 50176
```

The prepared ScanNet frame root must agree with the bindings and output root. Use cache roots in the saved run's original order. The preparer validates the original cache identity and settings, relocates paths and corresponding keys, then compares all cached tensors and numerical NPZ arrays. It preserves fields, coordinate values, scale and masks. A missing or mismatched cache stops preparation. `--assets-mode link` links instead of copying and requires the original files to remain available.

The retained geometry routes are CUT3R NPZ, scene-directory depth/pose selection and explicit depth/pose frames. Scene-directory preparation retains its color/depth/pose files so that finite-pose frame selection is unchanged. Depth images use the retained loader's divisor 1000. CUT3R estimates keep their original coordinate scale. Stored candidate banks may contain more than 32 frames; the original loader selects 32 with its original ordering.

Training reads saved JSON `simple3_aux` targets and inline objects. Auxiliary NPZ files remain dependency/provenance artifacts; this loader does not use them in place of JSON targets. The metric families occupy $1+3+3+3+3+3+1=17$ slots. Missing-object and missing-pair validity masks remain; QA is never removed or relabeled as zero. Current default sidecar generation is not a substitute for the historical targets.

## Answer-free inference and RGB reconstruction

Prepare a separate inference root with the same command plus `--inference-only`. `inference.jsonl` retains question/options, RGB/estimated geometry and the sourced reference, while excluding gold answers, object banks and saved auxiliary targets. Generate with `quasar-generate --inference-only`; score using a separate label file.

Install the checked CUT3R dependencies and build its original `curope` component as described in [ENVIRONMENT.md](../docs/ENVIRONMENT.md). The reconstruction runtime requires that component for the original negative camera-token positions. Code and model roots are independent caller-supplied paths.

For a separately recorded RGB reconstruction, supply the exact frozen frame manifest:

```bash
quasar-export-geometry \
  --video-frame-manifest "$RGB_MANIFEST" --output-root "$NEW_GEOMETRY" \
  --cut3r-repo-root "$CUT3R_REPO" --cut3r-model-path "$CUT3R_MODEL" \
  --image-size 512 --device cuda:0
quasar-bind-rebuilt-reference \
  --input-jsonl "$TARGET_CAMERA_INPUT" --geometry-npz "$NEW_GEOMETRY_NPZ" \
  --output-jsonl "$NEW_INPUT_JSONL" --max-visual-frames 32
quasar-export-visual-cache \
  --input-jsonl "$NEW_INPUT_JSONL" --output-root "$NEW_VISUAL_CACHE" \
  --qwen-model-path "$MODEL_PATH" --max-visual-frames 32 --max-pixels 50176 \
  --shard-index 0 --num-shards 1 --device cuda --inference-only
```

The target-camera input contains its original `target_frame_index` and the exact 32 RGB paths. The binder keeps that index and reads its new reference pose from the same new CUT3R estimate used by the observation geometry. It rejects changed RGB order. The new 12-component observation and reference-dependent 28-component descriptors therefore use the same reconstruction's coordinates. New estimates/caches have a separate root and do not replace historical targets or estimates. This reconstruction/visual-export/generation chain was checked on one preselected real target-camera sample, not on every dataset or reference route.

## Version boundaries

VSI 5,130, historical VSTI 6,042, corrected VSTI 5,736, historical parser-valid 5,598, revised intersection 5,292 and fixed 512 are separate sets. M01 raw-video input and Final candidate-frame input differ, including 671 option lists. Current corrected VSTI downloads cannot replace historical labels.

Preparation and real-model input checks cover all three reference types in four training and three validation samples. Source membership, dependency existence, inline metric widths and used cache identity/layout are checked. SFT/GRPO computations, paired final-model generations and the target-camera RGB reconstruction passed on those samples ([package checks](../docs/PACKAGE_CHECKS.md)). Gaps in historical unit/mask provenance are listed in [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).

Training IDs can repeat ([data guide](README.md)). `grpo_*_source_rows.jsonl` maps each of the 8,192 occurrences in each GRPO run to its source row and keeps the run's order; join through these files, not by ID alone, and keep repeated rows.
