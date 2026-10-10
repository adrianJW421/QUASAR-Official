# Public VSI evaluation

`public-vsi-v1` is a separately versioned input protocol for Base, QUASAR SFT (W02/44544) and QUASAR SFT+GRPO (W01/2048). Scores from this protocol are new measurements; manuscript values remain separate comparison columns. It does not assert equivalence to the historical Base loader or historical semantic viewpoints.

Validation status: Base is ready for a full run after twelve real-video processor checks. Both QUASAR cells are `BLOCKED_INPUT` because the reference estimator has not passed its fixed-sample check ([sample checks](PUBLIC_VSI_VALIDATION.md)); their full commands stop before generation. No new full-table answers or scores are included.

## Download and configure

Install the package as described in [the README](../README.md) and [environment setup](ENVIRONMENT.md). Download `test.jsonl` and the three official video archives using the pinned commands in [main-table downloads](../data/TABLE12_DOWNLOADS.md#vsi-bench-table-2). The provider supplies the benchmark videos; no ScanNet sensor exports or private object caches are needed.

Download the pinned [official Qwen weights](../data/TABLE12_DOWNLOADS.md#official-backbone). Base uses only these weights, the videos and questions. QUASAR additionally needs the [selected project weights](../checkpoints/README.md) and [CUT3R](../data/TABLE12_DOWNLOADS.md#cut3r-for-quasar). Download links for the project weights are not yet available.

Use one `reviewer_paths.json`, following `configs/reviewer_paths.example.json`. Its `source_root` points to this unpacked source; `run_root` must be a new output directory for this protocol. Each `vsi.video_roots` directory directly contains the corresponding `<scene>.mp4` files. The locator reuses the same pinned official Qwen backbone; it requires no additional model download or API service. Use the existing CUT3R and Qwen dependencies from the environment guide.

## Base commands

```bash
quasar-reviewer --paths reviewer_paths.json --protocol public-vsi-v1 --dataset vsi --role base --stage check
quasar-reviewer --paths reviewer_paths.json --protocol public-vsi-v1 --dataset vsi --role base --smoke --stage prepare
quasar-base-vsi-generate --input-jsonl RUN_ROOT/smoke/vsi/prepared_public_base/rgb/inference.jsonl --qwen-model-path BACKBONE --output-root RUN_ROOT/base_reader --inspect-only
quasar-reviewer --paths reviewer_paths.json --protocol public-vsi-v1 --dataset vsi --role base
quasar-reviewer --paths reviewer_paths.json --protocol public-vsi-v1 --dataset vsi --role base --stage table
```

Replace `RUN_ROOT` and `BACKBONE` with the paths configured in the JSON. Preparation extracts the registered original-video sequence and selects the package's final 32 frames. Generation reads original Base options (including two- and three-option questions), uses 50,176 maximum pixels/image, bf16/sdpa, seed 0, greedy decoding and 64 new tokens. Correct answers are kept in scoring records, outside the generation input. `--inspect-only` verifies the official processor's image grids and prompts without producing benchmark answers.

The full command prepares all 5,130 inputs, generates missing answers and scores them with the unchanged eight-task scorer. Repeat the command to resume; it checks the existing ordered answer prefix and frozen settings. Partial, duplicate or out-of-order records fail explicitly. `full/TABLE2.tsv` contains measured and manuscript scores separately; smoke output is not a main-table score.

## QUASAR reference preparation

The [frozen protocol](PUBLIC_VSI_PROTOCOL.md) describes the estimator and its assumptions. It locates only the station/facing objects in RGB, converts their boxes to points in this run's CUT3R coordinates, then constructs the estimated pose. In particular, an estimated visible object surface is not an annotated human position. There is no assumed metric scale, ScanNet-world transform or ground-truth viewpoint.

```bash
quasar-reviewer --paths reviewer_paths.json --protocol public-vsi-v1 --dataset vsi --role quasar_sft --smoke --stage prepare
```

Under `smoke/vsi/prepared_public_quasar/spatial/`, inspect:

- `station_inputs.jsonl`: identity, question text and type; no answer/options.
- `reference_estimates/<source_row>/REQUEST.jsonl`, `crop_*.png`, `LOCATOR.txt`: exact locator request, images and raw output.
- `reference_estimates/<source_row>/RESULT.json`: selected patches, reconstructed points, orientation calculation and result or failure.
- `reference_estimates/STATUS.json`: all retained sample outcomes. `ESTIMATED` records still require independent identity/direction review; matrix validity alone is insufficient.
- `prepared/inference.jsonl`: model-ready inputs, created only if every requested reference is available.

The twelve fixed sample IDs live in `configs/public_vsi_v1.json`. They cover all eight task groups, all three video sources and the standing/facing, robot facing, robot beginning-by and robot back-to expressions. A failed sample is not replaced. Missing/ambiguous objects, insufficient patches or a rejected front-face plane produce `BLOCKED_INPUT`, with no generated substitute pose. A cell with missing references does not proceed to answering or claim a complete score.

Because the estimator has not passed the sample check, full SFT/GRPO answering is disabled for public VSI. The seven-cell results are in [reproduction results](REPRODUCTION_RESULTS.md).

## Public components and extra information

The [VSI provider](https://github.com/vision-x-nyu/thinking-in-space/tree/51e089c3ae69b9435e9489058610f5b3964c56a8) supplies questions/videos and scene metadata. The estimator uses only question/video inference inputs; it does not use scene object boxes, room sizes, counts or relation answers. Qwen's [grounding documentation](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/cookbooks/2d_grounding.ipynb) defines normalized 0–1000 image coordinates. CUT3R's [fixed image loader](https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/src/dust3r/utils/image.py) defines the crop used for point correspondence.

Additional estimated information is explicitly introduced by this protocol: visual object identity/box, surface anchor, approximate up direction, and resulting station orientation. It is generated locally and audited, not downloaded as benchmark truth. Model and data component licenses remain those of their providers; the selected project terms are described in [license scope](../LICENSE_SCOPE.md).
