# Obtaining the inputs

The [source-by-source download guide](DOWNLOAD_GUIDE.md) lists official access routes, fixed VSI/VSTI versions, media/depth/camera layouts, preparation commands and the historical inputs that are not included.

The historical training/validation QA and M01 scoring records are included in `annotations/`; no download is needed for those fields. `manifests/DATASET_INDEX.tsv` identifies the separate saved native versions and their ordered members. Original RGB/depth/cameras and justified reference definitions are additional inputs. Generate geometry/features locally using [GENERATION.md](GENERATION.md); computed caches are not distributed.

## Maintainers with the original native records

If you hold the original native JSONL and caches, use the original native JSONL and asset/cache roots with `quasar-prepare-data` as described in [PREPARATION.md](PREPARATION.md). This is the tested route for preserving historical question, frame, reference and target versions. A current source download cannot replace a historical sidecar or cached target.

For ScanNet, obtain approved access and accept its [terms](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf), then use the official downloader provided with that approval. For ScanNet++, obtain approved access and signed [terms](https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf). Download through the official account instructions. The preparation command accepts those local files and does not bypass either access process. Obtain ARKitScenes from its [official repository](https://github.com/apple/ARKitScenes) under its own license.

## Upstream QA and model versions

The upstream training-QA source can be acquired with the Hugging Face CLI. Set `VLM3R_REVISION` to the desired upstream commit, not an implicit moving revision:

```bash
hf download Journey9ni/VLM-3R-DATA --repo-type dataset \
  --revision "$VLM3R_REVISION" --local-dir "$VLM3R_SOURCE_ROOT"
```

The upstream card documents the camera-label correction. Its current QA is a different version from the pre-correction training records in this package. The historical native sidecars are not contained in that upstream download.

The scoped saved Qwen model metadata identifies revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Acquire that upstream model separately if it is not already available:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir "$MODEL_PATH"
```

This revision identifies the checked training/cache assets; M01's resolved historical model revision remains unknown. Acquire CUT3R code and the 512-pixel checkpoint through the [official instructions](https://github.com/CUT3R/CUT3R#download-checkpoints), retain its license, and supply their paths to `quasar-export-geometry`. The checked source is commit `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` ([environment guide](../docs/ENVIRONMENT.md#cut3r-reconstruction-component)); the historical checkout did not record a git revision.

## Availability

The VSTI-Bench dataset card states no data license, so VSTI questions and answers are not redistributed here; download them from the provider and follow its terms. Sourced reference definitions and original annotations come from their providers. Computed geometry/features are generated locally. The member lists and donor IDs are included; full historical reproduction also needs source-specific reference/target generation records that are not available. Fixed-sample checks are listed in [PACKAGE_CHECKS.md](../docs/PACKAGE_CHECKS.md).
