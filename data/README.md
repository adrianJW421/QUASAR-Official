# Data package

The data release comprises 426,916 training QA records, 17,651 saved validation QA records, their ordered membership/target-availability manifests, the two actual 8,192-group GRPO member sequences, and M01's historical 5,130 scoring inputs. The training QA file is downloaded separately; the other records and manifests are included in this repository. Per-source licenses are listed in [TERMS.md](TERMS.md). Questions, options, labels and duplicate training rows are retained.

QA download links: `TOBEADD`. Save the external training file as `data/annotations/sft_426916_qa.jsonl` from the repository root. The [data file list](DOWNLOAD_GUIDE.md#included-records-and-additional-inputs) gives download locations and distinguishes external files from included records.

Original media, scans, measured depth/cameras and sourced reference definitions or original annotations are obtained from their providers. Users generate computed geometry, features and tensor caches under `runs/`; these are excluded from downloads. [GENERATION.md](GENERATION.md) specifies original-data answering routes and the missing historical reference/object/target generation. [PREPARATION.md](PREPARATION.md) is a separate route for maintainers who hold the original native records.

## Layout

- `manifests/`: source dataset identifiers, splits, original sample IDs, fixed question/option versions, video/frame mappings, geometry-cache descriptors, and reference assignment records.
- `annotations/`: included validation QA and M01 scoring records; place the separately downloaded training QA here. Complete auxiliary-target sidecars remain external.
- `examples/`: no benchmark media are included; synthetic reader fixtures are in `../tests/`.

Provider downloads belong under `upstream/`, with generated frames, geometry and features under `runs/`. These directories are excluded from ordinary version control. Their acquisition/reconstruction instructions are part of the source payload.

## Training data to account for

The checked SFT file contains 426,916 rows, including 2,696 repeated IDs at the end. It has scene/camera targets on all rows and full 17-component metric targets on 296,054; 130,862 have no object bank. Saved validation contains 17,651 rows. Each actual GRPO sequence contains 8,192 occurrences and 8,191 distinct IDs. These repetitions are retained in the QA files and member lists. Geometry sources include estimated CUT3R geometry and depth/camera-pose records; their original units and masks remain distinct.

The ordered member files in `manifests/` identify reference types, geometry routes and target availability. `grpo_*_source_rows.jsonl` resolves the logged groups to their exact original training rows, including repeated IDs, without changing either run's order. The QA is a field projection, not a complete model input: historical references/targets and visual inputs come from the original native records (see [PREPARATION.md](PREPARATION.md)), and new inputs are built with [GENERATION.md](GENERATION.md).

## Evaluation data to account for

VSI 5,130; historical VSTI 6,042; revised VSTI 5,736; pose-valid subset 5,598 and revised intersection 5,292; the fixed 512 subset and original donor mappings. Keep the associated question wording, option order, labels, and scoring protocol separate.

The separate Base VSTI version uses 6,042 questions with 312 RGB feature files and 9,984 frames. Its ordered member/cache keys are included in `manifests/base_vsti_6042_members.tsv`; its frozen run directory supplies the original prompts/features to `quasar-base-vsti-generate`. It has no inference geometry and is not replaced by the QUASAR spatial-cache input.

External test data cover SPAR, ViewSpatial, SPBench, CV-Bench, and MindCube-Tiny. Case-study manifests must preserve the exact rendered or original videos seen by the models. Display annotations, official verification geometry, and explanatory renders are not model inputs.

## Access and export

Official acquisition sources are [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench), [VSTI-Bench](https://huggingface.co/datasets/Journey9ni/vstibench), [VLM-3R training data](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA), [ScanNet](https://www.scan-net.org/), and [CUT3R](https://github.com/CUT3R/CUT3R). The [VLM-3R source](https://github.com/VITA-Group/VLM-3R) records the VSTI camera-label correction. Its current download must not be substituted for the frozen historical 6042 labels. The full revised set has 5736 records; the checked numerical-pose intersection has 5292.

The native-answer pack, original-data answering generator and historical migration entries are included under `quasar`. QA permissions are recorded per source. Exact historical reference, frame and object/target provenance require the original records; current default builders do not reproduce those historical target versions. Access to sourced reference definitions/annotations and a complete historical generation pipeline are not available in this release.

The existing visual cache identity includes model/input location metadata. `quasar-prepare-data` checks the original identity and settings, then relocates paths and matching keys and compares numerical arrays/tensors. The loader's checks are unchanged, and historical target values/cache priority are retained. Reconstruction from RGB uses a separate root and the exact original target-frame index.

Check each upstream dataset's terms before redistributing raw videos, scans, meshes, images or derived annotations. Included records use logical sample/video IDs and relative paths.

Per-source QA permissions are listed in [TERMS.md](TERMS.md). VSTI questions and answers are not redistributed because the dataset card states no data license; download them from the provider. Checks on fixed samples are summarized in [package checks](../docs/PACKAGE_CHECKS.md), and [release scope](../docs/RELEASE_SCOPE.md) lists what is included and what must be obtained elsewhere.
