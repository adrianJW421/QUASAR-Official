# Data download and preparation guide

For Tables 1 and 2, use the focused [evaluation download guide](TABLE12_DOWNLOADS.md) and the single-path-file [reviewer entry](../README.md). The training and external-comparison sections below are separate.

This guide lists original-provider access routes and the inputs needed by the recorded experiments. Provider pages and metadata were last checked on 2026-10-03. Each provider's terms decide whether its contents may be redistributed.

## Included records and additional inputs

`TOBEADD` marks a download link that will be added after upload. The training QA file is downloaded separately; the validation and historical Base VSI scoring files are included. Save the training file at the path below before using the training-data commands. Paths are relative to `data/`.

| Material | Delivery and use | Download / access |
| --- | --- | --- |
| Historical SFT QA | `annotations/sft_426916_qa.jsonl`: 426,916 ordered rows; keep repeated IDs | External download: `TOBEADD` |
| Saved validation QA | `annotations/validation_17651_qa.jsonl`: 17,651 ordered rows | `TOBEADD` |
| Historical Base VSI scoring QA | `annotations/vsi_m01_5130_scoring.jsonl`: 5,130 labels/options matching the retained Base answers | `TOBEADD` |
| Training, validation, GRPO and evaluation membership | `manifests/`; both 8,192-group GRPO runs retain their own order and source-row links | Included |
| Numerical-pose and fixed-512 controls | Member IDs and donor maps under `../experiments/interventions/`; these are not the missing VSTI questions or camera poses | Included |
| RGB, scans, depth and camera information | Obtain from the providers below; raw media is not included | Official sources below |
| Sourced reference definitions and original object/training annotation | Required where the source supplies them; [Generation](GENERATION.md) specifies coordinate/source requirements and gaps | Source-specific access |
| Computed geometry, visual features and tensor caches | Generated under the caller's `runs/` directory; excluded from downloads | Generated locally |

The QA retains source-specific licenses in [SOURCE_PERMISSIONS.tsv](manifests/SOURCE_PERMISSIONS.tsv). QA files alone are not model-ready inputs. The [experiment index](../docs/EXPERIMENT_INDEX.md) lists which experiment groups include answer files.

## Training questions

| Source | Official home and actual access route | Needed files and layout | Terms and version |
| --- | --- | --- | --- |
| VLM-3R | [Project](https://github.com/VITA-Group/VLM-3R); [dataset files](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA/tree/main) | `vsibench_train/merged_qa_*_train.json`; `vstibench_train/qa_*.json`; keep these directories under `upstream/vlm3r/` | Dataset card: Apache-2.0. July 13, 2026 camera-position corrections changed five training files and removed invalid items. The project training QA is pre-correction; its complete native sidecars are separate |
| ScanQA | [Official repository](https://github.com/ATR-DBI/ScanQA); [author-linked Google Drive folder](https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2?usp=sharing) | `ScanQA_v1.0_train.json`, `ScanQA_v1.0_val.json`, with scene/object IDs; use `upstream/scanqa/data/qa/`, as in the [preparation instructions](https://github.com/ATR-DBI/ScanQA/blob/main/docs/dataset.md) | [CC-BY-NC-SA-3.0](https://github.com/ATR-DBI/ScanQA/blob/main/LICENSE). The folder is reachable; Google may require login for an individual download. ScanNet access remains separate |
| SQA3D | [Official repository](https://github.com/SilongYong/SQA3D); [Zenodo record 7792397](https://zenodo.org/records/7792397); [file format](https://github.com/SilongYong/SQA3D/blob/master/assets/dataset.md) | Download `sqa_task.zip`; retain `sqa_task/balanced/v1_balanced_questions_{split}_scannetv2.json`, `v1_balanced_sqa_annotations_{split}_scannetv2.json`, and `sqa_task/answer_dict.json` under `upstream/sqa3d/assets/data/` | Data: CC-BY-4.0; code: Apache-2.0. Use the train/val split named by the native records, not all splits joined together |
| MSQA / MSR3D | [Official repository](https://github.com/MSR3D/MSR3D); [author's data-request form](https://docs.google.com/forms/d/e/1FAIpQLSdzj25kXcthCKtypAI0wQP8j16e9F7ODBroL4SCH_ly8_3rKw/viewform?usp=sf_link) | Request the ScanNet QA component; retain `upstream/msr3d/scannet/msqa_scannet_{split}.json` and any object images required by its records | Data: CC-BY-NC-SA-4.0. Access requires the authors' request form. The supplied training projection uses ScanNet MSQA, not an automatic expansion to 3RScan/ARKitScenes MSQA |

Use the project QA files listed above for the recorded training and validation sets. For a new corrected VLM-3R source download, this checked revision is available:

```bash
hf download Journey9ni/VLM-3R-DATA --repo-type dataset \
  --revision aae3535e513d36be5fd0a1edbaab523c8a3c45e4 \
  --local-dir ./upstream/vlm3r
```

That command obtains corrected upstream QA; it does not reconstruct the original 426,916-row training JSONL, repaired objects, reference records or saved targets. A pre-correction upstream training revision has not been bound to the complete native training set; use the separately downloaded historical training QA and, where available, the original native records for that experiment.

## VSI and VSTI evaluation

VSI uses [the official VSI-Bench repository](https://huggingface.co/datasets/nyu-visionx/VSI-Bench). The checked revision `bdcadb3fea447621a828a24911801faba3587c12` contains `test.jsonl` and the three video archives. Its card's generic full-Parquet description does not match the listed full-QA filename; use the actual `test.jsonl`. Keep the original full 5,130-question set; the debiased/pruned variants are different sets. QA is Apache-2.0; the underlying video sources have their own terms.

```bash
hf download nyu-visionx/VSI-Bench --repo-type dataset \
  --revision bdcadb3fea447621a828a24911801faba3587c12 \
  --include 'test.jsonl' 'README.md' 'scannet.zip' 'scannetpp.zip' 'arkitscenes.zip' \
  --local-dir ./upstream/vsi
unzip ./upstream/vsi/scannet.zip -d ./upstream/vsi/videos
unzip ./upstream/vsi/scannetpp.zip -d ./upstream/vsi/videos
unzip ./upstream/vsi/arkitscenes.zip -d ./upstream/vsi/videos
```

Preserve each archive's source/scene/video filenames. Inspect its directory listing before setting the video root; do not rename scene IDs. Historical Base raw-video inputs and QUASAR candidate-frame inputs differ, including 671 option lists. The supplied Base scoring file preserves that version; a current download is not a replacement for the recorded options or frame manifest.

VSTI's official [project](https://github.com/VITA-Group/VLM-3R) links to [Journey9ni/vstibench](https://huggingface.co/datasets/Journey9ni/vstibench). The card and metadata declare no dataset license, so this package does not redistribute VSTI questions, answers or videos; it contains member and donor IDs only. Download the benchmark from the provider and follow its terms. These are the two recorded releases:

| Evaluation release | Official revision | Files and expected scope |
| --- | --- | --- |
| Historical | `27ff0a2ddb5b22c1eb036a4249345e6f00ee8ea1` | `test.json`, `ScanNet.tar.gz`; original 6,042 questions, main historical five-task aggregate |
| July camera-label erratum | `0903eb0b0c9d03da730d43503c48302a1f3b412d` | `test.json`, `ScanNet.tar.gz`, `erratum/dropped_items.json`; 5,736 questions, nine subtypes |

Download the version you need:

```bash
hf download Journey9ni/vstibench --repo-type dataset \
  --revision "$VSTI_REVISION" --include 'test.json' 'ScanNet.tar.gz' 'erratum/*' \
  --local-dir "./upstream/vsti/$VSTI_REVISION"
tar -xzf "./upstream/vsti/$VSTI_REVISION/ScanNet.tar.gz" \
  -C "./upstream/vsti/$VSTI_REVISION"
```

Set `VSTI_REVISION` to one of the complete IDs above. Keep its `test.json` beside the extracted video tree and retain the archive's scene names. The 5,598 parser-valid historical set, corrected intersection of 5,292 questions, and fixed 512 questions from 127 scenes are project selections; reproduce them with their recorded member/donor maps, not a fresh random sample. Gold changes between releases must remain separate.

## External evaluation sources

These provider revisions identify currently accessible files; see [release scope](../docs/RELEASE_SCOPE.md#known-limitations) for historical conversion coverage.

| Source / experiment | Official home and files | Required subset/layout and access |
| --- | --- | --- |
| SPAR-Bench / T05 | [Official dataset](https://huggingface.co/datasets/jasonzhango/SPAR-Bench); [RGB-D version](https://huggingface.co/datasets/jasonzhango/SPAR-Bench-RGBD) | `data/test-00000-of-00004.parquet` through `00003`; retain under `upstream/spar/data/`. Recorded evaluator: 7,211 rows; card prose says 7,207, so do not truncate to that prose count. RGB-D files carry depth/pose/intrinsics. No explicit active data license was found in the checked cards; request terms rather than infer a code license |
| ViewSpatial / T06 | [Official dataset](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench) | `ViewSpatial-Bench.json`, `scannetv2_val.zip`, `val2017.zip`; extract under `upstream/viewspatial/` while retaining paths in the JSON. Recorded set: 5,712. Card: Apache-2.0; ScanNet and [COCO](https://cocodataset.org/#download) image terms remain applicable |
| SPBench / T07 | [Official project](https://github.com/ZJU-REAL/SpatialLadder); [dataset](https://huggingface.co/datasets/hongxingli/SPBench) | `SPBench-SI.parquet`, `SPBench-MV.parquet` and their matching `*-images.zip` archives under `upstream/spbench/`. Preserve both SI 1,009 and MV 319 rows; combined 1,328. Card: Apache-2.0; source ScanNet terms remain applicable |
| CV-Bench / T08 | [Official project](https://github.com/cambrian-mllm/cambrian); [dataset](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | `test_2d.parquet`, `test_3d.parquet`, matching JSONL and `build_img.py` under `upstream/cvbench/`. Parquet holds images; the upstream `build_img.py` creates `img/` for JSONL filename-based use. Preserve 1,438 2D + 1,200 3D = 2,638 rows. Card: Apache-2.0 |
| MindCube-Tiny / T09 | [Official project](https://github.com/mll-lab-nu/MindCube); [dataset](https://huggingface.co/datasets/MLL-Lab/MindCube) | Extract `data.zip` under `upstream/mindcube/`; use `data/raw/MindCube_tinybench.jsonl` and the referenced `data/other_all_image/{around,among,rotation,translation}/` files. Keep all 1,050 tiny-benchmark questions, not the training or full question set. Card: MIT |

| Source | Checked provider revision |
| --- | --- |
| SPAR-Bench | `ee122877c25c8bb08539b07e06d872152c9968f1` |
| SPAR-Bench-RGBD | `9e322d92065c8ee19213fe46d352092e1990ae8b` |
| ViewSpatial | `95d9ea3a94da7190bd19d2539ff661f93dba21dd` |
| SPBench | `03611025a4e6032c558117c0e86b76c8b084c305` |
| CV-Bench | `bc284db50d036958861cb60cdd7b77612052ce0d` |
| MindCube | `9c941b46a6bd65b6914669ef7a579948fc9c8467` |

Use the matching repository/revision above with the official CLI. Public metadata pages were reachable without login; authenticated downloads may depend on provider terms and the user's environment.

```bash
hf download "$DATASET_REPO" --repo-type dataset \
  --revision "$DATASET_REVISION" --local-dir "$DATASET_ROOT"
unzip ./upstream/viewspatial/scannetv2_val.zip -d ./upstream/viewspatial
unzip ./upstream/viewspatial/val2017.zip -d ./upstream/viewspatial
unzip ./upstream/spbench/SPBench-SI-images.zip -d ./upstream/spbench
unzip ./upstream/spbench/SPBench-MV-images.zip -d ./upstream/spbench
unzip ./upstream/mindcube/data.zip -d ./upstream/mindcube
```

The fixed image-count rule for T05–T07 combines existing answers and needs no additional dataset or trained weights. Native conversions must preserve question IDs, image order, choices and answer format. `quasar-prepare-data` relocates an existing native conversion; it does not convert these upstream schemas. The original conversions and scripts are not included.

## Original scenes, depth and cameras

### ScanNet v2

Use [the official project](https://github.com/ScanNet/ScanNet) and its [Terms of Use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf). Follow the institutional-email agreement process described by the project; use the downloader issued after approval. No credentials or approval are provided by this package.

Obtain only the scene IDs required by the training/evaluation members: training scenes for the training QA; validation scenes for VSI/VSTI. For a native scan route retain `scans/<scene_id>/<scene_id>.sens`, mesh, aggregation and segmentation files. Use the official [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) to export `color/`, `depth/`, `pose/` and `intrinsic/` under each scene. Preserve the `.sens` header's `depth_shift`, native depth conversion, actual frame-ID mapping and finite-pose selection. A newly spaced frame sample is not the historical frame list.

```bash
python "$SCANNET_REPO/SensReader/python/reader.py" \
  --filename "./upstream/scannet/scans/$SCENE_ID/$SCENE_ID.sens" \
  --output_path "./upstream/scannet/frames/$SCENE_ID" \
  --export_depth_images --export_color_images --export_poses --export_intrinsics
```

Use the environment specified by the upstream reader; this command is its extraction route, not a replacement for the missing historical frame-selection/native-conversion records.

The selected apartment/office/kitchen scenes are `scene0000_00`, `scene0010_00` and `scene0003_00`; the benchmark cases use `scene0354_00` and `scene0426_00`. Project-rendered camera trajectories, clips and exact model-frame pixels are separate derived materials, not official ScanNet downloads.

### ScanNet++

Use [the official portal](https://scannetpp.mlsg.cit.tum.de/scannetpp/), [login](https://scannetpp.mlsg.cit.tum.de/scannetpp/login), [signed terms](https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf), [file documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation), and [official preparation code](https://github.com/scannetpp/scannetpp). Downloads need an approved account.

Retain `data/<scene_id>/iphone/` RGB/depth/pose/calibration assets and `scans/` annotations required by the relevant native records; DSLR assets are needed only when those records select them. Use the portal's extraction instructions for its registered release. VSI's author-provided `scannetpp.zip` supplies benchmark videos but not native scans or historical auxiliary targets.

### ARKitScenes

Use [Apple's official repository](https://github.com/apple-aiml-research/ARKitScenes), [download instructions](https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md) and [license](https://github.com/apple-aiml-research/ARKitScenes/blob/main/LICENSE). The official downloader selects Training/Validation and video IDs. Obtain `mov`, `lowres_wide`, `lowres_depth`, `lowres_wide.traj`, `lowres_wide_intrinsics`, mesh and annotations only as required by the selected native records.

```bash
python3 "$ARKIT_REPO/download_data.py" raw --split Validation \
  --video_id "$ARKIT_VIDEO_ID" --download_dir ./upstream/arkitscenes \
  --raw_dataset_assets mov lowres_wide lowres_depth lowres_wide.traj \
  lowres_wide_intrinsics mesh annotation
```

The resulting tree uses `Validation/<video_id>/` (or `Training/` for training captures). PNG depth is millimetres; `.pincam` holds camera intrinsics and `.traj` holds timestamped poses. Keep their time association. The public preparer accepts already normalized native depth/pose records; a complete historical `.traj`/video-to-native conversion is not supplied.

## Geometry, references and historical targets

New RGB geometry can be generated with [CUT3R's official code/checkpoint](https://github.com/CUT3R/CUT3R#download-checkpoints), using `cut3r_512_dpt_4_64.pth` and its CC-BY-NC-SA-4.0 terms. Follow [the checked environment and component build](../docs/ENVIRONMENT.md). The pinned CUT3R source and component installation are in the main-table environment guide.

Supply the original ordered frame manifest. Each line requires `video_id`, `frame_candidate_id`, `frame_id`, `sample_order` and `frame_image_path`; orders run from 0 without gaps. For a target-camera input, keep its original `target_frame_index` and the same 32 RGB paths.

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

Store new estimates/features in separate roots. The reference pose and observation must come from that same reconstruction; preserve original RGB order. This chain was verified for one fixed target-camera example, not every source/reference route. The geometry-source oracle also uses [official VGGT](https://github.com/facebookresearch/vggt) and [VGGT-1B](https://huggingface.co/facebook/VGGT-1B), ground-truth masks, alignment and scale calibration; those masks/calibration selections remain missing public inputs.

The public original-data route is [GENERATION.md](GENERATION.md), using the new QA conversion, exact-frame extraction and batch reference/input entries. Maintainers who hold the original native records can migrate them, keeping their saved values and cache priority, with:

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

`bindings.json` lists `source_root`/`logical_root` pairs; logical roots name relative folders in the prepared root. Every absolute path in an original record must be bound. See [PREPARATION.md](PREPARATION.md) for the full input contract, including a separate `--inference-only` root and labels kept outside generation inputs.

For supported input routes and remaining reproduction requirements, see [release scope](../docs/RELEASE_SCOPE.md#known-limitations) and [reproducibility](../docs/REPRODUCIBILITY.md).
