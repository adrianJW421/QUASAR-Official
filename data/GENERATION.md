# Generate inputs from original data

For Tables 1 and 2, use the [README entry](../README.md), [input examples](../docs/INPUT_EXAMPLES.md) and [main-table input contract](../docs/TABLE12.md). That entry supplies public frame manifests for all 312 VSTI and 288 VSI videos and invokes the steps below automatically. The examples on this page also cover separate training/native-scan routes and do not replace main-table frame membership.

Computed visual features, estimated points/poses and auxiliary tensor caches are generated under the caller's `runs/` directory. They are not part of the source or the weight archives. Video evaluation uses provider questions/videos and local reconstruction; measured depth and camera calibration are required only by the separate native-scan route. Explicit station references require their own justified source.

`quasar-prepare-public-inputs` is the reconstruction entry. `quasar-prepare-data` is a separate historical-original migration tool: it requires existing native records/caches, preserves their values and rejects missing/mismatched files. Its requirements do not define the public reconstruction route.

## Original questions and frames

Use the licensed ordered training/validation QA for their respective members. Download the training file separately using the [data guide](DOWNLOAD_GUIDE.md#included-records-and-additional-inputs); validation QA is included. Join by `source_row` and `sample_id`; repeated training IDs are distinct occurrences. For the upstream VSI `test.jsonl`, convert its fields without renumbering the original IDs:

```bash
quasar-convert-vsi --input-jsonl ./upstream/vsi/test.jsonl \
  --output-jsonl ./runs/input/vsi_qa.jsonl
```

This current upstream conversion is distinct from the packaged M01 scoring records. It does not recreate M01's unknown loader/frame settings or QUASAR's historical candidate bank.

For each selected original video, prepare a JSONL video plan with `video_id`, `video_path`, `expected_frame_ids` and `image_long_edge`. Supply the intended frame-version record. The extractor supports the recorded VLM-3R exact-32 rule: `round(i * (N-1) / 31)`, for `i=0..31`, 720-pixel long edge, JPEG quality 95. It probes the video and rejects disagreement with the supplied frame list. Complete historical candidate-to-original-frame mapping is not included for every source; do not infer it from member IDs. Other historical candidate-selection rules require their own verified converter. Install working FFmpeg/FFprobe executables; explicit paths can be supplied with `--ffmpeg` and `--ffprobe`.

For the checked current upstream VSI video `arkitscenes/41069025.mp4` with 5,045 frames, `videos.jsonl` contains:

```json
{"video_id":"arkitscenes__41069025","video_path":"./upstream/vsi/videos/arkitscenes/41069025.mp4","image_long_edge":720,"expected_frame_ids":[0,163,325,488,651,814,976,1139,1302,1464,1627,1790,1953,2115,2278,2441,2603,2766,2929,3091,3254,3417,3580,3742,3905,4068,4230,4393,4556,4719,4881,5044]}
```

For its first original QA occurrence, `questions.jsonl` contains:

```json
{"source_row":1,"sample_id":"official_vsi::0","video_id":"arkitscenes__41069025","scene_id":"41069025","geometry_source":"cut3r"}
```

The video's path is the caller's download location; directory layouts may differ. Changed frame count/version fails explicitly. This example exercises current-source canonical answering, not M01 or historical Final benchmark inputs.

```bash
quasar-extract-frames --video-plan ./runs/input/videos.jsonl \
  --output-root ./runs/raw_rgb --num-frames 32
quasar-export-geometry \
  --video-frame-manifest ./runs/raw_rgb/video_frame_manifest.jsonl \
  --output-root ./runs/geometry \
  --cut3r-repo-root "$CUT3R_REPO" --cut3r-model-path "$CUT3R_MODEL" \
  --image-size 512 --device cuda:0
```

Official CUT3R code/weights are external downloads. Install its dependencies and build `curope` as specified in [ENVIRONMENT.md](../docs/ENVIRONMENT.md). The initial input is original media, not an old geometry NPZ.

## Reference definitions and native input

The input plan contains one row per selected QA occurrence, in source order. Required fields are `source_row`, `sample_id`, `video_id`, `scene_id` and `geometry_source` (`cut3r` or `scannet_depth_pose`). Target-camera questions also require the original zero-based `target_frame_index`, consistent with the question's original reference rule.

```bash
quasar-prepare-public-inputs --qa-jsonl ./runs/input/vsi_qa.jsonl \
  --input-plan ./runs/input/questions.jsonl \
  --frame-manifest ./runs/raw_rgb/video_frame_manifest.jsonl \
  --geometry-root ./runs/geometry \
  --output-root ./runs/prepared --max-visual-frames 32
```

The output separates `inference.jsonl` from `labels.jsonl`. Question text and choice order enter inference; gold answers, object annotations and training targets do not.

| Reference | Construction | Required material |
| --- | --- | --- |
| Target camera | Original target frame's pose from the new observation geometry | Target-frame identity and exact RGB order |
| Canonical scene | Mean of the same observation's valid points; existing maximum-span scale with minimum 1 | Selected observation; no object or answer labels |
| Semantic situated viewpoint | Explicit supplied definition in the observation's coordinates | Justified source and coordinate conversion; missing definitions stop processing |

`--reference-jsonl` supplies semantic definitions keyed by `source_row` and `sample_id`. Each row contains `source`, `geometry_gauge` and either a finite `pose_4x4` or original `loc3r_position_xy`/`loc3r_orientation_rad`. XY/yaw fields are supported only for `sqa3d_loc3r` with original ScanNet depth/pose, using the existing transform. Coordinate identifiers are `cut3r_reconstruction:VIDEO_ID` and `scannet_world:SCENE_ID`. Assigning an identifier does not prove correct conversion. This entry performs no alignment or scale fitting. A pose in another reconstruction cannot be relabelled as a matching reference.

For supplied station/object coordinates and known alignment, [construct a reference](../docs/INPUT_EXAMPLES.md#a-supplied-station-and-heading) with `quasar-construct-reference`. It does not locate objects or estimate missing alignment. Source-specific gaps are collected in [release scope](../docs/RELEASE_SCOPE.md#known-limitations).

## Original depth and cameras

The checked ScanNet route starts from the original `.sens`. Install `pip install '.[sensor]'` and obtain the upstream Python 3 reader separately. Its code and original scans are not included in this package:

```bash
mkdir -p ./upstream/scannet_reader
curl -fL https://raw.githubusercontent.com/VITA-Group/VLM-3R/cfeae6f7f5b93faaae439be0100d3b7dc74a297a/vlm_3r_data_process/src/metadata_generation/ScanNet/preprocess/SensorData.py \
  -o ./upstream/scannet_reader/SensorData.py
quasar-export-scannet-sensor --sensor-reader ./upstream/scannet_reader/SensorData.py \
  --sens ./upstream/scannet/scans/scene0380_00/scene0380_00.sens \
  --scene-id scene0380_00 --output-root ./runs/scannet_frames
```

This preserves the checked historical conversion: every twentieth native frame, independent nearest-neighbor resizing of color and depth to 480×640, unchanged original camera matrices saved to six decimal places, and depth in millimetres. No crop or depth filling is applied. The native model reader divides depth by 1000. This reproduces the recorded input convention; it does not recalibrate the original matrices after resizing. The fixed scene's 32 selected color/depth images match its retained export exactly; pose/calibration text matches as well. Sensor versions with a different depth unit fail explicitly.

The plan must include `expected_frame_ids` for the original finite-pose selection: sort native color frame IDs, retain existing depth/finite pose, then use the existing rounded 32-position selection. Changed membership/order is rejected.

```bash
quasar-prepare-public-inputs --qa-jsonl ./data/annotations/sft_426916_qa.jsonl \
  --input-plan ./runs/input/depth_questions.jsonl \
  --scannet-frame-root ./upstream/scannet/frames \
  --reference-jsonl ./runs/input/sourced_references.jsonl \
  --output-root ./runs/depth_prepared --max-visual-frames 32
```

For the supplied SQA3D example, generate the reference from the original balanced train questions and annotations, preserving alternate-situation order and the original XY/quaternion-yaw definition:

```bash
quasar-convert-sqa3d-reference --qa-jsonl ./data/annotations/sft_426916_qa.jsonl \
  --input-plan ./data/examples/sqa3d_depth_plan.jsonl \
  --questions ./upstream/sqa3d/assets/data/sqa_task/balanced/v1_balanced_questions_train_scannetv2.json \
  --annotations ./upstream/sqa3d/assets/data/sqa_task/balanced/v1_balanced_sqa_annotations_train_scannetv2.json \
  --split train --output ./runs/input/sqa3d_reference.jsonl
quasar-prepare-public-inputs --qa-jsonl ./data/annotations/sft_426916_qa.jsonl \
  --input-plan ./data/examples/sqa3d_depth_plan.jsonl \
  --scannet-frame-root ./runs/scannet_frames \
  --reference-jsonl ./runs/input/sqa3d_reference.jsonl \
  --output-root ./runs/depth_prepared --max-visual-frames 32
```

Only question-defined location/orientation enters this reference file. Answer and object annotations are not copied. A changed question, situation, scene or coordinate family is rejected. This route computes points from original measured depth/poses. ARKitScenes timestamp/calibration normalization, ScanNet++ conversion and other external benchmark converters remain source-specific delivery gaps.

## ScanNet target-frame examples

The supplied `vlm3r_train_plan.jsonl` and `vlm3r_val_plan.jsonl` retain the original questions, source rows and zero-based target indices 19 and 28. The raw scans contain 1,090 and 953 frames; `scannet_video_plan.jsonl` retains their recorded 32 frame IDs. VLM-3R does not distribute all raw training videos. Its pinned `export_video.py`, beside the same `SensorData.py`, provides the upstream conversion:

```bash
curl -fL https://raw.githubusercontent.com/VITA-Group/VLM-3R/cfeae6f7f5b93faaae439be0100d3b7dc74a297a/vlm_3r_data_process/src/metadata_generation/ScanNet/preprocess/export_video.py \
  -o ./upstream/scannet_reader/export_video.py
python ./upstream/scannet_reader/export_video.py \
  --scans_dir ./upstream/scannet/selected_scans --output_dir ./runs/scannet_videos \
  --width 640 --height 480 --fps 24 --frame_skip 1 --max_workers 1
quasar-extract-frames --video-plan ./data/examples/scannet_video_plan.jsonl \
  --output-root ./runs/scannet_rgb --num-frames 32
quasar-export-geometry --video-frame-manifest ./runs/scannet_rgb/video_frame_manifest.jsonl \
  --output-root ./runs/scannet_geometry --cut3r-repo-root "$CUT3R_REPO" \
  --cut3r-model-path "$CUT3R_MODEL" --image-size 512 --device cuda:0
quasar-prepare-public-inputs --qa-jsonl ./data/annotations/sft_426916_qa.jsonl \
  --input-plan ./data/examples/vlm3r_train_plan.jsonl \
  --frame-manifest ./runs/scannet_rgb/video_frame_manifest.jsonl \
  --geometry-root ./runs/scannet_geometry --output-root ./runs/train_example --max-visual-frames 32
quasar-prepare-public-inputs --qa-jsonl ./data/annotations/validation_17651_qa.jsonl \
  --input-plan ./data/examples/vlm3r_val_plan.jsonl \
  --frame-manifest ./runs/scannet_rgb/video_frame_manifest.jsonl \
  --geometry-root ./runs/scannet_geometry --output-root ./runs/val_example --max-visual-frames 32
```

`selected_scans` contains only `scene0191_00/scene0191_00.sens` and `scene0705_01/scene0705_01.sens`. Frame identity is checked.

## Features and answering

```bash
quasar-export-visual-cache --input-jsonl ./runs/prepared/inference.jsonl \
  --output-root ./runs/visual_cache --qwen-model-path "$MODEL_PATH" \
  --max-visual-frames 32 --max-pixels 50176 --depth-grid-h 24 --depth-grid-w 32 \
  --shard-index 0 --num-shards 1 --device cuda:0 --inference-only
python scripts/run_recorded.py --role decoderaux_grpo_alltask_final_vsi \
  --parameters ./runs/input/final_paths.json --inference-only --execute
```

`final_paths.json` binds the recorded recipe's six path parameters: `EVAL_JSONL_32` to the new `inference.jsonl`, `OUTPUT_DIR_42` to a new answer directory, `QWEN_MODEL_PATH` to the official backbone, `DECODERAUX_GRPO_ALLTASK_FINAL_2048` to the selected Final checkpoint, `VISUAL_CACHE_ROOT_38` to the new features, and `SCANNET_FRAME_ROOT` to the same original depth-frame root used for export. For RGB-only input without depth, bind the latter to an explicit empty directory. The runner retains every recorded model/generation setting and adds the answer-free input flag.

For depth, use its prepared input and a separate feature output; supply the same `--scannet-frame-root` during feature export and generation. Score with the independently supplied labels and matching protocol. The principal backbone is Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; the final project checkpoint is W01 (GRPO step 2,048); see the [checkpoint guide](../checkpoints/README.md).

## Training material still required

This entry creates answering inputs. Training additionally requires original object annotation, coordinate transformations/truncation, reference validity and the exact run-specific target/mask generator. The included `prepare_native_qwen_moe_metric_sidecar` reproduces the saved 17 components and missing-target masks on six retained historical inputs spanning VLM-3R, ScanQA, MSQA and SQA3D, with predeclared absolute/relative tolerances of 1e-6. That check used retained inputs, not regeneration from raw public data. Its camera component uses distances to the query where valid; the runtime's default target builder is a distinct path. Coordinates from estimated geometry are not automatically metres. Historical object-bank production, alignment and the effective object truncation limit remain incompletely bound. It is therefore not yet a complete public training-data recipe.

Training QA and target-availability/member records remain in the source package. Complete historical object/reference/auxiliary-target generation is not available.
