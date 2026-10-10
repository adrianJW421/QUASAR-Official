# Main-table downloads

Evaluation needs the selected benchmark's questions/videos, the official backbone and selected project weights. It does not need the training set. All paths below are relative to the source directory and match `configs/reviewer_paths.example.json`.

## VSI-Bench: Table 2

Provider: [nyu-visionx/VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench/tree/bdcadb3fea447621a828a24911801faba3587c12). Use revision `bdcadb3fea447621a828a24911801faba3587c12`, `test.jsonl` (5,130 rows), and the three video archives. Do not use a pruned/debiased split. QA terms are Apache-2.0; the underlying ScanNet, ScanNet++ and ARKitScenes terms remain applicable.

```bash
hf download nyu-visionx/VSI-Bench --repo-type dataset --revision bdcadb3fea447621a828a24911801faba3587c12 --include test.jsonl README.md scannet.zip scannetpp.zip arkitscenes.zip --local-dir upstream/vsi
mkdir -p upstream/vsi/videos
unzip upstream/vsi/scannet.zip -d upstream/vsi/videos
unzip upstream/vsi/scannetpp.zip -d upstream/vsi/videos
unzip upstream/vsi/arkitscenes.zip -d upstream/vsi/videos
```

Set each `vsi.video_roots` entry to the directory directly containing that source's `<scene>.mp4` files. Preserve scene names. Some previously extracted mirrors have two nested source directories; the configured root must directly contain the MP4 files. Example scene names are `scene0277_02.mp4`, `7b6477cb95.mp4`, and `41069025.mp4`.

Required video names and frame counts are in `manifests/table12_vsi_frames.jsonl` (288 videos). The original provider archives are the raw input; computed geometry/features are generated locally. Historical trained-model option expansion is implemented in `quasar.reviewer_data`; the separate Base M01 options and raw-video protocol are not substituted.

## Historical VSTI-Bench: Table 1

Provider: [Journey9ni/vstibench](https://huggingface.co/datasets/Journey9ni/vstibench/tree/27ff0a2ddb5b22c1eb036a4249345e6f00ee8ea1). Use revision `27ff0a2ddb5b22c1eb036a4249345e6f00ee8ea1`, with `test.json` (6,042 rows) and `ScanNet.tar.gz`. The VSTI-Bench dataset card does not state a data license. This package therefore does not redistribute VSTI questions, answers or videos; it includes only member IDs. Download the benchmark from the provider and follow the provider's terms. The [VLM-3R project](https://github.com/VITA-Group/VLM-3R) provides the benchmark contact/source.

```bash
hf download Journey9ni/vstibench --repo-type dataset --revision 27ff0a2ddb5b22c1eb036a4249345e6f00ee8ea1 --include test.json ScanNet.tar.gz --local-dir upstream/vsti/27ff0a2ddb5b22c1eb036a4249345e6f00ee8ea1
tar -xzf upstream/vsti/27ff0a2ddb5b22c1eb036a4249345e6f00ee8ea1/ScanNet.tar.gz -C upstream/vsti/27ff0a2ddb5b22c1eb036a4249345e6f00ee8ea1
```

Expected layout: `<vsti.video_root>/ScanNet/videos/val/<scene>.mp4`, with `test.json` under the configured QA path. `manifests/table12_vsti_frames.jsonl` lists 312 videos. The corrected 5,736-question revision is a different experiment; the checker rejects it.

## Official backbone

[Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b), Apache-2.0:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --local-dir upstream_models/B01
```

Retain the model configuration, tokenizer/processor, index and all weight shards. Use this directory for all four trained rows, Base VSTI and the new `public-vsi-v1` Base route. The new Base route explicitly adopts this revision; it does not claim that the historical March run used it.

## Project checkpoints

| Role | Directory | Saved step | Download link |
| --- | --- | ---: | --- |
| Visual-only SFT | `upstream_models/W06` | 46592 | `TOBEADD` |
| Visual-only SFT+GRPO | `upstream_models/W05` | 2048 | `TOBEADD` |
| QUASAR SFT | `upstream_models/W02` | 44544 | `TOBEADD` |
| QUASAR SFT+GRPO | `upstream_models/W01` | 2048 | `TOBEADD` |

Extract each archive into `upstream_models/` as shown in the [checkpoint guide](../checkpoints/README.md). The directory must contain `model_spec.json`, the named checkpoint and `qwen_lora_config.json`; the tokenizer/processor come from the pinned official backbone. The checker verifies model ID, step and backbone revision. Do not replace the selected SFT with the last training checkpoint or GRPO parent.

## CUT3R for QUASAR

Obtain [CUT3R source](https://github.com/CUT3R/CUT3R/tree/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf) at commit `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` and its [official 512 checkpoint](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view). Save the checkpoint as `upstream_models/cut3r_512_dpt_4_64.pth`; keep the CC-BY-NC-SA-4.0 terms. Install the component using [environment setup](../docs/ENVIRONMENT.md#cut3r-reconstruction-component). The preparation command calls it automatically.

## Native scans and training

These are separate routes, not additional requirements for Tables 1 and 2. For native ScanNet data, follow the [official project and access process](https://github.com/ScanNet/ScanNet); approval and the downloader are supplied by the provider. [Training downloads](DOWNLOAD_GUIDE.md#training-questions) list VLM-3R, ScanQA, SQA3D and MSQA sources. [Generation](GENERATION.md) covers ScanNet sensor export and auxiliary-target preparation. Historical training-object production is incomplete and is not needed to evaluate downloaded weights.
