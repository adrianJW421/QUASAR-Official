# Load the official backbone and added weights

The added-weight archives are an adaptation of the official model, not a compressed complete 8B model. Every role loads the full official `Qwen/Qwen3-VL-8B-Instruct` backbone, processor and tokenizer at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Download all four official weight shards, their index, model configuration and tokenizer/processor files using [the official download command](../data/TABLE12_DOWNLOADS.md#official-backbone).

Visual-only archives contain the trained LoRA adjustments. QUASAR archives additionally contain the spatial module and decoder-side parameters, including four learned spatial summaries and saved auxiliary readouts. Optimizer/RNG continuation state is excluded. The package's loaders restore the saved parameter names/shapes and architecture; these archives are not generic PEFT adapter directories and are not loaded as standalone `from_pretrained` models.

## First-release files

| Archive | Main-table role | Step | Archive bytes | Model-file bytes | Download link |
| --- | --- | ---: | ---: | ---: | --- |
| `W01.tar.gz` | QUASAR SFT+GRPO | 2,048 | 435,088,176 | 470,450,020 | `TOBEADD` |
| `W02.tar.gz` | QUASAR SFT | 44,544 | 437,140,031 | 472,551,578 | `TOBEADD` |
| `W05.tar.gz` | Visual-only SFT+GRPO | 2,048 | 162,031,225 | 174,765,183 | `TOBEADD` |
| `W06.tar.gz` | Visual-only SFT | 46,592 | 162,014,533 | 174,764,671 | `TOBEADD` |

`TOBEADD` marks a download link that will be added after upload.

The source archive contains none of these files. Base has no project-added weights. Only W01, W02, W05 and W06 are released; checkpoints for the other ablations are not part of this release. [WEIGHTS.tsv](WEIGHTS.tsv) indexes the four released archives.

Download the four archives and extract each one:

```bash
mkdir -p upstream_models
tar -xzf W01.tar.gz -C upstream_models
tar -xzf W02.tar.gz -C upstream_models
tar -xzf W05.tar.gz -C upstream_models
tar -xzf W06.tar.gz -C upstream_models
```

Each directory contains `model_spec.json`, `README.md`, `qwen_lora_config.json`, its model file, the complete `LICENSE-WEIGHTS`, `LICENSE_SCOPE.md`, `LICENSE-QWEN` and `NOTICE`. W01/W02 use `glow_patch_spatial.pt`; W05/W06 use `direct_qwen_sft.pt`. The backbone/tokenizer are downloaded separately, not copied out of these archives.

## Combined loading and answering

Set `backbone` to the official backbone directory (`upstream_models/B01` in the [download command](../data/TABLE12_DOWNLOADS.md#official-backbone)) and the four `weights` entries to their respective directories in `reviewer_paths.json`. The reviewer verifies model ID, saved step and backbone revision before using the matching recorded architecture. These commands exercise each first-release role with the fixed VSTI samples:

```bash
quasar-reviewer --paths reviewer_paths.json --dataset vsti --role visual_sft --smoke
quasar-reviewer --paths reviewer_paths.json --dataset vsti --role visual_grpo --smoke
quasar-reviewer --paths reviewer_paths.json --dataset vsti --role quasar_sft --smoke
quasar-reviewer --paths reviewer_paths.json --dataset vsti --role quasar_grpo --smoke
```

The QUASAR loader registers `<|spatial_query_0|>` through `<|spatial_query_3|>` at IDs 151669–151672 (tokenizer size 151673), constructs four summaries/four updates/five branches at the recorded width, applies the saved LoRA settings, and strictly loads spatial, decoder and LoRA tensors. Saved SIGReg readouts are restored even when their training loss is disabled. Visual-only loading preserves the original visual prompt and uses no spatial summaries. The original backbone and added tensors remain separate; no merging, quantization or re-export is required.

For existing prepared inputs, use the matching record from `configs/commands.json` through `scripts/run_recorded.py --inference-only`; [input examples](../docs/INPUT_EXAMPLES.md) explain the input and scoring files. The selected SFT endpoints W02/44544 and W06/46592 differ from their full-training endpoints and GRPO parents. Do not substitute another checkpoint merely because it was saved later.

The model files in all four archives passed exact tensor readback and strict real-model loading with output-token equality against the original checkpoints, and were used in the seven-cell independent evaluation; see [package checks](../docs/PACKAGE_CHECKS.md) and [known limitations](../docs/RELEASE_SCOPE.md#known-limitations).

## Licenses and archive verification

Each first-release archive identifies CC-BY-NC-4.0 as the license for project-added parameters. The complete official backbone retains Apache-2.0; CUT3R and data retain their own terms. See [license scope](../LICENSE_SCOPE.md). The model remains a backbone plus separate added parameters.

Each archive also contains license texts and a description. Their model `.pt` and `qwen_lora_config.json` files are byte-identical to the checked checkpoints. [WEIGHTS_INTEGRITY.json](WEIGHTS_INTEGRITY.json) lists the SHA-256 of each archive (to verify downloads) and of its model and configuration files, computed over raw file bytes.
