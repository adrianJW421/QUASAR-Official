# Dependencies and third-party material

Qwen visual-forward adaptations are included in nine project modules, with their original Apache-2.0 attribution and modification notices listed in [NOTICE](../NOTICE). Complete upstream libraries and model weights are acquired separately.

The imported external modules include PyTorch, torchvision, Transformers, NumPy, Pillow and qwen-vl-utils. Reconstruction also uses the caller's CUT3R frontend and its original compiled `curope` component. Installed-distribution license texts are retained in `licenses/`; data/model terms remain separate from project code licensing.

## Current checked CUDA runtime

| Component | Checked version/source | Notices |
| --- | --- | --- |
| PyTorch | 2.10.0, runtime cu128 build | `licenses/cuda_runtime/torch/` |
| torchvision | 0.25.0 | `licenses/cuda_runtime/torchvision/` |
| Transformers | 5.3.0.dev0; official commit `16a5b0936dcaae6efbc03ab1a4fd98dc324bfb9e` | `licenses/cuda_runtime/transformers/` |
| NumPy | 2.2.6 | `licenses/numpy/` |
| Pillow | 12.1.1 | `licenses/cuda_runtime/pillow/` |
| qwen-vl-utils | 0.0.14 | Apache-2.0 installed metadata label; the wheel contains no separate license file |
| Reconstruction dependencies | `requirements-geometry-tested.txt` | `licenses/cuda_runtime/` and `CUDA_SOURCE_NOTICES.json` |
| CUT3R / CroCo `curope` | Commit `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`; its 95 Python source files match the historical runtime copy file by file (the historical checkout did not record a git revision) | [upstream license](https://github.com/CUT3R/CUT3R/blob/main/LICENSE), `licenses/CUT3R-LICENSE.txt`; original Naver attribution retained in compiled-source headers and built wheel notices |

The current dependency snapshot is `requirements-cuda-tested.txt`. CUT3R source and weights are supplied separately; the build entry copies sources into the caller's new build directory and changes only the CUDA precision-dispatch API. Original rotation calculations and signed positions are preserved. [ENVIRONMENT.md](../docs/ENVIRONMENT.md) gives exact installation/build steps. This profile was used for the fixed-sample runtime checks; it is not the June training environment.

## Earlier CPU installation

The following versions and license labels were read from an earlier CPU installation. Its notices remain alongside the current runtime notices.

| Component | Version | License | Official source |
| --- | --- | --- | --- |
| PyTorch | 2.7.0 | BSD-3-Clause | [source](https://github.com/pytorch/pytorch/tree/v2.7.0) |
| torchvision | 0.22.0 | BSD-3-Clause | [source](https://github.com/pytorch/vision/tree/v0.22.0) |
| Transformers | 5.3.0 | Apache-2.0 | [source](https://github.com/huggingface/transformers/tree/v5.3.0) |
| NumPy | 2.2.6 | BSD-3-Clause; bundled notices retained | [source](https://github.com/numpy/numpy/tree/v2.2.6) |
| Pillow | 11.3.0 | MIT-CMU | [source](https://github.com/python-pillow/Pillow/tree/11.3.0) |
| qwen-vl-utils | 0.0.14 | Apache-2.0 | [source](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-utils) |
| Qwen3-VL-8B-Instruct | Recorded model revision in checkpoints/README.md | Apache-2.0 upstream model terms | [model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| CUT3R geometry frontend | Not installed in that CPU check; see the CUT3R row above | CC-BY-NC-SA-4.0 | [source](https://github.com/CUT3R/CUT3R) |

qwen-vl-utils supplies the Apache-2.0 license label in installed metadata; that wheel does not contain a separate license file. Reconstruction checks used separately obtained CUT3R code and weights, which are not in this package. SIGReg and the project scorer are implemented in the existing project modules. Original project contributions use Apache-2.0; upstream-derived portions retain their own notices.

Retain required upstream license texts and notices. Model downloads, third-party raw datasets, and derived assets have separate terms from this project's implementation. Publicly available material is not automatically redistributable.

The [project license scope](../LICENSE_SCOPE.md) distinguishes original Apache-2.0 code, CC-BY-NC-4.0 added weights and independently licensed third-party material. Third-party attribution is kept in full.

## Fixed upstream license references

- [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md) labels that exact model revision Apache-2.0. The model tree has no separate LICENSE file; the complete Apache text is retained here as [Qwen3-VL-Apache-2.0.txt](licenses/Qwen3-VL-Apache-2.0.txt), from the [official Qwen repository](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/LICENSE).
- [Transformers](https://github.com/huggingface/transformers/blob/16a5b0936dcaae6efbc03ab1a4fd98dc324bfb9e/LICENSE) and its Qwen code retain Apache-2.0, including the original Qwen/Hugging Face copyright. The checked revision identifies the current interface, not a reconstructed historical training checkout.
- [CUT3R](https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/LICENSE) and its [CroCo CUDA source](https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/src/croco/models/curope/kernels.cu) retain CC-BY-NC-SA-4.0. The separately built component carries a complete copy of that license and marks the precision-dispatch modification in the modified source. The project's CC-BY-NC-4.0 weight license does not replace CUT3R's share-alike requirement.

Only original QUASAR material is licensed by the project contributors. The package includes upstream attribution for adapted code and retained dependency license documents, not copies of full external distributions. ScanNet's sensor reader is supplied through `--sensor-reader`; no reader source is bundled. Included QA attributions, transformations and license texts are recorded in [data/TERMS.md](../data/TERMS.md).
