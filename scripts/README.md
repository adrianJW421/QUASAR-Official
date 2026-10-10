# Command entrypoints

`run_recorded.py` binds saved argument lists to explicit caller-supplied paths and prints the real training/generation command; `--execute` runs it. Installed console entrypoints call the actual module entry functions directly. `quasar-score-retained` never calls a model.

For recorded decoder generation, add `--inference-only` to keep gold/object/target reads outside the answering input. [GENERATION.md](../data/GENERATION.md) documents `quasar-convert-vsi`, `quasar-extract-frames` and `quasar-prepare-public-inputs`. They start from QA and original media/depth plus sourced references. `quasar-prepare-data` is for maintainers who hold the original native records ([PREPARATION.md](../data/PREPARATION.md)). `package_source.py` uses the exact public file list and excludes runtime outputs.

The collected entries cover SFT, all-task GRPO, Visual-only, Basic/NoFlow, generation and scoring. Imports are package-local and all locations are caller-provided paths. Historical target/cache readers keep their priority and identity checks.

`build_cut3r_extension.py` builds the original CUT3R `curope` component in a new independent directory, preserving negative camera-token calculations. It adapts the checked CUDA precision API and builds a wheel without exposing GPUs. Installation and the checked frontend dependencies are in [ENVIRONMENT.md](../docs/ENVIRONMENT.md).

Preparation, inference and analysis are separate commands; scoring and analysis commands do not train or generate answers.
