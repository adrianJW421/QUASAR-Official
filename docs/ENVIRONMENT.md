# Checked runtime environment

The [independent evaluation](REPRODUCTION_RESULTS.md) used Linux x86_64, Python 3.10, CUDA 12.8 and one NVIDIA A100 80GB, with the dependency versions below and four CPU cores. The earlier fixed-sample training/data checks recorded 65,928,167,424 bytes of reserved memory; full evaluation or training may need more.

| Dependency | Checked runtime version |
| --- | --- |
| PyTorch | 2.10.0; runtime build 2.10.0+cu128 |
| torchvision | 0.25.0 |
| Transformers | 5.3.0.dev0, official commit `16a5b0936dcaae6efbc03ab1a4fd98dc324bfb9e` |
| NumPy | 2.2.6 |
| Pillow | 12.1.1 |
| qwen-vl-utils | 0.0.14 |
| pytest | 8.4.2 |

`pyproject.toml` declares this runtime profile. [requirements-cuda-tested.txt](../requirements-cuda-tested.txt) records the actual installed dependency snapshot with relocatable package references. The implementation requires `Qwen3VLModel.compute_3d_position_ids`. The installed Transformers distribution records its [official source commit](https://github.com/huggingface/transformers/commit/16a5b0936dcaae6efbc03ab1a4fd98dc324bfb9e); its development version is installed from that source, not by assuming a public release with the same name.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'transformers @ https://github.com/huggingface/transformers/archive/16a5b0936dcaae6efbc03ab1a4fd98dc324bfb9e.zip'
python -m pip install '.[test]'
python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests -c pyproject.toml --rootdir=. -q
```

If the system Python reports that `ensurepip` is unavailable and you cannot install the system `python3.10-venv` package, create an isolated environment without pip, then install pip inside that environment using the official bootstrap:

```bash
python3.10 -m venv --without-pip .venv
curl --fail --location https://bootstrap.pypa.io/get-pip.py -o .venv/get-pip.py
.venv/bin/python .venv/get-pip.py
. .venv/bin/activate
```

Continue with the dependency installation commands above. This environment must not inherit another project's `site-packages` directory.

## CUT3R reconstruction component

Supply [CUT3R](https://github.com/CUT3R/CUT3R/tree/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf) commit `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` and its official 512 model separately. Its 95 Python source files match the historical runtime copy file by file; that historical checkout did not record a git revision. The required source layout is `src/dust3r` and `src/croco/models/curope`, with the original `tokens.type()` precision-dispatch call in `kernels.cu`.

Install the frontend dependencies checked for this reconstruction using [requirements-geometry-tested.txt](../requirements-geometry-tested.txt). The full upstream requirements file pins different NumPy/Pillow versions and is not the checked package profile.

```bash
mkdir -p upstream
curl -L https://github.com/CUT3R/CUT3R/archive/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf.zip -o upstream/CUT3R.zip
unzip upstream/CUT3R.zip -d upstream
export CUT3R_REPO="$PWD/upstream/CUT3R-8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"
export CUT3R_BUILD="$PWD/runs/curope_build"
export CUDA_HOME=/usr/local/cuda-12.8
python -m pip install -r requirements-geometry-tested.txt
python scripts/build_cut3r_extension.py \
  --cut3r-repo-root "$CUT3R_REPO" --build-root "$CUT3R_BUILD" \
  --cuda-home "$CUDA_HOME" --cuda-arch-list 8.0 --jobs 4
python -m pip install --no-deps "$CUT3R_BUILD"/dist/curope-*.whl
python -m pip check
python -c 'import curope; print(curope.__file__)'
```

Use a new build directory outside the source checkout. The build entry copies the original C++/CUDA sources and retains their license/attribution; it only adapts the precision API to `tokens.scalar_type()`. Original rotation equations and negative camera-token positions remain intact. Compiler flags match the checked component. Building explicitly targets A100 architecture 8.0 and hides all GPUs; a CUDA 12.8 toolkit/compiler is required even though compilation uses only CPU. Other architectures have not been checked.

The runtime imports CUT3R from the supplied code root independently of the weight location and requires the compiled `curope` implementation. [PREPARATION.md](../data/PREPARATION.md) gives the observation, same-source target-reference binding and visual-cache export commands.

Original-video extraction additionally requires working FFmpeg and FFprobe executables. Check both with `ffmpeg -version` and `ffprobe -version`; their system codec/math libraries must be available. The checked FFmpeg is 4.4.2 (Ubuntu 22.04). The extractor accepts explicit executable paths and retains the original JPEG encoder defaults; changing its output thread arguments changes JPEG pixels. It runs the recorded exact-frame selection and writes new RGB frames under `runs/`; [GENERATION.md](../data/GENERATION.md) describes the input plan and order checks.

On Ubuntu 22.04, install the system video and archive tools with `sudo apt-get install ffmpeg unzip`. The CUT3R build also needs the CUDA 12.8 toolkit and a compatible C++ compiler. Install them using the administrator's system setup; the Python package does not install system compilers.

If the Ubuntu FFmpeg binary reports missing `libblas.so.3` or `liblapack.so.3`, check the installed library files before running preparation. When the Ubuntu packages are present in their standard subdirectories but the system loader links are missing, the current shell can use them without changing the shared installation:

```bash
test -r /usr/lib/x86_64-linux-gnu/blas/libblas.so.3
test -r /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/blas:/usr/lib/x86_64-linux-gnu/lapack${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ffmpeg -version
ffprobe -version
```

If these files are absent, the required system packages must be installed before preparation. Keep the checked FFmpeg binary and extraction settings when comparing decoded pixels.

## Earlier CPU profile

An earlier CPU-only check ran on macOS arm64 with Python 3.11.3, torch 2.7.0, torchvision 0.22.0, Transformers 5.3.0, Pillow 11.3.0, NumPy 2.2.6, qwen-vl-utils 0.0.14 and pytest 8.4.2. Its dependency list is kept in `requirements-cpu-tested.txt`; the current runtime declaration is `pyproject.toml`. Neither profile reconstructs the June training environment.
