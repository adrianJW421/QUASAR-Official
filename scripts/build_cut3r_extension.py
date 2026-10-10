"""在独立目录编译 CUT3R 原计算组件；不改上游文件，也不加载显卡。"""
import argparse
import difflib
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut3r-repo-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--cuda-arch-list", default="8.0", help="8.0 is the checked A100 architecture")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    repo = args.cut3r_repo_root.resolve()
    work = args.build_root.resolve()
    source = repo / "src/croco/models/curope"
    if work.is_relative_to(repo):
        raise ValueError("Use a separate build root outside the read-only CUT3R checkout")
    kernel = (source / "kernels.cu").read_text()
    if kernel.count("tokens.type()") != 1:
        raise ValueError("Expected the checked CUT3R kernel with exactly one tokens.type() call")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    # 只换 torch 的精度接口名称，原旋转公式及相机 token 的负位置保持不变。
    adapted = ("// Modified for QUASAR: CUDA precision dispatch uses tokens.scalar_type(); "
               "rotation calculations are unchanged.\n"
               + kernel.replace("tokens.type()", "tokens.scalar_type()"))
    work.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "curope.cpp", work / "curope.cpp")
    shutil.copy2(repo / "LICENSE", work / "LICENSE")
    shutil.copy2(Path(__file__).resolve().parents[1] / "data/licenses/CC-BY-NC-SA-4.0.txt",
                 work / "LICENSE-CC-BY-NC-SA-4.0.txt")
    (work / "NOTICE").write_text(
        "CUT3R / CroCo curope component\n"
        "Copyright (C) 2022-present Naver Corporation. All rights reserved.\n"
        "Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n"
        "Upstream: https://github.com/CUT3R/CUT3R\n"
        "The CUDA precision dispatch uses tokens.scalar_type(); rotation calculations are unchanged.\n"
    )
    (work / "kernels.cu").write_text(adapted)
    (work / "kernels.cu.diff").write_text("".join(difflib.unified_diff(
        kernel.splitlines(keepends=True), adapted.splitlines(keepends=True),
        fromfile="CUT3R/kernels.cu", tofile="build/kernels.cu")))
    (work / "setup.py").write_text('''from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name="curope", version="0.0.0", license="CC-BY-NC-SA-4.0",
    license_files=["LICENSE", "LICENSE-CC-BY-NC-SA-4.0.txt", "NOTICE"],
    ext_modules=[CUDAExtension(name="curope", sources=["curope.cpp", "kernels.cu"],
        extra_compile_args={"cxx":["-O3"], "nvcc":["-O3", "--ptxas-options=-v", "--use_fast_math"]})],
    cmdclass={"build_ext":BuildExtension})
''')
    # 编译目标由参数给定，无须探测或占用显卡；构建失败直接保留日志并退出。
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES="", CUDA_HOME=str(args.cuda_home.resolve()),
        TORCH_CUDA_ARCH_LIST=args.cuda_arch_list, MAX_JOBS=str(args.jobs),
        OMP_NUM_THREADS=str(args.jobs), MKL_NUM_THREADS=str(args.jobs),
        OPENBLAS_NUM_THREADS=str(args.jobs), NUMEXPR_NUM_THREADS=str(args.jobs),
        PYTHONDONTWRITEBYTECODE="1")
    env["PATH"] = str(args.cuda_home.resolve() / "bin") + os.pathsep + env["PATH"]
    command = [sys.executable, "-I", "-B", "-m", "pip", "wheel", "--no-deps",
        "--no-build-isolation", "--no-cache-dir", "--wheel-dir", str(work / "dist"), str(work)]
    log = work / "build.log"
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=work, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"CUT3R extension build failed; see {log}", file=sys.stderr)
        raise SystemExit(result.returncode)
    wheels = list((work / "dist").glob("curope-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one curope wheel, found {len(wheels)} in {work / 'dist'}")
    print(wheels[0])


if __name__ == "__main__":
    main()
