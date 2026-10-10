"""将留存的命令参数绑定到使用者自己的文件；默认只显示命令。"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, help="Role in configs/commands.json.")
    parser.add_argument("--parameters", type=Path, required=True, help="JSON mapping parameter names to local paths.")
    parser.add_argument("--commands", type=Path, default=Path(__file__).resolve().parents[1] / "configs/commands.json")
    parser.add_argument("--execute", action="store_true", help="Execute the displayed training/evaluation command.")
    parser.add_argument("--inference-only", action="store_true", help="Generate answers without reading supervision.")
    parser.add_argument("--numeric-pose-mode", choices=["current", "hard_off"],
                        help="Explicit QUASAR comparison; omitted keeps the recorded original mode.")
    args = parser.parse_args()
    entries = json.loads(args.commands.read_text())["commands"]
    matches = [entry for entry in entries if entry["role"] == args.role]
    if len(matches) != 1:
        raise ValueError(f"expected one recorded command for role={args.role}, found {len(matches)}")
    entry = matches[0]
    if entry["unsupported_saved_fields"] or entry["not_expressible_saved_fields"]:
        raise ValueError("this saved configuration cannot be expressed by the packaged entry")
    parameters = json.loads(args.parameters.read_text())
    required = sorted(set(re.findall(r"\$\{([^}]+)\}", " ".join(entry["argv"]))))
    missing = [key for key in required if key not in parameters]
    if missing:
        raise KeyError(f"missing explicit path parameters: {missing}")
    # 路径替换只作用于已记录的参数字符串，不读取旧工程，也不补填历史缺项。
    argv = [re.sub(r"\$\{([^}]+)\}", lambda match: str(parameters[match[1]]), value) for value in entry["argv"]]
    if args.inference_only:
        if entry["module"] not in {"quasar.eval_native_qwen3vl2b_glow_patch_spatial_decoder_aux_generate",
                                   "quasar.eval_native_qwen3vl2b_direct_visual_generate"}:
            raise ValueError("inference-only requires a supported recorded generation entry")
        # 原配方的模型参数一项不换；仅明确要求读取器把答案与训练附加目标留在推理之外。
        argv.append("--inference-only")
    if args.numeric_pose_mode is not None:
        if not args.inference_only or entry["module"] != "quasar.eval_native_qwen3vl2b_glow_patch_spatial_decoder_aux_generate":
            raise ValueError("numeric-pose-mode requires answer-free QUASAR generation")
        from quasar.eval_native_qwen3vl2b_glow_patch_spatial_decoder_aux_generate import sha256_file
        from quasar.train_native_qwen3vl2b_glow_patch_spatial_decoder_aux import resolve_checkpoint_file
        checkpoint = resolve_checkpoint_file(Path(argv[argv.index('--checkpoint-path') + 1]))
        # 沿用原对照的权重身份校验；每个模式由使用者指定独立输出目录。
        output = Path(argv[argv.index('--output-dir') + 1])
        if output.exists():
            raise FileExistsError('Numeric-pose comparison needs a new output directory: ' + str(output))
        argv.extend(['--gate1-arm', args.numeric_pose_mode, '--gate1-reference-family', 'all',
                     '--gate1-expected-checkpoint-sha256', sha256_file(checkpoint)])
    command = [sys.executable, "-m", entry["module"], *argv]
    print(shlex.join(command), flush=True)
    if entry["historical_fields_missing"]:
        print("Fields not recorded historically: " + ", ".join(entry["historical_fields_missing"]), file=sys.stderr, flush=True)
    if args.execute:
        completed = subprocess.run(command)
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
