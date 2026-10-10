#!/usr/bin/env python3
"""Small helpers for paper-facing per-sample generation latency benchmarks.

旧的 cost table 统计了整条评测流水线的 wall-clock 时间，里面混了数据读取、
merge、lmms replay 和 IO。论文真正需要的是“模型回答一道题的 generate 有多慢”。
这里的工具只服务这个目的：在 batch 已经搬到 GPU 后，围绕 generate 调 CUDA 同步计时。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import torch


def add_latency_args(parser: argparse.ArgumentParser) -> None:
    """给现有 generate eval 脚本增加可选的 latency 输出参数。

    默认不启用，所以不会改变原来的 full eval 行为；只有显式传
    --latency-timed-rows 时，脚本才会截取 warmup+timed 条样本并写 latency 文件。
    """

    parser.add_argument("--latency-output-json", type=Path, default=None)
    parser.add_argument("--latency-output-jsonl", type=Path, default=None)
    parser.add_argument("--latency-warmup-rows", type=int, default=0)
    parser.add_argument("--latency-timed-rows", type=int, default=0)
    parser.add_argument("--latency-method-name", default="")
    parser.add_argument("--latency-benchmark-name", default="")


def latency_enabled(args: Any) -> bool:
    return int(getattr(args, "latency_timed_rows", 0)) > 0


def validate_latency_args(args: Any) -> None:
    if int(getattr(args, "latency_warmup_rows", 0)) < 0:
        raise ValueError("--latency-warmup-rows must be >= 0")
    if int(getattr(args, "latency_timed_rows", 0)) < 0:
        raise ValueError("--latency-timed-rows must be >= 0")
    if latency_enabled(args) and getattr(args, "latency_output_json", None) is None:
        raise ValueError("--latency-output-json is required when --latency-timed-rows > 0")


def slice_rows_for_latency(rows: Sequence[Mapping[str, Any]], args: Any) -> list[Mapping[str, Any]]:
    """按 question type 做确定性均衡抽样，只保留 warmup+timed 条样本。

    VSI/VSTI 的 JSONL 经常按题型排序，直接取前 110 条会只测到少数题型，
    延迟会被生成长度/题型分布带偏。这里不看答案、不看模型输出，只按题型轮流取样，
    让 100 条 timed samples 覆盖所有题型。
    """

    if not latency_enabled(args):
        return list(rows)
    need = int(args.latency_warmup_rows) + int(args.latency_timed_rows)
    if len(rows) < need:
        raise ValueError(f"latency benchmark needs {need} rows, but only got {len(rows)}")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        question = row.get("question")
        qtype = ""
        if isinstance(question, Mapping):
            qtype = str(question.get("canonical_question_type") or "").strip()
        if not qtype:
            qtype = str(row.get("canonical_question_type") or "").strip()
        if not qtype:
            raise KeyError("latency row missing question.canonical_question_type")
        groups[qtype].append(row)

    for qtype in groups:
        groups[qtype].sort(key=lambda item: str(item.get("sample_id") or ""))
    qtypes = sorted(groups)
    cursors = {qtype: 0 for qtype in qtypes}
    selected: list[Mapping[str, Any]] = []
    while len(selected) < need:
        made_progress = False
        for qtype in qtypes:
            cursor = cursors[qtype]
            if cursor < len(groups[qtype]):
                selected.append(groups[qtype][cursor])
                cursors[qtype] = cursor + 1
                made_progress = True
                if len(selected) == need:
                    break
        if not made_progress:
            raise ValueError(f"could not collect {need} latency rows from {len(rows)} rows")
    return selected


def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_payload(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "peak_allocated_gb": None,
            "peak_reserved_gb": None,
        }
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "device": str(device),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "peak_allocated_gb": allocated / 1024.0**3,
        "peak_reserved_gb": reserved / 1024.0**3,
    }


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        raise ValueError("cannot compute percentile for empty latency values")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def summarize_latency_records(
    *,
    records: Sequence[Mapping[str, Any]],
    args: Any,
    device: torch.device,
    total_elapsed_sec: float,
) -> dict[str, Any]:
    """把逐样本耗时变成论文表能直接使用的 mean / median / p90 / p95。

    tokens/s 用“计时样本生成 token 总数 / generate 总耗时”计算，避免每条样本 token 很少时
    单样本 tokens/s 过度抖动。
    """

    if len(records) != int(args.latency_timed_rows):
        raise ValueError(f"expected {args.latency_timed_rows} timed rows, got {len(records)}")
    latencies = [float(row["latency_sec"]) for row in records]
    tokens = [int(row["generated_token_count"]) for row in records]
    total_latency = sum(latencies)
    total_tokens = sum(tokens)
    return {
        "status": "completed",
        "method": str(getattr(args, "latency_method_name", "") or ""),
        "benchmark": str(getattr(args, "latency_benchmark_name", "") or ""),
        "warmup_rows": int(args.latency_warmup_rows),
        "timed_rows": int(args.latency_timed_rows),
        "total_elapsed_sec_including_load_and_warmup": float(total_elapsed_sec),
        "generate_latency_sec": {
            "mean": statistics.fmean(latencies),
            "std": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
            "median": statistics.median(latencies),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
            "sum": total_latency,
        },
        "generated_tokens": {
            "total": total_tokens,
            "mean_per_sample": statistics.fmean(tokens),
            "median_per_sample": statistics.median(tokens),
        },
        "tokens_per_sec": (total_tokens / total_latency) if total_latency > 0.0 else None,
        "peak_memory": peak_memory_payload(device),
        "notes": {
            "sample_selection": "deterministic round-robin over canonical_question_type before warmup/timed split",
            "timed_region": "batch already collated and moved to device; timer wraps only generate_one(...) with CUDA synchronize before and after",
            "excluded": "model loading, dataset filtering, DataLoader/collate IO, scoring, JSON writing, lmms replay, prediction merge",
        },
    }


def write_latency_outputs(
    *,
    args: Any,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    json_path = Path(args.latency_output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl_path = getattr(args, "latency_output_jsonl", None)
    if jsonl_path is not None:
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def timed_generate_call(device: torch.device, call):
    """同步计时一次 generate 调用，返回 `(elapsed_sec, result)`。

    这里接收 callable 是为了让 direct 和 DecoderAux 两个脚本共用同一套计时代码；
    真正的 generate 参数仍然写在各自脚本里，便于看清楚模型路径。
    """

    cuda_sync_if_needed(device)
    start = time.perf_counter()
    result = call()
    cuda_sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return elapsed, result
