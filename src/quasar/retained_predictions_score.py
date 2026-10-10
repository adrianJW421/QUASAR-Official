"""Check saved answer files against gold labels supplied by the user; no model is run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping, Sequence

from quasar.score_vsibench_official_like_predictions import (
    load_eval_rows_by_sample, official_like_score_for_vsi_prediction,
    official_like_score_for_vsti_prediction, read_jsonl, summarize_official_vsi,
    vsti_task_group_from_qtype,
)
from quasar.eval_native_qwen3vl2b_glow_patch_spatial_generate import score_native_answer


def rescore_row(prediction: Mapping, gold: Mapping, benchmark: str) -> dict:
    """把外部原题的答案交给已有评分函数，避免把保存的旧分数当作重算结果。"""
    if str(prediction["sample_id"]) != str(gold["sample_id"]):
        raise ValueError("prediction and evaluation sample IDs differ")
    if benchmark == "VSI":
        scored = official_like_score_for_vsi_prediction(prediction, gold)
    elif benchmark == "VSTI":
        native = score_native_answer(prediction=str(prediction["prediction_text"]), row=gold)
        scored = official_like_score_for_vsti_prediction({
            **prediction, **native,
            "canonical_question_type": gold["question"]["canonical_question_type"],
            "answer_type": gold["answer"]["type"], "target_text": gold["answer"]["text"],
            "target_numeric": gold["answer"].get("numeric_value"),
        })
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    return {"sample_id": str(prediction["sample_id"]), "prediction_text": str(prediction["prediction_text"]),
            "canonical_question_type": scored["question_type"], "score": float(scored["score"]),
            "metric": scored["metric"]}


def aggregate(records: Sequence[Mapping], protocol: str) -> dict:
    """按指定口径汇总逐题分数；五个任务、九个子类和逐题准确率分别计算。"""
    if not records:
        raise ValueError("cannot aggregate an empty answer file")
    ids = [str(row["sample_id"]) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs in retained answers")
    if protocol == "vsi_8task":
        summary = summarize_official_vsi([{"question_type": r["canonical_question_type"], "score": r["score"]} for r in records])
        value = summary["official_like_macro_over_present_tasks"]
    elif protocol in ["vsti_5task", "vsti_9subtype"]:
        grouped = defaultdict(list)
        for row in records:
            qtype = str(row["canonical_question_type"])
            group = vsti_task_group_from_qtype(qtype) if protocol == "vsti_5task" else qtype
            if not group:
                raise ValueError(f"unknown question type: {qtype}")
            grouped[group].append(float(row["score"]))
        means = {key: sum(values) / len(values) for key, values in sorted(grouped.items())}
        value = sum(means.values()) / len(means)
        summary = {"group_scores": means, "group_counts": {k: len(v) for k, v in sorted(grouped.items())}}
    elif protocol == "exact_micro":
        if any(float(row["score"]) not in [0., 1.] for row in records):
            raise ValueError("exact_micro requires binary correctness scores")
        value = sum(float(row["score"]) for row in records) / len(records)
        summary = {"correct": sum(int(row["score"]) for row in records)}
    else:
        raise ValueError(f"unknown aggregation protocol: {protocol}")
    return {"count": len(records), "protocol": protocol, "score_fraction": value,
            "score_percent": value * 100., "details": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--protocol", choices=["vsi_8task", "vsti_5task", "vsti_9subtype", "exact_micro"], required=True)
    parser.add_argument("--eval-jsonl", type=Path, help="External original questions and answers, for rescoring.")
    parser.add_argument("--benchmark", choices=["VSI", "VSTI"])
    parser.add_argument("--sample-ids", type=Path, help="Explicit one-ID-per-line subset, e.g. the revised 5292 intersection.")
    parser.add_argument("--aggregate-retained-scores", action="store_true", help="Only recompute the aggregate of retained per-item scores.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = list(read_jsonl(args.predictions))
    if args.sample_ids is not None:
        requested = {line.strip() for line in args.sample_ids.read_text().splitlines() if line.strip()}
        predictions = [row for row in predictions if str(row["sample_id"]) in requested]
        if {str(row["sample_id"]) for row in predictions} != requested:
            raise ValueError("requested sample IDs are missing from the retained predictions")
    if args.aggregate_retained_scores:
        if args.eval_jsonl is not None or args.benchmark is not None:
            parser.error("retained-score aggregation does not take evaluation data")
        rows = predictions
        verification = "aggregate_of_retained_scores"
    else:
        if args.eval_jsonl is None or args.benchmark is None:
            parser.error("rescoring requires --eval-jsonl and --benchmark")
        gold = load_eval_rows_by_sample(args.eval_jsonl)
        rows = [rescore_row(row, gold[str(row["sample_id"])], args.benchmark) for row in predictions]
        verification = "rescored_against_external_answers"
    result = {**aggregate(rows, args.protocol), "verification": verification}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
