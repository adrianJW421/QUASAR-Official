#!/usr/bin/env python3
"""Rescore native-answer predictions with the official-like VSI-Bench protocol.

较早的 native-answer eval 用的是“逐行 mean_score”，并且把 count 当 exact。
官方 VSI-Bench 不是这样算的：选择题算 accuracy，数值题算 MRA，最后按任务做 macro average。
这个脚本只重新打分已有 predictions，不重新生成答案，也不改任何训练产物。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}

NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}

OFFICIAL_TABULATED_TASKS = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
]

VSTI_TASK_GROUPS = [
    "camera_displacement",
    "camera_movement_direction",
    "camera_object_absolute_distance",
    "camera_object_relative_distance",
    "object_object_relative_position",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rescore native predictions with official-like VSI-Bench metrics.")
    parser.add_argument("--eval-jsonl", type=Path, required=True, help="Native-answer eval JSONL used to generate predictions.")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Repeatable METHOD_NAME=/path/to/predictions.jsonl.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSONL: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line={line_no} must be a JSON object")
            yield row


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--method must be NAME=/path/to/predictions.jsonl, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"--method has empty NAME or path: {value!r}")
    return name, Path(path)


def fuzzy_matching(prediction: str) -> str:
    # 官方 VSI-Bench 就取第一个空格前的 token，再去掉末尾句点。
    return prediction.split(" ")[0].rstrip(".").strip()


def exact_match(prediction: str, target: str) -> float:
    return 1.0 if prediction.lower() == target.lower() else 0.0


def to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError(f"linspace count must be positive, got {count}")
    if count == 1:
        return [float(start)]
    step = (float(end) - float(start)) / float(count - 1)
    return [float(start) + step * index for index in range(count)]


def mean_relative_accuracy(pred: float, target: float, *, start: float = 0.5, end: float = 0.95, interval: float = 0.05) -> float:
    # 完全贴近官方实现：int((end-start)/interval + 2) 在 .5/.95/.05 下会得到 10。
    # 含义很简单：预测越接近 GT，通过的相对误差阈值越多，分数越高。
    num_points = int((end - start) / interval + 2)
    thresholds = linspace(start, end, num_points)
    relative_error = abs(float(pred) - float(target)) / float(target)
    passed = [relative_error <= 1.0 - threshold for threshold in thresholds]
    return sum(1.0 for value in passed if value) / float(len(passed))


def load_eval_rows_by_sample(eval_jsonl: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    for row in read_jsonl(eval_jsonl):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise KeyError(f"eval row missing sample_id: {row}")
        if sample_id in rows:
            duplicates[sample_id] += 1
        rows[sample_id] = row
    if duplicates:
        raise ValueError(f"eval JSONL has duplicate sample_id, first examples={duplicates.most_common(5)}")
    return rows


def official_like_score_for_vsi_prediction(prediction_row: Mapping[str, Any], eval_row: Mapping[str, Any]) -> dict[str, Any]:
    question = eval_row.get("question")
    if not isinstance(question, Mapping):
        raise KeyError(f"sample_id={prediction_row.get('sample_id')} eval row missing question payload")
    answer = eval_row.get("answer")
    if not isinstance(answer, Mapping):
        raise KeyError(f"sample_id={prediction_row.get('sample_id')} eval row missing answer payload")
    qtype = str(question.get("canonical_question_type") or prediction_row.get("canonical_question_type") or "").strip()
    if not qtype:
        raise KeyError(f"sample_id={prediction_row.get('sample_id')} missing canonical question type")
    prediction_text = str(prediction_row.get("prediction_text") or "")
    first_token = fuzzy_matching(prediction_text)

    if qtype in MCA_QUESTION_TYPES:
        gold_letter = str(answer.get("option_letter") or answer.get("raw") or "").strip()
        if not gold_letter:
            raise KeyError(f"sample_id={prediction_row.get('sample_id')} MCA row missing answer.option_letter/raw")
        score = exact_match(first_token, gold_letter)
        return {
            "question_type": qtype,
            "metric": "accuracy",
            "score": score,
            "parsed_prediction": first_token,
            "target": gold_letter,
        }

    if qtype in NA_QUESTION_TYPES:
        pred_value = to_float(first_token)
        target_value = answer.get("numeric_value")
        if target_value is None:
            target_value = answer.get("text")
        gold = to_float(target_value)
        if pred_value is None or gold is None:
            score = 0.0
        else:
            score = mean_relative_accuracy(pred_value, gold)
        return {
            "question_type": qtype,
            "metric": "MRA:.5:.95:.05",
            "score": score,
            "parsed_prediction": pred_value,
            "target": gold,
        }

    raise ValueError(f"sample_id={prediction_row.get('sample_id')} unknown official VSI qtype={qtype!r}")


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / float(len(values))


def summarize_official_vsi(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "row_weighted_mean": None,
            "official_like_macro_over_present_tasks": None,
            "missing_official_tasks": list(OFFICIAL_TABULATED_TASKS),
            "task_scores": {},
            "raw_question_type_scores": {},
        }

    by_qtype: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_qtype[str(record["question_type"])].append(float(record["score"]))

    raw_qtype_scores = {qtype: mean(scores) for qtype, scores in sorted(by_qtype.items())}
    task_scores: dict[str, float] = {}
    for qtype, value in raw_qtype_scores.items():
        if value is None:
            continue
        if qtype.startswith("object_rel_direction_"):
            continue
        task_scores[qtype] = float(value)

    direction_parts = [
        raw_qtype_scores.get("object_rel_direction_easy"),
        raw_qtype_scores.get("object_rel_direction_medium"),
        raw_qtype_scores.get("object_rel_direction_hard"),
    ]
    if all(value is not None for value in direction_parts):
        task_scores["object_rel_direction"] = sum(float(value) for value in direction_parts if value is not None) / 3.0

    present_official_tasks = [task for task in OFFICIAL_TABULATED_TASKS if task in task_scores]
    missing_official_tasks = [task for task in OFFICIAL_TABULATED_TASKS if task not in task_scores]
    macro = mean([task_scores[task] for task in present_official_tasks])
    return {
        "count": len(records),
        "row_weighted_mean": mean([float(record["score"]) for record in records]),
        "official_like_macro_over_present_tasks": macro,
        "present_official_tasks": present_official_tasks,
        "missing_official_tasks": missing_official_tasks,
        "task_scores": {task: task_scores[task] for task in present_official_tasks},
        "raw_question_type_scores": raw_qtype_scores,
        "question_type_counts": dict(Counter(str(record["question_type"]) for record in records)),
    }


def summarize_existing_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "mean_score": None}
    scores = [float(row["score"]) for row in rows]
    return {
        "count": len(rows),
        "mean_score": mean(scores),
        "question_type_scores": {
            qtype: mean([float(row["score"]) for row in group])
            for qtype, group in sorted(group_by_question_type(rows).items())
        },
    }


def official_like_score_for_vsti_prediction(prediction_row: Mapping[str, Any]) -> dict[str, Any]:
    qtype = str(prediction_row.get("canonical_question_type") or "").strip()
    group = str(prediction_row.get("vsti_task_group") or "").strip() or vsti_task_group_from_qtype(qtype)
    answer_type = str(prediction_row.get("answer_type") or "").strip()
    prediction_text = str(prediction_row.get("prediction_text") or "")

    if answer_type == "numeric" or qtype in {"vsti_camera_displacement", "vsti_camera_obj_abs_dist"}:
        pred_value = to_float(fuzzy_matching(prediction_text))
        if pred_value is None:
            pred_value = to_float(prediction_row.get("parsed"))
        target_value = prediction_row.get("target_numeric")
        if target_value is None:
            target_value = prediction_row.get("target_text")
        gold = to_float(target_value)
        score = 0.0 if pred_value is None or gold is None else mean_relative_accuracy(pred_value, gold)
        return {
            "question_type": qtype,
            "task_group": group,
            "metric": "MRA:.5:.95:.05",
            "score": score,
            "parsed_prediction": pred_value,
            "target": gold,
        }

    return {
        "question_type": qtype,
        "task_group": group,
        "metric": str(prediction_row.get("metric") or "accuracy"),
        "score": float(prediction_row.get("score", 0.0)),
        "parsed_prediction": prediction_row.get("parsed"),
        "target": prediction_row.get("target_text"),
    }


def summarize_vsti_task_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按 VSTI 常用 5 个 task group 做 macro。

    VSTI 数值题必须按官方 NA MRA 现场重算，不能复用 native predictions
    里的 strict tolerance score；MCA 题继续使用逐题 accuracy。
    """

    if not rows:
        return {
            "count": 0,
            "row_weighted_mean": None,
            "task_group_macro": None,
            "missing_task_groups": list(VSTI_TASK_GROUPS),
            "task_group_scores": {},
        }
    scored_rows = [official_like_score_for_vsti_prediction(row) for row in rows]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in scored_rows:
        group = str(row.get("task_group") or "").strip()
        if group:
            grouped[group].append(float(row["score"]))
    task_group_scores = {group: mean(scores) for group, scores in sorted(grouped.items())}
    present = [group for group in VSTI_TASK_GROUPS if task_group_scores.get(group) is not None]
    missing = [group for group in VSTI_TASK_GROUPS if task_group_scores.get(group) is None]
    return {
        "count": len(rows),
        "row_weighted_mean": mean([float(row["score"]) for row in scored_rows]),
        "task_group_macro": mean([float(task_group_scores[group]) for group in present]),
        "present_task_groups": present,
        "missing_task_groups": missing,
        "task_group_scores": {group: task_group_scores[group] for group in present},
        "task_group_counts": {group: len(grouped[group]) for group in sorted(grouped)},
    }


def summarize_official_like_vsi_from_prediction_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    eval_rows_by_sample: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score in-memory prediction records with the official-like VSI protocol.

    This is the same VSI scorer used by the CLI path, but it avoids a
    predictions.jsonl round trip so generation evaluators can write the
    official-like summary into eval_summary.json before exiting.
    """

    official_records: list[dict[str, Any]] = []
    missing_eval_rows: list[str] = []
    for prediction_row in prediction_rows:
        if str(prediction_row.get("bench") or "") != "VSI":
            continue
        sample_id = str(prediction_row.get("sample_id") or "").strip()
        eval_row = eval_rows_by_sample.get(sample_id)
        if eval_row is None:
            missing_eval_rows.append(sample_id)
            continue
        scored = official_like_score_for_vsi_prediction(prediction_row, eval_row)
        scored["sample_id"] = sample_id
        official_records.append(scored)
    if missing_eval_rows:
        raise KeyError(f"VSI predictions missing from eval rows: first={missing_eval_rows[:5]}")
    return summarize_official_vsi(official_records)


def vsti_task_group_from_qtype(qtype: str) -> str:
    if qtype == "vsti_camera_displacement":
        return "camera_displacement"
    if qtype == "vsti_camera_movement_direction":
        return "camera_movement_direction"
    if qtype == "vsti_camera_obj_abs_dist":
        return "camera_object_absolute_distance"
    if qtype.startswith("vsti_camera_obj_rel_dist"):
        return "camera_object_relative_distance"
    if qtype.startswith("vsti_obj_obj_relative_pos"):
        return "object_object_relative_position"
    return ""


def group_by_question_type(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("canonical_question_type") or "")].append(row)
    return grouped


def score_method(*, name: str, predictions_path: Path, eval_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    prediction_rows = list(read_jsonl(predictions_path))
    if not prediction_rows:
        raise ValueError(f"prediction file is empty: {predictions_path}")

    vsi_predictions = [row for row in prediction_rows if str(row.get("bench") or "") == "VSI"]
    vsti_predictions = [row for row in prediction_rows if str(row.get("bench") or "") == "VSTI"]
    official_records: list[dict[str, Any]] = []
    missing_eval_rows: list[str] = []
    for prediction_row in vsi_predictions:
        sample_id = str(prediction_row.get("sample_id") or "").strip()
        eval_row = eval_rows.get(sample_id)
        if eval_row is None:
            missing_eval_rows.append(sample_id)
            continue
        scored = official_like_score_for_vsi_prediction(prediction_row, eval_row)
        scored["sample_id"] = sample_id
        official_records.append(scored)
    if missing_eval_rows:
        raise KeyError(f"{name} has predictions missing from eval JSONL: first={missing_eval_rows[:5]}")

    official_vsi = summarize_official_vsi(official_records)
    native_vsi = summarize_existing_scores(vsi_predictions)
    native_vsti = summarize_existing_scores(vsti_predictions)
    vsti_task_group = summarize_vsti_task_groups(vsti_predictions)
    native_all = summarize_existing_scores(prediction_rows)
    return {
        "method": name,
        "predictions_path": str(predictions_path),
        "prediction_rows": len(prediction_rows),
        "source_counts": dict(Counter(str(row.get("source_tag") or "") for row in prediction_rows)),
        "native_existing": {
            "overall": native_all,
            "vsi": native_vsi,
            "vsti": native_vsti,
        },
        "official_like_vsi": official_vsi,
        "vsti_task_group": vsti_task_group,
    }


def write_csv(path: Path, method_summaries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "native_overall",
        "native_vsi_row_mean",
        "official_like_vsi_row_mean",
        "official_like_vsi_macro_present",
        "native_vsti_row_mean",
        "vsti_task_group_macro",
        "vsi_count",
        "missing_official_tasks",
        "missing_vsti_task_groups",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in method_summaries:
            native = summary["native_existing"]
            official = summary["official_like_vsi"]
            vsti = summary["vsti_task_group"]
            writer.writerow(
                {
                    "method": summary["method"],
                    "native_overall": native["overall"]["mean_score"],
                    "native_vsi_row_mean": native["vsi"]["mean_score"],
                    "official_like_vsi_row_mean": official["row_weighted_mean"],
                    "official_like_vsi_macro_present": official["official_like_macro_over_present_tasks"],
                    "native_vsti_row_mean": native["vsti"]["mean_score"],
                    "vsti_task_group_macro": vsti["task_group_macro"],
                    "vsi_count": official["count"],
                    "missing_official_tasks": ",".join(official.get("missing_official_tasks") or []),
                    "missing_vsti_task_groups": ",".join(vsti.get("missing_task_groups") or []),
                }
            )


def pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.2f}"


def write_markdown(path: Path, method_summaries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Official-like VSIBench Rescore",
        "",
        "说明：`official-like VSI macro` 使用官方 VSIBench 的 MCA accuracy、NA MRA 和任务 macro；当前 subset 如果缺官方任务，会只对 present tasks 求 macro，并在 `missing` 列标出。",
        "",
        "`VSTI task-group macro` 对数值题现场重算 MRA、选择题使用 accuracy，再按 VSTI 五大 task group 平均；它用于检查 row mean 和天梯式 group mean 是否差很多。",
        "",
        "| method | native overall | native VSI row | official-like VSI row | official-like VSI macro | native VSTI row | VSTI group macro | missing |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in method_summaries:
        native = summary["native_existing"]
        official = summary["official_like_vsi"]
        vsti = summary["vsti_task_group"]
        lines.append(
            "| {method} | {native_all} | {native_vsi} | {official_row} | {official_macro} | {native_vsti} | {vsti_macro} | {missing} |".format(
                method=summary["method"],
                native_all=pct(native["overall"]["mean_score"]),
                native_vsi=pct(native["vsi"]["mean_score"]),
                official_row=pct(official["row_weighted_mean"]),
                official_macro=pct(official["official_like_macro_over_present_tasks"]),
                native_vsti=pct(native["vsti"]["mean_score"]),
                vsti_macro=pct(vsti["task_group_macro"]),
                missing=", ".join(official.get("missing_official_tasks") or []),
            )
        )

    lines.extend(["", "## Per-task official-like VSI scores", ""])
    for summary in method_summaries:
        official = summary["official_like_vsi"]
        lines.append(f"### {summary['method']}")
        lines.append("")
        lines.append("| task | score |")
        lines.append("|---|---:|")
        for task, score in official.get("task_scores", {}).items():
            lines.append(f"| {task} | {pct(score)} |")
        lines.append("")

    lines.extend(["", "## VSTI task-group scores", ""])
    for summary in method_summaries:
        vsti = summary["vsti_task_group"]
        lines.append(f"### {summary['method']}")
        lines.append("")
        lines.append("| task group | score |")
        lines.append("|---|---:|")
        for group, score in vsti.get("task_group_scores", {}).items():
            lines.append(f"| {group} | {pct(score)} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    eval_rows = load_eval_rows_by_sample(args.eval_jsonl)
    method_summaries = []
    for method_arg in args.method:
        name, predictions_path = parse_method(method_arg)
        method_summaries.append(score_method(name=name, predictions_path=predictions_path, eval_rows=eval_rows))

    payload = {
        "format_version": "glow_vsibench_official_like_rescore_v1",
        "eval_jsonl": str(args.eval_jsonl),
        "method_count": len(method_summaries),
        "methods": method_summaries,
        "protocol_notes": {
            "mca": "fuzzy_matching(prediction).lower() == ground_truth_letter.lower()",
            "na": "Mean Relative Accuracy over thresholds .5 to .95, matching official VSI-Bench code.",
            "overall": "Macro average over present official VSI tasks after merging direction easy/medium/hard.",
        },
    }
    write_json(args.output_json, payload)
    if args.output_csv is not None:
        write_csv(args.output_csv, method_summaries)
    if args.output_md is not None:
        write_markdown(args.output_md, method_summaries)


if __name__ == "__main__":
    main()
