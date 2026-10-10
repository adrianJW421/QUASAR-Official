"""Score all numerical-reference control conditions independently; question IDs, scoring protocol and per-scene sampling stay fixed."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
from types import ModuleType
from typing import Any, Mapping, Sequence
from quasar import score_vsibench_official_like_predictions as scorer


BOOTSTRAP_REPETITIONS = 5000

BOOTSTRAP_SEED = 20260812

EXPECTED_COUNTS = {"vsi": 5130, "vsti": 6042}

EXPECTED_SUBGROUPS = {
    "vsi": {"semantic_1162": 1162, "rest_3968": 3968},
    "vsti": {"legacy_5598": 5598, "rest_444": 444},
}

EXPECTED_BENCH = {"vsi": "VSI", "vsti": "VSTI"}

EXPECTED_QUERY_POSE_MODE = {"vsi": "legacy", "vsti": "provided_full"}

VSI_TASKS = (
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
)

VSTI_TASKS = (
    "camera_displacement",
    "camera_movement_direction",
    "camera_object_absolute_distance",
    "camera_object_relative_distance",
    "object_object_relative_position",
)

REQUIRED_RAW_FIELDS = (
    "sample_id",
    "bench",
    "source_tag",
    "canonical_question_type",
    "prediction_text",
    "prediction_text_raw",
    "answer_type",
    "query_pose_protocol",
    "gate1",
    "intervention_audit",
)

FORBIDDEN_RAW_FIELDS = {
    "answer",
    "correct",
    "gold",
    "ground_truth",
    "metric",
    "parsed",
    "score",
    "target",
    "target_numeric",
}

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)

RESULT_FIELDS = (
    "dataset",
    "arm",
    "group_kind",
    "group",
    "count",
    "primary_metric",
    "primary_score",
    "row_weighted_mean",
    "present_tasks",
    "missing_tasks",
    "delta_hard_off_minus_current_pp",
    "ci95_low_pp",
    "ci95_high_pp",
    "ci_status",
)

def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
    temporary.replace(path)

def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a mean from zero values")
    return sum(float(value) for value in values) / float(len(values))

def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile from zero values")
    location = (len(ordered) - 1) * float(quantile)
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def _source_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = row.get("source")
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(source, Mapping) or not isinstance(question, Mapping) or not isinstance(answer, Mapping):
        raise KeyError(f"sample_id={row.get('sample_id')} source row needs source/question/answer mappings")
    source_tag = str(source.get("tag") or "").strip()
    qtype = str(question.get("canonical_question_type") or "").strip()
    answer_type = str(answer.get("type") or "").strip()
    if not source_tag or not qtype or not answer_type:
        raise ValueError(f"sample_id={row.get('sample_id')} source identity is incomplete")
    return source_tag, qtype, answer_type

def validate_prediction_ids(actual_or_rows: Sequence[Any], expected_ids: Sequence[str], label: str) -> None:
    actual_ids = [
        str(item.get("sample_id") or "").strip() if isinstance(item, Mapping) else str(item).strip()
        for item in actual_or_rows
    ]
    if any(not sample_id for sample_id in actual_ids):
        raise ValueError(f"{label} contains an empty sample_id")
    duplicates = [sample_id for sample_id, count in Counter(actual_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} has duplicate sample IDs: {duplicates[:5]}")
    actual_set = set(actual_ids)
    expected_set = set(expected_ids)
    missing = [sample_id for sample_id in expected_ids if sample_id not in actual_set]
    extra = [sample_id for sample_id in actual_ids if sample_id not in expected_set]
    if missing or extra:
        raise ValueError(f"{label} ID set mismatch: missing={missing[:5]} extra={extra[:5]}")
    if actual_ids != list(expected_ids):
        raise ValueError(f"{label} ID order does not match the frozen manifest")

def _strict_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed

def validate_raw_prediction(
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    dataset: str,
    arm: str,
    query_pose_mode: str | None = None,
) -> dict[str, Any]:
    # 原始答卷是生成阶段的封存边界；这里先拒绝标签和分数，再允许读取 source 做独立计分。
    sample_id = str(row.get("sample_id") or "").strip()
    missing_fields = [field for field in REQUIRED_RAW_FIELDS if field not in row]
    if missing_fields:
        raise KeyError(f"sample_id={sample_id} raw prediction missing fields: {missing_fields}")
    leaked = sorted(field for field in FORBIDDEN_RAW_FIELDS if field in row)
    # 原生成器保留这个空字段作为输出格式；只有严格空串可接受，不能夹带标签。
    if "target_text" in row and row["target_text"] != "":
        leaked.append("target_text")
    if leaked:
        raise ValueError(f"sample_id={sample_id} raw prediction leaks labels/scores: {leaked}")
    source_tag, qtype, answer_type = _source_identity(source_row)
    expected_values = {
        "bench": EXPECTED_BENCH[dataset],
        "source_tag": source_tag,
        "canonical_question_type": qtype,
        "answer_type": answer_type,
    }
    for field, expected in expected_values.items():
        if str(row.get(field) or "") != expected:
            raise ValueError(
                f"sample_id={sample_id} {field} mismatch: raw={row.get(field)!r} source={expected!r}"
            )
    if not isinstance(row.get("prediction_text"), str) or not isinstance(row.get("prediction_text_raw"), str):
        raise TypeError(f"sample_id={sample_id} prediction text fields must be strings")

    protocol = row["query_pose_protocol"]
    gate1 = row["gate1"]
    audit = row["intervention_audit"]
    if not isinstance(protocol, Mapping) or not isinstance(gate1, Mapping) or not isinstance(audit, Mapping):
        raise TypeError(f"sample_id={sample_id} protocol/gate1/intervention_audit must be mappings")
    if "effective_pose_input_mask" not in protocol or not isinstance(protocol["effective_pose_input_mask"], bool):
        raise KeyError(f"sample_id={sample_id} missing boolean effective_pose_input_mask")
    if str(protocol.get("mode") or "") != (query_pose_mode or EXPECTED_QUERY_POSE_MODE[dataset]):
        raise ValueError(f"sample_id={sample_id} query pose mode mismatch")
    gate_required = (
        "arm",
        "effective_reference_type_id",
        "query_pose_vec12_all_zero",
        "descriptor_query_channels_all_zero",
    )
    gate_missing = [field for field in gate_required if field not in gate1]
    if gate_missing:
        raise KeyError(f"sample_id={sample_id} gate1 missing fields: {gate_missing}")
    if str(gate1.get("arm")) != arm:
        raise ValueError(f"sample_id={sample_id} gate1 arm mismatch")
    if isinstance(gate1["effective_reference_type_id"], bool) or not isinstance(
        gate1["effective_reference_type_id"], int
    ):
        raise TypeError(f"sample_id={sample_id} effective_reference_type_id must be an integer")
    for field in ("query_pose_vec12_all_zero", "descriptor_query_channels_all_zero"):
        if not isinstance(gate1[field], bool):
            raise TypeError(f"sample_id={sample_id} gate1.{field} must be boolean")

    audit_required = (
        "pose_projection_after_mask_max_abs",
        "visual_writeback_max_abs",
        "deepstack_layers",
        "spatial_query_count",
    )
    audit_missing = [field for field in audit_required if field not in audit]
    if audit_missing:
        raise KeyError(f"sample_id={sample_id} intervention_audit missing fields: {audit_missing}")
    projection = _strict_float(audit["pose_projection_after_mask_max_abs"], "pose projection audit")
    writeback = _strict_float(audit["visual_writeback_max_abs"], "visual writeback audit")
    deepstack_layers = audit["deepstack_layers"]
    spatial_query_count = audit["spatial_query_count"]
    if writeback <= 0.0:
        raise ValueError(f"sample_id={sample_id} visual writeback is not positive")
    if isinstance(deepstack_layers, bool) or not isinstance(deepstack_layers, int) or deepstack_layers <= 0:
        raise ValueError(f"sample_id={sample_id} deepstack_layers must be a positive integer")
    if isinstance(spatial_query_count, bool) or not isinstance(spatial_query_count, int) or spatial_query_count != 4:
        raise ValueError(f"sample_id={sample_id} spatial_query_count must equal 4")
    if arm == "hard_off":
        if protocol["effective_pose_input_mask"] is not False:
            raise ValueError(f"sample_id={sample_id} hard_off mask is not zero/false")
        if gate1["query_pose_vec12_all_zero"] is not True:
            raise ValueError(f"sample_id={sample_id} hard_off query pose vector is not all zero")
        if gate1["descriptor_query_channels_all_zero"] is not True:
            raise ValueError(f"sample_id={sample_id} hard_off descriptor query channels are not all zero")
        if projection != 0.0:
            raise ValueError(f"sample_id={sample_id} hard_off pose projection is not exactly zero")
    return {
        "projection": projection,
        "writeback": writeback,
        "deepstack_layers": deepstack_layers,
        "spatial_query_count": spatial_query_count,
        "effective_pose_input_mask": protocol["effective_pose_input_mask"],
        "query_pose_vec12_all_zero": gate1["query_pose_vec12_all_zero"],
        "descriptor_query_channels_all_zero": gate1["descriptor_query_channels_all_zero"],
        "effective_reference_type_id": gate1["effective_reference_type_id"],
    }

def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\.\-\+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _parse_first_float(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text.replace(",", ""))
    return None if match is None else float(match.group(0))

def _parse_choice_prediction(prediction: str, answer: Mapping[str, Any]) -> str | None:
    options = answer.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError("multiple_choice answer requires non-empty options")
    allowed = {str(option["letter"]).strip().upper(): str(option["text"]).strip() for option in options}
    match = re.match(r"^\s*([A-Z])(?:[\.|\)|:：]|\s|$)", prediction.strip(), flags=re.IGNORECASE)
    if match is not None and match.group(1).upper() in allowed:
        return match.group(1).upper()
    normalized_prediction = _normalize_text(prediction)
    for letter, text in allowed.items():
        normalized_option = _normalize_text(text)
        if normalized_option and (
            normalized_option == normalized_prediction or normalized_option in normalized_prediction
        ):
            return letter
    for token in re.findall(r"\b[A-Z]\b", prediction.upper()):
        if token in allowed:
            return token
    return None

def _vsti_native_payload(prediction: str, source_row: Mapping[str, Any]) -> dict[str, Any]:
    answer = source_row["answer"]
    answer_type = str(answer["type"])
    eval_payload = source_row.get("eval") if isinstance(source_row.get("eval"), Mapping) else {}
    if answer_type == "multiple_choice":
        parsed = _parse_choice_prediction(prediction, answer)
        target = str(answer.get("option_letter") or "").strip().upper()
        if not target:
            raise KeyError(f"sample_id={source_row.get('sample_id')} multiple choice label is missing")
        return {
            "metric": str(eval_payload.get("metric") or "choice_accuracy"),
            "score": 1.0 if parsed == target else 0.0,
            "parsed": parsed,
            "target_text": target,
        }
    if answer_type == "numeric":
        target = answer.get("numeric_value")
        if target is None:
            target = _parse_first_float(str(answer.get("text") or ""))
        if target is None:
            raise KeyError(f"sample_id={source_row.get('sample_id')} numeric label is missing")
        return {
            "metric": "MRA:.5:.95:.05",
            "score": 0.0,
            "parsed": _parse_first_float(prediction),
            "target_numeric": float(target),
            "target_text": str(answer.get("text") or ""),
        }
    raise ValueError(
        f"sample_id={source_row.get('sample_id')} unsupported historical VSTI answer type={answer_type!r}"
    )

def _vsi_task(qtype: str) -> str:
    return "object_rel_direction" if qtype.startswith("object_rel_direction_") else qtype

def _scene_cluster(dataset: str, source_row: Mapping[str, Any], raw_row: Mapping[str, Any]) -> str:
    if dataset == "vsi":
        scene = source_row.get("scene")
        cluster = str(scene.get("video_id") or scene.get("scene_id") or "").strip() if isinstance(scene, Mapping) else ""
    else:
        protocol = raw_row["query_pose_protocol"]
        cluster = str(protocol.get("geometry_gauge") or "").strip()
    if not cluster:
        raise ValueError(f"sample_id={raw_row.get('sample_id')} is missing its scene cluster")
    return cluster

def score_one(
    raw_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    dataset: str,
    arm: str,
    subgroup: str,
    preflight: bool,
    scorer: ModuleType,
) -> dict[str, Any]:
    # 标签只从按 sample_id 连接的冻结 source 进入，不能从原始 prediction 记录回收旧答案。
    if dataset == "vsi":
        payload = scorer.official_like_score_for_vsi_prediction(raw_row, source_row)
        qtype = str(payload["question_type"])
        task_group = _vsi_task(qtype)
    else:
        enriched = dict(raw_row)
        enriched.update(_vsti_native_payload(str(raw_row["prediction_text"]), source_row))
        payload = scorer.official_like_score_for_vsti_prediction(enriched)
        qtype = str(payload["question_type"])
        task_group = str(payload["task_group"])
    return {
        "sample_id": str(raw_row["sample_id"]),
        "dataset": dataset,
        "arm": arm,
        "bench": str(raw_row["bench"]),
        "source_tag": str(raw_row["source_tag"]),
        "canonical_question_type": qtype,
        "task_group": task_group,
        "subgroup": subgroup,
        "is_preflight": bool(preflight),
        "scene_cluster": _scene_cluster(dataset, source_row, raw_row),
        "answer_type": str(raw_row["answer_type"]),
        "prediction_text": str(raw_row["prediction_text"]),
        "prediction_text_raw": str(raw_row["prediction_text_raw"]),
        "metric": str(payload["metric"]),
        "score": float(payload["score"]),
        "parsed_prediction": payload.get("parsed_prediction"),
        "target": payload.get("target"),
    }

def aggregate_rows(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    scorer: ModuleType,
    *,
    group_kind: str,
    group: str,
    enforce_full_main: bool = True,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"cannot aggregate empty {dataset} group {group_kind}:{group}")
    qtype_buckets: dict[str, list[float]] = defaultdict(list)
    task_buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        qtype_buckets[str(row["canonical_question_type"])].append(float(row["score"]))
        task_buckets[str(row["task_group"])].append(float(row["score"]))
    qtype_scores = {name: mean(values) for name, values in sorted(qtype_buckets.items())}
    task_scores = {name: mean(values) for name, values in sorted(task_buckets.items())}
    task_order = VSI_TASKS if dataset == "vsi" else VSTI_TASKS
    present_tasks = [name for name in task_order if name in task_scores]
    missing_tasks = [name for name in task_order if name not in task_scores]
    row_mean = mean([float(row["score"]) for row in rows])

    if group_kind == "question_type":
        primary = row_mean
        definition = "canonical-question-type row mean"
    elif dataset == "vsi":
        historical = scorer.summarize_official_vsi(
            [
                {"question_type": str(row["canonical_question_type"]), "score": float(row["score"])}
                for row in rows
            ]
        )
        primary = historical["official_like_macro_over_present_tasks"]
        task_scores = dict(historical["task_scores"])
        present_tasks = list(historical.get("present_official_tasks") or ())
        missing_tasks = list(historical.get("missing_official_tasks") or ())
        if group_kind == "all":
            definition = "VSI eight-task macro"
        elif group_kind == "task_group":
            definition = "VSI task-group score"
        else:
            definition = "VSI macro over task groups present in this subset"
    else:
        primary = mean([task_scores[name] for name in present_tasks]) if present_tasks else None
        if group_kind == "all":
            definition = "VSTI five-task macro"
        elif group_kind == "task_group":
            definition = "VSTI task-group score"
        else:
            definition = "VSTI macro over task groups present in this subset"
    if enforce_full_main and group_kind == "all" and set(present_tasks) != set(task_order):
        raise ValueError(f"{dataset} full result is missing main task groups: {missing_tasks}")
    return {
        "group_kind": group_kind,
        "group": group,
        "count": len(rows),
        "primary_metric": definition,
        "primary_score": None if primary is None else float(primary),
        "row_weighted_mean": row_mean,
        "present_tasks": present_tasks,
        "missing_tasks": missing_tasks,
        "task_scores": task_scores,
        "task_counts": {name: len(values) for name, values in sorted(task_buckets.items())},
        "qtype_scores": qtype_scores,
        "qtype_counts": {name: len(values) for name, values in sorted(qtype_buckets.items())},
    }

def _group_specs(dataset: str, rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = [("all", "all")]
    specs.extend(("subgroup", name) for name in EXPECTED_SUBGROUPS[dataset])
    task_order = VSI_TASKS if dataset == "vsi" else VSTI_TASKS
    present_tasks = {str(row["task_group"]) for row in rows}
    specs.extend(("task_group", name) for name in task_order if name in present_tasks)
    specs.extend(
        ("question_type", name)
        for name in sorted({str(row["canonical_question_type"]) for row in rows})
    )
    return specs

def _select_group(rows: Sequence[Mapping[str, Any]], group_kind: str, group: str) -> list[Mapping[str, Any]]:
    if group_kind == "all":
        return list(rows)
    field = {
        "subgroup": "subgroup",
        "task_group": "task_group",
        "question_type": "canonical_question_type",
    }[group_kind]
    return [row for row in rows if str(row[field]) == group]

def build_summaries(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    scorer: ModuleType,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_kind, group in _group_specs(dataset, rows):
        selected = _select_group(rows, group_kind, group)
        result[f"{group_kind}::{group}"] = aggregate_rows(
            dataset, selected, scorer, group_kind=group_kind, group=group
        )
    return result

def _interval(
    values: Sequence[float],
    *,
    observed_delta: float | None,
    repetitions: int,
    unavailable_repetitions: int,
    unavailable_reason: str,
) -> dict[str, Any]:
    if observed_delta is None or unavailable_repetitions:
        return {
            "status": "unavailable",
            "reason": unavailable_reason,
            "observed_delta": observed_delta,
            "ci95": None,
            "direction_significant": "unavailable",
            "available_repetitions": len(values),
            "unavailable_repetitions": unavailable_repetitions,
        }
    lower = percentile(values, 0.025)
    upper = percentile(values, 0.975)
    return {
        "status": "available",
        "observed_delta": observed_delta,
        "ci95": [lower, upper],
        "direction_significant": "positive" if lower > 0 else ("negative" if upper < 0 else "inconclusive"),
        "available_repetitions": repetitions,
        "unavailable_repetitions": 0,
        "equivalence_claim_allowed": False,
    }

def paired_bootstrap(
    current_rows: Sequence[Mapping[str, Any]],
    hard_off_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    scorer: ModuleType,
    group_kind: str,
    group: str,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    # 复用旧分析器的有放回场景簇重采样；题型缺席时保留“不可用”，不补零或改用样本 bootstrap。
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    current_ids = [str(row["sample_id"]) for row in current_rows]
    hard_off_ids = [str(row["sample_id"]) for row in hard_off_rows]
    if current_ids != hard_off_ids:
        raise ValueError("paired bootstrap rows are not ID/order aligned")
    current_qtypes = [str(row["canonical_question_type"]) for row in current_rows]
    if current_qtypes != [str(row["canonical_question_type"]) for row in hard_off_rows]:
        raise ValueError("paired bootstrap rows have different question types")
    current_clusters = [str(row["scene_cluster"]) for row in current_rows]
    if current_clusters != [str(row["scene_cluster"]) for row in hard_off_rows] or any(
        not cluster for cluster in current_clusters
    ):
        raise ValueError("paired bootstrap rows have missing or unmatched scene clusters")
    cluster_indices = {
        cluster: [index for index, value in enumerate(current_clusters) if value == cluster]
        for cluster in sorted(set(current_clusters))
    }
    cluster_names = list(cluster_indices)
    qtypes = sorted(set(current_qtypes))
    values: dict[str, list[float]] = {
        "primary_score": [],
        "row_weighted_mean": [],
        **{f"qtype::{qtype}": [] for qtype in qtypes},
    }
    unavailable: Counter[str] = Counter()
    observed_current = aggregate_rows(
        dataset, current_rows, scorer, group_kind=group_kind, group=group
    )
    observed_hard_off = aggregate_rows(
        dataset, hard_off_rows, scorer, group_kind=group_kind, group=group
    )
    observed_task_set = set(observed_current["present_tasks"])
    if observed_task_set != set(observed_hard_off["present_tasks"]):
        raise ValueError("paired bootstrap observed arms have different task groups")
    rng = random.Random(int(seed))
    for _ in range(int(repetitions)):
        selected: list[int] = []
        for chosen_cluster in rng.choices(cluster_names, k=len(cluster_names)):
            selected.extend(cluster_indices[chosen_cluster])
        current_sample = [current_rows[index] for index in selected]
        hard_off_sample = [hard_off_rows[index] for index in selected]
        current_aggregate = aggregate_rows(
            dataset,
            current_sample,
            scorer,
            group_kind=group_kind,
            group=group,
            enforce_full_main=False,
        )
        hard_off_aggregate = aggregate_rows(
            dataset,
            hard_off_sample,
            scorer,
            group_kind=group_kind,
            group=group,
            enforce_full_main=False,
        )
        for metric in ("primary_score", "row_weighted_mean"):
            current_value = current_aggregate[metric]
            hard_off_value = hard_off_aggregate[metric]
            task_set_changed = metric == "primary_score" and (
                set(current_aggregate["present_tasks"]) != observed_task_set
                or set(hard_off_aggregate["present_tasks"]) != observed_task_set
            )
            if current_value is None or hard_off_value is None or task_set_changed:
                unavailable[metric] += 1
            else:
                values[metric].append(float(hard_off_value) - float(current_value))
        for qtype in qtypes:
            key = f"qtype::{qtype}"
            if qtype not in current_aggregate["qtype_scores"] or qtype not in hard_off_aggregate["qtype_scores"]:
                unavailable[key] += 1
            else:
                values[key].append(
                    float(hard_off_aggregate["qtype_scores"][qtype])
                    - float(current_aggregate["qtype_scores"][qtype])
                )

    intervals: dict[str, Any] = {}
    for metric in ("primary_score", "row_weighted_mean"):
        left = observed_current[metric]
        right = observed_hard_off[metric]
        observed = None if left is None or right is None else float(right) - float(left)
        intervals[metric] = _interval(
            values[metric],
            observed_delta=observed,
            repetitions=repetitions,
            unavailable_repetitions=unavailable[metric],
            unavailable_reason=(
                f"{metric} was undefined in {unavailable[metric]} scene-cluster bootstrap replicates"
            ),
        )
    for qtype in qtypes:
        key = f"qtype::{qtype}"
        observed = (
            float(observed_hard_off["qtype_scores"][qtype])
            - float(observed_current["qtype_scores"][qtype])
        )
        intervals[key] = _interval(
            values[key],
            observed_delta=observed,
            repetitions=repetitions,
            unavailable_repetitions=unavailable[key],
            unavailable_reason=(
                f"qtype {qtype} was absent from {unavailable[key]} scene-cluster bootstrap replicates; "
                "the prior method is not replaced with an invented score"
            ),
        )
    return {
        "method": "paired scene-cluster bootstrap",
        "direction": "hard_off-current",
        "resampling_unit": "scene_cluster",
        "repetitions": int(repetitions),
        "seed": int(seed),
        "cluster_count": len(cluster_indices),
        "qtype_counts": dict(Counter(current_qtypes)),
        "intervals": intervals,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['vsi', 'vsti'], required=True)
    parser.add_argument('--members', type=Path, required=True, help='Packaged numeric_pose_members.json')
    parser.add_argument('--eval-jsonl', type=Path, required=True, help='Independent native scoring rows with labels')
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--hard-off', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--query-pose-mode', choices=['legacy', 'provided_full'], required=True,
                        help='Actual mode shared by both runs; historical VSTI used provided_full')
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    contract = json.loads(args.members.read_text())[args.dataset]
    expected = contract['ids']
    if len(expected) != EXPECTED_COUNTS[args.dataset]:
        raise ValueError('Comparison requires the full original denominator')
    validate_prediction_ids(expected, expected, 'member manifest')
    subgroups = contract['subgroup_ids']
    if set(subgroups) != set(EXPECTED_SUBGROUPS[args.dataset]):
        raise ValueError('Unexpected comparison subsets')
    subset_ids = [sid for values in subgroups.values() for sid in values]
    if len(subset_ids) != len(set(subset_ids)) or set(subset_ids) != set(expected):
        raise ValueError('Subsets must partition the complete member set')
    for name, count in EXPECTED_SUBGROUPS[args.dataset].items():
        if len(subgroups[name]) != count:
            raise ValueError('Changed subset denominator: ' + name)
    labels = read_jsonl(args.eval_jsonl)
    validate_prediction_ids(labels, expected, 'scoring rows')
    raw = {'current': read_jsonl(args.current), 'hard_off': read_jsonl(args.hard_off)}
    scored = {}
    membership = {sid: group for group, ids in subgroups.items() for sid in ids}
    for arm, rows in raw.items():
        validate_prediction_ids(rows, expected, arm)
        scored[arm] = []
        for row, label in zip(rows, labels):
            validate_raw_prediction(row, label, dataset=args.dataset, arm=arm, query_pose_mode=args.query_pose_mode)
            if row['gate1']['reference_family'] != 'all':
                raise ValueError('Comparison requires all reference types')
            if row['gate1']['base_reference_type_id'] != row['gate1']['effective_reference_type_id']:
                raise ValueError('Reference type changed')
            scored[arm].append(score_one(row, label, dataset=args.dataset, arm=arm,
                subgroup=membership[row['sample_id']], preflight=False, scorer=scorer))
    for current, off in zip(raw['current'], raw['hard_off']):
        for field in ['source_query_pose_4x4_sha256', 'effective_reference_type_id']:
            if current['gate1'][field] != off['gate1'][field]:
                raise ValueError('Paired source reference changed: ' + field)
        if current['query_pose_protocol']['geometry_gauge'] != off['query_pose_protocol']['geometry_gauge']:
            raise ValueError('Paired observation coordinates changed')
    # 正确答案仅在两份答卷都完成后进入本进程；完整题序通过后才生成正式分数。
    summaries = {arm: build_summaries(args.dataset, rows, scorer) for arm, rows in scored.items()}
    comparisons = {}
    result_rows = []
    for kind, group in _group_specs(args.dataset, scored['current']):
        key = kind + '::' + group
        comparisons[key] = paired_bootstrap(
            _select_group(scored['current'], kind, group), _select_group(scored['hard_off'], kind, group),
            dataset=args.dataset, scorer=scorer, group_kind=kind, group=group)
        interval = comparisons[key]['intervals']['primary_score']
        ci = interval['ci95']
        for arm in ['current', 'hard_off']:
            summary = summaries[arm][key]
            result_rows.append(dict(dataset=args.dataset, arm=arm, **summary,
                delta_hard_off_minus_current_pp=interval['observed_delta'] * 100,
                ci95_low_pp=None if ci is None else ci[0] * 100,
                ci95_high_pp=None if ci is None else ci[1] * 100, ci_status=interval['status']))
    args.output_root.mkdir(parents=True)
    for arm, rows in scored.items():
        write_jsonl(args.output_root/(arm + '.scores.jsonl'), rows)
    write_json(args.output_root/'ANALYSIS.json', dict(dataset=args.dataset, summaries=summaries,
        comparisons=comparisons, input_scope='caller-supplied; historical input equivalence not established'))
    write_tsv(args.output_root/'RESULTS.tsv', result_rows)
    print(json.dumps({'status': 'COMPARISON_COMPLETE', 'rows_per_arm': len(expected)}))


if __name__ == '__main__':
    main()
