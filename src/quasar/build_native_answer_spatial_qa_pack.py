#!/usr/bin/env python3
"""Build the QUASAR native-answer spatial QA data pack.

过去很多实验为了快，把 VSI/VSTI/ScanQA 这些任务都压成 A/B/C/D 分类。
这一步反过来做：尽量保留每个数据集原本的答案形态，让后续 QwenVL 可以
像正常聊天模型一样自回归输出数字、短文本、yes/no 或选项。

这个脚本只做数据重处理和审计，不训练模型。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence, TextIO


FORMAT_VERSION = "glow_native_answer_spatial_qa_v1"

VSTI_QTYPE_MAP = {
    "camera_displacement": "vsti_camera_displacement",
    "camera_movement_direction": "vsti_camera_movement_direction",
    "camera_obj_abs_dist": "vsti_camera_obj_abs_dist",
    "camera_obj_rel_dist_v1": "vsti_camera_obj_rel_dist_v1",
    "camera_obj_rel_dist_v2": "vsti_camera_obj_rel_dist_v2",
    "camera_obj_rel_dist_v3": "vsti_camera_obj_rel_dist_v3",
    "obj_obj_relative_pos_lr": "vsti_obj_obj_relative_pos_lr",
    "obj_obj_relative_pos_nf": "vsti_obj_obj_relative_pos_nf",
    "obj_obj_relative_pos_ud": "vsti_obj_obj_relative_pos_ud",
}

COUNT_QTYPES = {
    "object_counting",
}

NUMERIC_QTYPES = {
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "vsti_camera_displacement",
    "vsti_camera_obj_abs_dist",
}

DISTANCE_QTYPES = {
    "object_abs_distance",
    "object_rel_distance",
    "vsti_camera_displacement",
    "vsti_camera_obj_abs_dist",
    "vsti_camera_obj_rel_dist_v1",
    "vsti_camera_obj_rel_dist_v2",
    "vsti_camera_obj_rel_dist_v3",
}

POSE_QTYPES = {
    "route_planning",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "vsti_camera_displacement",
    "vsti_camera_movement_direction",
    "vsti_camera_obj_abs_dist",
    "vsti_camera_obj_rel_dist_v1",
    "vsti_camera_obj_rel_dist_v2",
    "vsti_camera_obj_rel_dist_v3",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the QUASAR native-answer spatial QA JSONL pack.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vlm3r-unified-jsonl", type=Path, default=None)
    parser.add_argument("--vlm3r-canonical-jsonl", type=Path, default=None)
    parser.add_argument("--vsti-official-row32-jsonl", type=Path, default=None)
    parser.add_argument("--official-eval-jsonl", type=Path, default=None)
    parser.add_argument("--scanqa-msqa-lazy-jsonl", type=Path, default=None)
    parser.add_argument(
        "--sqa3d-loc3r-json",
        type=Path,
        action="append",
        default=[],
        help="SQA3D Loc3R processed JSON list. Pass train/val files separately.",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=0,
        help="Smoke test only. 0 means full source.",
    )
    parser.add_argument(
        "--include-vlm3r-vsibench-from-unified",
        action="store_true",
        help=(
            "By default, vsibench_train from the unified pack is not used because the canonical pack stores a cleaner VSI train split."
        ),
    )
    parser.add_argument(
        "--skip-vsti-from-official-eval",
        action="store_true",
        default=True,
        help="VSTI rows in the official_eval JSONL duplicate the row32 VSTI rows; by default only VSI legacy multiple-choice rows are taken from official_eval.",
    )
    return parser


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line={line_no} must be a JSON object")
            yield line_no, row


def iter_json_list(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON list not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON source must be a list: {path}")
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path} item={index} must be a JSON object")
        yield index, row


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def parse_number(text: str) -> float | None:
    stripped = str(text).strip().replace(",", "")
    match = re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", stripped)
    if match is None:
        return None
    return float(stripped)


COUNT_WORD_VALUES = {
    "zero": 0,
    "none": 0,
    "no": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

COUNT_TENS_VALUES = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

VAGUE_COUNT_ANSWERS = {
    "a few",
    "few",
    "several",
    "many",
    "some",
    "multiple",
    "various",
    "unknown",
    "unclear",
    "not sure",
}


def is_how_many_question(question: str) -> bool:
    return re.search(r"\bhow\s+many\b", normalized_text(question)) is not None


def _parse_integer_prefix(tokens: Sequence[str]) -> int | None:
    if not tokens:
        return None
    first = tokens[0]
    if re.fullmatch(r"\d+", first):
        return int(first)
    if first in COUNT_WORD_VALUES:
        value = COUNT_WORD_VALUES[first]
        if len(tokens) >= 2 and tokens[1] == "hundred":
            value *= 100
            rest = list(tokens[2:])
            if rest[:1] == ["and"]:
                rest = rest[1:]
            suffix = _parse_integer_prefix(rest)
            if suffix is not None:
                value += suffix
            return value
        return value
    if first in COUNT_TENS_VALUES:
        value = COUNT_TENS_VALUES[first]
        if len(tokens) >= 2 and tokens[1] in COUNT_WORD_VALUES and 0 < COUNT_WORD_VALUES[tokens[1]] < 10:
            value += COUNT_WORD_VALUES[tokens[1]]
        return value
    return None


def parse_count_answer(text: str) -> int | None:
    """Parse stable count answers while leaving vague quantities unnormalized."""

    # How many 的答案里经常出现 “1,000 chairs” 这种带千分位逗号的写法。
    # 这里先去掉逗号，否则正则会把它错误地看成开头的 “1”。
    lower = normalized_text(text).replace("-", " ").replace(",", "")
    if not lower:
        return None
    if lower in VAGUE_COUNT_ANSWERS:
        return None
    if re.search(r"\b(?:about|around|approximately|roughly|at least|at most|more than|less than)\b", lower):
        return None
    if re.search(r"\b\d+\s*(?:to|or)\s*\d+\b", lower):
        return None
    if re.search(
        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(?:to|or)\s+"
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety)\b",
        lower,
    ):
        return None

    tokens = re.findall(r"[a-z]+|\d+", lower)
    candidate_token_lists = [tokens]
    leading_phrases = (
        ("there", "are"),
        ("there", "is"),
        ("the", "answer", "is"),
        ("answer", "is"),
    )
    for phrase in leading_phrases:
        if tuple(tokens[: len(phrase)]) == phrase:
            candidate_token_lists.append(tokens[len(phrase) :])

    for candidate_tokens in candidate_token_lists:
        parsed = _parse_integer_prefix(candidate_tokens)
        if parsed is not None:
            return parsed
    return None


def infer_how_many_count_value(question: str, answer_text: str, numeric_value: float | int | None) -> int | None:
    if not is_how_many_question(question):
        return None
    if numeric_value is not None:
        numeric_float = float(numeric_value)
        if numeric_float.is_integer() and numeric_float >= 0:
            return int(numeric_float)
        return None
    return parse_count_answer(answer_text)


def normalize_options(raw_options: Any, *, context: str) -> list[dict[str, str]]:
    """把各种选项格式统一成 [{"letter": "A", "text": "..."}]。

    旧包里有的选项是 dict，有的是 "A. chair" 字符串。这里统一格式，
    但不猜测非法选项；格式不认识就直接报错，避免把答案对错搞乱。
    """

    if raw_options in (None, ""):
        return []
    if not isinstance(raw_options, list):
        raise ValueError(f"{context} options must be a list, got={type(raw_options).__name__}")

    options: list[dict[str, str]] = []
    for index, option in enumerate(raw_options):
        if isinstance(option, Mapping):
            if "letter" not in option or "text" not in option:
                raise KeyError(f"{context} option dict missing letter/text: {option}")
            letter = str(option["letter"]).strip()
            text = str(option["text"]).strip()
        else:
            raw = str(option).strip()
            match = re.match(r"^([A-Z])[\.\:\)]\s*(.+?)\s*$", raw)
            if match is None:
                raise ValueError(f"{context} cannot parse option string: {raw!r}")
            letter = match.group(1)
            text = match.group(2)
        if not letter or not text:
            raise ValueError(f"{context} empty option at index={index}: {option}")
        options.append({"letter": letter, "text": text})

    letters = [option["letter"] for option in options]
    if len(set(letters)) != len(letters):
        raise ValueError(f"{context} duplicate option letters: {letters}")
    return options


def infer_numeric_unit(canonical_qtype: str, question: str) -> str | None:
    lower = normalized_text(question)
    if canonical_qtype in {"object_counting"}:
        return "count"
    if "centimeter" in lower or "cm" in lower:
        return "centimeters"
    if "square meter" in lower or "area" in lower or canonical_qtype == "room_size_estimation":
        return "square_meters"
    if "meter" in lower or canonical_qtype in DISTANCE_QTYPES:
        return "meters"
    return None


def infer_answer_type(
    *,
    question: str,
    answer_text: str,
    answer_raw: str,
    answer_kind: str,
    canonical_qtype: str,
    options: Sequence[Mapping[str, str]],
    legacy_mca: bool,
    numeric_value: float | None,
) -> str:
    if legacy_mca:
        return "legacy_mca"
    if infer_how_many_count_value(question, answer_text, numeric_value) is not None:
        return "count"
    if answer_kind == "numeric" or numeric_value is not None:
        if canonical_qtype in COUNT_QTYPES:
            return "count"
        return "numeric"
    if options and answer_kind == "option_letter":
        return "multiple_choice"
    normalized_answer = normalized_text(answer_text or answer_raw)
    if normalized_answer in {"yes", "no", "true", "false"}:
        return "yes_no"
    if canonical_qtype in COUNT_QTYPES and parse_number(answer_text) is not None:
        return "count"
    return "short_text"


def infer_reference_condition(*, canonical_qtype: str, question: str, dataset: str, row: Mapping[str, Any]) -> str:
    """判断这道题的参考系主要是什么。

    这不是模型输入，而是数据审计标签。它告诉我们这个问题更像
    “站到某个相机位置看”、还是“从全局/鸟瞰参考系看”、还是“按语义 pose 看”。
    """

    lower = normalized_text(question)
    if canonical_qtype.startswith("vsti_camera_") or re.search(r"\bframe\s+\d+\s+of\s+32\b", lower):
        return "target_camera_pose"
    if canonical_qtype in {
        "route_planning",
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
    }:
        return "semantic_query_pose"
    if str(row.get("situation", "")).strip() or str(row.get("loc3r_situation", "")).strip():
        if dataset in {"sqa3d", "msqa", "msqa_scannet"}:
            return "semantic_query_pose"
    if canonical_qtype in {"object_counting", "room_size_estimation", "object_size_estimation"}:
        return "canonical_bev_scene_pose"
    if "left of" in lower or "right of" in lower or "closest" in lower or "farthest" in lower:
        return "object_relation_reference"
    return "global_scene_reference"


def infer_query_pose_policy(reference_condition: str) -> str:
    if reference_condition == "target_camera_pose":
        return "parse_target_camera_from_question_or_frame_reference"
    if reference_condition == "semantic_query_pose":
        return "infer_semantic_pose_from_question_or_situation"
    if reference_condition == "canonical_bev_scene_pose":
        return "use_canonical_bev_placeholder_pose"
    if reference_condition == "object_relation_reference":
        return "condition_on_mentioned_objects"
    return "use_global_scene_placeholder_pose"


def infer_eval(answer_type: str, *, source_benchmark: str, canonical_qtype: str) -> dict[str, Any]:
    if answer_type == "legacy_mca":
        return {
            "metric": "choice_accuracy",
            "official_metric": "legacy_vsi_mca_accuracy",
            "parse_policy": "parse_option_letter_first_then_option_text",
            "tolerance": None,
        }
    if answer_type == "multiple_choice":
        official = "vsti_choice_accuracy" if source_benchmark == "vsti_bench_test" else "choice_accuracy"
        return {
            "metric": "choice_accuracy",
            "official_metric": official,
            "parse_policy": "parse_option_letter_first_then_option_text",
            "tolerance": None,
        }
    if answer_type == "count":
        return {
            "metric": "count_exact",
            "official_metric": "integer_exact_after_numeric_parse",
            "parse_policy": "parse_integer_or_number_word",
            "tolerance": 0,
        }
    if answer_type == "numeric":
        unit = infer_numeric_unit(canonical_qtype, "")
        tolerance = 0.1 if unit == "meters" else None
        return {
            "metric": "numeric_value",
            "official_metric": "numeric_mae_and_tolerance",
            "parse_policy": "parse_first_float_with_optional_unit",
            "tolerance": tolerance,
        }
    if answer_type == "yes_no":
        return {
            "metric": "yes_no_exact",
            "official_metric": "normalized_yes_no_exact",
            "parse_policy": "normalize_yes_no_token",
            "tolerance": None,
        }
    return {
        "metric": "short_text_normalized",
        "official_metric": "normalized_exact_match_and_token_f1",
        "parse_policy": "lowercase_strip_articles_punctuation_then_em_f1",
        "tolerance": None,
    }


def canonical_qtype_from_row(row: Mapping[str, Any], *, dataset: str, source_tag: str) -> str:
    if row.get("canonical_question_type"):
        return str(row["canonical_question_type"])
    raw_qtype = str(row.get("question_type") or "").strip()
    if raw_qtype in VSTI_QTYPE_MAP:
        return VSTI_QTYPE_MAP[raw_qtype]
    if raw_qtype:
        return raw_qtype
    answer_task_type = str(row.get("answer_task_type") or "").strip()
    if dataset == "scanqa":
        return "scanqa_open_spatial_qa"
    if dataset == "sqa3d":
        return "sqa3d_situated_spatial_qa"
    if dataset in {"msqa", "msqa_scannet"}:
        return "msqa_multiview_spatial_qa"
    if answer_task_type:
        return answer_task_type
    return f"{source_tag}_unknown_question_type"


def source_split_from_row(row: Mapping[str, Any], *, source_tag: str, source_file: str) -> str:
    split = str(row.get("split") or row.get("source_split") or "").strip()
    if split:
        return split
    if source_tag in {"vsti_official_row32", "official_eval_legacy_vsi"}:
        return "test"
    if "train" in source_file:
        return "train"
    if "val" in source_file or "valid" in source_file:
        return "val"
    if "test" in source_file:
        return "test"
    return "train"


def adapt_sqa3d_loc3r_row(row: Mapping[str, Any], *, source_path: Path) -> dict[str, Any]:
    """把 Loc3R 版本 SQA3D 转成 build_record 能读的普通行。

    SQA3D 不是 JSONL，答案藏在 conversations 的 gpt 回复里，pose 藏在
    loc3r_position_xy / loc3r_orientation_rad 里。这里只做字段搬运，不改问题和答案。
    """

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"SQA3D Loc3R row missing metadata: {row.get('id')}")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"SQA3D Loc3R row missing two-turn conversations: {row.get('id')}")
    answer_text = str(conversations[1].get("value") or "").strip()
    if not answer_text:
        raise ValueError(f"SQA3D Loc3R row has empty answer: {row.get('id')}")
    scene_id = str(metadata.get("scene_id") or "")
    split = str(metadata.get("split") or "")
    if not scene_id or not split:
        raise ValueError(f"SQA3D Loc3R row missing scene_id/split: {row.get('id')}")
    return {
        "row_id": str(row["id"]),
        "source_dataset": "SQA3D",
        "dataset": "sqa3d",
        "split": split,
        "source_file": str(source_path),
        "scene_id": scene_id,
        "scene_name": scene_id,
        "video_id": f"sqa3d__{scene_id}",
        "video_path": row.get("video"),
        "question": str(row.get("loc3r_question") or "").strip(),
        "question_type": f"sqa3d_{metadata.get('question_type') or 'unknown'}",
        "answer_text": answer_text,
        "answer_task_type": "sqa3d_native_short_text",
        "situation": row.get("loc3r_situation"),
        "loc3r_position_xy": row.get("loc3r_position_xy"),
        "loc3r_orientation_rad": row.get("loc3r_orientation_rad"),
        "source_question_id": metadata.get("question_id"),
        "has_loc3r_position": metadata.get("has_loc3r_position"),
        "has_loc3r_orientation": metadata.get("has_loc3r_orientation"),
    }


def build_answer_payload(
    row: Mapping[str, Any],
    *,
    canonical_qtype: str,
    options: Sequence[Mapping[str, str]],
    legacy_mca: bool,
    context: str,
) -> dict[str, Any]:
    answer_raw = str(row.get("answer_raw") or row.get("answer") or row.get("answer_text") or "").strip()
    answer_text = str(row.get("answer_text") or row.get("answer") or row.get("answer_raw") or "").strip()
    answer_kind = str(row.get("answer_kind") or "").strip()
    option_letter = str(row.get("answer_option_letter") or "").strip() or None

    if not answer_kind:
        if option_letter or (len(answer_raw) == 1 and answer_raw.isalpha() and options):
            answer_kind = "option_letter"
        elif row.get("answer_numeric_value") is not None:
            answer_kind = "numeric"
        else:
            answer_kind = "text"

    if answer_kind == "option_letter":
        if option_letter is None:
            option_letter = answer_raw
        option_by_letter = {option["letter"]: option["text"] for option in options}
        if option_letter not in option_by_letter:
            raise ValueError(f"{context} answer option {option_letter!r} not in options")
        if not answer_text or answer_text == option_letter:
            answer_text = option_by_letter[option_letter]

    numeric_value: float | None
    if row.get("answer_numeric_value") is not None:
        numeric_value = float(row["answer_numeric_value"])
    else:
        numeric_value = parse_number(answer_text)
    how_many_count_value = infer_how_many_count_value(str(row.get("question") or ""), answer_text, numeric_value)
    if how_many_count_value is not None:
        numeric_value = how_many_count_value

    answer_type = infer_answer_type(
        question=str(row.get("question") or ""),
        answer_text=answer_text,
        answer_raw=answer_raw,
        answer_kind=answer_kind,
        canonical_qtype=canonical_qtype,
        options=options,
        legacy_mca=legacy_mca,
        numeric_value=numeric_value,
    )
    semantic_native_type = answer_type
    if legacy_mca:
        semantic_native_type = infer_answer_type(
            question=str(row.get("question") or ""),
            answer_text=answer_text,
            answer_raw=answer_raw,
            answer_kind="numeric" if parse_number(answer_text) is not None else "text",
            canonical_qtype=canonical_qtype,
            options=[],
            legacy_mca=False,
            numeric_value=parse_number(answer_text),
        )

    numeric_unit = row.get("answer_numeric_unit") or infer_numeric_unit(canonical_qtype, str(row.get("question") or ""))
    if answer_type == "count":
        numeric_unit = "count"

    return {
        "type": answer_type,
        "semantic_native_type": semantic_native_type,
        "text": answer_text,
        "raw": answer_raw,
        "numeric_value": numeric_value,
        "numeric_unit": numeric_unit,
        "options": list(options),
        "option_letter": option_letter,
        "legacy_mca": bool(legacy_mca),
        "native_answer_preserved": not legacy_mca,
    }


def build_record(
    row: Mapping[str, Any],
    *,
    source_tag: str,
    source_path: Path,
    source_line_no: int,
    legacy_mca: bool,
    is_official_eval: bool,
) -> dict[str, Any]:
    source_file = str(row.get("source_file") or source_path)
    dataset = str(row.get("dataset") or row.get("source_dataset") or row.get("source_benchmark") or source_tag)
    source_benchmark = str(row.get("source_benchmark") or row.get("source_dataset") or dataset)
    split = source_split_from_row(row, source_tag=source_tag, source_file=source_file)
    question = str(row.get("question") or row.get("loc3r_question") or "").strip()
    if not question:
        raise ValueError(f"{source_tag} line={source_line_no} missing question")
    canonical_qtype = canonical_qtype_from_row(row, dataset=dataset, source_tag=source_tag)
    raw_qtype = str(row.get("question_type") or row.get("raw_question_type") or canonical_qtype)
    options = normalize_options(row.get("options") or [], context=f"{source_tag} line={source_line_no}")
    answer = build_answer_payload(
        row,
        canonical_qtype=canonical_qtype,
        options=options,
        legacy_mca=legacy_mca,
        context=f"{source_tag} line={source_line_no}",
    )
    reference_condition = infer_reference_condition(
        canonical_qtype=canonical_qtype,
        question=question,
        dataset=dataset,
        row=row,
    )

    source_row_id = str(row.get("row_id") or row.get("id") or row.get("question_id") or source_line_no)
    sample_id = f"{source_tag}::{source_row_id}"
    scene_id = str(row.get("scene_id") or row.get("scene_name") or "")
    scene_name = str(row.get("scene_name") or row.get("scene_id") or "")
    video_id = str(row.get("video_id") or scene_name or scene_id)
    frame_ids = row.get("candidate_frame_ids") or row.get("frame_ids") or []
    image_paths = row.get("image_input") or row.get("candidate_color_paths") or row.get("frame_paths") or []
    if image_paths is None:
        image_paths = []
    if frame_ids is None:
        frame_ids = []

    derived_vlm3r_cut3r_npz_path = None
    if row.get("cut3r_npz_path") is None and source_tag.startswith("vlm3r") and video_id:
        vlm3r_root = source_path.parent.parent
        candidate_cut3r = vlm3r_root / "geometry_cut3r_32f_v1" / "video_npz" / f"{video_id}.npz"
        # VLM3R unified 的行里早期没有直接写 CUT3R 路径，但同一数据根下面有
        # 标准 geometry_cut3r_32f_v1/video_npz/{video_id}.npz。存在就挂上；不存在就让
        # refs 里保持 None，并在 summary 里通过缺失比例继续审计。
        if candidate_cut3r.is_file():
            derived_vlm3r_cut3r_npz_path = str(candidate_cut3r)

    feature_refs = {
        "clip_embedding_path": row.get("clip_embedding_path"),
        "dino_pooled_embedding_path": row.get("dino_pooled_embedding_path"),
        "dino_token_path": row.get("dino_token_path"),
        "dino_patch_npz_path": row.get("dino_patch_npz_path"),
        "dino_frame_manifest": row.get("dino_frame_manifest"),
        "candidate_dino_token_row_indices": row.get("candidate_dino_token_row_indices"),
    }
    geometry_refs = {
        "cut3r_npz_path": row.get("cut3r_npz_path") or derived_vlm3r_cut3r_npz_path,
        "geometry_npz_path": row.get("geometry_npz_path"),
        "pointmap_npz_path": row.get("pointmap_npz_path"),
        "cut3r_frame_indices": row.get("cut3r_frame_indices"),
        "candidate_indices": row.get("candidate_indices"),
        "candidate_depth_paths": row.get("candidate_depth_paths"),
        "candidate_intrinsic_color_paths": row.get("candidate_intrinsic_color_paths"),
        "candidate_intrinsic_depth_paths": row.get("candidate_intrinsic_depth_paths"),
        "candidate_geometry_sources": row.get("candidate_geometry_sources"),
        "geometry_source": row.get("geometry_source"),
        "candidate_sample_orders": row.get("candidate_sample_orders"),
    }
    object_refs = {
        "object_token_npz_path": row.get("object_token_npz_path"),
        "global_object_token_npz_path": row.get("global_object_token_npz_path"),
        "hybrid_object_token_npz_path": row.get("hybrid_object_token_npz_path"),
        "target_objects": row.get("target_objects"),
    }
    pose_refs = {
        "candidate_pose_paths": row.get("candidate_pose_paths"),
        "pose_path": row.get("pose_path"),
        "loc3r_position_xy": row.get("loc3r_position_xy"),
        "loc3r_orientation_rad": row.get("loc3r_orientation_rad"),
        "situation": row.get("situation") or row.get("loc3r_situation"),
    }
    eval_payload = infer_eval(
        str(answer["type"]),
        source_benchmark=source_benchmark,
        canonical_qtype=canonical_qtype,
    )

    return {
        "format_version": FORMAT_VERSION,
        "sample_id": sample_id,
        "source": {
            "tag": source_tag,
            "dataset": dataset,
            "benchmark": source_benchmark,
            "split": split,
            "path": str(source_path),
            "row_id": source_row_id,
            "line_no": source_line_no,
            "source_file": source_file,
        },
        "scene": {
            "dataset": dataset,
            "scene_id": scene_id,
            "scene_name": scene_name,
            "video_id": video_id,
        },
        "question": {
            "text": question,
            "canonical_question_type": canonical_qtype,
            "raw_question_type": raw_qtype,
            "reference_condition": reference_condition,
            "query_pose_policy": infer_query_pose_policy(reference_condition),
            "situation": row.get("situation") or row.get("loc3r_situation"),
        },
        "answer": answer,
        "eval": eval_payload,
        "inputs": {
            "video_path": row.get("video_path"),
            "image_paths": image_paths,
            "frame_ids": frame_ids,
            "frame_candidate_ids": row.get("candidate_ids") or row.get("candidate_frame_ids") or [],
        },
        "refs": {
            "pose": pose_refs,
            "geometry": geometry_refs,
            "object": object_refs,
            "feature": feature_refs,
        },
        "flags": {
            "is_official_eval": bool(is_official_eval),
            "is_legacy_mca": bool(legacy_mca),
            "has_options": bool(options),
            "has_numeric_value": answer["numeric_value"] is not None,
            "native_answer_preserved": bool(answer["native_answer_preserved"]),
        },
    }


class PackWriter:
    """集中写多个 JSONL，并同步统计。

    一条样本会进 all；按 split 进 train/val/test；
    official 和 legacy 还会额外打一份，方便后续训练时明确排除旧 MCA。
    """

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.handles: dict[str, TextIO] = {
            "all": (output_dir / "native_answer.all.jsonl").open("w", encoding="utf-8"),
            "source_train": (output_dir / "native_answer.source_train.jsonl").open("w", encoding="utf-8"),
            "source_val": (output_dir / "native_answer.source_val.jsonl").open("w", encoding="utf-8"),
            "source_test": (output_dir / "native_answer.source_test.jsonl").open("w", encoding="utf-8"),
            "official_eval": (output_dir / "native_answer.official_eval.jsonl").open("w", encoding="utf-8"),
            "legacy_mca": (output_dir / "native_answer.legacy_mca.jsonl").open("w", encoding="utf-8"),
        }
        self.counts: Counter[str] = Counter()
        self.by_source: Counter[str] = Counter()
        self.by_split: Counter[str] = Counter()
        self.by_answer_type: Counter[str] = Counter()
        self.by_semantic_answer_type: Counter[str] = Counter()
        self.by_qtype: Counter[str] = Counter()
        self.by_reference: Counter[str] = Counter()
        self.by_metric: Counter[str] = Counter()
        self.by_ref_presence: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.split_keys: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()

    def write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.handles["all"].write(line)
        split = str(record["source"]["split"])
        source_split_handle = f"source_{split}"
        if source_split_handle in self.handles:
            self.handles[source_split_handle].write(line)
        if record["flags"]["is_official_eval"]:
            self.handles["official_eval"].write(line)
        if record["flags"]["is_legacy_mca"]:
            self.handles["legacy_mca"].write(line)

        source_tag = str(record["source"]["tag"])
        answer_type = str(record["answer"]["type"])
        semantic_type = str(record["answer"]["semantic_native_type"])
        canonical_qtype = str(record["question"]["canonical_question_type"])
        reference = str(record["question"]["reference_condition"])
        metric = str(record["eval"]["metric"])
        self.counts["total"] += 1
        self.by_source[source_tag] += 1
        self.by_split[split] += 1
        self.by_answer_type[answer_type] += 1
        self.by_semantic_answer_type[semantic_type] += 1
        self.by_qtype[canonical_qtype] += 1
        self.by_reference[reference] += 1
        self.by_metric[metric] += 1
        refs = record["refs"]
        inputs = record["inputs"]
        if inputs.get("image_paths"):
            self.by_ref_presence["has_image_paths"] += 1
        if inputs.get("video_path"):
            self.by_ref_presence["has_video_path"] += 1
        if refs["geometry"].get("cut3r_npz_path") or refs["geometry"].get("candidate_depth_paths"):
            self.by_ref_presence["has_geometry_or_depth"] += 1
        if refs["object"].get("object_token_npz_path") or refs["object"].get("target_objects"):
            self.by_ref_presence["has_object_reference"] += 1
        if refs["feature"].get("dino_token_path"):
            self.by_ref_presence["has_dino_token_path"] += 1
        if refs["pose"].get("candidate_pose_paths") or refs["pose"].get("pose_path"):
            self.by_ref_presence["has_pose_paths"] += 1
        if (
            refs["pose"].get("candidate_pose_paths")
            or refs["pose"].get("pose_path")
            or refs["pose"].get("loc3r_position_xy") is not None
            or refs["pose"].get("loc3r_orientation_rad") is not None
        ):
            self.by_ref_presence["has_pose_reference"] += 1
        if len(self.examples[source_tag]) < 3:
            self.examples[source_tag].append(dict(record))

        self._update_overlap_keys(record)

    def _update_overlap_keys(self, record: Mapping[str, Any]) -> None:
        split = str(record["source"]["split"])
        question = str(record["question"]["text"])
        scene = str(record["scene"]["scene_name"] or record["scene"]["scene_id"])
        video = str(record["scene"]["video_id"])
        sample_id = str(record["sample_id"])
        question_hash = stable_hash(normalized_text(question))
        scene_question_hash = stable_hash(f"{scene}::{normalized_text(question)}")
        self.split_keys[split]["sample_id"].add(sample_id)
        self.split_keys[split]["question_hash"].add(question_hash)
        if scene:
            self.split_keys[split]["scene"].add(scene)
            self.split_keys[split]["scene_question_hash"].add(scene_question_hash)
        if video:
            self.split_keys[split]["video"].add(video)

    def summary(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "output_dir": str(self.output_dir),
            "counts": dict(self.counts),
            "by_source": dict(self.by_source),
            "by_split": dict(self.by_split),
            "by_answer_type": dict(self.by_answer_type),
            "by_semantic_answer_type": dict(self.by_semantic_answer_type),
            "by_canonical_question_type": dict(self.by_qtype),
            "by_reference_condition": dict(self.by_reference),
            "by_metric": dict(self.by_metric),
            "by_ref_presence": dict(self.by_ref_presence),
            "examples": self.examples,
        }

    def overlap_audit(self) -> dict[str, Any]:
        splits = sorted(self.split_keys)
        pairs: dict[str, Any] = {}
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                label = f"{left}__vs__{right}"
                pairs[label] = {}
                key_names = sorted(set(self.split_keys[left]) | set(self.split_keys[right]))
                for key_name in key_names:
                    left_values = self.split_keys[left].get(key_name, set())
                    right_values = self.split_keys[right].get(key_name, set())
                    pairs[label][key_name] = len(left_values & right_values)
        sizes = {
            split: {key_name: len(values) for key_name, values in buckets.items()}
            for split, buckets in self.split_keys.items()
        }
        return {
            "format_version": "glow_native_answer_overlap_audit_v1",
            "split_key_sizes": sizes,
            "pairwise_intersections": pairs,
            "note": (
                "这个审计只报告重叠，不自动判失败；某些官方数据集原始 split 可能共享 scene。"
                "后续正式训练是否排除这些重叠，需要在训练采样策略里显式决定。"
            ),
        }


def should_include_vlm3r_unified_row(row: Mapping[str, Any], *, include_vsi: bool) -> bool:
    source_file = str(row.get("source_file") or "")
    if source_file.startswith("vstibench_train/"):
        return True
    if include_vsi and source_file.startswith("vsibench_train/"):
        return True
    return False


def process_source(
    *,
    writer: PackWriter,
    path: Path,
    source_tag: str,
    legacy_mca: bool,
    is_official_eval: bool,
    limit: int,
    schema_keys: dict[str, Counter[str]],
    row_filter: Any | None = None,
    source_format: str = "jsonl",
) -> dict[str, Any]:
    processed = 0
    skipped = 0
    iterator = iter_json_list(path) if source_format == "json_list" else iter_jsonl(path)
    for line_no, row in iterator:
        schema_keys[source_tag].update(str(key) for key in row.keys())
        if row_filter is not None and not row_filter(row):
            skipped += 1
            continue
        if limit and processed >= limit:
            break
        if source_tag == "sqa3d_loc3r":
            row = adapt_sqa3d_loc3r_row(row, source_path=path)
        record = build_record(
            row,
            source_tag=source_tag,
            source_path=path,
            source_line_no=line_no,
            legacy_mca=legacy_mca,
            is_official_eval=is_official_eval,
        )
        writer.write(record)
        processed += 1
    return {"processed": processed, "skipped": skipped, "path": str(path)}


def update_split_keys_for_audit(split_keys: dict[str, dict[str, set[str]]], record: Mapping[str, Any], *, split: str) -> None:
    question = str(record["question"]["text"])
    scene = str(record["scene"]["scene_name"] or record["scene"]["scene_id"])
    video = str(record["scene"]["video_id"])
    sample_id = str(record["sample_id"])
    question_hash = stable_hash(normalized_text(question))
    scene_question_hash = stable_hash(f"{scene}::{normalized_text(question)}")
    split_keys[split]["sample_id"].add(sample_id)
    split_keys[split]["question_hash"].add(question_hash)
    if scene:
        split_keys[split]["scene"].add(scene)
        split_keys[split]["scene_question_hash"].add(scene_question_hash)
    if video:
        split_keys[split]["video"].add(video)


def build_overlap_audit_from_keys(split_keys: Mapping[str, Mapping[str, set[str]]], *, note: str) -> dict[str, Any]:
    splits = sorted(split_keys)
    pairs: dict[str, Any] = {}
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            label = f"{left}__vs__{right}"
            pairs[label] = {}
            key_names = sorted(set(split_keys[left]) | set(split_keys[right]))
            for key_name in key_names:
                left_values = split_keys[left].get(key_name, set())
                right_values = split_keys[right].get(key_name, set())
                pairs[label][key_name] = len(left_values & right_values)
    sizes = {
        split: {key_name: len(values) for key_name, values in buckets.items()}
        for split, buckets in split_keys.items()
    }
    return {
        "format_version": "glow_native_answer_overlap_audit_v1",
        "split_key_sizes": sizes,
        "pairwise_intersections": pairs,
        "note": note,
    }


def write_strict_split_files(output_dir: Path, source_split_keys: Mapping[str, Mapping[str, set[str]]]) -> dict[str, Any]:
    """从 all.jsonl 生成严格的 train/val/test。

    官方/原始 split 里可能 train 和 val/test 共享 scene/video。训练时这会让
    模型在同一个房间里“见过题外信息”。所以最终给训练用的 train/val/test 不是
    直接复制 source split，而是做一遍 scene/video 隔离：
    - source test 永远保留为 test；
    - source val 若不碰 test，保留为 val；
    - source train 只要碰到 val/test 的 scene 或 video，就进 quarantine，不进训练。
    """

    test_scenes = set(source_split_keys.get("test", {}).get("scene", set()))
    test_videos = set(source_split_keys.get("test", {}).get("video", set()))
    val_scenes = set(source_split_keys.get("val", {}).get("scene", set()))
    val_videos = set(source_split_keys.get("val", {}).get("video", set()))

    all_path = output_dir / "native_answer.all.jsonl"
    if not all_path.is_file():
        raise FileNotFoundError(f"all JSONL missing before strict split: {all_path}")

    handles = {
        "train": (output_dir / "native_answer.train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / "native_answer.val.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / "native_answer.test.jsonl").open("w", encoding="utf-8"),
        "quarantine": (output_dir / "native_answer.quarantine_overlap.jsonl").open("w", encoding="utf-8"),
    }
    strict_counts: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    strict_keys: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    try:
        for _, record in iter_jsonl(all_path):
            source_split = str(record["source"]["split"])
            scene = str(record["scene"]["scene_name"] or record["scene"]["scene_id"])
            video = str(record["scene"]["video_id"])
            target_split = source_split
            reason = ""
            if source_split == "test":
                target_split = "test"
            elif source_split == "val":
                if (scene and scene in test_scenes) or (video and video in test_videos):
                    target_split = "quarantine"
                    reason = "source_val_overlaps_test_scene_or_video"
                else:
                    target_split = "val"
            elif source_split == "train":
                overlaps_test = (scene and scene in test_scenes) or (video and video in test_videos)
                overlaps_val = (scene and scene in val_scenes) or (video and video in val_videos)
                if overlaps_test and overlaps_val:
                    target_split = "quarantine"
                    reason = "source_train_overlaps_val_and_test_scene_or_video"
                elif overlaps_test:
                    target_split = "quarantine"
                    reason = "source_train_overlaps_test_scene_or_video"
                elif overlaps_val:
                    target_split = "quarantine"
                    reason = "source_train_overlaps_val_scene_or_video"
                else:
                    target_split = "train"
            else:
                target_split = "train"

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            handles[target_split].write(line)
            strict_counts[target_split] += 1
            if target_split == "quarantine":
                quarantine_reasons[reason or "unknown"] += 1
            else:
                update_split_keys_for_audit(strict_keys, record, split=target_split)
    finally:
        for handle in handles.values():
            handle.close()

    return {
        "format_version": "glow_native_answer_strict_split_summary_v1",
        "strict_counts": dict(strict_counts),
        "quarantine_reasons": dict(quarantine_reasons),
        "policy": (
            "source test kept as test; source val kept unless it overlaps test scene/video; "
            "source train removed to quarantine if it overlaps val/test scene/video"
        ),
        "overlap_audit": build_overlap_audit_from_keys(
            strict_keys,
            note="strict train/val/test after scene/video quarantine; sample/question overlaps are reported for visibility.",
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.limit_per_source < 0:
        raise ValueError("--limit-per-source must be non-negative")

    writer = PackWriter(args.output_dir)
    schema_keys: dict[str, Counter[str]] = defaultdict(Counter)
    source_results: dict[str, Any] = {}
    try:
        if args.vlm3r_canonical_jsonl is not None:
            source_results["vlm3r_canonical_vsi"] = process_source(
                writer=writer,
                path=args.vlm3r_canonical_jsonl,
                source_tag="vlm3r_canonical_vsi",
                legacy_mca=False,
                is_official_eval=False,
                limit=args.limit_per_source,
                schema_keys=schema_keys,
            )

        if args.vlm3r_unified_jsonl is not None:
            source_results["vlm3r_unified_vsti"] = process_source(
                writer=writer,
                path=args.vlm3r_unified_jsonl,
                source_tag="vlm3r_unified_vsti",
                legacy_mca=False,
                is_official_eval=False,
                limit=args.limit_per_source,
                schema_keys=schema_keys,
                row_filter=lambda row: should_include_vlm3r_unified_row(
                    row,
                    include_vsi=bool(args.include_vlm3r_vsibench_from_unified),
                ),
            )

        if args.vsti_official_row32_jsonl is not None:
            source_results["vsti_official_row32"] = process_source(
                writer=writer,
                path=args.vsti_official_row32_jsonl,
                source_tag="vsti_official_row32",
                legacy_mca=False,
                is_official_eval=True,
                limit=args.limit_per_source,
                schema_keys=schema_keys,
            )

        if args.official_eval_jsonl is not None:
            source_results["official_eval_legacy_vsi"] = process_source(
                writer=writer,
                path=args.official_eval_jsonl,
                source_tag="official_eval_legacy_vsi",
                legacy_mca=True,
                is_official_eval=True,
                limit=args.limit_per_source,
                schema_keys=schema_keys,
                row_filter=lambda row: str(row.get("source_benchmark") or "") == "vsi_bench_all_mca",
            )

        if args.scanqa_msqa_lazy_jsonl is not None:
            source_results["scanqa_sqa3d_msqa"] = process_source(
                writer=writer,
                path=args.scanqa_msqa_lazy_jsonl,
                source_tag="scanqa_sqa3d_msqa",
                legacy_mca=False,
                is_official_eval=False,
                limit=args.limit_per_source,
                schema_keys=schema_keys,
            )

        for sqa3d_path in args.sqa3d_loc3r_json:
            key = f"sqa3d_loc3r::{sqa3d_path.name}"
            source_results[key] = process_source(
                writer=writer,
                path=sqa3d_path,
                source_tag="sqa3d_loc3r",
                legacy_mca=False,
                is_official_eval=False,
                limit=args.limit_per_source,
                schema_keys=schema_keys,
                source_format="json_list",
            )
    finally:
        writer.close()

    summary = writer.summary()
    summary["source_results"] = source_results
    summary["args"] = {
        "output_dir": str(args.output_dir),
        "vlm3r_unified_jsonl": str(args.vlm3r_unified_jsonl) if args.vlm3r_unified_jsonl else None,
        "vlm3r_canonical_jsonl": str(args.vlm3r_canonical_jsonl) if args.vlm3r_canonical_jsonl else None,
        "vsti_official_row32_jsonl": str(args.vsti_official_row32_jsonl) if args.vsti_official_row32_jsonl else None,
        "official_eval_jsonl": str(args.official_eval_jsonl) if args.official_eval_jsonl else None,
        "scanqa_msqa_lazy_jsonl": str(args.scanqa_msqa_lazy_jsonl) if args.scanqa_msqa_lazy_jsonl else None,
        "sqa3d_loc3r_json": [str(path) for path in args.sqa3d_loc3r_json],
        "limit_per_source": int(args.limit_per_source),
        "include_vlm3r_vsibench_from_unified": bool(args.include_vlm3r_vsibench_from_unified),
    }
    schema_audit = {
        "format_version": "glow_native_answer_schema_audit_v1",
        "source_key_counts": {source: dict(counter) for source, counter in schema_keys.items()},
        "source_keys": {source: sorted(counter) for source, counter in schema_keys.items()},
    }
    source_overlap_audit = writer.overlap_audit()
    strict_split_summary = write_strict_split_files(args.output_dir, writer.split_keys)
    summary["strict_split_summary"] = {
        key: value for key, value in strict_split_summary.items() if key != "overlap_audit"
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "schema_audit.json", schema_audit)
    write_json(args.output_dir / "source_overlap_audit.json", source_overlap_audit)
    write_json(args.output_dir / "overlap_audit.json", strict_split_summary["overlap_audit"])
    print(json.dumps({"event": "completed", "summary_json": str(args.output_dir / "summary.json"), **summary["counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
