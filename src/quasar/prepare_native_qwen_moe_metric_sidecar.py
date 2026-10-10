#!/usr/bin/env python3
"""Prepare explicit MoE metric auxiliary supervision for native-answer Qwen/QUASAR SFT.

这个脚本不覆盖原始数据集。它读取 native-answer train/val/test JSONL，
为每条样本生成一个新的 `refs.simple3_aux` 字段，并额外写入 `metric_aux`
审计字段，把较大的 patch-level 几何监督保存到独立 npz sidecar。

之前 simple3 SFT 还是在 loader 里临时猜 reference type / query pose。
这会让 semantic-query 这类题到底有没有监督变得不透明。这个脚本把三类 reference
的监督先离线写清楚：能监督就写 pose/坐标；不能监督就写清失败原因。
MoE metric 版本还会把 scene/object/camera 的米制统计量写进 sidecar，方便后续确认
metric_scale expert 到底有多少训练信号。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


from quasar.train_flow_as_picker_vsi import write_json  # noqa: E402
from quasar.train_native_qwen3vl2b_glow_patch_spatial import (  # noqa: E402
    NativeAnswerPatchSpatialDataset,
    build_bev_coord_targets,
    build_bev_occupancy,
    compute_scene_center_scale,
    filter_native_rows,
    infer_simple_reference_condition,
    parse_target_frame_index,
    pose_from_xy_yaw,
    read_native_jsonl,
    reference_type_mapping,
    require_mapping,
    stable_hash,
    transform_world_points_to_camera,
)


FORMAT_VERSION = "native_qwen_moe_metric_sidecar_v1"


def normalize_label(text: str) -> str:
    """把物体名变成粗粒度可匹配形式。

    object bank 里的 prompt 可能是 `trash bin`，题干里可能写 `trash cans`。
    这里只做可审计的轻量归一化，不调用 VLM，也不静默猜复杂同义词。
    """

    value = str(text or "").lower()
    value = re.sub(r"<|>|-?\d+-img", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    words = []
    for word in value.split():
        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def labels_match(requested: str, candidate: str) -> bool:
    req = normalize_label(requested)
    cand = normalize_label(candidate)
    if not req or not cand:
        return False
    if req == cand:
        return True
    req_words = set(req.split())
    cand_words = set(cand.split())
    if req_words and cand_words and req_words.issubset(cand_words):
        return True
    if req_words and cand_words and cand_words.issubset(req_words):
        return True
    return False


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line={line_no} must be JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"input JSONL has zero rows: {path}")
    return rows


def write_jsonl_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_object_manifest(path: Optional[Path]) -> dict[str, Path]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"object token manifest does not exist: {path}")
    mapping: dict[str, Path] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path} line={line_no} must be JSON object")
            if str(row.get("status") or "ok") != "ok":
                continue
            video_id = str(row.get("video_id") or "").strip()
            npz_path = Path(str(row.get("object_token_npz_path") or ""))
            if not video_id:
                raise ValueError(f"{path} line={line_no} missing video_id")
            if not npz_path.is_file():
                raise FileNotFoundError(f"{path} line={line_no} object npz missing: {npz_path}")
            if video_id in mapping:
                raise ValueError(f"duplicate object token video_id={video_id} in {path}")
            mapping[video_id] = npz_path
    return mapping


def object_manifest_video_candidates(row: Mapping[str, Any]) -> list[str]:
    scene = require_mapping(row, "scene")
    raw_candidates = [
        str(scene.get("video_id") or ""),
        str(scene.get("scene_id") or ""),
        str(scene.get("scene_name") or ""),
    ]
    output: list[str] = []
    for item in raw_candidates:
        item = item.strip()
        if item and item not in output:
            output.append(item)
        if item and not item.startswith("scannet__"):
            with_prefix = f"scannet__{item}"
            if with_prefix not in output:
                output.append(with_prefix)
    return output


def load_object_npz(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path)
    if key in cache:
        return cache[key]
    with np.load(path, allow_pickle=True) as payload:
        required = ("prompts", "object_center_world", "object_size_xyz")
        for name in required:
            if name not in payload.files:
                raise KeyError(f"object token npz missing {name}: {path}")
        prompts = [str(value) for value in payload["prompts"].tolist()]
        centers = np.asarray(payload["object_center_world"], dtype=np.float32)
        sizes = np.asarray(payload["object_size_xyz"], dtype=np.float32)
        source_types = (
            [str(value) for value in payload["hybrid_source_type"].tolist()]
            if "hybrid_source_type" in payload.files
            else ["unknown"] * len(prompts)
        )
        areas = (
            np.asarray(payload["total_area_px"], dtype=np.float32)
            if "total_area_px" in payload.files
            else np.ones((len(prompts),), dtype=np.float32)
        )
    if centers.shape != (len(prompts), 3):
        raise ValueError(f"object_center_world shape mismatch: {centers.shape} prompts={len(prompts)} path={path}")
    if sizes.shape != (len(prompts), 3):
        raise ValueError(f"object_size_xyz shape mismatch: {sizes.shape} prompts={len(prompts)} path={path}")
    data = {"prompts": prompts, "centers": centers, "sizes": sizes, "source_types": source_types, "areas": areas}
    cache[key] = data
    return data


def resolve_object_bank(row: Mapping[str, Any], manifest: Mapping[str, Path], cache: dict[str, dict[str, Any]]) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[Path]]:
    object_refs = require_mapping(require_mapping(row, "refs"), "object")
    explicit_path = object_refs.get("hybrid_object_token_npz_path") or object_refs.get("global_object_token_npz_path") or object_refs.get("object_token_npz_path")
    if explicit_path:
        path = Path(str(explicit_path))
        if not path.is_file():
            raise FileNotFoundError(f"explicit object token npz missing: {path}")
        return load_object_npz(path, cache), "explicit_refs_object_npz_path", path
    for video_id in object_manifest_video_candidates(row):
        path = manifest.get(video_id)
        if path is not None:
            return load_object_npz(path, cache), f"manifest_video_id:{video_id}", path
    return None, None, None


def match_object_center(
    *,
    label: str,
    object_bank: Optional[Mapping[str, Any]],
) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
    if object_bank is None:
        return None, None
    prompts = list(object_bank["prompts"])
    centers = np.asarray(object_bank["centers"], dtype=np.float32)
    sizes = np.asarray(object_bank["sizes"], dtype=np.float32)
    areas = np.asarray(object_bank["areas"], dtype=np.float32)
    source_types = list(object_bank["source_types"])
    matched: list[int] = [i for i, prompt in enumerate(prompts) if labels_match(label, prompt)]
    if not matched:
        return None, None
    # 同名物体可能有多个。这里选 mask 面积最大的那个，不随机、不静默采样。
    matched.sort(key=lambda i: float(areas[i]), reverse=True)
    index = int(matched[0])
    audit = {
        "requested_label": label,
        "matched_prompt": prompts[index],
        "matched_index": index,
        "matched_count": len(matched),
        "center_world": centers[index].astype(float).tolist(),
        "size_xyz": sizes[index].astype(float).tolist(),
        "source_type": str(source_types[index]),
        "area_px": float(areas[index]),
    }
    return centers[index].astype(np.float32), audit


def summarize_object_centers_for_scene(
    *,
    row: Mapping[str, Any],
    object_manifest: Mapping[str, Path],
    object_npz_cache: dict[str, dict[str, Any]],
    max_object_centers: int,
) -> dict[str, Any]:
    """为 BEV/canonical 监督保存可审计的 object centers。

    BEV 不只是 occupancy，也应该知道场景里有哪些物体中心。
    object bank 有就写入；没有就明确记录 missing，不让后续误以为已经有物体中心监督。
    """

    object_bank, bank_source, bank_path = resolve_object_bank(row, object_manifest, object_npz_cache)
    if object_bank is None:
        return {
            "object_center_supervision": False,
            "object_bank_source": None,
            "object_bank_path": None,
            "object_centers": [],
            "failure_reason": "missing_object_bank",
        }
    prompts = list(object_bank["prompts"])
    centers = np.asarray(object_bank["centers"], dtype=np.float32)
    sizes = np.asarray(object_bank["sizes"], dtype=np.float32)
    areas = np.asarray(object_bank["areas"], dtype=np.float32)
    source_types = list(object_bank["source_types"])
    order = list(range(len(prompts)))
    order.sort(key=lambda i: float(areas[i]), reverse=True)
    if max_object_centers > 0:
        order = order[: int(max_object_centers)]
    records = []
    for index in order:
        records.append(
            {
                "prompt": prompts[index],
                "center_world": centers[index].astype(float).tolist(),
                "size_xyz": sizes[index].astype(float).tolist(),
                "source_type": str(source_types[index]),
                "area_px": float(areas[index]),
            }
        )
    return {
        "object_center_supervision": bool(records),
        "object_bank_source": bank_source,
        "object_bank_path": str(bank_path) if bank_path is not None else None,
        "object_centers": records,
        "object_center_count": len(records),
        "failure_reason": None if records else "empty_object_bank",
    }


def parse_vlm3r_standing_facing(question: str) -> tuple[Optional[str], Optional[str], str]:
    text = str(question or "").strip()
    patterns = [
        r"standing by the ([a-zA-Z0-9 _-]+?) and facing the ([a-zA-Z0-9 _-]+?)[,?.]",
        r"standing near the ([a-zA-Z0-9 _-]+?) and facing the ([a-zA-Z0-9 _-]+?)[,?.]",
        r"standing at the ([a-zA-Z0-9 _-]+?) and facing the ([a-zA-Z0-9 _-]+?)[,?.]",
        r"standing beside the ([a-zA-Z0-9 _-]+?) and facing the ([a-zA-Z0-9 _-]+?)[,?.]",
        r"standing next to the ([a-zA-Z0-9 _-]+?) and facing the ([a-zA-Z0-9 _-]+?)[,?.]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip(), f"regex:{pattern}"
    facing = re.search(r"facing the ([a-zA-Z0-9 _-]+?)[,?.]", text, flags=re.IGNORECASE)
    if facing:
        return None, facing.group(1).strip(), "regex:facing_only"
    return None, None, "no_standing_facing_pattern"


def parse_tagged_activity_anchor(text: str) -> tuple[Optional[str], Optional[str], str]:
    """从 SQA3D/MSQA situation 里找 `I am sitting on <chair-6-IMG>` 这类锚点。"""

    raw = str(text or "")
    tagged = re.search(
        r"I am (?:sitting on|cleaning|decorating|removing|opening|unlocking|reading|looking at|standing by|standing near) (?:the )?<([a-zA-Z _-]+)-(\d+)-IMG>",
        raw,
        flags=re.IGNORECASE,
    )
    if tagged:
        return tagged.group(1).strip(), tagged.group(2).strip(), "regex:tagged_activity_anchor"
    plain = re.search(
        r"I am (?:sitting on|cleaning|decorating|removing|opening|unlocking|reading|looking at|standing by|standing near) (?:the |a |an )?([a-zA-Z _-]+?)(?:\\.|,|$)",
        raw,
        flags=re.IGNORECASE,
    )
    if plain:
        label = plain.group(1).strip()
        # 去掉动作短语后残留的长描述，只保留最后两个词以内的物体名。
        words = normalize_label(label).split()
        if words:
            return " ".join(words[-2:]), None, "regex:plain_activity_anchor"
    return None, None, "no_activity_anchor"


def pose_from_anchor_facing(
    *,
    anchor_center: np.ndarray,
    facing_center: np.ndarray,
    anchor_size: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    anchor = np.asarray(anchor_center, dtype=np.float32)
    facing = np.asarray(facing_center, dtype=np.float32)
    delta = facing - anchor
    horiz = np.asarray([float(delta[0]), float(delta[2])], dtype=np.float32)
    norm = float(np.linalg.norm(horiz))
    if norm <= 1e-4:
        raise ValueError(f"anchor/facing centers overlap horizontally: anchor={anchor.tolist()} facing={facing.tolist()}")
    direction = horiz / norm
    offset = 0.5
    if anchor_size is not None:
        size = np.asarray(anchor_size, dtype=np.float32)
        if size.shape == (3,) and np.isfinite(size).all():
            offset = max(offset, float(np.linalg.norm(size[[0, 2]])) * 0.5)
    camera_xz = np.asarray([anchor[0], anchor[2]], dtype=np.float32) - direction * offset
    yaw = float(np.arctan2(direction[0], direction[1]))
    pose = pose_from_xy_yaw(float(camera_xz[0]), float(camera_xz[1]), yaw).numpy().astype(np.float32)
    pose[1, 3] = float(anchor[1])
    return pose, yaw


def build_semantic_query_pose(
    *,
    row: Mapping[str, Any],
    object_manifest: Mapping[str, Path],
    object_npz_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = require_mapping(row, "source")
    question = require_mapping(row, "question")
    pose_refs = require_mapping(require_mapping(row, "refs"), "pose")
    source_tag = str(source.get("tag") or "unknown")

    loc_xy = pose_refs.get("loc3r_position_xy")
    loc_yaw = pose_refs.get("loc3r_orientation_rad")
    if isinstance(loc_xy, list) and len(loc_xy) == 2 and loc_yaw is not None:
        pose = pose_from_xy_yaw(float(loc_xy[0]), float(loc_xy[1]), float(loc_yaw)).numpy().astype(np.float32)
        return {
            "pose_valid": True,
            "pose_4x4": pose.astype(float).tolist(),
            "confidence": 1.0,
            "source": "loc3r_position_xy_orientation",
            "anchor": {"status": "valid", "source": "loc3r_pose_field"},
            "facing": {"status": "valid", "source": "loc3r_pose_field"},
            "failure_reason": None,
        }

    object_bank, bank_source, bank_path = resolve_object_bank(row, object_manifest, object_npz_cache)
    q_text = str(question.get("text") or "")
    situation = str(question.get("situation") or "")
    anchor_label: Optional[str]
    facing_label: Optional[str]
    parse_source: str
    if source_tag == "vlm3r_canonical_vsi":
        anchor_label, facing_label, parse_source = parse_vlm3r_standing_facing(q_text)
    else:
        anchor_label, _anchor_id, parse_source = parse_tagged_activity_anchor(situation)
        # MSQA/SQA3D 的题干常写 “at my N o'clock”，这不是一个可直接确定朝向的物体；
        # 如果没有 Loc3R pose，就只把站位 anchor 记录下来，不伪造 yaw。
        facing_label = None

    anchor_center = None
    anchor_audit = None
    facing_center = None
    facing_audit = None
    if anchor_label:
        anchor_center, anchor_audit = match_object_center(label=anchor_label, object_bank=object_bank)
    if facing_label:
        facing_center, facing_audit = match_object_center(label=facing_label, object_bank=object_bank)

    overlap_failure: Optional[dict[str, Any]] = None
    if anchor_label and facing_label and anchor_center is not None and facing_center is not None:
        anchor_size = np.asarray(anchor_audit["size_xyz"], dtype=np.float32) if anchor_audit else None
        # 有些样本会把 anchor 和 facing 匹配到几乎同一个物体中心。
        # 这种情况下“从 anchor 朝向 facing”的 yaw 没有几何意义，不能硬造 pose；
        # 但它只是 semantic pose 监督不可用，不应该让整份 20k sidecar 中断。
        anchor_arr = np.asarray(anchor_center, dtype=np.float32)
        facing_arr = np.asarray(facing_center, dtype=np.float32)
        horiz_delta = np.asarray([float(facing_arr[0] - anchor_arr[0]), float(facing_arr[2] - anchor_arr[2])], dtype=np.float32)
        horiz_norm = float(np.linalg.norm(horiz_delta))
        if horiz_norm <= 1e-4:
            overlap_failure = {
                "reason": "anchor_facing_overlap_horiz",
                "horizontal_norm": horiz_norm,
                "anchor_center": anchor_arr.astype(float).tolist(),
                "facing_center": facing_arr.astype(float).tolist(),
            }
        else:
            pose, yaw = pose_from_anchor_facing(anchor_center=anchor_center, facing_center=facing_center, anchor_size=anchor_size)
            return {
                "pose_valid": True,
                "pose_4x4": pose.astype(float).tolist(),
                "confidence": 0.75 if bank_source else 0.0,
                "source": "object_center_anchor_facing",
                "anchor": {"status": "matched", **(anchor_audit or {})},
                "facing": {"status": "matched", **(facing_audit or {})},
                "object_bank_source": bank_source,
                "object_bank_path": str(bank_path) if bank_path is not None else None,
                "parse_source": parse_source,
                "yaw_rad": yaw,
                "failure_reason": None,
            }

    missing: list[str] = []
    if object_bank is None:
        missing.append("missing_object_bank")
    if not anchor_label:
        missing.append("missing_anchor_label")
    if anchor_label and anchor_center is None:
        missing.append("anchor_not_matched")
    if not facing_label:
        missing.append("missing_facing_label")
    if facing_label and facing_center is None:
        missing.append("facing_not_matched")
    if overlap_failure is not None:
        missing.append(str(overlap_failure["reason"]))
    return {
        "pose_valid": False,
        "pose_4x4": None,
        "confidence": 0.0,
        "source": "semantic_parse_object_bank",
        "anchor": {"status": "missing" if anchor_audit is None else "matched", "requested_label": anchor_label, "match": anchor_audit},
        "facing": {"status": "missing" if facing_audit is None else "matched", "requested_label": facing_label, "match": facing_audit},
        "object_bank_source": bank_source,
        "object_bank_path": str(bank_path) if bank_path is not None else None,
        "parse_source": parse_source,
        "overlap_failure": overlap_failure,
        "failure_reason": "+".join(missing) if missing else "unknown_semantic_pose_failure",
    }


def tensor_to_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().float().tolist()


def finite_min_max(points: torch.Tensor, valid: torch.Tensor) -> tuple[list[float], list[float]]:
    valid_points = points[valid]
    if valid_points.numel() == 0:
        raise ValueError("cannot compute scene extent from zero valid points")
    return tensor_to_list(valid_points.min(dim=0).values), tensor_to_list(valid_points.max(dim=0).values)


def log1p_float(value: float) -> float:
    return float(np.log1p(max(0.0, float(value))))


def parse_frame_indices(question: str, *, max_frames: int) -> list[int]:
    matches = re.findall(r"\bframe\s+(\d+)\s+of\s+32\b", str(question).lower())
    output: list[int] = []
    for match in matches:
        index = max(0, min(max_frames - 1, int(match) - 1))
        if index not in output:
            output.append(index)
    return output


def build_metric_aux_for_row(
    *,
    row: Mapping[str, Any],
    sample: Any,
    query_pose: torch.Tensor,
    query_pose_valid: bool,
    center: torch.Tensor,
    scale: torch.Tensor,
    scene_min: Sequence[float],
    scene_max: Sequence[float],
    object_center_summary: Mapping[str, Any],
    bev_grid_size: int,
) -> dict[str, Any]:
    """写入 MoE metric expert 的可审计监督统计。

    第一版不假装已经解析出了每道 numeric 题的具体物体。这里先写稳定的
    scene/object/camera 米制统计，目标是让 metric_scale expert 学会“米”的数量级。
    """

    question = require_mapping(row, "question")
    answer = require_mapping(row, "answer")
    qtype = str(question.get("canonical_question_type") or "")
    answer_type = str(answer.get("type") or "")
    scene_min_arr = np.asarray(scene_min, dtype=np.float32)
    scene_max_arr = np.asarray(scene_max, dtype=np.float32)
    room_extent = np.maximum(scene_max_arr - scene_min_arr, 0.0)

    translations = sample.source_poses[:, :3, 3].detach().cpu().float().numpy()
    query_t = query_pose[:3, 3].detach().cpu().float().numpy()
    frame_indices = parse_frame_indices(str(question.get("text") or ""), max_frames=int(translations.shape[0]))
    if len(frame_indices) >= 2:
        camera_primary = float(np.linalg.norm(translations[frame_indices[-1]] - translations[frame_indices[0]]))
        camera_source = "question_frame_pair"
    elif query_pose_valid:
        camera_primary = float(np.linalg.norm(query_t - translations[0]))
        camera_source = "first_to_query_pose"
    else:
        camera_primary = float(np.linalg.norm(translations[-1] - translations[0]))
        camera_source = "sequence_start_to_end"
    camera_start_end = float(np.linalg.norm(translations[-1] - translations[0]))
    camera_to_query = np.linalg.norm(translations - query_t.reshape(1, 3), axis=-1) if query_pose_valid else np.asarray([camera_start_end], dtype=np.float32)

    object_records = list(object_center_summary.get("object_centers") or [])
    object_stats: dict[str, Any]
    if object_records:
        centers = np.asarray([record["center_world"] for record in object_records], dtype=np.float32)
        sizes = np.asarray([record["size_xyz"] for record in object_records], dtype=np.float32)
        camera_object_dist = np.linalg.norm(centers - query_t.reshape(1, 3), axis=-1)
        if centers.shape[0] >= 2:
            diff = centers[:, None, :] - centers[None, :, :]
            pair = np.linalg.norm(diff, axis=-1)
            pair = pair[np.triu_indices(centers.shape[0], k=1)]
            pair_stats = {
                "valid": True,
                "min": float(pair.min()),
                "mean": float(pair.mean()),
                "max": float(pair.max()),
                "log1p_min_mean_max": [log1p_float(pair.min()), log1p_float(pair.mean()), log1p_float(pair.max())],
            }
        else:
            pair_stats = {"valid": False, "failure_reason": "fewer_than_two_objects"}
        object_stats = {
            "valid": True,
            "count": int(centers.shape[0]),
            "log1p_count": log1p_float(float(centers.shape[0])),
            "mean_size_xyz": sizes.mean(axis=0).astype(float).tolist(),
            "log1p_mean_size_xyz": np.log1p(np.maximum(sizes.mean(axis=0), 0.0)).astype(float).tolist(),
            "pair_distance": pair_stats,
            "camera_object_distance": {
                "valid": True,
                "min": float(camera_object_dist.min()),
                "mean": float(camera_object_dist.mean()),
                "max": float(camera_object_dist.max()),
                "log1p_min_mean_max": [
                    log1p_float(camera_object_dist.min()),
                    log1p_float(camera_object_dist.mean()),
                    log1p_float(camera_object_dist.max()),
                ],
            },
        }
    else:
        object_stats = {
            "valid": False,
            "failure_reason": str(object_center_summary.get("failure_reason") or "missing_object_centers"),
        }

    return {
        "enabled": True,
        "source": "prepare_native_qwen_moe_metric_sidecar.py",
        "qtype": qtype,
        "answer_type": answer_type,
        "metric_scale_recommended": bool(answer_type in {"numeric", "count"} or "distance" in qtype or "size" in qtype or "count" in qtype),
        "scene": {
            "scene_center": tensor_to_list(center),
            "scene_scale": float(scale.detach().cpu().item()),
            "log1p_scene_scale": log1p_float(float(scale.detach().cpu().item())),
            "scene_min": list(scene_min),
            "scene_max": list(scene_max),
            "room_extent_xyz": room_extent.astype(float).tolist(),
            "log1p_room_extent_xyz": np.log1p(np.maximum(room_extent, 0.0)).astype(float).tolist(),
            "bev_cell_size_meter": float(2.0 * float(scale.detach().cpu().item()) / max(1, int(bev_grid_size) - 1)),
        },
        "camera": {
            "valid": True,
            "source": camera_source,
            "primary_displacement": camera_primary,
            "start_to_end_displacement": camera_start_end,
            "max_to_query_displacement": float(camera_to_query.max()),
            "log1p_primary_first_max": [
                log1p_float(camera_primary),
                log1p_float(float(np.linalg.norm(query_t - translations[0])) if query_pose_valid else camera_primary),
                log1p_float(float(camera_to_query.max())),
            ],
        },
        "objects": object_stats,
    }


def save_aux_npz(
    *,
    path: Path,
    reference_type: str,
    query_pose: Optional[torch.Tensor],
    query_pose_valid: bool,
    points: torch.Tensor,
    valid: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    bev_grid_size: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    bev_coord = build_bev_coord_targets(points=points, center=center, scale=scale)
    occupancy = build_bev_occupancy(points=points, valid=valid, center=center, scale=scale, grid_size=bev_grid_size)
    if query_pose_valid:
        if query_pose is None:
            raise ValueError("query_pose_valid=True but query_pose is None")
        query_coord = transform_world_points_to_camera(points=points, camera_pose=query_pose, scene_scale=scale)
    else:
        query_coord = torch.zeros((*points.shape[:-1], 3), dtype=torch.float32)
    scene_min, scene_max = finite_min_max(points, valid)
    np.savez_compressed(
        path,
        format_version=np.asarray(FORMAT_VERSION),
        reference_type=np.asarray(reference_type),
        query_pose_valid=np.asarray(bool(query_pose_valid)),
        query_pose_4x4=np.asarray(tensor_to_list(query_pose) if query_pose is not None else np.eye(4, dtype=np.float32), dtype=np.float32),
        scene_center=np.asarray(tensor_to_list(center), dtype=np.float32),
        scene_scale=np.asarray(float(scale.detach().cpu().item()), dtype=np.float32),
        scene_min=np.asarray(scene_min, dtype=np.float32),
        scene_max=np.asarray(scene_max, dtype=np.float32),
        patch_valid_grid=valid.detach().cpu().numpy().astype(np.bool_),
        query_frame_coord_grid=query_coord.detach().cpu().numpy().astype(np.float16),
        bev_coord_grid=bev_coord.detach().cpu().numpy().astype(np.float16),
        bev_occupancy_grid=occupancy.detach().cpu().numpy().astype(np.float16),
    )
    return {
        "path": str(path),
        "query_frame_coord_grid_shape": list(query_coord.shape),
        "bev_coord_grid_shape": list(bev_coord.shape),
        "bev_occupancy_grid_shape": list(occupancy.shape),
        "scene_min": scene_min,
        "scene_max": scene_max,
    }


def build_simple3_aux_for_row(
    *,
    row: Mapping[str, Any],
    sample: Any,
    split: str,
    out_root: Path,
    object_manifest: Mapping[str, Path],
    object_npz_cache: dict[str, dict[str, Any]],
    bev_grid_size: int,
    max_object_centers: int,
) -> dict[str, Any]:
    question = require_mapping(row, "question")
    sample_id = str(sample.sample_id)
    mapping = reference_type_mapping("simple3")
    reference_type = infer_simple_reference_condition(row=row, question_payload=question)
    if reference_type not in mapping:
        raise ValueError(f"sample_id={sample_id} invalid simple3 reference_type={reference_type!r}")
    center, scale = compute_scene_center_scale(points=sample.points, valid=sample.patch_valid)
    scene_min, scene_max = finite_min_max(sample.points, sample.patch_valid)
    object_center_summary = summarize_object_centers_for_scene(
        row=row,
        object_manifest=object_manifest,
        object_npz_cache=object_npz_cache,
        max_object_centers=max_object_centers,
    )
    query_pose: Optional[torch.Tensor] = None
    query_pose_valid = False
    target_camera: dict[str, Any] = {"enabled": False}
    semantic_query: dict[str, Any] = {"enabled": False}
    canonical_bev: dict[str, Any] = {"enabled": False}

    if reference_type == "target_camera_pose":
        frame_index = parse_target_frame_index(str(question.get("text") or ""), max_frames=int(sample.source_poses.shape[0]))
        query_pose = sample.source_poses[frame_index].float()
        query_pose_valid = True
        target_camera = {
            "enabled": True,
            "target_frame_index": int(frame_index),
            "pose_valid": True,
            "pose_4x4": tensor_to_list(query_pose),
            "coord_supervision": True,
            "source": "parse_last_frame_reference_from_question",
        }
    elif reference_type == "semantic_query_pose":
        semantic_query = build_semantic_query_pose(row=row, object_manifest=object_manifest, object_npz_cache=object_npz_cache)
        semantic_query["enabled"] = True
        query_pose_valid = bool(semantic_query.get("pose_valid"))
        if query_pose_valid:
            query_pose = torch.tensor(semantic_query["pose_4x4"], dtype=torch.float32)
        else:
            # semantic pose 解析失败时，不让 loader 再猜 pose。
            # 我们显式写入一个“场景中心占位 pose”，并把 query_pose_valid=false。
            # 这样主干输入稳定，但 semantic coord aux loss 不会启用。
            placeholder_pose = torch.eye(4, dtype=torch.float32)
            placeholder_pose[:3, 3] = center.float()
            query_pose = placeholder_pose
            semantic_query["placeholder_pose_4x4"] = tensor_to_list(placeholder_pose)
            semantic_query["placeholder_source"] = "scene_center_placeholder_for_invalid_semantic_pose"
    elif reference_type == "canonical_bev_scene_pose":
        canonical_pose = torch.eye(4, dtype=torch.float32)
        canonical_pose[:3, 3] = center.float()
        query_pose = canonical_pose
        query_pose_valid = False
        canonical_bev = {
            "enabled": True,
            "pose_valid": False,
            "canonical_pose_4x4": tensor_to_list(canonical_pose),
            "scene_center": tensor_to_list(center),
            "scene_scale": float(scale.detach().cpu().item()),
            "scene_min": scene_min,
            "scene_max": scene_max,
            "bev_coord_supervision": True,
            "occupancy_supervision": True,
            **object_center_summary,
            "source": "scene_points_center_scale",
        }
    else:
        raise ValueError(f"sample_id={sample_id} unsupported reference_type={reference_type}")

    npz_name = stable_hash(f"{split}|{sample_id}|{reference_type}") + ".npz"
    npz_path = out_root / "aux_npz" / split / npz_name[:2] / npz_name
    npz_meta = save_aux_npz(
        path=npz_path,
        reference_type=reference_type,
        query_pose=query_pose,
        query_pose_valid=query_pose_valid,
        points=sample.points.float(),
        valid=sample.patch_valid.bool(),
        center=center.float(),
        scale=scale.float(),
        bev_grid_size=bev_grid_size,
    )
    metric_aux = build_metric_aux_for_row(
        row=row,
        sample=sample,
        query_pose=query_pose if query_pose is not None else torch.eye(4, dtype=torch.float32),
        query_pose_valid=bool(query_pose_valid),
        center=center.float(),
        scale=scale.float(),
        scene_min=scene_min,
        scene_max=scene_max,
        object_center_summary=object_center_summary,
        bev_grid_size=bev_grid_size,
    )
    return {
        "format_version": FORMAT_VERSION,
        "reference_schema": "simple3",
        "reference_type": reference_type,
        "reference_type_id": int(mapping[reference_type]),
        "route": {
            "raw_reference_condition": str(question.get("reference_condition") or ""),
            "canonical_question_type": str(question.get("canonical_question_type") or ""),
            "source": "prepare_native_qwen_simple3_aux_sidecar.py",
        },
        "query_pose_valid": bool(query_pose_valid),
        "query_pose_4x4": tensor_to_list(query_pose) if query_pose is not None else None,
        "query_pose_source": (
            target_camera.get("source")
            if reference_type == "target_camera_pose"
            else semantic_query.get("source")
            if reference_type == "semantic_query_pose"
            else canonical_bev.get("source")
        ),
        "target_camera": target_camera,
        "canonical_bev": canonical_bev,
        "semantic_query": semantic_query,
        "metric_aux": metric_aux,
        "aux_npz_path": str(npz_path),
        "aux_npz": npz_meta,
        "coverage_flags": {
            "reference_type": True,
            "target_camera_coord": bool(reference_type == "target_camera_pose" and query_pose_valid),
            "semantic_query_coord": bool(reference_type == "semantic_query_pose" and query_pose_valid),
            "bev_coord": bool(reference_type == "canonical_bev_scene_pose"),
            "occupancy": bool(reference_type == "canonical_bev_scene_pose"),
            "object_center": bool(
                reference_type == "canonical_bev_scene_pose"
                and canonical_bev.get("object_center_supervision")
            ),
            "metric_scene": True,
            "metric_camera": True,
            "metric_object": bool(metric_aux["objects"].get("valid")),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build explicit simple3 auxiliary supervision sidecars.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scannet-frame-root", type=Path, default=None)
    parser.add_argument("--object-token-manifest", type=Path, default=None)
    parser.add_argument("--max-visual-frames", type=int, default=32)
    parser.add_argument("--depth-grid-h", type=int, default=24)
    parser.add_argument("--depth-grid-w", type=int, default=32)
    parser.add_argument("--bev-grid-size", type=int, default=32)
    parser.add_argument("--max-object-centers", type=int, default=64)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument(
        "--allow-legacy-mca",
        action="store_true",
        help="Only for official VSI-Bench eval rows. Training sidecars should leave this disabled.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    object_manifest = load_object_manifest(args.object_token_manifest)
    object_npz_cache: dict[str, dict[str, Any]] = {}

    split_inputs = {
        "train": args.train_jsonl,
        "val": args.val_jsonl,
        "test": args.test_jsonl,
    }
    global_audit: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "output_root": str(args.output_root),
        "object_token_manifest": str(args.object_token_manifest) if args.object_token_manifest is not None else None,
        "splits": {},
    }

    for split, input_path in split_inputs.items():
        rows_raw = read_native_jsonl(input_path)
        rows, filter_audit = filter_native_rows(
            rows=rows_raw,
            max_rows=int(args.max_rows_per_split),
            require_patch_geometry=True,
            scannet_frame_root=args.scannet_frame_root,
            max_visual_frames=int(args.max_visual_frames),
            allow_legacy_mca=bool(args.allow_legacy_mca),
        )
        dataset = NativeAnswerPatchSpatialDataset(
            rows=rows,
            max_visual_frames=int(args.max_visual_frames),
            scannet_frame_root=args.scannet_frame_root,
            depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
            reference_schema="simple3",
        )

        out_rows: list[dict[str, Any]] = []
        counters: Counter[str] = Counter()
        by_ref: dict[str, Counter[str]] = defaultdict(Counter)
        semantic_failures: Counter[str] = Counter()
        for index, row in enumerate(rows):
            sample = dataset[index]
            aux = build_simple3_aux_for_row(
                row=row,
                sample=sample,
                split=split,
                out_root=args.output_root,
                object_manifest=object_manifest,
                object_npz_cache=object_npz_cache,
                bev_grid_size=int(args.bev_grid_size),
                max_object_centers=int(args.max_object_centers),
            )
            new_row = dict(row)
            refs = dict(require_mapping(new_row, "refs"))
            refs["simple3_aux"] = aux
            new_row["refs"] = refs
            out_rows.append(new_row)

            ref = str(aux["reference_type"])
            counters["rows"] += 1
            counters[f"reference_type:{ref}"] += 1
            for flag, enabled in aux["coverage_flags"].items():
                if bool(enabled):
                    counters[f"coverage:{flag}"] += 1
                    by_ref[ref][f"coverage:{flag}"] += 1
            by_ref[ref]["rows"] += 1
            if ref == "semantic_query_pose":
                if bool(aux["semantic_query"].get("pose_valid")):
                    counters["semantic_pose_valid"] += 1
                    by_ref[ref]["pose_valid"] += 1
                else:
                    reason = str(aux["semantic_query"].get("failure_reason") or "unknown")
                    semantic_failures[reason] += 1
                    by_ref[ref][f"failure:{reason}"] += 1

        out_jsonl = args.output_root / f"native_answer.simple3_aux.{split}.jsonl"
        write_jsonl_rows(out_jsonl, out_rows)
        split_audit = {
            "input_jsonl": str(input_path),
            "output_jsonl": str(out_jsonl),
            "filter_audit": filter_audit,
            "rows": len(out_rows),
            "counters": dict(counters),
            "by_reference_type": {key: dict(value) for key, value in sorted(by_ref.items())},
            "semantic_failure_reasons": dict(semantic_failures),
        }
        write_json(args.output_root / f"coverage_audit.{split}.json", split_audit)
        global_audit["splits"][split] = split_audit
        print(json.dumps({"event": "split_completed", "split": split, **split_audit}, ensure_ascii=False), flush=True)

    write_json(args.output_root / "coverage_audit.all.json", global_audit)
    write_json(
        args.output_root / "manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "train_jsonl": str(args.output_root / "native_answer.simple3_aux.train.jsonl"),
            "val_jsonl": str(args.output_root / "native_answer.simple3_aux.val.jsonl"),
            "test_jsonl": str(args.output_root / "native_answer.simple3_aux.test.jsonl"),
            "coverage_audit": str(args.output_root / "coverage_audit.all.json"),
            "aux_npz_root": str(args.output_root / "aux_npz"),
        },
    )


if __name__ == "__main__":
    main()
