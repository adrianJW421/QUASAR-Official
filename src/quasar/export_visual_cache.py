#!/usr/bin/env python3
"""Export video/scene-level Qwen3VL visual cache + QUASAR base spatial cache.

这个脚本只算“题目无关”的重活：
- 同一个 video/scene 的 32 帧，只过一次 Qwen vision tower；
- CUT3R/depth/pose 也只对齐一次到 Qwen visual grid；
- query/reference-conditioned latent flow bias 不在这里算，因为它依赖具体问题。

所以它和 How-many/count schema 修复基本独立。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import transformers
from transformers import AutoConfig, AutoProcessor

from quasar.train_native_qwen3vl2b_glow_patch_spatial import (  # noqa: E402
    ANSWER_TYPES,
    FORMAT_VERSION as TRAIN_FORMAT_VERSION,
    NativeAnswerPatchSpatialDataset,
    compute_scene_center_scale,
    process_vision_inputs,
    read_native_jsonl,
    resize_points_grid,
    resize_scalar_grid,
    scene_dir_has_min_frames,
)
from quasar.train_flow_as_picker_vsi import write_json, append_jsonl  # noqa: E402


CACHE_FORMAT_VERSION = "native_qwen3vl2b_patch_spatial_cache_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export QUASAR video-level visual/spatial cache shards.")
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--qwen-cache-dir", type=Path, default=None)
    parser.add_argument("--scannet-frame-root", type=Path, default=None)
    parser.add_argument("--max-visual-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--depth-grid-h", type=int, default=24)
    parser.add_argument("--depth-grid-w", type=int, default=32)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-jobs", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inference-only", action="store_true", help="Read answer-free records without training targets.")
    return parser


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def require_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"row missing mapping field={key!r}")
    return value


def as_str_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return [str(item) for item in value]


def cache_identity_from_row(
    row: Mapping[str, Any],
    *,
    scannet_frame_root: Path | None,
    max_visual_frames: int,
    qwen_model_path: Path,
    max_pixels: int,
) -> tuple[str, str] | None:
    scene = require_mapping(row, "scene")
    inputs = require_mapping(row, "inputs")
    refs = require_mapping(row, "refs")
    geometry = require_mapping(refs, "geometry")
    pose = require_mapping(refs, "pose")

    model_part = f"qwen={qwen_model_path}|pixels={max_pixels}|frames={max_visual_frames}"
    cut3r_path = geometry.get("cut3r_npz_path")
    if cut3r_path:
        identity = f"cut3r|{cut3r_path}|{model_part}"
        return stable_hash(identity), identity

    if scannet_frame_root is not None:
        scene_name = str(scene.get("scene_name") or scene.get("scene_id") or "").strip()
        if scene_name:
            scene_dir = scannet_frame_root / scene_name
            if scene_dir_has_min_frames(scene_dir, min_frames=max_visual_frames):
                identity = f"scannet_scene|{scene_dir}|{model_part}"
                return stable_hash(identity), identity

    image_paths = as_str_list(inputs.get("image_paths"))
    depth_paths = as_str_list(geometry.get("candidate_depth_paths"))
    pose_paths = as_str_list(pose.get("candidate_pose_paths"))
    intrinsic_paths = as_str_list(geometry.get("candidate_intrinsic_depth_paths"))
    if image_paths and depth_paths and pose_paths and intrinsic_paths:
        identity = "explicit_depth_pose|" + "|".join(image_paths + depth_paths + pose_paths + intrinsic_paths) + "|" + model_part
        return stable_hash(identity), identity
    return None


def collect_unique_jobs(
    *,
    input_jsonl: Sequence[Path],
    scannet_frame_root: Path | None,
    max_visual_frames: int,
    qwen_model_path: Path,
    max_pixels: int,
) -> tuple[list[tuple[str, Mapping[str, Any], str]], Dict[str, Any]]:
    jobs: dict[str, tuple[Mapping[str, Any], str]] = {}
    skipped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for path in input_jsonl:
        rows = read_native_jsonl(path)
        for row in rows:
            flags = require_mapping(row, "flags")
            answer = require_mapping(row, "answer")
            answer_type = str(answer.get("type") or "")
            if bool(flags.get("is_legacy_mca")) or bool(answer.get("legacy_mca")) or answer_type not in ANSWER_TYPES:
                skipped["unsupported_or_legacy_answer_schema"] += 1
                continue
            identity = cache_identity_from_row(
                row,
                scannet_frame_root=scannet_frame_root,
                max_visual_frames=max_visual_frames,
                qwen_model_path=qwen_model_path,
                max_pixels=max_pixels,
            )
            if identity is None:
                skipped["missing_visual_geometry"] += 1
                continue
            key, identity_text = identity
            if key not in jobs:
                jobs[key] = (row, identity_text)
                source = require_mapping(row, "source")
                source_counts[str(source.get("tag") or "unknown")] += 1
    sorted_jobs = [(key, row, identity_text) for key, (row, identity_text) in sorted(jobs.items())]
    audit = {
        "input_files": [str(p) for p in input_jsonl],
        "unique_jobs": len(sorted_jobs),
        "skipped": dict(skipped),
        "unique_jobs_by_representative_source": dict(source_counts),
    }
    if not sorted_jobs:
        raise ValueError(f"zero cache jobs collected: {audit}")
    return sorted_jobs, audit


def load_processor_and_qwen_for_cache(args: argparse.Namespace) -> tuple[Any, torch.nn.Module, int, Dict[str, Any]]:
    config_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    model_kwargs: Dict[str, Any] = {"dtype": torch.bfloat16, "trust_remote_code": True}
    if args.qwen_cache_dir is not None:
        config_kwargs["cache_dir"] = str(args.qwen_cache_dir)
        model_kwargs["cache_dir"] = str(args.qwen_cache_dir)
    config = AutoConfig.from_pretrained(str(args.qwen_model_path), **config_kwargs)
    model_type = str(getattr(config, "model_type", ""))
    if model_type.startswith("qwen3_vl"):
        model_cls_name = "Qwen3VLForConditionalGeneration"
    elif model_type.startswith("qwen2_5_vl"):
        model_cls_name = "Qwen2_5_VLForConditionalGeneration"
    else:
        raise ValueError(f"expected Qwen3-VL or Qwen2.5-VL model_type, got {model_type!r}")
    vision_config = getattr(config, "vision_config", None)
    spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
    if spatial_merge_size <= 0:
        raise ValueError("cannot resolve Qwen vision spatial_merge_size")
    model_cls = getattr(transformers, model_cls_name, None)
    if model_cls is None:
        raise ValueError(f"{model_cls_name} missing in this environment")
    processor = AutoProcessor.from_pretrained(str(args.qwen_model_path), **config_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = int(args.max_pixels)
    qwen = model_cls.from_pretrained(str(args.qwen_model_path), **model_kwargs)
    qwen.eval()
    for parameter in qwen.parameters():
        parameter.requires_grad = False
    return processor, qwen, spatial_merge_size, {
        "model_type": model_type,
        "spatial_merge_size": spatial_merge_size,
        "max_pixels": int(args.max_pixels),
        "train_format_version": TRAIN_FORMAT_VERSION,
    }


def encode_qwen_images(
    *,
    processor: Any,
    qwen: torch.nn.Module,
    image_paths: Sequence[str],
    max_pixels: int,
    device: torch.device,
) -> Dict[str, Any]:
    content = [{"type": "image", "image": str(path), "max_pixels": max_pixels} for path in image_paths]
    content.append({"type": "text", "text": "Cache visual features."})
    messages = [[{"role": "user", "content": content}]]
    text = [processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=False)]
    image_inputs, video_inputs = process_vision_inputs(messages)
    if video_inputs is not None:
        raise ValueError("cache exporter expects image inputs only")
    encoded = dict(processor(text=text, images=image_inputs, padding=True, return_tensors="pt"))
    pixel_values = encoded["pixel_values"].to(device)
    image_grid_thw = encoded["image_grid_thw"].to(device)
    with torch.no_grad():
        outputs = qwen.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    image_embeds = torch.cat(outputs.pooler_output, dim=0).detach().cpu().to(torch.bfloat16)
    deepstack = [tensor.detach().cpu().to(torch.bfloat16) for tensor in (getattr(outputs, "deepstack_features", ()) or ())]
    return {
        "image_grid_thw": image_grid_thw.detach().cpu().to(torch.int64),
        "image_embeds": image_embeds,
        "deepstack_features": deepstack,
    }


def build_base_spatial_cache(
    *,
    row: Mapping[str, Any],
    image_grid_thw: torch.Tensor,
    scannet_frame_root: Path | None,
    max_visual_frames: int,
    depth_grid_hw: tuple[int, int],
    spatial_merge_size: int,
    inference_only: bool = False,
) -> Dict[str, Any]:
    dataset = NativeAnswerPatchSpatialDataset(
        rows=[row],
        max_visual_frames=max_visual_frames,
        scannet_frame_root=scannet_frame_root,
        depth_grid_hw=depth_grid_hw,
        inference_only=inference_only,
    )
    sample = dataset[0]
    expected_images = len(sample.image_paths)
    if image_grid_thw.shape[0] != expected_images:
        raise ValueError(f"image_grid_thw image count mismatch: {tuple(image_grid_thw.shape)} vs images={expected_images}")
    center, scale = compute_scene_center_scale(points=sample.points, valid=sample.patch_valid)
    all_points: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    all_conf: list[torch.Tensor] = []
    token_frame_indices: list[torch.Tensor] = []
    token_slices: list[list[int]] = []
    cursor = 0
    for frame_i in range(expected_images):
        thw = image_grid_thw[frame_i]
        if int(thw[0].item()) != 1:
            raise ValueError(f"expected image T=1, got thw={thw.tolist()}")
        llm_h = int(thw[1].item()) // spatial_merge_size
        llm_w = int(thw[2].item()) // spatial_merge_size
        points = resize_points_grid(sample.points[frame_i], out_hw=(llm_h, llm_w)).reshape(llm_h * llm_w, 3)
        valid = (resize_scalar_grid(sample.patch_valid[frame_i].float(), out_hw=(llm_h, llm_w), mode="nearest") > 0.5).reshape(-1)
        conf = resize_scalar_grid(sample.patch_confidence[frame_i].float(), out_hw=(llm_h, llm_w), mode="bilinear").reshape(-1)
        count = int(points.shape[0])
        all_points.append(points.float())
        all_valid.append(valid.bool())
        all_conf.append(conf.float())
        token_frame_indices.append(torch.full((count,), frame_i, dtype=torch.long))
        token_slices.append([cursor, cursor + count])
        cursor += count
    return {
        "image_paths": list(sample.image_paths),
        "points": torch.cat(all_points, dim=0).to(torch.float16),
        "valid": torch.cat(all_valid, dim=0),
        "confidence": torch.cat(all_conf, dim=0).to(torch.float16),
        "token_frame_indices": torch.cat(token_frame_indices, dim=0),
        "token_slices": token_slices,
        "source_poses": sample.source_poses.to(torch.float32),
        "scene_center": center.to(torch.float32),
        "scene_scale": scale.to(torch.float32),
        "representative_sample_id": sample.sample_id,
        "representative_video_id": sample.video_id,
        "representative_scene_id": sample.scene_id,
    }


def cache_path_for_key(root: Path, key: str) -> Path:
    return root / "cache" / key[:2] / f"{key}.pt"


def main() -> None:
    args = build_parser().parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"invalid shard {args.shard_index}/{args.num_shards}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs, audit = collect_unique_jobs(
        input_jsonl=args.input_jsonl,
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
    )
    shard_jobs = [job for i, job in enumerate(jobs) if i % int(args.num_shards) == int(args.shard_index)]
    if int(args.limit_jobs) > 0:
        shard_jobs = shard_jobs[: int(args.limit_jobs)]
    shard_audit = {
        **audit,
        "format_version": CACHE_FORMAT_VERSION,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "shard_jobs": len(shard_jobs),
        "dry_run": bool(args.dry_run),
    }
    write_json(args.output_root / f"shard_{args.shard_index:03d}_audit.json", shard_audit)
    if args.dry_run:
        return

    device = torch.device(args.device)
    processor, qwen, spatial_merge_size, qwen_meta = load_processor_and_qwen_for_cache(args)
    qwen = qwen.to(device)
    write_json(args.output_root / f"shard_{args.shard_index:03d}_qwen_meta.json", qwen_meta)

    manifest_path = args.output_root / f"shard_{args.shard_index:03d}_manifest.jsonl"
    failure_path = args.output_root / f"shard_{args.shard_index:03d}_failures.jsonl"
    done = 0
    skipped_existing = 0
    failed = 0
    start = time.time()
    for local_i, (key, row, identity_text) in enumerate(shard_jobs, start=1):
        out_path = cache_path_for_key(args.output_root, key)
        if out_path.is_file():
            skipped_existing += 1
            continue
        source = require_mapping(row, "source")
        sample_id = str(row.get("sample_id") or source.get("row_id") or key)
        try:
            # 这里允许捕获“单个视频/场景的坏数据”，因为这是批量数据审计。
            # 不做任何替代特征；坏样本写 failure manifest，后续训练只用成功 cache。
            dataset = NativeAnswerPatchSpatialDataset(
                rows=[row],
                max_visual_frames=int(args.max_visual_frames),
                scannet_frame_root=args.scannet_frame_root,
                depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
                inference_only=bool(args.inference_only),
            )
            sample = dataset[0]
            qwen_cache = encode_qwen_images(
                processor=processor,
                qwen=qwen,
                image_paths=sample.image_paths,
                max_pixels=int(args.max_pixels),
                device=device,
            )
            spatial_cache = build_base_spatial_cache(
                row=row,
                image_grid_thw=qwen_cache["image_grid_thw"],
                scannet_frame_root=args.scannet_frame_root,
                max_visual_frames=int(args.max_visual_frames),
                depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
                spatial_merge_size=spatial_merge_size,
                inference_only=bool(args.inference_only),
            )
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            failed += 1
            append_jsonl(
                failure_path,
                {
                    "cache_key": key,
                    "representative_sample_id": sample_id,
                    "identity_text": identity_text,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "local_i": local_i,
                    "elapsed_sec": time.time() - start,
                },
            )
            continue
        if int(qwen_cache["image_embeds"].shape[0]) != int(spatial_cache["points"].shape[0]):
            raise ValueError(
                f"cache token mismatch key={key}: qwen={tuple(qwen_cache['image_embeds'].shape)} "
                f"spatial={tuple(spatial_cache['points'].shape)}"
            )
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "cache_key": key,
            "identity_text": identity_text,
            "created_time": time.time(),
            "qwen_model_path": str(args.qwen_model_path),
            "max_pixels": int(args.max_pixels),
            "max_visual_frames": int(args.max_visual_frames),
            "spatial_merge_size": int(spatial_merge_size),
            "qwen": qwen_cache,
            "spatial": spatial_cache,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.rename(out_path)
        done += 1
        append_jsonl(
            manifest_path,
            {
                "cache_key": key,
                "path": str(out_path),
                "representative_sample_id": spatial_cache["representative_sample_id"],
                "representative_video_id": spatial_cache["representative_video_id"],
                "token_count": int(qwen_cache["image_embeds"].shape[0]),
                "local_i": local_i,
                "elapsed_sec": time.time() - start,
            },
        )
        if int(args.log_every) > 0 and (done % int(args.log_every) == 0 or done == 1):
            print(
                json.dumps(
                    {
                        "shard": int(args.shard_index),
                        "done": done,
                        "skipped_existing": skipped_existing,
                        "failed": failed,
                        "local_i": local_i,
                        "total_shard_jobs": len(shard_jobs),
                        "elapsed_sec": time.time() - start,
                        "last_key": key,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    write_json(
        args.output_root / f"shard_{args.shard_index:03d}_summary.json",
        {
            "format_version": CACHE_FORMAT_VERSION,
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "done": done,
            "skipped_existing": skipped_existing,
            "failed": failed,
            "shard_jobs": len(shard_jobs),
            "elapsed_sec": time.time() - start,
        },
    )
    if failed:
        raise RuntimeError(f"{failed} cache jobs failed; see {failure_path}")


if __name__ == "__main__":
    main()
