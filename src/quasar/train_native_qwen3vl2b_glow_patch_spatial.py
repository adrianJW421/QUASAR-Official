#!/usr/bin/env python3
# Qwen visual-forward portions adapted for QUASAR.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 修改说明：沿用 Qwen 视觉输入流程，接入项目的已保存视觉特征、空间修正、
# 摘要读出或逐步生成；各文件保留原有计算。对应上游版本与适用范围见 NOTICE。
"""Native-Qwen QUASAR patch-level spatial positional injection trainer.

这不是 direct Qwen SFT，也不是旧的 MCA classifier teacher。

正式技术目标：保持 Qwen3-VL 原生自回归 forward / causal mask / next-token CE，
只在“视觉 patch token 进入语言模型之前”注入 query/reference-conditioned 的空间位置偏置：

    v_hat_{i,j} = v_{i,j} + S_phi(X_{i,j}, p_i, r_q)

Qwen 仍然按自己原来的方式看图和生成答案；QUASAR 只负责把每个图像 patch
放进三维空间里，再根据题目要求的参考视角/参考系，把这个空间信息变成视觉 token 的位置偏置。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoConfig, AutoProcessor



from quasar.train_flow_as_picker_vsi import (  # noqa: E402
    append_jsonl,
    normalize_args_for_json,
    set_seed,
    write_json,
)


FORMAT_VERSION = "native_qwen3vl2b_glow_patch_spatial_v1"
CACHE_FORMAT_VERSION = "native_qwen3vl2b_patch_spatial_cache_v1"
LEGACY_REF_TYPE_TO_ID = {
    "target_camera_pose": 0,
    "semantic_query_pose": 1,
    "canonical_bev_scene_pose": 2,
    "object_relation_reference": 3,
    "global_scene_reference": 4,
}
SIMPLE_REF_TYPE_TO_ID = {
    "target_camera_pose": 0,
    "semantic_query_pose": 1,
    "canonical_bev_scene_pose": 2,
}
REF_TYPE_TO_ID = LEGACY_REF_TYPE_TO_ID
ID_TO_REF_TYPE = {value: key for key, value in REF_TYPE_TO_ID.items()}
REFERENCE_SCHEMA_TO_MAPPING = {
    "legacy5": LEGACY_REF_TYPE_TO_ID,
    "simple3": SIMPLE_REF_TYPE_TO_ID,
}
ANSWER_TYPES = {"multiple_choice", "numeric", "count", "yes_no", "short_text"}
SPATIAL_DESCRIPTOR_DIM = 28
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
QWEN_VISION_KEYS = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids"}
SEMANTIC_QUERY_QTYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "route_planning",
    "msqa_multiview_spatial_qa",
    "sqa3d_how",
    "sqa3d_others",
    "sqa3d_can",
    "sqa3d_is",
    "sqa3d_what",
    "sqa3d_which",
}


def reference_type_mapping(reference_schema: str) -> Dict[str, int]:
    """按实验 schema 返回 reference type 表。

    旧实验是 5 类；新简化路线只保留 3 类。这里不让两套 silently 混用。
    """

    schema = str(reference_schema)
    if schema not in REFERENCE_SCHEMA_TO_MAPPING:
        raise ValueError(f"unsupported reference_schema={schema!r}; expected one of {sorted(REFERENCE_SCHEMA_TO_MAPPING)}")
    return dict(REFERENCE_SCHEMA_TO_MAPPING[schema])


def infer_simple_reference_condition(*, row: Mapping[str, Any], question_payload: Mapping[str, Any]) -> str:
    """用题型规则把样本路由到三类 reference condition。

    这一步故意不用复杂 learned router。先用 benchmark/qtype 的稳定结构，把问题归到
    目标相机、语义人称视角、鸟瞰全局三类。这样容易审计，也不把系统做成一堆组件。
    """

    source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
    source_tag = str(source.get("tag") or "")
    qtype = str(question_payload.get("canonical_question_type") or "").strip()
    question = str(question_payload.get("text") or "").lower()
    if source_tag == "vlm3r_unified_vsti" or qtype.startswith("vsti_") or re.search(r"\bframe\s+\d+\s+of\s+32\b", question):
        return "target_camera_pose"
    if qtype in SEMANTIC_QUERY_QTYPES:
        return "semantic_query_pose"
    return "canonical_bev_scene_pose"


@dataclass(frozen=True)
class NativeGlowSample:
    sample_id: str
    source_tag: str
    scene_id: str
    video_id: str
    question: str
    canonical_question_type: str
    answer_type: str
    target_text: str
    prompt: str
    image_paths: tuple[str, ...]
    points: torch.Tensor          # [F,H,W,3] raw world/canonical points
    patch_valid: torch.Tensor     # [F,H,W]
    patch_confidence: torch.Tensor  # [F,H,W]
    source_poses: torch.Tensor    # [F,4,4]
    reference_type_id: int
    query_pose: torch.Tensor      # [4,4]
    query_pose_valid: bool
    aux_bev_center: Optional[torch.Tensor] = None
    aux_bev_scale: Optional[torch.Tensor] = None
    aux_object_centers: Optional[torch.Tensor] = None  # [K,3] canonical-BEV object center supervision
    visual_cache_key: Optional[str] = None
    cached_image_grid_thw: Optional[torch.Tensor] = None
    cached_image_embeds: Optional[torch.Tensor] = None
    cached_deepstack_features: Optional[tuple[torch.Tensor, ...]] = None


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_identity_from_row(
    row: Mapping[str, Any],
    *,
    scannet_frame_root: Path | None,
    max_visual_frames: int,
    qwen_model_path: Path,
    max_pixels: int,
) -> tuple[str, str] | None:
    """生成和 cache exporter 完全一致的视觉缓存 key。

    cache 是按“同一个视频/scene 的同一组 32 帧”去重的，不是按 QA 样本去重。
    训练时必须用同一套 key 找回对应的 Qwen image_embeds/deepstack，否则就会把 A 视频的视觉 token
    错接到 B 问题上。
    """

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
    if image_paths and not geometry and not pose:
        # 纯视觉模型只需要画面；键绑定同一组画面和底座，不要求先生成三维信息。
        identity = "rgb_only|" + "|".join(image_paths) + "|" + model_part
        return stable_hash(identity), identity
    return None


def visual_cache_path_for_key(root: Path, key: str) -> Path:
    return root / "cache" / key[:2] / f"{key}.pt"


def normalize_visual_cache_roots(
    visual_cache_root: Optional[Path],
    extra_visual_cache_roots: Optional[Sequence[Path]] = None,
) -> list[Path]:
    """把主 cache root 和额外 cache root 合成一个有序、去重的列表。

    full repaired 的训练样本和 dev/val 样本并不总在同一个视觉 cache 目录里。
    我们只按显式传进来的目录顺序查找，缺了就报错，不静默回退到实时视觉塔。
    """

    roots: list[Path] = []
    if visual_cache_root is not None:
        roots.append(Path(visual_cache_root))
    for root in extra_visual_cache_roots or []:
        roots.append(Path(root))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def find_visual_cache_path(roots: Sequence[Path], key: str) -> Path | None:
    for root in roots:
        candidate = visual_cache_path_for_key(root, key)
        if candidate.is_file():
            return candidate
    return None


class NativeAnswerPatchSpatialDataset(Dataset):
    """读取 nested native-answer schema，并构造 patch-level 3D/pose 输入。

    这里不把题重新变成 MCA，也不把 3D 信息塞成旁边的文本。
    每个样本必须拿到图像帧、对应的 patch 点云/深度、source camera pose，
    以及题目诱导出来的 reference condition。
    """

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        max_visual_frames: int,
        scannet_frame_root: Optional[Path],
        depth_grid_hw: tuple[int, int],
        reference_schema: str = "legacy5",
        visual_cache_root: Optional[Path] = None,
        visual_cache_roots: Optional[Sequence[Path]] = None,
        qwen_model_path: Optional[Path] = None,
        max_pixels: Optional[int] = None,
        simple3_aux_required: bool = False,
        inference_only: bool = False,
    ) -> None:
        if not rows:
            raise ValueError("NativeAnswerPatchSpatialDataset rows must not be empty")
        if max_visual_frames <= 0:
            raise ValueError(f"max_visual_frames must be positive, got {max_visual_frames}")
        self.rows = list(rows)
        self.max_visual_frames = int(max_visual_frames)
        self.scannet_frame_root = Path(scannet_frame_root) if scannet_frame_root is not None else None
        self.depth_grid_hw = (int(depth_grid_hw[0]), int(depth_grid_hw[1]))
        self.reference_schema = str(reference_schema)
        self.reference_type_to_id = reference_type_mapping(self.reference_schema)
        self.visual_cache_roots = normalize_visual_cache_roots(visual_cache_root, visual_cache_roots)
        self.visual_cache_root = self.visual_cache_roots[0] if self.visual_cache_roots else None
        self.qwen_model_path = Path(qwen_model_path) if qwen_model_path is not None else None
        self.max_pixels = int(max_pixels) if max_pixels is not None else None
        self.simple3_aux_required = bool(simple3_aux_required)
        self.inference_only = bool(inference_only)
        if self.visual_cache_roots:
            if self.qwen_model_path is None or self.max_pixels is None:
                raise ValueError("visual_cache_root requires qwen_model_path and max_pixels")
            for root in self.visual_cache_roots:
                if not root.is_dir():
                    raise FileNotFoundError(f"visual cache root does not exist: {root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> NativeGlowSample:
        row = self.rows[index]
        sample_id = require_str(row, "sample_id")
        source = require_mapping(row, "source")
        scene = require_mapping(row, "scene")
        question_payload = require_mapping(row, "question")
        answer = require_mapping(row, "answer")
        inputs = require_mapping(row, "inputs")
        refs = require_mapping(row, "refs")
        geometry_refs = require_mapping(refs, "geometry")
        pose_refs = require_mapping(refs, "pose")

        answer_type = str(answer.get("type") or "").strip()
        if answer_type not in ANSWER_TYPES:
            raise ValueError(f"sample_id={sample_id} unsupported native answer type={answer_type!r}")
        if bool(answer.get("legacy_mca")):
            raise ValueError(f"sample_id={sample_id} legacy_mca reached dataset; loader filter is wrong")

        question_text = require_str(question_payload, "text")
        # 生成只需要作答格式和选项；真值留在独立评分文件，不能为读取图像而索取答案。
        target_text = "" if self.inference_only else build_target_text(answer=answer, answer_type=answer_type, sample_id=sample_id)
        prompt = build_native_prompt(question_payload=question_payload, answer=answer, answer_type=answer_type)

        image_paths, points, valid, confidence, source_poses = self._load_visual_geometry(
            sample_id=sample_id,
            scene=scene,
            inputs=inputs,
            geometry_refs=geometry_refs,
            pose_refs=pose_refs,
        )
        if len(image_paths) != int(points.shape[0]):
            raise ValueError(
                f"sample_id={sample_id} image/geometry frame mismatch: images={len(image_paths)} points={tuple(points.shape)}"
            )
        if len(image_paths) > self.max_visual_frames:
            keep = torch.linspace(0, len(image_paths) - 1, steps=self.max_visual_frames).round().to(torch.long).tolist()
            image_paths = [image_paths[int(i)] for i in keep]
            points = points[keep]
            valid = valid[keep]
            confidence = confidence[keep]
            source_poses = source_poses[keep]
        if len(image_paths) != self.max_visual_frames:
            raise ValueError(
                f"sample_id={sample_id} resolved {len(image_paths)} frames; formal patch-spatial pipeline requires "
                f"exactly {self.max_visual_frames} frames"
            )
        for path in image_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"sample_id={sample_id} missing image frame: {path}")

        aux_bev_center = None
        aux_bev_scale = None
        aux_object_centers = None
        simple3_aux = refs.get("simple3_aux")
        if self.reference_schema == "simple3" and (self.simple3_aux_required or simple3_aux is not None):
            (
                reference_type_id,
                query_pose,
                query_pose_valid,
                aux_bev_center,
                aux_bev_scale,
                aux_object_centers,
            ) = resolve_reference_condition_from_simple3_aux(
                row=row,
                sample_id=sample_id,
                simple3_aux=simple3_aux,
                reference_schema=self.reference_schema,
                include_targets=not self.inference_only,
            )
        else:
            reference_type_id, query_pose, query_pose_valid = resolve_reference_condition(
                row=row,
                question_payload=question_payload,
                pose_refs=pose_refs,
                source_poses=source_poses,
                points=points,
                valid=valid,
                reference_schema=self.reference_schema,
            )
        visual_cache_key = None
        cached_image_grid_thw = None
        cached_image_embeds = None
        cached_deepstack_features = None
        if self.visual_cache_roots:
            assert self.qwen_model_path is not None and self.max_pixels is not None
            identity = cache_identity_from_row(
                row,
                scannet_frame_root=self.scannet_frame_root,
                max_visual_frames=self.max_visual_frames,
                qwen_model_path=self.qwen_model_path,
                max_pixels=self.max_pixels,
            )
            if identity is None:
                raise ValueError(f"sample_id={sample_id} cannot build visual cache identity")
            visual_cache_key, identity_text = identity
            cache_path = find_visual_cache_path(self.visual_cache_roots, visual_cache_key)
            if cache_path is None:
                probed = [str(visual_cache_path_for_key(root, visual_cache_key)) for root in self.visual_cache_roots]
                raise FileNotFoundError(f"sample_id={sample_id} missing visual cache in all roots: {probed}")
            cache_payload = torch.load(cache_path, map_location="cpu")
            if not isinstance(cache_payload, Mapping):
                raise ValueError(f"sample_id={sample_id} visual cache payload must be mapping: {cache_path}")
            if str(cache_payload.get("format_version")) != CACHE_FORMAT_VERSION:
                raise ValueError(f"sample_id={sample_id} visual cache format mismatch: {cache_path}")
            if str(cache_payload.get("cache_key")) != visual_cache_key:
                raise ValueError(f"sample_id={sample_id} visual cache key mismatch: {cache_path}")
            if str(cache_payload.get("identity_text")) != identity_text:
                raise ValueError(f"sample_id={sample_id} visual cache identity mismatch: {cache_path}")
            if str(cache_payload.get("qwen_model_path")) != str(self.qwen_model_path):
                raise ValueError(f"sample_id={sample_id} visual cache Qwen path mismatch: {cache_path}")
            if int(cache_payload.get("max_pixels")) != int(self.max_pixels):
                raise ValueError(f"sample_id={sample_id} visual cache max_pixels mismatch: {cache_path}")
            if int(cache_payload.get("max_visual_frames")) != int(self.max_visual_frames):
                raise ValueError(f"sample_id={sample_id} visual cache max_visual_frames mismatch: {cache_path}")
            qwen_cache = cache_payload.get("qwen")
            if not isinstance(qwen_cache, Mapping):
                raise ValueError(f"sample_id={sample_id} visual cache missing qwen payload: {cache_path}")
            cached_image_grid_thw = qwen_cache["image_grid_thw"].to(torch.long)
            cached_image_embeds = qwen_cache["image_embeds"].to(torch.bfloat16)
            cached_deepstack_features = tuple(tensor.to(torch.bfloat16) for tensor in qwen_cache["deepstack_features"])
            if cached_image_grid_thw.ndim != 2 or cached_image_grid_thw.shape[0] != self.max_visual_frames or cached_image_grid_thw.shape[1] != 3:
                raise ValueError(
                    f"sample_id={sample_id} cached image_grid_thw must be [F,3], got {tuple(cached_image_grid_thw.shape)}"
                )
            cache_merge_size = int(cache_payload.get("spatial_merge_size"))
            if cache_merge_size <= 0:
                raise ValueError(f"sample_id={sample_id} invalid visual cache spatial_merge_size: {cache_path}")
            expected_tokens = int((cached_image_grid_thw.prod(dim=1) // (cache_merge_size ** 2)).sum().item())
            if int(cached_image_embeds.shape[0]) != expected_tokens:
                raise ValueError(f"sample_id={sample_id} cached image_embeds token mismatch: {cache_path}")
            for layer_i, deep in enumerate(cached_deepstack_features):
                if tuple(deep.shape) != tuple(cached_image_embeds.shape):
                    raise ValueError(f"sample_id={sample_id} cached deepstack layer {layer_i} shape mismatch: {cache_path}")
        return NativeGlowSample(
            sample_id=sample_id,
            source_tag=str(source.get("tag") or "unknown"),
            scene_id=str(scene.get("scene_id") or scene.get("scene_name") or ""),
            video_id=str(scene.get("video_id") or scene.get("scene_id") or ""),
            question=question_text,
            canonical_question_type=str(question_payload.get("canonical_question_type") or "unknown"),
            answer_type=answer_type,
            target_text=target_text,
            prompt=prompt,
            image_paths=tuple(str(p) for p in image_paths),
            points=points.float(),
            patch_valid=valid.bool(),
            patch_confidence=confidence.float(),
            source_poses=source_poses.float(),
            reference_type_id=int(reference_type_id),
            query_pose=query_pose.float(),
            query_pose_valid=bool(query_pose_valid),
            aux_bev_center=aux_bev_center.float() if aux_bev_center is not None else None,
            aux_bev_scale=aux_bev_scale.float() if aux_bev_scale is not None else None,
            aux_object_centers=aux_object_centers.float() if aux_object_centers is not None else None,
            visual_cache_key=visual_cache_key,
            cached_image_grid_thw=cached_image_grid_thw,
            cached_image_embeds=cached_image_embeds,
            cached_deepstack_features=cached_deepstack_features,
        )

    def _load_visual_geometry(
        self,
        *,
        sample_id: str,
        scene: Mapping[str, Any],
        inputs: Mapping[str, Any],
        geometry_refs: Mapping[str, Any],
        pose_refs: Mapping[str, Any],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cut3r_path = geometry_refs.get("cut3r_npz_path")
        if cut3r_path:
            return self._load_cut3r_npz(sample_id=sample_id, path=Path(str(cut3r_path)))

        if self.scannet_frame_root is not None:
            scene_name = str(scene.get("scene_name") or scene.get("scene_id") or "").strip()
            if scene_name:
                scene_dir = self.scannet_frame_root / scene_name
                if scene_dir_has_min_frames(scene_dir, min_frames=self.max_visual_frames):
                    # ScanQA/MSQA/SQA3D 是 scene-level QA，不是指定某个 frame 的 VSTI。
                    # 因此优先从场景目录抽取 pose/depth 都有效的 32 帧，避免旧候选帧里混入无效 pose。
                    return self._load_scannet_scene_frames(sample_id=sample_id, scene_dir=scene_dir)

        depth_paths = as_str_list(geometry_refs.get("candidate_depth_paths"))
        pose_paths = as_str_list(pose_refs.get("candidate_pose_paths"))
        intrinsic_depth_paths = as_str_list(geometry_refs.get("candidate_intrinsic_depth_paths"))
        image_paths = as_str_list(inputs.get("image_paths"))
        if depth_paths and pose_paths and intrinsic_depth_paths and image_paths:
            return self._load_depth_pose_frames(
                sample_id=sample_id,
                image_paths=image_paths,
                depth_paths=depth_paths,
                pose_paths=pose_paths,
                intrinsic_depth_paths=intrinsic_depth_paths,
            )

        raise ValueError(
            f"sample_id={sample_id} has no usable patch geometry. Need CUT3R npz, depth+pose frames, "
            "or explicit --scannet-frame-root scene frames."
        )

    def _load_cut3r_npz(self, *, sample_id: str, path: Path) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not path.is_file():
            raise FileNotFoundError(f"sample_id={sample_id} missing CUT3R npz: {path}")
        npz = np.load(path, allow_pickle=True)
        required = ["frame_image_paths", "target_points", "camera_pose", "patch_grid_hw"]
        for key in required:
            if key not in npz:
                raise KeyError(f"sample_id={sample_id} CUT3R npz missing key={key}: {path}")
        image_paths = [str(value) for value in npz["frame_image_paths"].tolist()]
        raw_points = np.asarray(npz["target_points"], dtype=np.float32)
        raw_poses = np.asarray(npz["camera_pose"], dtype=np.float32)
        patch_grid_hw = np.asarray(npz["patch_grid_hw"], dtype=np.int64)
        if raw_points.ndim != 3 or raw_points.shape[-1] != 3:
            raise ValueError(f"sample_id={sample_id} target_points must be [F,N,3], got {raw_points.shape}")
        if raw_poses.shape != (raw_points.shape[0], 4, 4):
            raise ValueError(f"sample_id={sample_id} camera_pose mismatch: {raw_poses.shape} vs points={raw_points.shape}")
        if patch_grid_hw.shape[0] != raw_points.shape[0]:
            raise ValueError(f"sample_id={sample_id} patch_grid_hw frame count mismatch: {patch_grid_hw.shape}")
        h0, w0 = int(patch_grid_hw[0, 0]), int(patch_grid_hw[0, 1])
        if h0 * w0 != raw_points.shape[1]:
            raise ValueError(f"sample_id={sample_id} patch grid {h0}x{w0} does not match N={raw_points.shape[1]}")
        if not bool(np.all(patch_grid_hw[:, 0] == h0) and np.all(patch_grid_hw[:, 1] == w0)):
            raise ValueError(f"sample_id={sample_id} variable CUT3R grid is not supported in one sample: {patch_grid_hw[:4]}")
        points = torch.from_numpy(raw_points.reshape(raw_points.shape[0], h0, w0, 3))
        if "token_mask" in npz:
            valid = torch.from_numpy(np.asarray(npz["token_mask"], dtype=np.bool_).reshape(raw_points.shape[0], h0, w0))
        else:
            valid = torch.isfinite(points).all(dim=-1)
        if "token_confidence" in npz:
            confidence = torch.from_numpy(np.asarray(npz["token_confidence"], dtype=np.float32).reshape(raw_points.shape[0], h0, w0))
        else:
            confidence = valid.float()
        finite = torch.isfinite(points).all(dim=-1)
        valid = valid & finite
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        return image_paths, points, valid, confidence, torch.from_numpy(raw_poses)

    def _load_depth_pose_frames(
        self,
        *,
        sample_id: str,
        image_paths: Sequence[str],
        depth_paths: Sequence[str],
        pose_paths: Sequence[str],
        intrinsic_depth_paths: Sequence[str],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = len(image_paths)
        if not (len(depth_paths) == count and len(pose_paths) == count and len(intrinsic_depth_paths) == count):
            raise ValueError(
                f"sample_id={sample_id} depth/pose/intrinsic/image counts mismatch: "
                f"images={count}, depth={len(depth_paths)}, pose={len(pose_paths)}, K={len(intrinsic_depth_paths)}"
            )
        points_list: list[torch.Tensor] = []
        valid_list: list[torch.Tensor] = []
        pose_list: list[torch.Tensor] = []
        for depth_path, pose_path, intrinsic_path in zip(depth_paths, pose_paths, intrinsic_depth_paths):
            points, valid, pose = load_depth_point_grid(
                depth_path=Path(depth_path),
                pose_path=Path(pose_path),
                intrinsic_path=Path(intrinsic_path),
                grid_hw=self.depth_grid_hw,
                sample_id=sample_id,
            )
            points_list.append(points)
            valid_list.append(valid)
            pose_list.append(pose)
        points = torch.stack(points_list, dim=0)
        valid = torch.stack(valid_list, dim=0)
        poses = torch.stack(pose_list, dim=0)
        confidence = valid.float()
        return list(image_paths), points, valid, confidence, poses

    def _load_scannet_scene_frames(self, *, sample_id: str, scene_dir: Path) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        color_dir = scene_dir / "color"
        depth_dir = scene_dir / "depth"
        pose_dir = scene_dir / "pose"
        intrinsic_depth = scene_dir / "intrinsic" / "intrinsic_depth.txt"
        for required_path in [color_dir, depth_dir, pose_dir, intrinsic_depth]:
            if not required_path.exists():
                raise FileNotFoundError(f"sample_id={sample_id} missing ScanNet scene component: {required_path}")
        color_paths = sorted(color_dir.glob("*.jpg"), key=frame_path_sort_key)
        if not color_paths:
            color_paths = sorted(color_dir.glob("*.png"), key=frame_path_sort_key)
        if len(color_paths) < self.max_visual_frames:
            raise ValueError(f"sample_id={sample_id} scene has only {len(color_paths)} color frames: {color_dir}")
        valid_color_paths: list[Path] = []
        for color_path in color_paths:
            stem = color_path.stem
            depth_path = depth_dir / f"{stem}.png"
            pose_path = pose_dir / f"{stem}.txt"
            if not depth_path.is_file() or not pose_path.is_file():
                continue
            pose_matrix = np.loadtxt(pose_path).astype(np.float32)
            if pose_matrix.shape == (4, 4) and bool(np.isfinite(pose_matrix).all()):
                valid_color_paths.append(color_path)
        if len(valid_color_paths) < self.max_visual_frames:
            raise ValueError(
                f"sample_id={sample_id} scene has only {len(valid_color_paths)} finite-pose frames: {scene_dir}"
            )
        keep = torch.linspace(0, len(valid_color_paths) - 1, steps=self.max_visual_frames).round().to(torch.long).tolist()
        selected_color = [valid_color_paths[int(i)] for i in keep]
        image_paths: list[str] = []
        depth_paths: list[str] = []
        pose_paths: list[str] = []
        intrinsic_paths: list[str] = []
        for color_path in selected_color:
            stem = color_path.stem
            depth_path = depth_dir / f"{stem}.png"
            pose_path = pose_dir / f"{stem}.txt"
            if not depth_path.is_file() or not pose_path.is_file():
                raise FileNotFoundError(f"sample_id={sample_id} missing depth/pose for frame={stem} under {scene_dir}")
            image_paths.append(str(color_path))
            depth_paths.append(str(depth_path))
            pose_paths.append(str(pose_path))
            intrinsic_paths.append(str(intrinsic_depth))
        return self._load_depth_pose_frames(
            sample_id=sample_id,
            image_paths=image_paths,
            depth_paths=depth_paths,
            pose_paths=pose_paths,
            intrinsic_depth_paths=intrinsic_paths,
        )


def require_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"expected mapping field {key!r}")
    return value


def require_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"expected non-empty string field {key!r}")
    return value.strip()


def as_str_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return [str(item) for item in value]


def read_native_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 native-answer JSONL。

    旧工具里的 read_jsonl 绑定了旧 flat MCA schema，会强制找
    canonical_question_type/video_id；这里必须直接读取新的 nested schema。
    """

    if not path.is_file():
        raise FileNotFoundError(f"native jsonl does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line={line_no} must be JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"native jsonl has zero rows: {path}")
    return rows


def frame_path_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), stem
    return 10**12, stem


@lru_cache(maxsize=20000)
def scene_dir_has_min_frames(scene_dir: Path, *, min_frames: int) -> bool:
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_depth = scene_dir / "intrinsic" / "intrinsic_depth.txt"
    if not (color_dir.is_dir() and depth_dir.is_dir() and pose_dir.is_dir() and intrinsic_depth.is_file()):
        return False
    color_paths = sorted(color_dir.glob("*.jpg"), key=frame_path_sort_key)
    if not color_paths:
        color_paths = sorted(color_dir.glob("*.png"), key=frame_path_sort_key)
    valid_count = 0
    for color_path in color_paths:
        stem = color_path.stem
        pose_path = pose_dir / f"{stem}.txt"
        if (depth_dir / f"{stem}.png").is_file() and pose_path.is_file():
            pose_matrix = np.loadtxt(pose_path).astype(np.float32)
            if pose_matrix.shape == (4, 4) and bool(np.isfinite(pose_matrix).all()):
                valid_count += 1
                if valid_count >= min_frames:
                    return True
    return False


def load_depth_point_grid(
    *,
    depth_path: Path,
    pose_path: Path,
    intrinsic_path: Path,
    grid_hw: tuple[int, int],
    sample_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 ScanNet depth+pose 变成固定 patch grid 的世界坐标点。

    Qwen 的每个视觉 patch 需要一个 3D 位置。CUT3R 已经给了 point map；
    ScanQA/MSQA 这类数据没有 CUT3R npz，但有真实 depth/pose/intrinsic，
    所以这里按 patch 网格采样深度并反投影到世界坐标。
    """

    for path in [depth_path, pose_path, intrinsic_path]:
        if not path.is_file():
            raise FileNotFoundError(f"sample_id={sample_id} missing geometry file: {path}")
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"sample_id={sample_id} depth image must be HxW, got {depth.shape}: {depth_path}")
    depth_m = depth / 1000.0
    pose = np.loadtxt(pose_path).astype(np.float32)
    intrinsic = np.loadtxt(intrinsic_path).astype(np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"sample_id={sample_id} pose must be 4x4, got {pose.shape}: {pose_path}")
    if intrinsic.shape[0] < 3 or intrinsic.shape[1] < 3:
        raise ValueError(f"sample_id={sample_id} intrinsic must contain 3x3 matrix, got {intrinsic.shape}: {intrinsic_path}")
    if not bool(np.isfinite(pose).all()):
        raise ValueError(f"sample_id={sample_id} pose contains non-finite values: {pose_path}")
    if not bool(np.isfinite(intrinsic[:3, :3]).all()):
        raise ValueError(f"sample_id={sample_id} intrinsic contains non-finite values: {intrinsic_path}")
    gh, gw = grid_hw
    ys = np.linspace(0, depth_m.shape[0] - 1, gh).round().astype(np.int64)
    xs = np.linspace(0, depth_m.shape[1] - 1, gw).round().astype(np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    z = depth_m[yy, xx]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError(f"sample_id={sample_id} invalid intrinsic fx/fy: {intrinsic_path}")
    x_cam = (xx.astype(np.float32) - cx) * z / fx
    y_cam = (yy.astype(np.float32) - cy) * z / fy
    ones = np.ones_like(z, dtype=np.float32)
    cam = np.stack([x_cam, y_cam, z, ones], axis=-1).reshape(-1, 4).T
    world = (pose @ cam).T[:, :3].reshape(gh, gw, 3)
    valid = np.isfinite(world).all(axis=-1) & np.isfinite(z) & (z > 0.0)
    world = np.nan_to_num(world, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(world).float(), torch.from_numpy(valid), torch.from_numpy(pose).float()


def build_target_text(*, answer: Mapping[str, Any], answer_type: str, sample_id: str) -> str:
    if answer_type == "multiple_choice":
        letter = str(answer.get("option_letter") or "").strip()
        text = str(answer.get("text") or "").strip()
        if not letter or not text:
            raise ValueError(f"sample_id={sample_id} multiple_choice needs option_letter and text")
        return f"{letter}. {text}"
    text = str(answer.get("text") or answer.get("raw") or "").strip()
    if not text:
        raise ValueError(f"sample_id={sample_id} empty native answer text")
    return text


def build_native_prompt(*, question_payload: Mapping[str, Any], answer: Mapping[str, Any], answer_type: str) -> str:
    question = require_str(question_payload, "text")
    qtype = str(question_payload.get("canonical_question_type") or "unknown")
    reference = str(question_payload.get("reference_condition") or "unknown")
    situation = str(question_payload.get("situation") or "").strip()
    lines = [
        "You are a spatial reasoning assistant.",
        "Use the video frames and the injected 3D spatial evidence to answer the question.",
        f"Question type: {qtype}",
        f"Reference condition: {reference}",
    ]
    if situation:
        lines.append(f"Situation: {situation}")
    lines.append(f"Question: {question}")
    if answer_type == "multiple_choice":
        options = answer.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError("multiple_choice prompt requires answer.options")
        lines.append("Options:")
        for option in options:
            if not isinstance(option, Mapping):
                raise ValueError(f"option must be mapping, got {option!r}")
            letter = str(option.get("letter") or "").strip()
            text = str(option.get("text") or "").strip()
            if not letter or not text:
                raise ValueError(f"invalid option payload: {option}")
            lines.append(f"{letter}. {text}")
        lines.append("Answer with the option letter and the option text, e.g. 'A. chair'.")
    elif answer_type in {"numeric", "count"}:
        unit = str(answer.get("numeric_unit") or "").strip()
        if unit:
            lines.append(f"Answer as a concise number in {unit}.")
        else:
            lines.append("Answer as a concise number.")
    elif answer_type == "yes_no":
        lines.append("Answer yes or no.")
    else:
        lines.append("Answer with a concise text phrase.")
    lines.append("Answer:")
    return "\n".join(lines)


def resolve_reference_condition(
    *,
    row: Mapping[str, Any],
    question_payload: Mapping[str, Any],
    pose_refs: Mapping[str, Any],
    source_poses: torch.Tensor,
    points: torch.Tensor,
    valid: torch.Tensor,
    reference_schema: str = "legacy5",
) -> tuple[int, torch.Tensor, bool]:
    mapping = reference_type_mapping(reference_schema)
    if reference_schema == "simple3":
        reference = infer_simple_reference_condition(row=row, question_payload=question_payload)
    else:
        reference = str(question_payload.get("reference_condition") or "global_scene_reference").strip()
        if reference not in mapping:
            reference = "global_scene_reference"
    if reference not in mapping:
        raise ValueError(f"reference={reference!r} not available under reference_schema={reference_schema!r}")
    ref_type_id = mapping[reference]

    canonical_pose = build_canonical_bev_pose(points=points, valid=valid)
    query_pose = canonical_pose.clone()
    query_pose_valid = False

    if reference == "target_camera_pose":
        frame_index = parse_target_frame_index(str(question_payload.get("text") or ""), max_frames=int(source_poses.shape[0]))
        query_pose = source_poses[frame_index].clone()
        query_pose_valid = True
    elif reference == "semantic_query_pose":
        loc_xy = pose_refs.get("loc3r_position_xy")
        loc_yaw = pose_refs.get("loc3r_orientation_rad")
        if isinstance(loc_xy, list) and len(loc_xy) == 2 and loc_yaw is not None:
            query_pose = pose_from_xy_yaw(float(loc_xy[0]), float(loc_xy[1]), float(loc_yaw))
            query_pose_valid = True
    return ref_type_id, query_pose, query_pose_valid


def _pose_tensor_from_simple3_aux(value: Any, *, sample_id: str, field_name: str) -> torch.Tensor:
    """严格把 sidecar 里的 4x4 pose 转成 tensor。

    派生数据的 pose 是正式监督信号。这里不允许缺字段时悄悄改用旧规则；
    shape 或数值不对就直接报错，方便发现数据预处理问题。
    """

    pose = torch.tensor(value, dtype=torch.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"sample_id={sample_id} {field_name} must be 4x4, got {tuple(pose.shape)}")
    if not bool(torch.isfinite(pose).all().item()):
        raise ValueError(f"sample_id={sample_id} {field_name} contains non-finite values")
    return pose


def resolve_reference_condition_from_simple3_aux(
    *,
    row: Mapping[str, Any],
    sample_id: str,
    simple3_aux: Any,
    reference_schema: str,
    include_targets: bool = True,
) -> tuple[int, torch.Tensor, bool, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """从派生 `refs.simple3_aux` 严格读取 reference condition。

    这是 simple3 新数据包的关键入口。只要打开 `--simple3-aux-required`，
    loader 就不再自己猜 target/semantic/BEV，而是完全按预处理写入的显式字段来训练。
    """

    if not isinstance(simple3_aux, Mapping):
        raise KeyError(f"sample_id={sample_id} requires refs.simple3_aux for reference_schema={reference_schema}")
    if str(simple3_aux.get("reference_schema")) != "simple3":
        raise ValueError(f"sample_id={sample_id} refs.simple3_aux.reference_schema must be simple3")
    mapping = reference_type_mapping(reference_schema)
    reference = str(simple3_aux.get("reference_type") or "").strip()
    if reference not in mapping:
        raise ValueError(f"sample_id={sample_id} invalid simple3 reference_type={reference!r}")
    ref_type_id = int(simple3_aux.get("reference_type_id"))
    if ref_type_id != int(mapping[reference]):
        raise ValueError(
            f"sample_id={sample_id} simple3 reference_type_id mismatch: "
            f"{ref_type_id} vs {mapping[reference]} for {reference}"
        )

    query_pose_valid = bool(simple3_aux.get("query_pose_valid"))
    query_pose_payload = simple3_aux.get("query_pose_4x4")
    if query_pose_payload is None:
        query_pose = torch.eye(4, dtype=torch.float32)
    else:
        query_pose = _pose_tensor_from_simple3_aux(query_pose_payload, sample_id=sample_id, field_name="query_pose_4x4")

    aux_bev_center = None
    aux_bev_scale = None
    aux_object_centers = None
    if reference == "target_camera_pose":
        target = simple3_aux.get("target_camera")
        if not isinstance(target, Mapping) or not bool(target.get("enabled")):
            raise ValueError(f"sample_id={sample_id} target_camera_pose requires enabled target_camera supervision")
        if not query_pose_valid:
            raise ValueError(f"sample_id={sample_id} target_camera_pose requires query_pose_valid=true")
    elif reference == "semantic_query_pose":
        semantic = simple3_aux.get("semantic_query")
        if not isinstance(semantic, Mapping) or not bool(semantic.get("enabled")):
            raise ValueError(f"sample_id={sample_id} semantic_query_pose requires enabled semantic_query block")
        # semantic pose 可以显式失败。失败样本仍保留 reference type，但不参与 semantic coord aux loss。
        if query_pose_valid and simple3_aux.get("query_pose_4x4") is None:
            raise ValueError(f"sample_id={sample_id} semantic_query_pose valid sample missing query_pose_4x4")
    elif reference == "canonical_bev_scene_pose":
        canonical = simple3_aux.get("canonical_bev")
        if not isinstance(canonical, Mapping) or not bool(canonical.get("enabled")):
            raise ValueError(f"sample_id={sample_id} canonical_bev_scene_pose requires enabled canonical_bev block")
        center = torch.tensor(canonical.get("scene_center"), dtype=torch.float32)
        if center.shape != (3,) or not bool(torch.isfinite(center).all().item()):
            raise ValueError(f"sample_id={sample_id} canonical_bev.scene_center must be finite [3], got {tuple(center.shape)}")
        scale = torch.tensor(float(canonical.get("scene_scale")), dtype=torch.float32)
        if not bool(torch.isfinite(scale).item()) or float(scale.item()) <= 0.0:
            raise ValueError(f"sample_id={sample_id} canonical_bev.scene_scale must be positive finite")
        aux_bev_center = center
        aux_bev_scale = scale
        if not include_targets:
            return ref_type_id, query_pose, query_pose_valid, aux_bev_center, aux_bev_scale, None
        if "object_center_supervision" not in canonical:
            raise KeyError(f"sample_id={sample_id} canonical_bev missing object_center_supervision")
        object_center_supervision = bool(canonical.get("object_center_supervision"))
        object_centers_payload = canonical.get("object_centers")
        if not isinstance(object_centers_payload, list):
            raise ValueError(f"sample_id={sample_id} canonical_bev.object_centers must be a list")
        if object_center_supervision:
            centers: list[list[float]] = []
            for item_i, item in enumerate(object_centers_payload):
                if not isinstance(item, Mapping):
                    raise ValueError(f"sample_id={sample_id} canonical_bev.object_centers[{item_i}] must be an object")
                center_world = item.get("center_world")
                center_tensor = torch.tensor(center_world, dtype=torch.float32)
                if center_tensor.shape != (3,) or not bool(torch.isfinite(center_tensor).all().item()):
                    raise ValueError(
                        f"sample_id={sample_id} canonical_bev.object_centers[{item_i}].center_world must be finite [3]"
                    )
                centers.append(center_tensor.tolist())
            if not centers:
                raise ValueError(f"sample_id={sample_id} object_center_supervision=true but object_centers is empty")
            aux_object_centers = torch.tensor(centers, dtype=torch.float32)
        else:
            # 这里不是 fallback。sidecar 明确说该 scene 没有 object-center 监督，
            # 所以训练时只关闭 object-center loss，但仍保留 BEV coord / occupancy 监督。
            failure_reason = str(canonical.get("failure_reason") or "")
            if not failure_reason:
                raise ValueError(f"sample_id={sample_id} object_center_supervision=false requires failure_reason")
    else:
        raise ValueError(f"sample_id={sample_id} unsupported simple3 reference_type={reference!r}")

    return ref_type_id, query_pose, query_pose_valid, aux_bev_center, aux_bev_scale, aux_object_centers


def parse_target_frame_index(question: str, *, max_frames: int) -> int:
    matches = re.findall(r"\bframe\s+(\d+)\s+of\s+32\b", question.lower())
    if not matches:
        return max_frames - 1
    # 像“between frame 14 and frame 20”这种题，reference 是最终要对齐/比较的目标帧，取最后一个出现的帧号。
    frame_number = int(matches[-1])
    return max(0, min(max_frames - 1, frame_number - 1))


def build_canonical_bev_pose(*, points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_points = points[valid]
    pose = torch.eye(4, dtype=torch.float32)
    if valid_points.numel() > 0:
        center = valid_points.float().mean(dim=0)
        pose[:3, 3] = center
    return pose


def pose_from_xy_yaw(x: float, y: float, yaw: float) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    pose[0, 0] = c
    pose[0, 2] = s
    pose[2, 0] = -s
    pose[2, 2] = c
    pose[0, 3] = float(x)
    pose[2, 3] = float(y)
    return pose


def filter_native_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    max_rows: int,
    require_patch_geometry: bool,
    scannet_frame_root: Optional[Path],
    max_visual_frames: int,
    allow_legacy_mca: bool = False,
) -> tuple[list[Mapping[str, Any]], Dict[str, Any]]:
    kept: list[Mapping[str, Any]] = []
    skipped: Counter[str] = Counter()
    by_answer: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    for row in rows:
        flags = require_mapping(row, "flags")
        answer = require_mapping(row, "answer")
        source = require_mapping(row, "source")
        scene = require_mapping(row, "scene")
        refs = require_mapping(row, "refs")
        geometry = require_mapping(refs, "geometry")
        pose = require_mapping(refs, "pose")
        inputs = require_mapping(row, "inputs")
        if not bool(allow_legacy_mca) and (bool(flags.get("is_legacy_mca")) or bool(answer.get("legacy_mca"))):
            skipped["legacy_mca"] += 1
            continue
        answer_type = str(answer.get("type") or "")
        if answer_type not in ANSWER_TYPES:
            skipped[f"unsupported_answer::{answer_type}"] += 1
            continue
        has_cut3r = bool(geometry.get("cut3r_npz_path"))
        has_depth_pose = bool(geometry.get("candidate_depth_paths") and pose.get("candidate_pose_paths") and inputs.get("image_paths"))
        has_scene_frames = False
        if scannet_frame_root is not None:
            scene_name = str(scene.get("scene_name") or scene.get("scene_id") or "").strip()
            if scene_name:
                scene_dir = scannet_frame_root / scene_name
                has_scene_frames = scene_dir_has_min_frames(scene_dir, min_frames=max_visual_frames)
        if require_patch_geometry and not (has_cut3r or has_depth_pose or has_scene_frames):
            skipped["missing_patch_geometry"] += 1
            continue
        kept.append(row)
        by_answer[answer_type] += 1
        by_source[str(source.get("tag") or "unknown")] += 1
        if max_rows > 0 and len(kept) >= max_rows:
            break
    audit = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "skipped": dict(skipped),
        "kept_by_answer_type": dict(by_answer),
        "kept_by_source": dict(by_source),
        "require_patch_geometry": bool(require_patch_geometry),
        "allow_legacy_mca": bool(allow_legacy_mca),
        "scannet_frame_root": str(scannet_frame_root) if scannet_frame_root is not None else None,
        "max_visual_frames": int(max_visual_frames),
    }
    if not kept:
        raise ValueError(f"native row filter kept zero rows: {audit}")
    return kept, audit


def audit_reference_schema_rows(*, rows: Sequence[Mapping[str, Any]], reference_schema: str, split_name: str) -> Dict[str, Any]:
    """审计 qtype 规则路由后的 reference type 分布。

    这一步不跑模型，只看数据标签。我们要确认 simple3 真的把样本收束到三类，
    并知道每个题型被分到哪一类，避免悄悄把 VSI 尺度题路由到错误的人称 pose。
    """

    mapping = reference_type_mapping(reference_schema)
    routed_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    by_qtype: dict[str, Counter[str]] = {}
    for row in rows:
        question = require_mapping(row, "question")
        qtype = str(question.get("canonical_question_type") or "unknown")
        raw = str(question.get("reference_condition") or "missing")
        raw_counts[raw] += 1
        if reference_schema == "simple3":
            routed = infer_simple_reference_condition(row=row, question_payload=question)
        else:
            routed = raw if raw in mapping else "global_scene_reference"
        if routed not in mapping:
            raise ValueError(f"split={split_name} routed reference={routed!r} not in schema={reference_schema!r}")
        routed_counts[routed] += 1
        if qtype not in by_qtype:
            by_qtype[qtype] = Counter()
        by_qtype[qtype][routed] += 1
    return {
        "split": split_name,
        "reference_schema": str(reference_schema),
        "rows": len(rows),
        "raw_reference_condition_counts": dict(raw_counts),
        "routed_reference_type_counts": dict(routed_counts),
        "by_canonical_question_type": {key: dict(value) for key, value in sorted(by_qtype.items())},
    }


def audit_visual_cache_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    visual_cache_root: Path,
    visual_cache_roots: Optional[Sequence[Path]] = None,
    scannet_frame_root: Optional[Path],
    max_visual_frames: int,
    qwen_model_path: Path,
    max_pixels: int,
    split_name: str,
) -> Dict[str, Any]:
    """训练前严格检查 cache 覆盖率。

    cache 训练不能静默退回实时视觉塔；否则速度和实验设置都会改变。
    所以这里在 DataLoader 启动前就把缺失 cache 的样本找出来，缺一条就报错。
    """

    missing_identity: list[str] = []
    missing_file: list[dict[str, str]] = []
    covered = 0
    roots = normalize_visual_cache_roots(visual_cache_root, visual_cache_roots)
    if not roots:
        raise ValueError("audit_visual_cache_rows requires at least one visual cache root")
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        identity = cache_identity_from_row(
            row,
            scannet_frame_root=scannet_frame_root,
            max_visual_frames=max_visual_frames,
            qwen_model_path=qwen_model_path,
            max_pixels=max_pixels,
        )
        if identity is None:
            missing_identity.append(sample_id)
            continue
        key, _ = identity
        path = find_visual_cache_path(roots, key)
        if path is None:
            missing_file.append(
                {
                    "sample_id": sample_id,
                    "cache_key": key,
                    "probed_paths": [str(visual_cache_path_for_key(root, key)) for root in roots],
                }
            )
            continue
        covered += 1
    audit = {
        "split": split_name,
        "rows": len(rows),
        "covered": covered,
        "coverage": float(covered) / float(len(rows)) if rows else 0.0,
        "missing_identity_count": len(missing_identity),
        "missing_file_count": len(missing_file),
        "first_missing_identity": missing_identity[:20],
        "first_missing_file": missing_file[:20],
        "visual_cache_root": str(visual_cache_root),
        "visual_cache_roots": [str(root) for root in roots],
    }
    if missing_identity or missing_file:
        raise FileNotFoundError(f"visual cache audit failed for {split_name}: {audit}")
    return audit


class NativeGlowPatchSpatialCollator:
    """把 native-answer 样本打包成 Qwen 输入和 patch-level spatial descriptor。

    Qwen processor 决定最终视觉 token 网格大小；所以 patch 3D grid 必须在 collate 时
    按 image_grid_thw 精确重采样，保证每个 Qwen image token 都有一条对应的空间 descriptor。
    """

    def __init__(
        self,
        *,
        processor: Any,
        max_pixels: int,
        spatial_merge_size: int,
        bev_grid_size: int,
        reference_schema: str = "legacy5",
    ) -> None:
        if max_pixels <= 0:
            raise ValueError(f"max_pixels must be positive, got {max_pixels}")
        if spatial_merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
        if bev_grid_size <= 1:
            raise ValueError(f"bev_grid_size must be >1, got {bev_grid_size}")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_pixels = int(max_pixels)
        self.spatial_merge_size = int(spatial_merge_size)
        self.bev_grid_size = int(bev_grid_size)
        self.reference_schema = str(reference_schema)
        self.reference_type_to_id = reference_type_mapping(self.reference_schema)

    def __call__(self, samples: Sequence[NativeGlowSample]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("collator samples must not be empty")
        encoded = self._encode_prompt_batch(samples=samples)
        batch = self._append_answer_span(encoded=encoded, samples=samples)
        spatial_payload = self._build_spatial_payload(samples=samples, image_grid_thw=batch["image_grid_thw"])
        for key, value in spatial_payload.items():
            if key in batch:
                raise KeyError(f"spatial payload collides with Qwen key: {key}")
            batch[key] = value
        self._append_cached_visual_payload(batch=batch, samples=samples)
        batch["sample_ids"] = [sample.sample_id for sample in samples]
        batch["source_tags"] = [sample.source_tag for sample in samples]
        batch["answer_types"] = [sample.answer_type for sample in samples]
        batch["target_texts"] = [sample.target_text for sample in samples]
        batch["prompt_texts"] = [sample.prompt for sample in samples]
        return batch

    def collate_generation_prompts(self, samples: Sequence[NativeGlowSample]) -> Dict[str, Any]:
        """只打包 prompt，不拼接 gold answer。

        真实评测和 RL rollout 必须让模型自己生成答案，不能读取标准答案。
        但空间 descriptor 仍然要和训练时完全同一套逻辑构造，避免 train/eval 输入不一致。
        """

        if not samples:
            raise ValueError("generation collator samples must not be empty")
        batch = self._encode_prompt_batch(samples=samples)
        spatial_payload = self._build_spatial_payload(samples=samples, image_grid_thw=batch["image_grid_thw"])
        for key, value in spatial_payload.items():
            if key in batch:
                raise KeyError(f"spatial payload collides with Qwen key during generation: {key}")
            batch[key] = value
        self._append_cached_visual_payload(batch=batch, samples=samples)
        batch["sample_ids"] = [sample.sample_id for sample in samples]
        batch["source_tags"] = [sample.source_tag for sample in samples]
        batch["answer_types"] = [sample.answer_type for sample in samples]
        batch["target_texts"] = [sample.target_text for sample in samples]
        batch["prompt_texts"] = [sample.prompt for sample in samples]
        return batch

    def _encode_prompt_batch(self, *, samples: Sequence[NativeGlowSample]) -> Dict[str, Any]:
        cached_flags = [sample.cached_image_embeds is not None for sample in samples]
        if any(cached_flags) and not all(cached_flags):
            raise ValueError("cannot mix cached and uncached visual samples in one batch")
        if all(cached_flags):
            return self._encode_prompt_batch_from_visual_cache(samples=samples)

        messages_batch = []
        for sample in samples:
            content = []
            for image_path in sample.image_paths:
                content.append({"type": "image", "image": image_path, "max_pixels": self.max_pixels})
            content.append({"type": "text", "text": sample.prompt})
            messages_batch.append([{"role": "user", "content": content}])
        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        image_inputs, video_inputs = process_vision_inputs(messages_batch)
        if video_inputs is not None:
            raise ValueError("QUASAR patch spatial trainer expects image inputs only, got video_inputs")
        encoded = dict(self.processor(text=texts, images=image_inputs, padding=True, return_tensors="pt"))
        if "mm_token_type_ids" not in encoded:
            raise KeyError("Qwen3VL processor did not return mm_token_type_ids; cannot preserve native M-RoPE safely")
        if "image_grid_thw" not in encoded:
            raise KeyError("Qwen3VL processor did not return image_grid_thw")
        return encoded

    def _encode_prompt_batch_from_visual_cache(self, *, samples: Sequence[NativeGlowSample]) -> Dict[str, Any]:
        """不用读图，直接用 cache 的 image_grid_thw 展开 Qwen 图像占位符。

        Qwen processor 平时会先读图片，再根据图片大小把一个 <|image_pad|> 扩成很多个
        image token。我们已经提前缓存了 image_grid_thw，所以这里手工做同一件事，避免训练时重复跑
        image processor / vision tower。
        """

        texts: list[str] = []
        image_grid_chunks: list[torch.Tensor] = []
        for sample in samples:
            if sample.cached_image_grid_thw is None:
                raise ValueError(f"sample_id={sample.sample_id} missing cached_image_grid_thw")
            content = []
            for image_path in sample.image_paths:
                content.append({"type": "image", "image": image_path, "max_pixels": self.max_pixels})
            content.append({"type": "text", "text": sample.prompt})
            messages = [{"role": "user", "content": content}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            image_grid_chunks.append(sample.cached_image_grid_thw)
        image_grid_thw = torch.cat(image_grid_chunks, dim=0).long()
        merge_length = int(self.spatial_merge_size) ** 2
        image_cursor = 0
        expanded_texts: list[str] = []
        for text in texts:
            expanded = str(text)
            while self.processor.image_token in expanded:
                if image_cursor >= int(image_grid_thw.shape[0]):
                    raise ValueError("cached prompt has more image placeholders than cached image_grid_thw rows")
                num_image_tokens = int(image_grid_thw[image_cursor].prod().item()) // merge_length
                expanded = expanded.replace(self.processor.image_token, "<|placeholder|>" * num_image_tokens, 1)
                image_cursor += 1
            expanded_texts.append(expanded.replace("<|placeholder|>", self.processor.image_token))
        if image_cursor != int(image_grid_thw.shape[0]):
            raise ValueError(
                f"cached prompt/image_grid mismatch: used {image_cursor} image placeholders, "
                f"but cache has {int(image_grid_thw.shape[0])} grids"
            )
        text_inputs = self.tokenizer(expanded_texts, padding=True, return_attention_mask=True, return_tensors="pt")
        input_ids = text_inputs["input_ids"]
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == int(self.processor.image_token_id)] = 1
        image_token_count = int((input_ids == int(self.processor.image_token_id)).sum().item())
        expected_tokens = int((image_grid_thw.prod(dim=1) // merge_length).sum().item())
        if image_token_count != expected_tokens:
            raise ValueError(f"cached image token count mismatch: text={image_token_count}, grid={expected_tokens}")
        return {
            "input_ids": input_ids,
            "attention_mask": text_inputs["attention_mask"],
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": image_grid_thw,
        }

    def _append_answer_span(self, *, encoded: Mapping[str, torch.Tensor], samples: Sequence[NativeGlowSample]) -> Dict[str, Any]:
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        mm_token_type_ids = encoded["mm_token_type_ids"]
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                f"Qwen text tensors must be [B,L] with same shape, got input={tuple(input_ids.shape)}, "
                f"attention={tuple(attention_mask.shape)}, mm={tuple(mm_token_type_ids.shape)}"
            )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Qwen tokenizer pad_token_id must be set")
        eos_id = self.tokenizer.eos_token_id
        answer_token_lists: list[list[int]] = []
        for sample in samples:
            ids = self.tokenizer(sample.target_text, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"sample_id={sample.sample_id} target text tokenized to empty ids: {sample.target_text!r}")
            if eos_id is not None:
                ids = list(ids) + [int(eos_id)]
            answer_token_lists.append([int(x) for x in ids])

        prompt_lens = attention_mask.sum(dim=1).to(torch.long)
        max_len = max(int(prompt_lens[i].item()) + len(answer_token_lists[i]) for i in range(len(samples)))
        new_input = torch.full((len(samples), max_len), int(pad_id), dtype=torch.long)
        new_attention = torch.zeros((len(samples), max_len), dtype=torch.long)
        new_mm = torch.zeros((len(samples), max_len), dtype=mm_token_type_ids.dtype)
        labels = torch.full((len(samples), max_len), -100, dtype=torch.long)
        answer_span_starts: list[int] = []
        answer_span_lengths: list[int] = []
        for i, ids in enumerate(answer_token_lists):
            prompt_len = int(prompt_lens[i].item())
            answer_len = len(ids)
            new_input[i, :prompt_len] = input_ids[i, :prompt_len]
            new_attention[i, : prompt_len + answer_len] = 1
            new_mm[i, :prompt_len] = mm_token_type_ids[i, :prompt_len]
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            new_input[i, prompt_len : prompt_len + answer_len] = ids_tensor
            labels[i, prompt_len : prompt_len + answer_len] = ids_tensor
            answer_span_starts.append(prompt_len)
            answer_span_lengths.append(answer_len)
        batch: Dict[str, Any] = {key: value for key, value in encoded.items() if key not in {"input_ids", "attention_mask", "mm_token_type_ids"}}
        batch["input_ids"] = new_input
        batch["attention_mask"] = new_attention
        batch["mm_token_type_ids"] = new_mm
        batch["labels"] = labels
        batch["answer_span_starts"] = torch.tensor(answer_span_starts, dtype=torch.long)
        batch["answer_span_lengths"] = torch.tensor(answer_span_lengths, dtype=torch.long)
        return batch

    def _build_spatial_payload(self, *, samples: Sequence[NativeGlowSample], image_grid_thw: torch.Tensor) -> Dict[str, torch.Tensor]:
        expected_images = sum(len(sample.image_paths) for sample in samples)
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[0] != expected_images or image_grid_thw.shape[1] != 3:
            raise ValueError(
                f"image_grid_thw must be [num_images,3] matching sample image count, got {tuple(image_grid_thw.shape)} "
                f"expected_images={expected_images}"
            )
        descriptor_chunks: list[torch.Tensor] = []
        token_sample_ids: list[torch.Tensor] = []
        token_points: list[torch.Tensor] = []
        token_valid: list[torch.Tensor] = []
        query_frame_coord_targets: list[torch.Tensor] = []
        bev_coord_targets: list[torch.Tensor] = []
        target_camera_token_masks: list[torch.Tensor] = []
        semantic_query_token_masks: list[torch.Tensor] = []
        canonical_bev_token_masks: list[torch.Tensor] = []
        bev_targets: list[torch.Tensor] = []
        object_center_targets: list[torch.Tensor] = []
        object_center_valids: list[float] = []
        query_pose_vecs: list[torch.Tensor] = []
        query_pose_valids: list[float] = []
        ref_type_ids: list[int] = []
        image_cursor = 0
        target_camera_id = self.reference_type_to_id["target_camera_pose"]
        semantic_query_id = self.reference_type_to_id["semantic_query_pose"]
        canonical_bev_id = self.reference_type_to_id["canonical_bev_scene_pose"]
        for sample_i, sample in enumerate(samples):
            if sample.aux_bev_center is not None or sample.aux_bev_scale is not None:
                if sample.aux_bev_center is None or sample.aux_bev_scale is None:
                    raise ValueError(f"sample_id={sample.sample_id} aux_bev_center/aux_bev_scale must appear together")
                center = sample.aux_bev_center.float()
                scale = sample.aux_bev_scale.float()
            else:
                center, scale = compute_scene_center_scale(points=sample.points, valid=sample.patch_valid)
            bev_targets.append(build_bev_occupancy(points=sample.points, valid=sample.patch_valid, center=center, scale=scale, grid_size=self.bev_grid_size))
            if sample.aux_object_centers is not None:
                object_center_target = build_object_center_heatmap(
                    object_centers=sample.aux_object_centers,
                    scene_center=center,
                    scene_scale=scale,
                    grid_size=self.bev_grid_size,
                    sample_id=sample.sample_id,
                )
                object_center_targets.append(object_center_target)
                # 少量 repaired full 样本的 object center 会落在当前 BEV crop 外。
                # 这些样本仍可用于主答案和其它空间监督，只是不该参与 object-center heatmap loss。
                object_center_valids.append(1.0 if bool(object_center_target.any().item()) else 0.0)
            else:
                object_center_targets.append(torch.zeros((self.bev_grid_size, self.bev_grid_size), dtype=torch.float32))
                object_center_valids.append(0.0)
            query_pose_vecs.append(pose_to_vec12(sample.query_pose))
            query_pose_valids.append(1.0 if sample.query_pose_valid else 0.0)
            ref_type_ids.append(int(sample.reference_type_id))
            for frame_i in range(len(sample.image_paths)):
                thw = image_grid_thw[image_cursor]
                if int(thw[0].item()) != 1:
                    raise ValueError(f"sample_id={sample.sample_id} expected image grid T=1, got {thw.tolist()}")
                llm_h = int(thw[1].item()) // self.spatial_merge_size
                llm_w = int(thw[2].item()) // self.spatial_merge_size
                if llm_h <= 0 or llm_w <= 0:
                    raise ValueError(f"sample_id={sample.sample_id} invalid LLM visual grid from thw={thw.tolist()}")
                frame_points = resize_points_grid(sample.points[frame_i], out_hw=(llm_h, llm_w))
                frame_valid = resize_scalar_grid(sample.patch_valid[frame_i].float(), out_hw=(llm_h, llm_w), mode="nearest") > 0.5
                frame_conf = resize_scalar_grid(sample.patch_confidence[frame_i].float(), out_hw=(llm_h, llm_w), mode="bilinear")
                query_frame_coord = transform_world_points_to_camera(points=frame_points, camera_pose=sample.query_pose, scene_scale=scale)
                bev_coord = build_bev_coord_targets(points=frame_points, center=center, scale=scale)
                desc = build_patch_descriptor(
                    points=frame_points,
                    valid=frame_valid,
                    confidence=frame_conf,
                    source_pose=sample.source_poses[frame_i],
                    query_pose=sample.query_pose,
                    scene_center=center,
                    scene_scale=scale,
                )
                flat_desc = desc.reshape(llm_h * llm_w, SPATIAL_DESCRIPTOR_DIM)
                flat_points = frame_points.reshape(llm_h * llm_w, 3)
                flat_valid = frame_valid.reshape(llm_h * llm_w)
                flat_query_frame_coord = query_frame_coord.reshape(llm_h * llm_w, 3)
                flat_bev_coord = bev_coord.reshape(llm_h * llm_w, 2)
                descriptor_chunks.append(flat_desc)
                token_points.append(flat_points)
                token_valid.append(flat_valid)
                query_frame_coord_targets.append(flat_query_frame_coord)
                bev_coord_targets.append(flat_bev_coord)
                ref_id = int(sample.reference_type_id)
                query_valid = bool(sample.query_pose_valid)
                target_camera_token_masks.append(flat_valid & bool(ref_id == target_camera_id and query_valid))
                semantic_query_token_masks.append(flat_valid & bool(ref_id == semantic_query_id and query_valid))
                canonical_bev_token_masks.append(flat_valid & bool(ref_id == canonical_bev_id))
                token_sample_ids.append(torch.full((llm_h * llm_w,), sample_i, dtype=torch.long))
                image_cursor += 1
        descriptors = torch.cat(descriptor_chunks, dim=0).float()
        points = torch.cat(token_points, dim=0).float()
        valids = torch.cat(token_valid, dim=0).bool()
        query_frame_targets = torch.cat(query_frame_coord_targets, dim=0).float()
        bev_coord_targets_tensor = torch.cat(bev_coord_targets, dim=0).float()
        target_camera_masks = torch.cat(target_camera_token_masks, dim=0).bool()
        semantic_query_masks = torch.cat(semantic_query_token_masks, dim=0).bool()
        canonical_bev_masks = torch.cat(canonical_bev_token_masks, dim=0).bool()
        sample_ids = torch.cat(token_sample_ids, dim=0).long()
        expected_tokens = int((image_grid_thw.prod(dim=1) // (self.spatial_merge_size ** 2)).sum().item())
        if descriptors.shape[0] != expected_tokens:
            raise ValueError(f"spatial descriptor token count mismatch: {descriptors.shape[0]} vs expected {expected_tokens}")
        return {
            "spatial_descriptors": descriptors,
            "spatial_token_sample_ids": sample_ids,
            "spatial_points": points,
            "spatial_token_valid": valids,
            "query_frame_coord_targets": query_frame_targets,
            "bev_coord_targets": bev_coord_targets_tensor,
            "target_camera_token_mask": target_camera_masks,
            "semantic_query_token_mask": semantic_query_masks,
            "canonical_bev_token_mask": canonical_bev_masks,
            "reference_type_ids": torch.tensor(ref_type_ids, dtype=torch.long),
            "query_pose_vec12": torch.stack(query_pose_vecs, dim=0).float(),
            "query_pose_valid": torch.tensor(query_pose_valids, dtype=torch.float32),
            "bev_targets": torch.stack(bev_targets, dim=0).float(),
            "object_center_targets": torch.stack(object_center_targets, dim=0).float(),
            "object_center_valid": torch.tensor(object_center_valids, dtype=torch.float32),
        }

    def _append_cached_visual_payload(self, *, batch: Dict[str, Any], samples: Sequence[NativeGlowSample]) -> None:
        cached_flags = [sample.cached_image_embeds is not None for sample in samples]
        if not any(cached_flags):
            return
        if not all(cached_flags):
            raise ValueError("cannot mix cached and uncached visual samples in one batch")
        image_embeds_chunks: list[torch.Tensor] = []
        deepstack_layers: list[list[torch.Tensor]] = []
        cache_keys: list[str] = []
        for sample in samples:
            if sample.cached_image_embeds is None or sample.cached_deepstack_features is None or sample.visual_cache_key is None:
                raise ValueError(f"sample_id={sample.sample_id} has incomplete visual cache payload")
            image_embeds_chunks.append(sample.cached_image_embeds)
            cache_keys.append(sample.visual_cache_key)
            if not deepstack_layers:
                deepstack_layers = [[] for _ in sample.cached_deepstack_features]
            if len(sample.cached_deepstack_features) != len(deepstack_layers):
                raise ValueError(f"sample_id={sample.sample_id} cached deepstack layer count mismatch")
            for layer_i, deep in enumerate(sample.cached_deepstack_features):
                deepstack_layers[layer_i].append(deep)
        batch["cached_image_embeds"] = torch.cat(image_embeds_chunks, dim=0)
        batch["cached_deepstack_features"] = [torch.cat(layer, dim=0) for layer in deepstack_layers]
        batch["visual_cache_keys"] = cache_keys


def process_vision_inputs(messages_batch: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[Any, Any]:
    from qwen_vl_utils import process_vision_info

    outputs = process_vision_info(messages_batch)
    if len(outputs) != 2:
        raise ValueError(f"expected process_vision_info to return 2 values, got {len(outputs)}")
    return outputs[0], outputs[1]


def resize_points_grid(points: torch.Tensor, *, out_hw: tuple[int, int]) -> torch.Tensor:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points grid must be [H,W,3], got {tuple(points.shape)}")
    tensor = points.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=out_hw, mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).contiguous()


def resize_scalar_grid(values: torch.Tensor, *, out_hw: tuple[int, int], mode: str) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"scalar grid must be [H,W], got {tuple(values.shape)}")
    tensor = values.view(1, 1, values.shape[0], values.shape[1]).float()
    if mode == "nearest":
        resized = F.interpolate(tensor, size=out_hw, mode="nearest")
    elif mode == "bilinear":
        resized = F.interpolate(tensor, size=out_hw, mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"unsupported resize mode={mode}")
    return resized.view(out_hw[0], out_hw[1])


def compute_scene_center_scale(*, points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid_points = points[valid]
    if valid_points.numel() == 0:
        raise ValueError("cannot compute scene center/scale from zero valid points")
    center = valid_points.float().mean(dim=0)
    span = (valid_points.float() - center).abs().amax(dim=0)
    scale = span.max().clamp_min(torch.tensor(1.0, dtype=torch.float32))
    return center.float(), scale.float()


def pose_to_vec12(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be 4x4, got {tuple(pose.shape)}")
    return pose[:3, :4].reshape(12).float()


def build_patch_descriptor(
    *,
    points: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    source_pose: torch.Tensor,
    query_pose: torch.Tensor,
    scene_center: torch.Tensor,
    scene_scale: torch.Tensor,
) -> torch.Tensor:
    h, w = points.shape[:2]
    if source_pose.shape != (4, 4) or query_pose.shape != (4, 4):
        raise ValueError(f"source/query pose must be 4x4, got {tuple(source_pose.shape)} and {tuple(query_pose.shape)}")
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, w, dtype=torch.float32),
        indexing="ij",
    )
    source_t = source_pose[:3, 3].float()
    query_t = query_pose[:3, 3].float()
    source_forward = source_pose[:3, 2].float()
    source_right = source_pose[:3, 0].float()
    source_up = source_pose[:3, 1].float()
    norm_points = (points.float() - scene_center.view(1, 1, 3)) / scene_scale
    rel_source = (points.float() - source_t.view(1, 1, 3)) / scene_scale
    rel_query = (points.float() - query_t.view(1, 1, 3)) / scene_scale
    cam_delta = (source_t - query_t).view(1, 1, 3).expand(h, w, 3) / scene_scale
    descriptor = torch.cat(
        [
            norm_points,
            rel_source,
            rel_query,
            cam_delta,
            source_t.view(1, 1, 3).expand(h, w, 3) / scene_scale,
            source_forward.view(1, 1, 3).expand(h, w, 3),
            source_right.view(1, 1, 3).expand(h, w, 3),
            source_up.view(1, 1, 3).expand(h, w, 3),
            confidence.float().view(h, w, 1),
            valid.float().view(h, w, 1),
            xx.view(h, w, 1),
            yy.view(h, w, 1),
        ],
        dim=-1,
    )
    if descriptor.shape[-1] != SPATIAL_DESCRIPTOR_DIM:
        raise RuntimeError(f"descriptor dim mismatch: {descriptor.shape[-1]} vs {SPATIAL_DESCRIPTOR_DIM}")
    return descriptor


def transform_world_points_to_camera(*, points: torch.Tensor, camera_pose: torch.Tensor, scene_scale: torch.Tensor) -> torch.Tensor:
    """把世界坐标点转成某个 reference camera 的局部坐标。

    如果我们声称 source patch 被 flow 到 target/semantic pose 下，那监督信号就不能只看
    世界坐标差值；至少要让它知道“在那个相机自己看来，这个点在前后左右上下哪里”。
    """

    if camera_pose.shape != (4, 4):
        raise ValueError(f"camera_pose must be 4x4, got {tuple(camera_pose.shape)}")
    rotation = camera_pose[:3, :3].float()
    translation = camera_pose[:3, 3].float()
    local = torch.matmul(points.float() - translation.view(1, 1, 3), rotation)
    return local / scene_scale


def build_bev_coord_targets(*, points: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """给每个 patch 点生成连续 BEV 坐标 target。

    occupancy 只告诉模型“地图哪里有东西”，不够精确；这个 target 直接要求每个
    patch-flow latent 能落回鸟瞰图里的具体二维坐标。
    """

    xy = (points.float()[..., [0, 2]] - center[[0, 2]].view(1, 1, 2)) / scale
    return xy


def build_bev_occupancy(*, points: torch.Tensor, valid: torch.Tensor, center: torch.Tensor, scale: torch.Tensor, grid_size: int) -> torch.Tensor:
    valid_points = points[valid].float()
    if valid_points.numel() == 0:
        raise ValueError("cannot build BEV from zero valid points")
    xy = (valid_points[:, [0, 2]] - center[[0, 2]].view(1, 2)) / scale
    uv = ((xy + 1.0) * 0.5 * (grid_size - 1)).round().to(torch.long)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < grid_size) & (uv[:, 1] >= 0) & (uv[:, 1] < grid_size)
    bev = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    uv = uv[inside]
    if uv.numel() > 0:
        bev[uv[:, 1], uv[:, 0]] = 1.0
    return bev


def build_object_center_heatmap(
    *,
    object_centers: torch.Tensor,
    scene_center: torch.Tensor,
    scene_scale: torch.Tensor,
    grid_size: int,
    sample_id: str,
) -> torch.Tensor:
    """把 object center 变成 BEV 网格监督。

    occupancy 只告诉模型“哪里有几何点”，object center 进一步告诉模型“物体中心在哪里”。
    这一步直接服务 canonical BEV reference，不让 object-center sidecar 只停留在数据字段里。
    """

    if object_centers.ndim != 2 or object_centers.shape[-1] != 3:
        raise ValueError(f"sample_id={sample_id} object_centers must be [K,3], got {tuple(object_centers.shape)}")
    if int(object_centers.shape[0]) <= 0:
        raise ValueError(f"sample_id={sample_id} object_centers is empty")
    xy = (object_centers.float()[:, [0, 2]] - scene_center[[0, 2]].view(1, 2)) / scene_scale.float()
    uv = ((xy + 1.0) * 0.5 * (grid_size - 1)).round().to(torch.long)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < grid_size) & (uv[:, 1] >= 0) & (uv[:, 1] < grid_size)
    heatmap = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    if int(inside.sum().item()) <= 0:
        # 这类样本的 object center sidecar 有值，但按当前 BEV crop 投影后完全落到网格外。
        # 这不是“空物体图”的正样本，而是 object-center 热图监督不可用；
        # 因此返回全零，让 collator 把 object_center_valid 置 0，主 answer/geometry 训练继续。
        return heatmap
    uv = uv[inside]
    heatmap[uv[:, 1], uv[:, 0]] = 1.0
    return heatmap


class QueryReferenceSpatialLatentFlow(nn.Module):
    """从 source patch spatial code 流到 query/reference-conditioned spatial code。

    这个模块只处理“空间位置编码”，不接管 Qwen 的语言推理。
    它根据 patch 的 3D 点、source pose 和题目参考系，输出同维度的视觉 token bias。
    """

    def __init__(
        self,
        *,
        descriptor_dim: int,
        qwen_hidden_size: int,
        flow_dim: int,
        num_flow_steps: int,
        dropout: float,
        bev_grid_size: int,
        reference_type_count: int,
        reference_schema: str,
    ) -> None:
        super().__init__()
        if num_flow_steps <= 0:
            raise ValueError(f"num_flow_steps must be positive, got {num_flow_steps}")
        self.descriptor_dim = int(descriptor_dim)
        self.qwen_hidden_size = int(qwen_hidden_size)
        self.flow_dim = int(flow_dim)
        self.num_flow_steps = int(num_flow_steps)
        self.bev_grid_size = int(bev_grid_size)
        self.reference_type_count = int(reference_type_count)
        self.reference_schema = str(reference_schema)
        if self.reference_type_count <= 0:
            raise ValueError(f"reference_type_count must be positive, got {reference_type_count}")
        self.reference_type_embedding = nn.Embedding(self.reference_type_count, flow_dim)
        self.query_pose_proj = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, flow_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flow_dim, flow_dim),
        )
        self.source_proj = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, flow_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flow_dim, flow_dim),
        )
        self.flow_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(flow_dim * 4 + descriptor_dim),
                    nn.Linear(flow_dim * 4 + descriptor_dim, flow_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(flow_dim * 2, flow_dim),
                )
                for _ in range(num_flow_steps)
            ]
        )
        self.output_norm = nn.LayerNorm(flow_dim)
        self.to_qwen_bias = nn.Linear(flow_dim, qwen_hidden_size)
        self.ref_type_head = nn.Sequential(
            nn.LayerNorm(flow_dim),
            nn.Linear(flow_dim, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, self.reference_type_count),
        )
        self.query_pose_head = nn.Sequential(
            nn.LayerNorm(flow_dim),
            nn.Linear(flow_dim, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, 12),
        )
        self.bev_head = nn.Sequential(
            nn.LayerNorm(flow_dim),
            nn.Linear(flow_dim, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, bev_grid_size * bev_grid_size),
        )
        self.object_center_head = nn.Sequential(
            nn.LayerNorm(flow_dim),
            nn.Linear(flow_dim, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, bev_grid_size * bev_grid_size),
        )
        self.patch_recon_head = nn.Sequential(
            nn.LayerNorm(flow_dim),
            nn.Linear(flow_dim, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, 3),
        )
        self.query_frame_coord_head = nn.Sequential(
            nn.LayerNorm(flow_dim),
            nn.Linear(flow_dim, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, 3),
        )
        self.bev_coord_head = nn.Sequential(
            nn.LayerNorm(flow_dim),
            nn.Linear(flow_dim, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, 2),
        )

    def forward(
        self,
        *,
        descriptors: torch.Tensor,
        token_sample_ids: torch.Tensor,
        reference_type_ids: torch.Tensor,
        query_pose_vec12: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        if descriptors.ndim != 2 or descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError(f"descriptors must be [N,{self.descriptor_dim}], got {tuple(descriptors.shape)}")
        if token_sample_ids.shape != (descriptors.shape[0],):
            raise ValueError(f"token_sample_ids shape mismatch: {tuple(token_sample_ids.shape)}")
        ref_emb = self.reference_type_embedding(reference_type_ids)
        pose_emb = self.query_pose_proj(query_pose_vec12)
        sample_ref = ref_emb + pose_emb
        token_ref = sample_ref[token_sample_ids]
        z = self.source_proj(descriptors)
        for block in self.flow_blocks:
            # 像 flow matching 的一步步“从 source 空间码走向 query/reference 空间码”。
            # 但它只更新空间 code，不更新 Qwen 的文本 hidden state。
            dz = block(torch.cat([z, token_ref, token_ref - z, z * token_ref, descriptors], dim=-1))
            z = z + dz / float(self.num_flow_steps)
        z = self.output_norm(z)
        spatial_bias = self.to_qwen_bias(z)
        pooled = scatter_mean(z, token_sample_ids, batch_size=batch_size)
        return {
            "spatial_bias": spatial_bias,
            "flow_code": z,
            "sample_flow_pooled": pooled,
            "ref_type_logits": self.ref_type_head(pooled),
            "query_pose_pred": self.query_pose_head(pooled),
            "bev_logits": self.bev_head(pooled).view(batch_size, self.bev_grid_size, self.bev_grid_size),
            "object_center_logits": self.object_center_head(pooled).view(batch_size, self.bev_grid_size, self.bev_grid_size),
            "patch_recon": self.patch_recon_head(z),
            "query_frame_coord_pred": self.query_frame_coord_head(z),
            "bev_coord_pred": self.bev_coord_head(z),
        }


def scatter_mean(values: torch.Tensor, sample_ids: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"scatter_mean values must be [N,D], got {tuple(values.shape)}")
    out = values.new_zeros((batch_size, values.shape[-1]))
    counts = values.new_zeros((batch_size, 1))
    out.index_add_(0, sample_ids, values)
    counts.index_add_(0, sample_ids, torch.ones((values.shape[0], 1), device=values.device, dtype=values.dtype))
    return out / counts.clamp_min(1.0)


class NativeQwenGlowPatchSpatialModel(nn.Module):
    """Qwen3VL + patch-level spatial positional injection。

    关键原则：Qwen 的自回归 language_model 不被替换；我们只复刻 Qwen3VLModel.forward 中
    image_embeds masked_scatter 之前的几行，在那里把 spatial_bias 加进 image patch features。
    """

    def __init__(
        self,
        *,
        qwen: nn.Module,
        tokenizer: Any,
        qwen_hidden_size: int,
        flow_dim: int,
        num_flow_steps: int,
        dropout: float,
        bev_grid_size: int,
        reference_schema: str,
        aux_weights: Mapping[str, float],
        disable_spatial_flow_bias: bool = False,
    ) -> None:
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.qwen_hidden_size = int(qwen_hidden_size)
        self.reference_schema = str(reference_schema)
        self.reference_type_to_id = reference_type_mapping(self.reference_schema)
        self.spatial_flow = QueryReferenceSpatialLatentFlow(
            descriptor_dim=SPATIAL_DESCRIPTOR_DIM,
            qwen_hidden_size=qwen_hidden_size,
            flow_dim=flow_dim,
            num_flow_steps=num_flow_steps,
            dropout=dropout,
            bev_grid_size=bev_grid_size,
            reference_type_count=len(self.reference_type_to_id),
            reference_schema=self.reference_schema,
        )
        self.aux_weights = {key: float(value) for key, value in aux_weights.items()}
        self.disable_spatial_flow_bias = bool(disable_spatial_flow_bias)
        self.last_forward_audit: Dict[str, Any] = {}

    def base_qwen(self) -> nn.Module:
        if hasattr(self.qwen, "get_base_model"):
            return self.qwen.get_base_model()
        return self.qwen

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        image_grid_thw = batch["image_grid_thw"]
        mm_token_type_ids = batch["mm_token_type_ids"]
        batch_size = int(input_ids.shape[0])

        flow_out = self.spatial_flow(
            descriptors=batch["spatial_descriptors"].float(),
            token_sample_ids=batch["spatial_token_sample_ids"].long(),
            reference_type_ids=batch["reference_type_ids"].long(),
            query_pose_vec12=batch["query_pose_vec12"].float(),
            batch_size=batch_size,
        )
        raw_spatial_bias = flow_out["spatial_bias"]
        spatial_bias = self.spatial_bias_for_qwen(raw_spatial_bias)
        qwen_out = self._qwen_forward_with_spatial_bias(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            spatial_bias=spatial_bias,
            cached_image_embeds=batch.get("cached_image_embeds"),
            cached_deepstack_features=batch.get("cached_deepstack_features"),
        )
        logits = qwen_out["logits"]
        answer_ce = compute_answer_span_ce(logits=logits, labels=labels)
        aux_losses = self._compute_aux_losses(batch=batch, flow_out=flow_out)
        total_loss = answer_ce
        for key, value in aux_losses.items():
            total_loss = total_loss + value
        self.last_forward_audit = {
            "image_feature_tokens": int(qwen_out["image_feature_tokens"].detach().cpu().item()),
            "visual_cache_used": bool(int(qwen_out["visual_cache_used"].detach().cpu().item())),
            "spatial_bias_tokens": int(spatial_bias.shape[0]),
            "disable_spatial_flow_bias": bool(self.disable_spatial_flow_bias),
            "raw_spatial_bias_abs_mean": float(raw_spatial_bias.detach().float().abs().mean().cpu().item()),
            "spatial_bias_abs_mean": float(spatial_bias.detach().float().abs().mean().cpu().item()),
            "answer_label_tokens": int((labels != -100).sum().detach().cpu().item()),
            "reference_schema": self.reference_schema,
            "target_camera_aux_tokens": int(batch["target_camera_token_mask"].detach().sum().cpu().item()),
            "semantic_query_aux_tokens": int(batch["semantic_query_token_mask"].detach().sum().cpu().item()),
            "canonical_bev_aux_tokens": int(batch["canonical_bev_token_mask"].detach().sum().cpu().item()),
            "object_center_aux_samples": int(batch["object_center_valid"].detach().sum().cpu().item()),
        }
        output = {
            "loss": total_loss,
            "answer_ce_loss": answer_ce.detach(),
            "logits": logits,
            "spatial_bias": spatial_bias,
            "raw_spatial_bias": raw_spatial_bias,
        }
        for key, value in aux_losses.items():
            output[key] = value.detach()
        return output

    def spatial_bias_for_qwen(self, spatial_bias: torch.Tensor) -> torch.Tensor:
        """返回真正注入 Qwen 视觉 token 的 bias。

        NoFlow 消融不是把整套几何辅助任务删掉，而是只拔掉“把 latent flow
        加进 Qwen 视觉特征”这根线。这样 BaseSpatial 和 BaseSpatial-NoFlow 的差别
        就集中在 latent flow injection 本身，辅助 loss、数据和 LoRA 仍保持一致。
        """

        if self.disable_spatial_flow_bias:
            return spatial_bias.detach().new_zeros(spatial_bias.shape)
        return spatial_bias

    def _qwen_forward_with_spatial_bias(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        spatial_bias: torch.Tensor,
        cached_image_embeds: Optional[torch.Tensor] = None,
        cached_deepstack_features: Optional[Sequence[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        base = self.base_qwen()
        qwen_model = base.model
        inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
        if cached_image_embeds is None:
            if pixel_values is None:
                raise ValueError("pixel_values is required when cached_image_embeds is not provided")
            image_outputs = qwen_model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_source = list(image_outputs.deepstack_features)
            visual_cache_used = False
        else:
            if cached_deepstack_features is None:
                raise ValueError("cached_deepstack_features is required when cached_image_embeds is provided")
            image_embeds = cached_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_source = list(cached_deepstack_features)
            visual_cache_used = True
        if image_embeds.shape != spatial_bias.shape:
            raise ValueError(
                f"spatial bias must match Qwen image embeds exactly, got bias={tuple(spatial_bias.shape)} "
                f"image_embeds={tuple(image_embeds.shape)}"
            )
        image_embeds = image_embeds + spatial_bias.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_mask_1d = image_mask[..., 0]
        deepstack_visual_embeds = []
        for layer_i, deep in enumerate(deepstack_source):
            if deep.shape != spatial_bias.shape:
                raise ValueError(
                    f"deepstack layer {layer_i} shape must match spatial bias for rigorous injection, "
                    f"got deep={tuple(deep.shape)} bias={tuple(spatial_bias.shape)}"
                )
            deepstack_visual_embeds.append(deep.to(inputs_embeds.device, inputs_embeds.dtype) + spatial_bias.to(inputs_embeds.device, inputs_embeds.dtype))
        position_ids = qwen_model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        language_outputs = qwen_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask_1d,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=False,
        )
        hidden_states = language_outputs[0]
        logits = base.lm_head(hidden_states)
        return {
            "logits": logits,
            "image_feature_tokens": torch.tensor(image_embeds.shape[0], device=logits.device),
            "visual_cache_used": torch.tensor(1 if visual_cache_used else 0, device=logits.device),
        }

    def _compute_aux_losses(self, *, batch: Mapping[str, Any], flow_out: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        ref_weight = self.aux_weights.get("reference_type", 0.0)
        pose_weight = self.aux_weights.get("query_pose", 0.0)
        bev_weight = self.aux_weights.get("bev", 0.0)
        patch_weight = self.aux_weights.get("patch_consistency", 0.0)
        target_camera_weight = self.aux_weights.get("target_camera_coord", 0.0)
        semantic_query_weight = self.aux_weights.get("semantic_query_coord", 0.0)
        bev_coord_weight = self.aux_weights.get("bev_coord", 0.0)
        object_center_weight = self.aux_weights.get("object_center", 0.0)
        if ref_weight > 0.0:
            losses["loss_ref_type"] = F.cross_entropy(flow_out["ref_type_logits"].float(), batch["reference_type_ids"].long()) * ref_weight
        if pose_weight > 0.0:
            valid = batch["query_pose_valid"].float().view(-1, 1)
            if float(valid.sum().detach().cpu().item()) > 0.0:
                pose_l1 = (flow_out["query_pose_pred"].float() - batch["query_pose_vec12"].float()).abs() * valid
                losses["loss_query_pose"] = pose_l1.sum() / valid.sum().clamp_min(1.0) / 12.0 * pose_weight
            else:
                losses["loss_query_pose"] = flow_out["query_pose_pred"].float().sum() * 0.0
        if bev_weight > 0.0:
            losses["loss_bev"] = F.binary_cross_entropy_with_logits(
                flow_out["bev_logits"].float(), batch["bev_targets"].float()
            ) * bev_weight
        if object_center_weight > 0.0:
            valid = batch["object_center_valid"].float().view(-1, 1, 1)
            if float(valid.sum().detach().cpu().item()) > 0.0:
                # 只对确实有 object-center 监督的 canonical-BEV 样本算 loss。
                # 没有 object bank 的样本不假装有中心点，也不把空图当负样本压模型。
                object_center_bce = F.binary_cross_entropy_with_logits(
                    flow_out["object_center_logits"].float(),
                    batch["object_center_targets"].float(),
                    reduction="none",
                )
                grid_area = float(batch["object_center_targets"].shape[-1] * batch["object_center_targets"].shape[-2])
                losses["loss_object_center"] = (object_center_bce * valid).sum() / valid.sum().clamp_min(1.0) / grid_area * object_center_weight
            else:
                losses["loss_object_center"] = flow_out["object_center_logits"].float().sum() * 0.0
        if patch_weight > 0.0:
            target = batch["spatial_descriptors"].float()[:, 6:9]
            valid = batch["spatial_token_valid"].float().view(-1, 1)
            patch_l1 = (flow_out["patch_recon"].float() - target).abs() * valid
            losses["loss_patch_consistency"] = patch_l1.sum() / valid.sum().clamp_min(1.0) / 3.0 * patch_weight
        if target_camera_weight > 0.0:
            valid = batch["target_camera_token_mask"].float().view(-1, 1)
            if float(valid.sum().detach().cpu().item()) > 0.0:
                target_l1 = (flow_out["query_frame_coord_pred"].float() - batch["query_frame_coord_targets"].float()).abs() * valid
                losses["loss_target_camera_coord"] = target_l1.sum() / valid.sum().clamp_min(1.0) / 3.0 * target_camera_weight
            else:
                losses["loss_target_camera_coord"] = flow_out["query_frame_coord_pred"].float().sum() * 0.0
        if semantic_query_weight > 0.0:
            valid = batch["semantic_query_token_mask"].float().view(-1, 1)
            if float(valid.sum().detach().cpu().item()) > 0.0:
                semantic_l1 = (flow_out["query_frame_coord_pred"].float() - batch["query_frame_coord_targets"].float()).abs() * valid
                losses["loss_semantic_query_coord"] = semantic_l1.sum() / valid.sum().clamp_min(1.0) / 3.0 * semantic_query_weight
            else:
                losses["loss_semantic_query_coord"] = flow_out["query_frame_coord_pred"].float().sum() * 0.0
        if bev_coord_weight > 0.0:
            valid = batch["canonical_bev_token_mask"].float().view(-1, 1)
            if float(valid.sum().detach().cpu().item()) > 0.0:
                bev_l1 = (flow_out["bev_coord_pred"].float() - batch["bev_coord_targets"].float()).abs() * valid
                losses["loss_bev_coord"] = bev_l1.sum() / valid.sum().clamp_min(1.0) / 2.0 * bev_coord_weight
            else:
                losses["loss_bev_coord"] = flow_out["bev_coord_pred"].float().sum() * 0.0
        return losses


def compute_answer_span_ce(*, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"logits/labels shape mismatch: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits/labels sequence shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    if int((shift_labels != -100).sum().detach().cpu().item()) <= 0:
        raise ValueError("answer-span CE has zero supervised tokens after shift")
    return F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)


class LoRALinear(nn.Module):
    """项目内置 LoRA Linear。

    本项目运行环境不依赖 peft。这里不退回 frozen Qwen，也不改成 direct SFT，
    而是明确把目标 Linear 替换成标准 LoRA：base linear 冻结，只训练低秩 A/B。
    """

    def __init__(self, base: nn.Linear, *, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if bool(getattr(self, "disable_lora", False)):
            return base_out
        lora_out = self.lora_b(self.lora_a(self.dropout(x.float()))) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype, device=base_out.device)


def apply_lora_to_qwen_linears(
    module: nn.Module,
    *,
    target_modules: set[str],
    r: int,
    alpha: int,
    dropout: float,
    prefix: str = "",
) -> list[str]:
    replaced: list[str] = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and child_name in target_modules:
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
        else:
            replaced.extend(
                apply_lora_to_qwen_linears(
                    child,
                    target_modules=target_modules,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    prefix=full_name,
                )
            )
    return replaced


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, value in module.state_dict().items():
        if ".lora_a." in name or ".lora_b." in name:
            state[name] = value.detach().cpu()
    if not state:
        raise ValueError("LoRA state dict is empty; Qwen LoRA injection did not happen")
    return state


def load_lora_state_dict(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """把项目内置 LoRA 权重严格装回 Qwen。

    我们不用 peft adapter 文件，而是自己保存 LoRA A/B 矩阵。
    加载时每个 key 都必须真实存在、shape 必须一致；否则直接报错，避免静默变成未训练 Qwen。
    """

    if not state:
        raise ValueError("cannot load empty LoRA state dict")
    own_state = module.state_dict()
    expected_lora_keys = set(lora_state_dict(module).keys())
    provided_keys = set(str(key) for key in state.keys())
    missing = sorted(expected_lora_keys - provided_keys)
    unexpected = sorted(provided_keys - expected_lora_keys)
    if missing or unexpected:
        raise KeyError(f"LoRA checkpoint key mismatch: missing={missing[:20]} unexpected={unexpected[:20]}")
    for key, value in state.items():
        if key not in own_state:
            raise KeyError(f"LoRA checkpoint key not found in model: {key}")
        target = own_state[key]
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(f"LoRA checkpoint shape mismatch for {key}: ckpt={tuple(value.shape)} model={tuple(target.shape)}")
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def resolve_checkpoint_file(path: Path) -> Path:
    """接受 checkpoint 文件或目录，统一解析到 glow_patch_spatial.pt。"""

    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "glow_patch_spatial.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve Native-Qwen patch-spatial checkpoint from: {path}")


def load_patch_spatial_checkpoint(
    path: Path,
    model: NativeQwenGlowPatchSpatialModel,
    *,
    map_location: str | torch.device = "cpu",
    trusted_checkpoint: bool = False,
) -> Dict[str, Any]:
    """严格加载 Native-Qwen patch-spatial checkpoint。

    这个 checkpoint 同时包含 spatial_flow 和 Qwen LoRA。
    两者缺一不可；否则评测/RL 用的就不是完整的训练后模型。

    ``trusted_checkpoint=True`` 只允许用于已经核验来源与 SHA 的项目自产训练 checkpoint。
    这类 checkpoint 还含 numpy RNG、optimizer 等非纯 tensor 状态，PyTorch 2.6+ 的安全模式
    无法反序列化；普通评测和外部输入仍保持 ``weights_only=True``，不会扩大 pickle 信任边界。
    """

    checkpoint_file = resolve_checkpoint_file(path)
    payload = torch.load(
        checkpoint_file,
        map_location=map_location,
        weights_only=not bool(trusted_checkpoint),
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {checkpoint_file}")
    if str(payload.get("format_version")) != FORMAT_VERSION:
        raise ValueError(f"checkpoint format mismatch: {payload.get('format_version')!r} vs {FORMAT_VERSION!r}")
    if "spatial_flow_state_dict" not in payload or "qwen_lora_state_dict" not in payload:
        raise KeyError(f"checkpoint missing spatial_flow_state_dict or qwen_lora_state_dict: {checkpoint_file}")
    payload_args = payload.get("args") if isinstance(payload.get("args"), Mapping) else {}
    checkpoint_reference_schema = str(payload_args.get("reference_schema") or "legacy5")
    model_reference_schema = str(getattr(model, "reference_schema", "legacy5"))
    if checkpoint_reference_schema != model_reference_schema:
        raise ValueError(
            f"checkpoint reference_schema mismatch: ckpt={checkpoint_reference_schema!r} model={model_reference_schema!r}. "
            "Pass the same --reference-schema as the checkpoint when evaluating, running GRPO or resuming SFT."
        )
    model.spatial_flow.load_state_dict(payload["spatial_flow_state_dict"])
    load_lora_state_dict(model.qwen, payload["qwen_lora_state_dict"])
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the QUASAR patch-level spatial model.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--qwen-cache-dir", type=Path, default=None)
    parser.add_argument("--visual-cache-root", type=Path, default=None)
    parser.add_argument(
        "--extra-visual-cache-root",
        type=Path,
        action="append",
        default=[],
        help="Additional cache roots searched after --visual-cache-root, useful when train and dev caches live separately.",
    )
    parser.add_argument("--scannet-frame-root", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--require-patch-geometry", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-visual-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--depth-grid-h", type=int, default=24)
    parser.add_argument("--depth-grid-w", type=int, default=32)
    parser.add_argument("--reference-schema", choices=tuple(sorted(REFERENCE_SCHEMA_TO_MAPPING.keys())), default="legacy5")
    parser.add_argument(
        "--simple3-aux-required",
        action="store_true",
        help="For reference-schema=simple3, require explicit refs.simple3_aux fields and do not infer query/reference in the loader.",
    )
    parser.add_argument("--flow-dim", type=int, default=512)
    parser.add_argument("--num-flow-steps", type=int, default=4)
    parser.add_argument(
        "--disable-spatial-flow-bias",
        action="store_true",
        help="Ablation: keep spatial_flow auxiliary heads, but inject zero bias into Qwen visual tokens.",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--bev-grid-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--reference-type-loss-weight", type=float, default=0.2)
    parser.add_argument("--query-pose-loss-weight", type=float, default=0.1)
    parser.add_argument("--bev-loss-weight", type=float, default=0.05)
    parser.add_argument("--patch-consistency-loss-weight", type=float, default=0.05)
    parser.add_argument("--target-camera-coord-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-query-coord-loss-weight", type=float, default=0.0)
    parser.add_argument("--bev-coord-loss-weight", type=float, default=0.0)
    parser.add_argument("--object-center-loss-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-optimizer-state", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser


def load_processor_and_qwen(args: argparse.Namespace) -> tuple[Any, nn.Module, int, int, Dict[str, Any]]:
    if not args.qwen_model_path.exists():
        raise FileNotFoundError(f"Qwen model path does not exist: {args.qwen_model_path}")
    config_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    model_kwargs: Dict[str, Any] = {"dtype": torch.bfloat16, "trust_remote_code": True}
    if args.qwen_cache_dir is not None:
        config_kwargs["cache_dir"] = str(args.qwen_cache_dir)
        model_kwargs["cache_dir"] = str(args.qwen_cache_dir)
    config = AutoConfig.from_pretrained(str(args.qwen_model_path), **config_kwargs)
    model_type = str(getattr(config, "model_type", ""))
    if not model_type.startswith("qwen3_vl"):
        raise ValueError(f"expected Qwen3-VL model_type, got {model_type!r}")
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or getattr(config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError(f"cannot resolve Qwen hidden size from config: {args.qwen_model_path}")
    spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
    if spatial_merge_size <= 0:
        raise ValueError("cannot resolve Qwen vision spatial_merge_size")
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        raise ValueError("Qwen3VLForConditionalGeneration is missing in this transformers environment")
    processor = AutoProcessor.from_pretrained(str(args.qwen_model_path), **config_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = int(args.max_pixels)
    qwen = model_cls.from_pretrained(str(args.qwen_model_path), **model_kwargs)
    qwen.config.use_cache = False
    if getattr(qwen, "generation_config", None) is not None:
        qwen.generation_config.use_cache = False
    if args.gradient_checkpointing:
        qwen.gradient_checkpointing_enable()
        if hasattr(qwen, "enable_input_require_grads"):
            qwen.enable_input_require_grads()
    for parameter in qwen.parameters():
        parameter.requires_grad = False

    if bool(getattr(args, "no_lora", False)):
        meta = {
            "model_type": model_type,
            "hidden_size": hidden_size,
            "spatial_merge_size": spatial_merge_size,
            "max_pixels": int(args.max_pixels),
            "lora": {"enabled": False},
        }
        return processor, qwen, hidden_size, spatial_merge_size, meta

    target_modules = {item.strip() for item in str(args.lora_target_modules).split(",") if item.strip()}
    if not target_modules:
        raise ValueError("--lora-target-modules must not be empty; formal pipeline requires Qwen LoRA trainability")
    replaced = apply_lora_to_qwen_linears(
        qwen,
        target_modules=target_modules,
        r=int(args.lora_r),
        alpha=int(args.lora_alpha),
        dropout=float(args.lora_dropout),
    )
    if not replaced:
        raise ValueError(f"no Qwen Linear modules matched LoRA targets: {sorted(target_modules)}")
    meta = {
        "model_type": model_type,
        "hidden_size": hidden_size,
        "spatial_merge_size": spatial_merge_size,
        "max_pixels": int(args.max_pixels),
        "lora": {
            "enabled": True,
            "r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "target_modules": sorted(target_modules),
            "replaced_module_count": len(replaced),
            "first_replaced_modules": replaced[:20],
        },
    }
    return processor, qwen, hidden_size, spatial_merge_size, meta


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
            moved[key] = [item.to(device) for item in value]
        else:
            moved[key] = value
    return moved


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for parameter in model.parameters():
        if parameter.requires_grad:
            yield parameter


def gradient_audit(model: NativeQwenGlowPatchSpatialModel) -> Dict[str, Any]:
    groups: Dict[str, float] = {"qwen_lora": 0.0, "spatial_flow": 0.0}
    nonzero: Counter[str] = Counter()
    trainable_total = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_total += int(parameter.numel())
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().float().norm().cpu().item())
        if grad_norm <= 0.0:
            continue
        if name.startswith("qwen.") and "lora_" in name:
            group = "qwen_lora"
        elif name.startswith("spatial_flow."):
            group = "spatial_flow"
        else:
            group = "other"
            groups.setdefault(group, 0.0)
        groups[group] += grad_norm
        nonzero[group] += 1
    required = ["qwen_lora", "spatial_flow"]
    missing = [group for group in required if groups.get(group, 0.0) <= 0.0]
    return {
        "grad_norm_by_group": groups,
        "nonzero_grad_params_by_group": dict(nonzero),
        "missing_required_grad_groups": missing,
        "trainable_parameters": trainable_total,
        "passed": not missing,
    }


def decode_label_audit(*, tokenizer: Any, labels: torch.Tensor, target_texts: Sequence[str]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for i in range(labels.shape[0]):
        ids = labels[i][labels[i] != -100].detach().cpu().tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        audits.append({"target_text": target_texts[i], "label_token_count": len(ids), "decoded_label_span": decoded})
    return audits


@torch.no_grad()
def evaluate_loss(model: NativeQwenGlowPatchSpatialModel, loader: DataLoader, *, device: torch.device, max_batches: int) -> Dict[str, float]:
    model.eval()
    sums: Counter[str] = Counter()
    count = 0
    for batch_i, batch in enumerate(loader, start=1):
        if max_batches > 0 and batch_i > max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        if not bool(torch.isfinite(out["loss"]).detach().cpu().item()):
            raise RuntimeError(f"non-finite eval loss at batch={batch_i}, sample_ids={batch['sample_ids']}")
        sums["loss"] += float(out["loss"].detach().cpu().item())
        sums["answer_ce_loss"] += float(out["answer_ce_loss"].detach().cpu().item())
        for key, value in out.items():
            if key.startswith("loss_"):
                sums[key] += float(value.detach().cpu().item())
        count += 1
    if count == 0:
        raise ValueError("evaluate_loss saw zero batches")
    return {key: value / count for key, value in sums.items()}


def checkpoint_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state["torch_cuda_all"]))


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    *,
    path: Path,
    model: NativeQwenGlowPatchSpatialModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    batch_i: int,
    args: argparse.Namespace,
    save_optimizer_state: bool,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "epoch": int(epoch),
        "batch_i": int(batch_i),
        "spatial_flow_state_dict": model.spatial_flow.state_dict(),
        "qwen_lora_state_dict": lora_state_dict(model.qwen),
        "args": normalize_args_for_json(args),
        "rng_state": checkpoint_rng_state(),
    }
    if save_optimizer_state:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path / "glow_patch_spatial.pt")
    write_json(
        path / "qwen_lora_config.json",
        {
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_target_modules": str(args.lora_target_modules),
            "note": "Project-local LoRA Linear state is stored in glow_patch_spatial.pt::qwen_lora_state_dict.",
        },
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "args.json", normalize_args_for_json(args))

    train_rows_raw = read_native_jsonl(args.train_jsonl)
    val_rows_raw = read_native_jsonl(args.val_jsonl)
    train_rows, train_audit = filter_native_rows(
        rows=train_rows_raw,
        max_rows=int(args.max_train_rows),
        require_patch_geometry=bool(args.require_patch_geometry),
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
    )
    val_rows, val_audit = filter_native_rows(
        rows=val_rows_raw,
        max_rows=int(args.max_val_rows),
        require_patch_geometry=bool(args.require_patch_geometry),
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
    )
    write_json(args.output_dir / "data_filter_audit.json", {"train": train_audit, "val": val_audit})
    write_json(
        args.output_dir / "reference_schema_audit.json",
        {
            "train": audit_reference_schema_rows(rows=train_rows, reference_schema=str(args.reference_schema), split_name="train"),
            "val": audit_reference_schema_rows(rows=val_rows, reference_schema=str(args.reference_schema), split_name="val"),
        },
    )
    visual_cache_roots = normalize_visual_cache_roots(args.visual_cache_root, args.extra_visual_cache_root)
    primary_visual_cache_root = visual_cache_roots[0] if visual_cache_roots else None
    extra_visual_cache_roots = visual_cache_roots[1:] if visual_cache_roots else []
    if primary_visual_cache_root is not None:
        train_cache_audit = audit_visual_cache_rows(
            rows=train_rows,
            visual_cache_root=primary_visual_cache_root,
            visual_cache_roots=extra_visual_cache_roots,
            scannet_frame_root=args.scannet_frame_root,
            max_visual_frames=int(args.max_visual_frames),
            qwen_model_path=args.qwen_model_path,
            max_pixels=int(args.max_pixels),
            split_name="train",
        )
        val_cache_audit = audit_visual_cache_rows(
            rows=val_rows,
            visual_cache_root=primary_visual_cache_root,
            visual_cache_roots=extra_visual_cache_roots,
            scannet_frame_root=args.scannet_frame_root,
            max_visual_frames=int(args.max_visual_frames),
            qwen_model_path=args.qwen_model_path,
            max_pixels=int(args.max_pixels),
            split_name="val",
        )
        write_json(args.output_dir / "visual_cache_audit.json", {"train": train_cache_audit, "val": val_cache_audit})

    processor, qwen, hidden_size, spatial_merge_size, qwen_meta = load_processor_and_qwen(args)
    write_json(args.output_dir / "qwen_meta.json", qwen_meta)
    collator = NativeGlowPatchSpatialCollator(
        processor=processor,
        max_pixels=int(args.max_pixels),
        spatial_merge_size=spatial_merge_size,
        bev_grid_size=int(args.bev_grid_size),
        reference_schema=str(args.reference_schema),
    )
    train_dataset = NativeAnswerPatchSpatialDataset(
        rows=train_rows,
        max_visual_frames=int(args.max_visual_frames),
        scannet_frame_root=args.scannet_frame_root,
        depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
        reference_schema=str(args.reference_schema),
        visual_cache_root=primary_visual_cache_root,
        visual_cache_roots=extra_visual_cache_roots,
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
        simple3_aux_required=bool(args.simple3_aux_required),
    )
    val_dataset = NativeAnswerPatchSpatialDataset(
        rows=val_rows,
        max_visual_frames=int(args.max_visual_frames),
        scannet_frame_root=args.scannet_frame_root,
        depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
        reference_schema=str(args.reference_schema),
        visual_cache_root=primary_visual_cache_root,
        visual_cache_roots=extra_visual_cache_roots,
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
        simple3_aux_required=bool(args.simple3_aux_required),
    )
    def build_train_loader_for_epoch(epoch_index: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(int(args.seed) + int(epoch_index))
        return DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            generator=generator,
            num_workers=int(args.num_workers),
            collate_fn=collator,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collator,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    aux_weights = {
        "reference_type": float(args.reference_type_loss_weight),
        "query_pose": float(args.query_pose_loss_weight),
        "bev": float(args.bev_loss_weight),
        "patch_consistency": float(args.patch_consistency_loss_weight),
        "target_camera_coord": float(args.target_camera_coord_loss_weight),
        "semantic_query_coord": float(args.semantic_query_coord_loss_weight),
        "bev_coord": float(args.bev_coord_loss_weight),
        "object_center": float(args.object_center_loss_weight),
    }
    model = NativeQwenGlowPatchSpatialModel(
        qwen=qwen,
        tokenizer=processor.tokenizer,
        qwen_hidden_size=hidden_size,
        flow_dim=int(args.flow_dim),
        num_flow_steps=int(args.num_flow_steps),
        dropout=float(args.dropout),
        bev_grid_size=int(args.bev_grid_size),
        reference_schema=str(args.reference_schema),
        aux_weights=aux_weights,
        disable_spatial_flow_bias=bool(args.disable_spatial_flow_bias),
    ).to(device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_log_path = args.output_dir / "train_log.jsonl"
    first_batch_audit_path = args.output_dir / "first_batch_audit.json"
    global_step = 0
    start_epoch = 0
    resume_skip_batch_i = 0
    if args.resume_from is not None:
        resume_payload = load_patch_spatial_checkpoint(args.resume_from, model, map_location="cpu")
        if "optimizer_state_dict" not in resume_payload:
            raise KeyError(
                f"resume checkpoint has no optimizer_state_dict: {args.resume_from}. "
                "Formal full resume requires checkpoints saved with --save-optimizer-state."
            )
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        global_step = int(resume_payload.get("step", 0))
        start_epoch = int(resume_payload.get("epoch", 0))
        checkpoint_batch_i = resume_payload.get("batch_i")
        if checkpoint_batch_i is None:
            resume_skip_batch_i = global_step * int(args.gradient_accumulation_steps)
            batch_i_source = "derived_from_step"
        else:
            resume_skip_batch_i = int(checkpoint_batch_i)
            batch_i_source = "checkpoint"
        if isinstance(resume_payload.get("rng_state"), Mapping):
            restore_rng_state(resume_payload["rng_state"])
        write_json(
            args.output_dir / "resume_audit.json",
            {
                "resume_from": str(args.resume_from),
                "global_step": global_step,
                "start_epoch": start_epoch,
                "resume_skip_batch_i": resume_skip_batch_i,
                "resume_skip_batch_i_source": batch_i_source,
                "optimizer_state_loaded": True,
                "rng_state_loaded": isinstance(resume_payload.get("rng_state"), Mapping),
            },
        )

    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    last_batch_i = resume_skip_batch_i
    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        train_loader = build_train_loader_for_epoch(epoch)
        for batch_i, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and resume_skip_batch_i > 0 and batch_i <= resume_skip_batch_i:
                continue
            last_batch_i = batch_i
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            if not bool(torch.isfinite(out["loss"]).detach().cpu().item()):
                raise RuntimeError(
                    f"non-finite training loss at epoch={epoch} batch_i={batch_i}, sample_ids={batch['sample_ids']}"
                )
            loss = out["loss"] / int(args.gradient_accumulation_steps)
            loss.backward()
            if global_step == 0 and batch_i == 1:
                audit = gradient_audit(model)
                audit.update(model.last_forward_audit)
                audit["label_audit"] = decode_label_audit(
                    tokenizer=processor.tokenizer,
                    labels=batch["labels"].detach().cpu(),
                    target_texts=batch["target_texts"],
                )
                audit["sample_ids"] = batch["sample_ids"]
                audit["passed"] = bool(audit["passed"]) and audit["spatial_bias_tokens"] == audit["image_feature_tokens"] and audit["answer_label_tokens"] > 0
                write_json(first_batch_audit_path, audit)
                if not audit["passed"]:
                    raise RuntimeError(f"first-batch audit failed: {audit}")
            if batch_i % int(args.gradient_accumulation_steps) == 0:
                if float(args.grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % int(args.log_every) == 0 or global_step == 1:
                    record = {
                        "time": time.time(),
                        "elapsed_sec": time.time() - start_time,
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_i": batch_i,
                        "loss": float(out["loss"].detach().cpu().item()),
                        "answer_ce_loss": float(out["answer_ce_loss"].detach().cpu().item()),
                    }
                    for key, value in out.items():
                        if key.startswith("loss_"):
                            record[key] = float(value.detach().cpu().item())
                    append_jsonl(train_log_path, record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                if int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0:
                    metrics = evaluate_loss(model, val_loader, device=device, max_batches=int(args.eval_max_batches))
                    metrics_record = {"global_step": global_step, "epoch": epoch, "metrics": metrics, "time": time.time()}
                    append_jsonl(args.output_dir / "val_log.jsonl", metrics_record)
                    print(json.dumps(metrics_record, ensure_ascii=False), flush=True)
                    model.train()
                if int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
                    save_checkpoint(
                        path=args.output_dir / "checkpoints" / f"step_{global_step:06d}",
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        epoch=epoch,
                        batch_i=batch_i,
                        args=args,
                        save_optimizer_state=bool(args.save_optimizer_state),
                    )
        # 梯度累积不足一个整步时，不静默丢掉最后一段梯度。
        if batch_i % int(args.gradient_accumulation_steps) != 0:
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), float(args.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
    final_metrics = evaluate_loss(model, val_loader, device=device, max_batches=int(args.eval_max_batches))
    write_json(args.output_dir / "final_val_metrics.json", final_metrics)
    save_checkpoint(
        path=args.output_dir / "final_checkpoint",
        model=model,
        optimizer=optimizer,
        step=global_step,
        epoch=int(args.epochs),
        batch_i=last_batch_i,
        args=args,
        save_optimizer_state=bool(args.save_optimizer_state),
    )
    write_json(
        args.output_dir / "run_summary.json",
        {
            "format_version": FORMAT_VERSION,
            "reference_schema": str(args.reference_schema),
            "global_step": global_step,
            "elapsed_sec": time.time() - start_time,
            "final_val_metrics": final_metrics,
            "first_batch_audit_path": str(first_batch_audit_path),
        },
    )


if __name__ == "__main__":
    main()
