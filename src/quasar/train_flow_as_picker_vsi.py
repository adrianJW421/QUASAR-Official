#!/usr/bin/env python3
"""Flow-as-Picker / All-Evidence Bottleneck teacher for VSI canonical train pack.

这份脚本刻意独立于现有 QUASAR 训练链路：
- 不改 Stage1/Stage2 trainer；
- 不复用旧的 CLIP top-k picker；
- 直接读取 VSI canonical 32 帧 compact evidence；
- 用 query-conditioned bottleneck latent 对全部 evidence 做软选择；
- 用答案分类损失 + 很弱的题型先验正则训练第一版 teacher。

第一版目标不是替代正式 MLLM SFT，而是验证“全证据交给 latent flow 自己选”这条路能否形成
稳定的 forward/loss/gradient/per-type eval 闭环。后续如果有效，再把这里的 bottleneck 输出接回
更重的 Qwen/3DThinker 训练链路。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


from quasar.train_spa3r_lite_hlatent_aux import (
    HLatentSpa3RLiteAuxModel,
    make_even_indices,
    make_patch_xy,
)


FRAME_COUNT = 32
FRAME_FEATURE_DIM = 768 * 2 + 1
POSE_FEATURE_DIM = 20
GEO_FEATURE_DIM = 15
OBJECT_FEATURE_DIM = 768 * 3 + 12
DEFAULT_TEXT_HASH_DIM = 256
DEFAULT_MAX_OPTIONS = 4
DEFAULT_MAX_OBJECT_TOKENS = 64
NODE_TYPE_FRAME = 0
NODE_TYPE_POSE = 1
NODE_TYPE_GEO = 2
NODE_TYPE_OBJECT = 3
HLATENT_CONTEXT_POSITIONS = (-2, 0, 2)


CANONICAL_QTYPES = (
    "object_abs_distance",
    "object_counting",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "object_size_estimation",
    "room_size_estimation",
    "route_planning",
    "obj_appearance_order",
    "vsti_camera_displacement",
    "vsti_camera_movement_direction",
    "vsti_camera_obj_abs_dist",
    "vsti_camera_obj_rel_dist_v1",
    "vsti_camera_obj_rel_dist_v2",
    "vsti_camera_obj_rel_dist_v3",
    "vsti_obj_obj_relative_pos_lr",
    "vsti_obj_obj_relative_pos_nf",
    "vsti_obj_obj_relative_pos_ud",
)


UNIT_TO_ID = {
    "none": 0,
    "meters": 1,
    "square_meters": 2,
    "centimeters": 3,
    "count": 4,
}


WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


@dataclass(frozen=True)
class NumericCandidate:
    text: str
    value: float
    unit: str


@dataclass(frozen=True)
class EvidenceSample:
    row_id: str
    video_id: str
    source_benchmark: str
    qtype: str
    question: str
    answer_label: int
    answer_texts: Tuple[str, ...]
    answer_mask: np.ndarray
    candidate_text_hash: np.ndarray
    candidate_numeric: np.ndarray
    question_hash: np.ndarray
    frame_features: np.ndarray
    pose_features: np.ndarray
    geo_features: np.ndarray
    pose_xyz: np.ndarray
    object_features: np.ndarray
    object_prompts: Tuple[str, ...]
    h_context_dino: Optional[np.ndarray] = None
    h_context_cut3r: Optional[np.ndarray] = None
    h_context_points: Optional[np.ndarray] = None
    h_context_pose: Optional[np.ndarray] = None
    h_precomputed_latent: Optional[np.ndarray] = None


class SmallArrayCache:
    """一个很小的 LRU 缓存，避免同一个 video 的 32 帧特征被同一轮反复从磁盘读。

    VSI train 里同一个 scene/video 会对应很多问题。如果每个问题都重新 np.load 一遍，
    smoke 也会很慢。这里缓存的是已经读出的 compact array，不缓存 CUT3R 的 context_tokens，
    所以不会把显存/内存吃爆。
    """

    def __init__(self, max_items: int) -> None:
        if max_items <= 0:
            raise ValueError(f"cache max_items must be positive, got {max_items}")
        self.max_items = int(max_items)
        self._items: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: str, value: Any) -> None:
        if key in self._items:
            self._items.pop(key)
        self._items[key] = value
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


class VsiCanonicalAllEvidenceDataset(Dataset):
    """从 canonical JSONL 逐题读取全部 compact evidence。

    这层只做“把磁盘证据变成统一 tensor 所需的 numpy 数组”，不做 top-k picker。
    也不会静默从 GT label 字段构造 geometry 答案；答案只用于分类监督。
    """

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        qtype_to_index: Mapping[str, int],
        object_manifest: Optional[Mapping[str, Path]],
        text_hash_dim: int,
        max_answer_options: int,
        max_object_tokens: int,
        feature_cache_items: int,
        require_object_tokens: bool,
        use_spa3r_hlatent: bool = False,
        spa3r_hlatent_windows: int = 0,
        spa3r_hlatent_dino_tokens: int = 0,
        spa3r_hlatent_cut3r_tokens: int = 0,
        spa3r_hlatent_cache_items: int = 4,
    ) -> None:
        if not rows:
            raise ValueError("dataset rows must not be empty")
        if text_hash_dim <= 0:
            raise ValueError(f"text_hash_dim must be positive, got {text_hash_dim}")
        if max_answer_options < 2:
            raise ValueError(f"max_answer_options must be >=2, got {max_answer_options}")
        if max_object_tokens <= 0:
            raise ValueError(f"max_object_tokens must be positive, got {max_object_tokens}")
        self.rows = list(rows)
        self.qtype_to_index = dict(qtype_to_index)
        self.object_manifest = dict(object_manifest) if object_manifest is not None else None
        self.text_hash_dim = int(text_hash_dim)
        self.max_answer_options = int(max_answer_options)
        self.max_object_tokens = int(max_object_tokens)
        self.require_object_tokens = bool(require_object_tokens)
        self.cache = SmallArrayCache(feature_cache_items)
        self.use_spa3r_hlatent = bool(use_spa3r_hlatent)
        self.spa3r_hlatent_windows = int(spa3r_hlatent_windows)
        self.h_dino_indices = make_even_indices(256, int(spa3r_hlatent_dino_tokens)) if self.use_spa3r_hlatent else None
        self.h_cut3r_indices = make_even_indices(768, int(spa3r_hlatent_cut3r_tokens)) if self.use_spa3r_hlatent else None
        self.h_cache = SmallArrayCache(spa3r_hlatent_cache_items) if self.use_spa3r_hlatent else None
        if self.use_spa3r_hlatent:
            if self.spa3r_hlatent_windows <= 0:
                raise ValueError("--spa3r-hlatent-windows must be positive when --use-spa3r-hlatent")
            if self.spa3r_hlatent_windows > FRAME_COUNT - 4:
                raise ValueError(f"spa3r_hlatent_windows={self.spa3r_hlatent_windows} is too large for {FRAME_COUNT} frames")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> EvidenceSample:
        row = self.rows[index]
        row_id = require_str(row, "row_id")
        video_id = require_str(row, "video_id")
        source_benchmark = str(row.get("source_benchmark") or row.get("source_dataset") or "unknown")
        qtype = require_str(row, "canonical_question_type")
        if qtype not in self.qtype_to_index:
            raise KeyError(f"row_id={row_id} unknown canonical_question_type={qtype!r}")
        question = require_str(row, "question")

        answer_label, answer_texts, answer_mask, candidate_numeric = build_answer_candidates(
            row,
            max_answer_options=self.max_answer_options,
        )
        candidate_hash_rows = [hashed_bow(text, dim=self.text_hash_dim) for text in answer_texts]
        candidate_text_hash = np.stack(candidate_hash_rows, axis=0).astype(np.float32)
        question_hash = hashed_bow(question, dim=self.text_hash_dim).astype(np.float32)

        frame_indices = require_int_list(row, "candidate_cut3r_row_indices", expected_len=FRAME_COUNT)
        clip_path = require_existing_path(row, "clip_embedding_path")
        dino_pooled_path = require_existing_path(row, "dino_pooled_embedding_path")
        cut3r_path = require_existing_path(row, "cut3r_npz_path")

        frame_features = self._load_frame_features(
            clip_path=clip_path,
            dino_pooled_path=dino_pooled_path,
            frame_indices=frame_indices,
        )
        pose_features, geo_features, pose_xyz = self._load_cut3r_features(
            cut3r_path=cut3r_path,
            frame_indices=frame_indices,
        )
        object_features, object_prompts = self._load_object_features(row=row, video_id=video_id, row_id=row_id)
        h_context_dino = None
        h_context_cut3r = None
        h_context_points = None
        h_context_pose = None
        h_precomputed_latent = None
        if self.use_spa3r_hlatent:
            precomputed_path = optional_path_value(row.get("spa3r_hlatent_npz_path"))
            if precomputed_path is not None:
                h_precomputed_latent = self._load_precomputed_hlatent(precomputed_path=precomputed_path, row_id=row_id)
            else:
                dino_token_path = require_existing_path(row, "dino_token_path")
                h_context_dino, h_context_cut3r, h_context_points, h_context_pose = self._load_spa3r_hlatent_context(
                    cut3r_path=cut3r_path,
                    dino_token_path=dino_token_path,
                    frame_indices=frame_indices,
                    row_id=row_id,
                )

        return EvidenceSample(
            row_id=row_id,
            video_id=video_id,
            source_benchmark=source_benchmark,
            qtype=qtype,
            question=question,
            answer_label=answer_label,
            answer_texts=tuple(answer_texts),
            answer_mask=answer_mask,
            candidate_text_hash=candidate_text_hash,
            candidate_numeric=candidate_numeric,
            question_hash=question_hash,
            frame_features=frame_features,
            pose_features=pose_features,
            geo_features=geo_features,
            pose_xyz=pose_xyz,
            object_features=object_features,
            object_prompts=object_prompts,
            h_context_dino=h_context_dino,
            h_context_cut3r=h_context_cut3r,
            h_context_points=h_context_points,
            h_context_pose=h_context_pose,
            h_precomputed_latent=h_precomputed_latent,
        )

    def _load_precomputed_hlatent(self, *, precomputed_path: Path, row_id: str) -> np.ndarray:
        if not precomputed_path.is_file():
            raise FileNotFoundError(f"row_id={row_id} precomputed H-latent npz missing: {precomputed_path}")
        with np.load(precomputed_path, allow_pickle=False) as payload:
            if "h_latent" not in payload.files:
                raise KeyError(f"precomputed H-latent npz missing h_latent: {precomputed_path}")
            h_latent = np.asarray(payload["h_latent"], dtype=np.float32)
        if h_latent.ndim != 2:
            raise ValueError(f"precomputed h_latent must be [tokens, hidden_dim], got {h_latent.shape}: {precomputed_path}")
        require_finite_np("h_precomputed_latent", h_latent)
        return h_latent

    def _load_frame_features(self, *, clip_path: Path, dino_pooled_path: Path, frame_indices: Sequence[int]) -> np.ndarray:
        cache_key = f"frame::{clip_path}::{dino_pooled_path}"
        cached = self.cache.get(cache_key)
        if cached is None:
            clip = np.asarray(np.load(clip_path), dtype=np.float32)
            dino = np.asarray(np.load(dino_pooled_path), dtype=np.float32)
            if clip.ndim != 2 or int(clip.shape[1]) != 768:
                raise ValueError(f"clip embeddings must be [T,768], got {clip.shape}: {clip_path}")
            if dino.ndim != 2 or int(dino.shape[1]) != 768:
                raise ValueError(f"dino pooled embeddings must be [T,768], got {dino.shape}: {dino_pooled_path}")
            if int(clip.shape[0]) != int(dino.shape[0]):
                raise ValueError(f"clip/dino frame count mismatch: {clip.shape[0]} vs {dino.shape[0]}")
            cached = (clip, dino)
            self.cache.put(cache_key, cached)
        clip, dino = cached
        selected_clip = clip[np.asarray(frame_indices, dtype=np.int64)]
        selected_dino = dino[np.asarray(frame_indices, dtype=np.int64)]
        order = np.linspace(0.0, 1.0, FRAME_COUNT, dtype=np.float32).reshape(FRAME_COUNT, 1)
        frame_features = np.concatenate([selected_clip, selected_dino, order], axis=1).astype(np.float32)
        if frame_features.shape != (FRAME_COUNT, FRAME_FEATURE_DIM):
            raise ValueError(f"frame feature shape mismatch: {frame_features.shape}")
        require_finite_np("frame_features", frame_features)
        return frame_features

    def _load_cut3r_features(
        self,
        *,
        cut3r_path: Path,
        frame_indices: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_key = f"cut3r::{cut3r_path}"
        cached = self.cache.get(cache_key)
        if cached is None:
            with np.load(cut3r_path, allow_pickle=False) as payload:
                for key in ("camera_pose", "target_points", "token_confidence", "token_mask"):
                    if key not in payload.files:
                        raise KeyError(f"CUT3R npz missing key={key}: {cut3r_path}")
                camera_pose = np.asarray(payload["camera_pose"], dtype=np.float32)
                target_points = np.asarray(payload["target_points"], dtype=np.float32)
                token_confidence = np.asarray(payload["token_confidence"], dtype=np.float32)
                token_mask = np.asarray(payload["token_mask"], dtype=np.bool_)
            if camera_pose.ndim != 3 or camera_pose.shape[1:] != (4, 4):
                raise ValueError(f"camera_pose must be [T,4,4], got {camera_pose.shape}: {cut3r_path}")
            if target_points.ndim != 3 or int(target_points.shape[2]) != 3:
                raise ValueError(f"target_points must be [T,P,3], got {target_points.shape}: {cut3r_path}")
            if token_confidence.shape != token_mask.shape:
                raise ValueError(f"token_confidence/token_mask shape mismatch: {token_confidence.shape} vs {token_mask.shape}")
            if int(target_points.shape[0]) != int(camera_pose.shape[0]) or int(token_confidence.shape[0]) != int(camera_pose.shape[0]):
                raise ValueError("CUT3R frame count mismatch among pose/points/confidence")
            cached = (camera_pose, target_points, token_confidence, token_mask)
            self.cache.put(cache_key, cached)
        camera_pose, target_points, token_confidence, token_mask = cached
        index_array = np.asarray(frame_indices, dtype=np.int64)
        selected_pose = camera_pose[index_array]
        selected_points = target_points[index_array]
        selected_conf = token_confidence[index_array]
        selected_mask = token_mask[index_array]

        translation = selected_pose[:, :3, 3].astype(np.float32)
        first_translation = translation[0:1]
        query_translation = translation[-1:]
        delta_from_first = translation - first_translation
        delta_to_query = query_translation - translation
        rotation_flat = selected_pose[:, :3, :3].reshape(FRAME_COUNT, 9).astype(np.float32)
        order = np.linspace(0.0, 1.0, FRAME_COUNT, dtype=np.float32).reshape(FRAME_COUNT, 1)
        path_distance = np.linalg.norm(delta_from_first, axis=1, keepdims=True).astype(np.float32)
        pose_features = np.concatenate(
            [translation, delta_from_first, delta_to_query, rotation_flat, order, path_distance],
            axis=1,
        ).astype(np.float32)
        if pose_features.shape != (FRAME_COUNT, POSE_FEATURE_DIM):
            raise ValueError(f"pose feature shape mismatch: {pose_features.shape}")

        geo_rows: List[np.ndarray] = []
        for frame_i in range(FRAME_COUNT):
            mask = selected_mask[frame_i].astype(np.bool_)
            if not bool(mask.any()):
                raise ValueError(f"CUT3R token_mask has no valid point: path={cut3r_path} frame_index={frame_indices[frame_i]}")
            conf = selected_conf[frame_i][mask].astype(np.float32)
            points = selected_points[frame_i][mask].astype(np.float32)
            valid_ratio = np.asarray([float(mask.mean())], dtype=np.float32)
            row = np.concatenate(
                [
                    np.asarray([float(conf.mean()), float(conf.std())], dtype=np.float32),
                    valid_ratio,
                    points.mean(axis=0).astype(np.float32),
                    points.std(axis=0).astype(np.float32),
                    points.min(axis=0).astype(np.float32),
                    points.max(axis=0).astype(np.float32),
                ],
                axis=0,
            )
            geo_rows.append(row)
        geo_features = np.stack(geo_rows, axis=0).astype(np.float32)
        if geo_features.shape != (FRAME_COUNT, GEO_FEATURE_DIM):
            raise ValueError(f"geo feature shape mismatch: {geo_features.shape}")
        require_finite_np("pose_features", pose_features)
        require_finite_np("geo_features", geo_features)
        require_finite_np("pose_xyz", translation)
        return pose_features, geo_features, translation.astype(np.float32)

    def _load_spa3r_hlatent_context(
        self,
        *,
        cut3r_path: Path,
        dino_token_path: Path,
        frame_indices: Sequence[int],
        row_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.h_cache is None or self.h_dino_indices is None or self.h_cut3r_indices is None:
            raise RuntimeError("spa3r h-latent cache/indices are not initialized")
        cache_key = f"spa3r_hlatent::{cut3r_path}::{dino_token_path}"
        cached = self.h_cache.get(cache_key)
        if cached is None:
            with np.load(dino_token_path, allow_pickle=False) as dino_payload:
                if "dino_token_features" not in dino_payload.files:
                    raise KeyError(f"DINO token npz missing dino_token_features: {dino_token_path}")
                dino_tokens = np.asarray(dino_payload["dino_token_features"], dtype=np.float32)
            with np.load(cut3r_path, allow_pickle=False) as cut3r_payload:
                for key in ("context_tokens", "target_points", "camera_pose"):
                    if key not in cut3r_payload.files:
                        raise KeyError(f"CUT3R npz missing key={key}: {cut3r_path}")
                cut3r_tokens = np.asarray(cut3r_payload["context_tokens"], dtype=np.float32)
                points = np.asarray(cut3r_payload["target_points"], dtype=np.float32)
                camera_pose = np.asarray(cut3r_payload["camera_pose"], dtype=np.float32)
            if dino_tokens.ndim != 3 or dino_tokens.shape[1:] != (256, 768):
                raise ValueError(f"dino_token_features must be [T,256,768], got {dino_tokens.shape}: {dino_token_path}")
            if cut3r_tokens.ndim != 3 or cut3r_tokens.shape[1:] != (768, 768):
                raise ValueError(f"context_tokens must be [T,768,768], got {cut3r_tokens.shape}: {cut3r_path}")
            if points.ndim != 3 or points.shape[1:] != (768, 3):
                raise ValueError(f"target_points must be [T,768,3], got {points.shape}: {cut3r_path}")
            if camera_pose.ndim != 3 or camera_pose.shape[1:] != (4, 4):
                raise ValueError(f"camera_pose must be [T,4,4], got {camera_pose.shape}: {cut3r_path}")
            if not (dino_tokens.shape[0] == cut3r_tokens.shape[0] == points.shape[0] == camera_pose.shape[0]):
                raise ValueError(f"row_id={row_id} DINO/CUT3R frame count mismatch for H-latent context")
            cached = (dino_tokens, cut3r_tokens, points, camera_pose)
            self.h_cache.put(cache_key, cached)
        dino_tokens, cut3r_tokens, points, camera_pose = cached

        # 把 32 帧切成几个局部小窗口。每个窗口只拿 [-2, 0, +2] 三帧，
        # 和 auxiliary-only 训练时的 Spa3R-lite 输入格式保持一致。
        anchor_positions = np.rint(
            np.linspace(2, FRAME_COUNT - 3, num=self.spa3r_hlatent_windows, dtype=np.float32)
        ).astype(np.int64)
        if len(set(int(item) for item in anchor_positions.tolist())) != int(self.spa3r_hlatent_windows):
            raise ValueError(f"spa3r_hlatent_windows={self.spa3r_hlatent_windows} produced duplicate anchors")
        frame_index_array = np.asarray(frame_indices, dtype=np.int64)
        context_offsets = np.asarray(HLATENT_CONTEXT_POSITIONS, dtype=np.int64)
        window_rows = []
        for anchor in anchor_positions:
            candidate_positions = anchor + context_offsets
            if int(candidate_positions.min()) < 0 or int(candidate_positions.max()) >= FRAME_COUNT:
                raise IndexError(f"row_id={row_id} H-latent window out of 32-frame range: anchor={anchor}")
            window_rows.append(frame_index_array[candidate_positions])
        row_index = np.stack(window_rows, axis=0).astype(np.int64)

        h_context_dino = dino_tokens[row_index][:, :, self.h_dino_indices, :].astype(np.float32)
        h_context_cut3r = cut3r_tokens[row_index][:, :, self.h_cut3r_indices, :].astype(np.float32)
        h_context_points = points[row_index][:, :, self.h_cut3r_indices, :].astype(np.float32)
        h_context_pose = camera_pose[row_index].astype(np.float32)
        require_finite_np("h_context_dino", h_context_dino)
        require_finite_np("h_context_cut3r", h_context_cut3r)
        require_finite_np("h_context_points", h_context_points)
        require_finite_np("h_context_pose", h_context_pose)
        return h_context_dino, h_context_cut3r, h_context_points, h_context_pose

    def _load_object_features(
        self,
        *,
        row: Mapping[str, Any],
        video_id: str,
        row_id: str,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        object_path = optional_path_value(row.get("object_token_npz_path"))
        if object_path is None:
            if self.object_manifest is None:
                raise KeyError(
                    f"row_id={row_id} has no object_token_npz_path; provide --object-token-manifest "
                    "or disable --require-object-tokens explicitly"
                )
            if video_id not in self.object_manifest:
                raise KeyError(f"row_id={row_id} video_id={video_id} missing from object token manifest")
            object_path = self.object_manifest[video_id]
        if not object_path.is_file():
            raise FileNotFoundError(f"row_id={row_id} object token npz missing: {object_path}")

        cache_key = f"object::{object_path}"
        cached = self.cache.get(cache_key)
        if cached is None:
            with np.load(object_path, allow_pickle=False) as payload:
                required = (
                    "prompts",
                    "object_context_features",
                    "object_dino_features",
                    "object_semantic_features",
                    "object_center_world",
                    "object_size_xyz",
                    "visible_frame_mask",
                    "visible_patch_count",
                    "total_area_px",
                )
                for key in required:
                    if key not in payload.files:
                        raise KeyError(f"object token npz missing key={key}: {object_path}")
                prompts = tuple(str(value) for value in payload["prompts"].tolist())
                context = np.asarray(payload["object_context_features"], dtype=np.float32)
                dino = np.asarray(payload["object_dino_features"], dtype=np.float32)
                semantic = np.asarray(payload["object_semantic_features"], dtype=np.float32)
                center = np.asarray(payload["object_center_world"], dtype=np.float32)
                size = np.asarray(payload["object_size_xyz"], dtype=np.float32)
                visible = np.asarray(payload["visible_frame_mask"], dtype=np.bool_)
                patch_count = np.asarray(payload["visible_patch_count"], dtype=np.float32)
                area = np.asarray(payload["total_area_px"], dtype=np.float32)
            object_count = len(prompts)
            if context.shape != (object_count, 768) or dino.shape != (object_count, 768) or semantic.shape != (object_count, 768):
                raise ValueError(f"object token feature shape mismatch in {object_path}")
            if center.shape != (object_count, 3) or size.shape != (object_count, 3):
                raise ValueError(f"object center/size shape mismatch in {object_path}")
            if visible.ndim != 2 or visible.shape[0] != object_count:
                raise ValueError(f"visible_frame_mask must be [N,T], got {visible.shape}: {object_path}")
            if patch_count.shape != visible.shape:
                raise ValueError(f"visible_patch_count shape mismatch: {patch_count.shape} vs {visible.shape}")
            if area.shape != (object_count,):
                raise ValueError(f"total_area_px must be [N], got {area.shape}: {object_path}")
            visible_ratio = visible.astype(np.float32).mean(axis=1, keepdims=True)
            first_visible = np.argmax(visible, axis=1).astype(np.float32).reshape(object_count, 1) / max(1, visible.shape[1] - 1)
            last_visible = (visible.shape[1] - 1 - np.argmax(visible[:, ::-1], axis=1)).astype(np.float32).reshape(object_count, 1)
            last_visible = last_visible / max(1, visible.shape[1] - 1)
            patch_stats = np.concatenate(
                [
                    patch_count.mean(axis=1, keepdims=True),
                    patch_count.max(axis=1, keepdims=True),
                    np.log1p(area).reshape(object_count, 1),
                ],
                axis=1,
            ).astype(np.float32)
            features = np.concatenate(
                [context, dino, semantic, center, size, visible_ratio, first_visible, last_visible, patch_stats],
                axis=1,
            ).astype(np.float32)
            if features.shape != (object_count, OBJECT_FEATURE_DIM):
                raise ValueError(f"object feature dim mismatch: {features.shape}")
            require_finite_np("object_features", features)
            cached = (features, prompts)
            self.cache.put(cache_key, cached)
        features, prompts = cached
        if self.require_object_tokens and int(features.shape[0]) == 0:
            raise ValueError(f"row_id={row_id} video_id={video_id} has zero object tokens")
        keep = min(int(features.shape[0]), self.max_object_tokens)
        return features[:keep].astype(np.float32), tuple(prompts[:keep])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train independent Flow-as-Picker all-evidence teacher on VSI canonical pack.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-token-manifest", type=Path, default=None)
    parser.add_argument(
        "--filter-to-object-manifest",
        action="store_true",
        help="Keep only videos whose object manifest has a token bank, so training never meets a missing object token.",
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-train-rows-per-qtype", type=int, default=0)
    parser.add_argument("--max-val-rows-per-qtype", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-bottleneck-tokens", type=int, default=32)
    parser.add_argument("--num-flow-steps", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--text-hash-dim", type=int, default=DEFAULT_TEXT_HASH_DIM)
    parser.add_argument("--max-answer-options", type=int, default=DEFAULT_MAX_OPTIONS)
    parser.add_argument("--max-object-tokens", type=int, default=DEFAULT_MAX_OBJECT_TOKENS)
    parser.add_argument("--feature-cache-items", type=int, default=64)
    parser.add_argument(
        "--use-spa3r-hlatent",
        action="store_true",
        help="Attach the auxiliary-only H-latent encoder to Flow and add local environment latents as extra evidence nodes.",
    )
    parser.add_argument("--spa3r-hlatent-checkpoint", type=Path, default=None)
    parser.add_argument("--spa3r-hlatent-hidden-dim", type=int, default=128)
    parser.add_argument("--spa3r-hlatent-num-latents", type=int, default=8)
    parser.add_argument("--spa3r-hlatent-num-layers", type=int, default=2)
    parser.add_argument("--spa3r-hlatent-num-heads", type=int, default=4)
    parser.add_argument("--spa3r-hlatent-dropout", type=float, default=0.05)
    parser.add_argument("--spa3r-hlatent-windows", type=int, default=4)
    parser.add_argument("--spa3r-hlatent-dino-tokens", type=int, default=64)
    parser.add_argument("--spa3r-hlatent-cut3r-tokens", type=int, default=64)
    parser.add_argument("--spa3r-hlatent-cache-items", type=int, default=4)
    parser.add_argument(
        "--spa3r-hlatent-integration",
        choices=("evidence_nodes", "latent_residual"),
        default="evidence_nodes",
        help=(
            "How the H-latent is attached to Flow. evidence_nodes (earlier version): append the H-latent as extra evidence nodes; "
            "latent_residual (H-transform version): add the H-latent to the Flow latent as a soft-gated residual."
        ),
    )
    parser.add_argument(
        "--train-spa3r-hlatent",
        action="store_true",
        help="By default the H-latent encoder is frozen and only the projection and Flow are trained; enable this to fine-tune the H-latent encoder as well.",
    )
    parser.add_argument("--prior-loss-weight", type=float, default=0.05)
    parser.add_argument("--sparsity-loss-weight", type=float, default=0.005)
    parser.add_argument("--diversity-loss-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--require-object-tokens", action="store_true", default=True)
    parser.add_argument(
        "--allow-zero-object-tokens",
        action="store_true",
        help="Explicitly allow videos without object tokens; by default this is an error, to avoid silently falling back to frame-only input.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_zero_object_tokens:
        args.require_object_tokens = False
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spa3r_hlatent_config: Optional[Dict[str, Any]] = None
    spa3r_hlatent_checkpoint_payload: Optional[Mapping[str, Any]] = None
    if args.use_spa3r_hlatent:
        if args.spa3r_hlatent_checkpoint is not None:
            if not args.spa3r_hlatent_checkpoint.is_file():
                raise FileNotFoundError(f"spa3r h-latent checkpoint not found: {args.spa3r_hlatent_checkpoint}")
            # 这个 checkpoint 由本项目训练脚本保存，里面除了 tensor 还有训练 args。
            # 显式 weights_only=False，避免新版 PyTorch 默认只读 tensor 时拒绝加载 Path/argparse 这类元信息。
            spa3r_hlatent_checkpoint_payload = torch.load(args.spa3r_hlatent_checkpoint, map_location="cpu", weights_only=False)
            if "model_state_dict" not in spa3r_hlatent_checkpoint_payload or "args" not in spa3r_hlatent_checkpoint_payload:
                raise KeyError(f"invalid spa3r h-latent checkpoint: {args.spa3r_hlatent_checkpoint}")
            checkpoint_args = dict(spa3r_hlatent_checkpoint_payload["args"])
            # 接回 Flow 时，H-latent encoder 的结构必须和 auxiliary checkpoint 完全一致；
            # 所以这些结构参数以 checkpoint 为准，避免手动参数写错后悄悄只加载一部分。
            args.spa3r_hlatent_hidden_dim = int(checkpoint_args["hidden_dim"])
            args.spa3r_hlatent_num_latents = int(checkpoint_args["num_h_latents"])
            args.spa3r_hlatent_num_layers = int(checkpoint_args["num_layers"])
            args.spa3r_hlatent_num_heads = int(checkpoint_args["num_heads"])
            args.spa3r_hlatent_dropout = float(checkpoint_args["dropout"])
            args.spa3r_hlatent_dino_tokens = int(checkpoint_args["dino_token_subsample"])
            args.spa3r_hlatent_cut3r_tokens = int(checkpoint_args["cut3r_token_subsample"])
        spa3r_hlatent_config = {
            "hidden_dim": int(args.spa3r_hlatent_hidden_dim),
            "num_h_latents": int(args.spa3r_hlatent_num_latents),
            "num_layers": int(args.spa3r_hlatent_num_layers),
            "num_heads": int(args.spa3r_hlatent_num_heads),
            "dropout": float(args.spa3r_hlatent_dropout),
            "dino_token_subsample": int(args.spa3r_hlatent_dino_tokens),
            "cut3r_token_subsample": int(args.spa3r_hlatent_cut3r_tokens),
            "windows": int(args.spa3r_hlatent_windows),
            "frozen": not bool(args.train_spa3r_hlatent),
            "checkpoint_path": str(args.spa3r_hlatent_checkpoint) if args.spa3r_hlatent_checkpoint is not None else None,
            "integration": str(args.spa3r_hlatent_integration),
        }

    object_manifest = load_object_manifest(args.object_token_manifest) if args.object_token_manifest is not None else None
    allowed_video_ids = None
    if args.filter_to_object_manifest:
        if object_manifest is None:
            raise ValueError("--filter-to-object-manifest requires --object-token-manifest")
        allowed_video_ids = set(object_manifest.keys())
    train_rows = read_jsonl(
        args.train_jsonl,
        max_rows=args.max_train_rows,
        max_rows_per_qtype=args.max_train_rows_per_qtype,
        allowed_video_ids=allowed_video_ids,
    )
    val_rows = read_jsonl(
        args.val_jsonl,
        max_rows=args.max_val_rows,
        max_rows_per_qtype=args.max_val_rows_per_qtype,
        allowed_video_ids=allowed_video_ids,
    )
    qtype_to_index = build_qtype_to_index(train_rows, val_rows)

    run_config = {
        "args": normalize_args_for_json(args),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_qtype_counts": dict(Counter(require_str(row, "canonical_question_type") for row in train_rows)),
        "val_qtype_counts": dict(Counter(require_str(row, "canonical_question_type") for row in val_rows)),
        "train_source_counts": dict(Counter(str(row.get("source_benchmark") or row.get("source_dataset") or "unknown") for row in train_rows)),
        "val_source_counts": dict(Counter(str(row.get("source_benchmark") or row.get("source_dataset") or "unknown") for row in val_rows)),
        "qtype_to_index": qtype_to_index,
        "spa3r_hlatent_config": spa3r_hlatent_config,
        "format": "flow_as_picker_vsi_v1",
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps({"event": "run_config", **run_config}, ensure_ascii=False), flush=True)

    train_dataset = VsiCanonicalAllEvidenceDataset(
        rows=train_rows,
        qtype_to_index=qtype_to_index,
        object_manifest=object_manifest,
        text_hash_dim=args.text_hash_dim,
        max_answer_options=args.max_answer_options,
        max_object_tokens=args.max_object_tokens,
        feature_cache_items=args.feature_cache_items,
        require_object_tokens=args.require_object_tokens,
        use_spa3r_hlatent=args.use_spa3r_hlatent,
        spa3r_hlatent_windows=args.spa3r_hlatent_windows,
        spa3r_hlatent_dino_tokens=args.spa3r_hlatent_dino_tokens,
        spa3r_hlatent_cut3r_tokens=args.spa3r_hlatent_cut3r_tokens,
        spa3r_hlatent_cache_items=args.spa3r_hlatent_cache_items,
    )
    val_dataset = VsiCanonicalAllEvidenceDataset(
        rows=val_rows,
        qtype_to_index=qtype_to_index,
        object_manifest=object_manifest,
        text_hash_dim=args.text_hash_dim,
        max_answer_options=args.max_answer_options,
        max_object_tokens=args.max_object_tokens,
        feature_cache_items=args.feature_cache_items,
        require_object_tokens=args.require_object_tokens,
        use_spa3r_hlatent=args.use_spa3r_hlatent,
        spa3r_hlatent_windows=args.spa3r_hlatent_windows,
        spa3r_hlatent_dino_tokens=args.spa3r_hlatent_dino_tokens,
        spa3r_hlatent_cut3r_tokens=args.spa3r_hlatent_cut3r_tokens,
        spa3r_hlatent_cache_items=args.spa3r_hlatent_cache_items,
    )
    hlatent_node_count = (
        int(args.spa3r_hlatent_windows) * int(args.spa3r_hlatent_num_latents)
        if args.use_spa3r_hlatent and args.spa3r_hlatent_integration == "evidence_nodes"
        else 0
    )
    collate = AllEvidenceCollator(
        qtype_to_index=qtype_to_index,
        max_answer_options=args.max_answer_options,
        include_hlatent_context=args.use_spa3r_hlatent,
        hlatent_node_count=hlatent_node_count,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=str(args.device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=str(args.device).startswith("cuda"),
    )

    device = torch.device(args.device)
    model = FlowAsPickerAnswerTeacher(
        qtype_count=len(qtype_to_index),
        text_hash_dim=args.text_hash_dim,
        max_answer_options=args.max_answer_options,
        d_model=args.d_model,
        num_bottleneck_tokens=args.num_bottleneck_tokens,
        num_flow_steps=args.num_flow_steps,
        num_attention_heads=args.num_attention_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        spa3r_hlatent_config=spa3r_hlatent_config,
    )
    if args.use_spa3r_hlatent:
        if model.spa3r_hlatent is None:
            raise RuntimeError("model was built without spa3r_hlatent despite --use-spa3r-hlatent")
        if spa3r_hlatent_checkpoint_payload is not None:
            model.spa3r_hlatent.load_state_dict(spa3r_hlatent_checkpoint_payload["model_state_dict"], strict=True)
        if not args.train_spa3r_hlatent:
            for parameter in model.spa3r_hlatent.parameters():
                parameter.requires_grad = False
    model = model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    first_batch = next(iter(train_loader))
    first_batch = move_batch_to_device(first_batch, device)
    model.train()
    first_output = model(first_batch)
    first_loss, first_metrics = compute_total_loss(
        output=first_output,
        batch=first_batch,
        prior_loss_weight=args.prior_loss_weight,
        sparsity_loss_weight=args.sparsity_loss_weight,
        diversity_loss_weight=args.diversity_loss_weight,
    )
    optimizer.zero_grad(set_to_none=True)
    first_loss.backward()
    grad_audit = gradient_audit(model)
    optimizer.zero_grad(set_to_none=True)
    first_batch_audit = {
        "loss": float(first_loss.detach().cpu().item()),
        "metrics": tensor_metrics_to_json(first_metrics),
        "grad_audit": grad_audit,
        "batch_shapes": batch_shape_summary(first_batch),
    }
    write_json(args.output_dir / "first_batch_audit.json", first_batch_audit)
    print(json.dumps({"event": "first_batch_audit", **first_batch_audit}, ensure_ascii=False), flush=True)

    train_log_path = args.output_dir / "train_log.jsonl"
    best_val_acc = -1.0
    best_step = 0
    global_step = 0
    started_at = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            loss, metrics = compute_total_loss(
                output=output,
                batch=batch,
                prior_loss_weight=args.prior_loss_weight,
                sparsity_loss_weight=args.sparsity_loss_weight,
                diversity_loss_weight=args.diversity_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()

            if global_step == 1 or global_step % args.log_every == 0:
                append_jsonl(
                    train_log_path,
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(loss.detach().cpu().item()),
                        "metrics": tensor_metrics_to_json(metrics),
                        "elapsed_sec": round(time.time() - started_at, 3),
                    },
                )
            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                eval_summary = evaluate(model=model, loader=val_loader, device=device, max_batches=0)
                append_jsonl(
                    train_log_path,
                    {
                        "event": "eval_step",
                        "epoch": epoch,
                        "step": global_step,
                        "eval": eval_summary,
                        "elapsed_sec": round(time.time() - started_at, 3),
                    },
                )
                if float(eval_summary["overall_acc"]) > best_val_acc:
                    best_val_acc = float(eval_summary["overall_acc"])
                    best_step = global_step
                    save_checkpoint(args.output_dir / "best_checkpoint.pt", model, optimizer, args, qtype_to_index, global_step, eval_summary)

        eval_summary = evaluate(model=model, loader=val_loader, device=device, max_batches=0)
        eval_event = {
            "event": "eval_epoch",
            "epoch": epoch,
            "step": global_step,
            "eval": eval_summary,
            "elapsed_sec": round(time.time() - started_at, 3),
        }
        append_jsonl(train_log_path, eval_event)
        print(json.dumps(eval_event, ensure_ascii=False), flush=True)
        if float(eval_summary["overall_acc"]) > best_val_acc:
            best_val_acc = float(eval_summary["overall_acc"])
            best_step = global_step
            save_checkpoint(args.output_dir / "best_checkpoint.pt", model, optimizer, args, qtype_to_index, global_step, eval_summary)

    final_eval = evaluate(model=model, loader=val_loader, device=device, max_batches=0)
    write_json(args.output_dir / "eval_summary.json", final_eval)
    final_summary = {
        "status": "completed",
        "format": "flow_as_picker_vsi_v1",
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "global_step": global_step,
        "best_step": best_step,
        "best_val_acc": best_val_acc,
        "final_eval": final_eval,
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    write_json(args.output_dir / "summary.json", final_summary)
    print(json.dumps({"event": "completed", **final_summary}, ensure_ascii=False), flush=True)
    save_checkpoint(args.output_dir / "last_checkpoint.pt", model, optimizer, args, qtype_to_index, global_step, final_eval)


class FlowBlock(nn.Module):
    """一个离散 latent-flow 更新步：先读证据，再让 bottleneck token 彼此整理信息。"""

    def __init__(self, *, d_model: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latent: torch.Tensor,
        nodes: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cross_out, cross_attn = self.cross_attn(
            query=latent,
            key=nodes,
            value=nodes,
            key_padding_mask=~node_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        latent = self.cross_norm(latent + cross_out)
        self_out, _ = self.self_attn(
            query=latent,
            key=latent,
            value=latent,
            need_weights=False,
        )
        latent = self.self_norm(latent + self_out)
        latent = self.ffn_norm(latent + self.ffn(latent))
        # MultiheadAttention 返回 [B,H,K,N]，loss 需要知道每个 bottleneck token 读了哪些 evidence。
        return latent, cross_attn


class FlowAsPickerAnswerTeacher(nn.Module):
    """全证据 bottleneck teacher。

    所有 frame/pose/geometry/object 节点先投到同一个 hidden 空间；32 个可学习 latent token
    带着 question/type 条件去 cross-attend 全部节点。attention 本身就是软 picker，输出的 32 个
    updated evidence tokens 再和候选答案文本一起做 MCA 分类。
    """

    def __init__(
        self,
        *,
        qtype_count: int,
        text_hash_dim: int,
        max_answer_options: int,
        d_model: int,
        num_bottleneck_tokens: int,
        num_flow_steps: int,
        num_attention_heads: int,
        ffn_dim: int,
        dropout: float,
        spa3r_hlatent_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if d_model % num_attention_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={num_attention_heads}")
        if num_bottleneck_tokens <= 0:
            raise ValueError("num_bottleneck_tokens must be positive")
        if num_flow_steps <= 0:
            raise ValueError("num_flow_steps must be positive")
        self.max_answer_options = int(max_answer_options)
        self.d_model = int(d_model)
        self.spa3r_hlatent_config = dict(spa3r_hlatent_config) if spa3r_hlatent_config is not None else None
        self.frame_proj = build_projector(FRAME_FEATURE_DIM, d_model, dropout)
        self.pose_proj = build_projector(POSE_FEATURE_DIM, d_model, dropout)
        self.geo_proj = build_projector(GEO_FEATURE_DIM, d_model, dropout)
        self.object_proj = build_projector(OBJECT_FEATURE_DIM, d_model, dropout)
        self.node_type_embedding = nn.Embedding(4, d_model)
        self.qtype_embedding = nn.Embedding(qtype_count, d_model)
        self.question_proj = build_projector(text_hash_dim, d_model, dropout)
        self.candidate_text_proj = build_projector(text_hash_dim, d_model, dropout)
        self.candidate_numeric_proj = build_projector(4, d_model, dropout)
        self.bottleneck_tokens = nn.Parameter(torch.empty(1, num_bottleneck_tokens, d_model))
        self.slot_embedding = nn.Parameter(torch.empty(1, num_bottleneck_tokens, d_model))
        self.spa3r_hlatent: Optional[HLatentSpa3RLiteAuxModel] = None
        self.spa3r_hlatent_proj: Optional[nn.Sequential] = None
        self.spa3r_hlatent_type_embedding: Optional[nn.Parameter] = None
        self.spa3r_hlatent_integration = "none"
        self.spa3r_hlatent_residual_attn: Optional[nn.MultiheadAttention] = None
        self.spa3r_hlatent_residual_norm: Optional[nn.LayerNorm] = None
        self.spa3r_hlatent_residual_proj: Optional[nn.Sequential] = None
        self.spa3r_hlatent_gate: Optional[nn.Sequential] = None
        self.spa3r_hlatent_frozen = True
        if self.spa3r_hlatent_config is not None:
            h_hidden_dim = int(self.spa3r_hlatent_config["hidden_dim"])
            h_num_latents = int(self.spa3r_hlatent_config["num_h_latents"])
            h_num_layers = int(self.spa3r_hlatent_config["num_layers"])
            h_num_heads = int(self.spa3r_hlatent_config["num_heads"])
            h_dropout = float(self.spa3r_hlatent_config["dropout"])
            h_dino_tokens = int(self.spa3r_hlatent_config["dino_token_subsample"])
            h_cut3r_tokens = int(self.spa3r_hlatent_config["cut3r_token_subsample"])
            h_dino_indices = make_even_indices(256, h_dino_tokens)
            h_cut3r_indices = make_even_indices(768, h_cut3r_tokens)
            self.spa3r_hlatent = HLatentSpa3RLiteAuxModel(
                hidden_dim=h_hidden_dim,
                num_h_latents=h_num_latents,
                num_layers=h_num_layers,
                num_heads=h_num_heads,
                dropout=h_dropout,
                dino_patch_xy=make_patch_xy(height=16, width=16, indices=h_dino_indices),
                cut3r_patch_xy=make_patch_xy(height=24, width=32, indices=h_cut3r_indices),
            )
            self.spa3r_hlatent_proj = build_projector(h_hidden_dim, d_model, dropout)
            self.spa3r_hlatent_type_embedding = nn.Parameter(torch.empty(1, 1, d_model))
            self.spa3r_hlatent_frozen = bool(self.spa3r_hlatent_config.get("frozen", True))
            self.spa3r_hlatent_integration = str(self.spa3r_hlatent_config.get("integration", "evidence_nodes"))
            if self.spa3r_hlatent_integration not in {"evidence_nodes", "latent_residual"}:
                raise ValueError(f"unknown spa3r_hlatent integration: {self.spa3r_hlatent_integration}")
            if self.spa3r_hlatent_integration == "latent_residual":
                self.spa3r_hlatent_residual_attn = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.spa3r_hlatent_residual_norm = nn.LayerNorm(d_model)
                self.spa3r_hlatent_residual_proj = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
                # 这个 gate 是 H-transform 的“阀门”。
                # 它同时看当前 latent、问题文本、题型和 H 分支给出的空间更新量，
                # 再决定每个 bottleneck slot 要吃进多少 H 空间修正。
                self.spa3r_hlatent_gate = nn.Sequential(
                    nn.LayerNorm(d_model * 4),
                    nn.Linear(d_model * 4, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )
        self.flow_blocks = nn.ModuleList(
            [FlowBlock(d_model=d_model, num_heads=num_attention_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in range(num_flow_steps)]
        )
        self.latent_norm = nn.LayerNorm(d_model)
        self.answer_scorer = nn.Sequential(
            nn.LayerNorm(d_model * 5),
            nn.Linear(d_model * 5, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bottleneck_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.slot_embedding, mean=0.0, std=0.02)
        if self.spa3r_hlatent_type_embedding is not None:
            nn.init.normal_(self.spa3r_hlatent_type_embedding, mean=0.0, std=0.02)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        frame = batch["frame_features"]
        pose = batch["pose_features"]
        geo = batch["geo_features"]
        obj = batch["object_features"]
        object_mask = batch["object_mask"]
        qtype_idx = batch["qtype_idx"]
        question_hash = batch["question_hash"]
        candidate_text_hash = batch["candidate_text_hash"]
        candidate_numeric = batch["candidate_numeric"]
        candidate_mask = batch["candidate_mask"]

        batch_size = int(frame.shape[0])
        frame_nodes = self.frame_proj(frame) + self.node_type_embedding.weight[NODE_TYPE_FRAME].view(1, 1, -1)
        pose_nodes = self.pose_proj(pose) + self.node_type_embedding.weight[NODE_TYPE_POSE].view(1, 1, -1)
        geo_nodes = self.geo_proj(geo) + self.node_type_embedding.weight[NODE_TYPE_GEO].view(1, 1, -1)
        object_nodes = self.object_proj(obj) + self.node_type_embedding.weight[NODE_TYPE_OBJECT].view(1, 1, -1)
        nodes = torch.cat([frame_nodes, pose_nodes, geo_nodes, object_nodes], dim=1)
        fixed_mask = torch.ones((batch_size, FRAME_COUNT * 3), device=nodes.device, dtype=torch.bool)
        node_mask = torch.cat([fixed_mask, object_mask], dim=1)

        h_nodes: Optional[torch.Tensor] = None
        if self.spa3r_hlatent is not None:
            if self.spa3r_hlatent_proj is None or self.spa3r_hlatent_type_embedding is None:
                raise RuntimeError("spa3r h-latent projection/type embedding is not initialized")
            if "h_precomputed_latent" in batch:
                # H encoder 是冻结的，所以主实验可直接读预计算 H-latent；
                # 这样不会反复解压同一个 video 的 DINO/CUT3R 大数组，训练才不会被 I/O 拖死。
                h_latent = batch["h_precomputed_latent"]
                if h_latent.ndim != 3:
                    raise ValueError(f"h_precomputed_latent must be [B,T,H], got {tuple(h_latent.shape)}")
            else:
                h_context_dino = batch["h_context_dino"]
                h_context_cut3r = batch["h_context_cut3r"]
                h_context_points = batch["h_context_points"]
                h_context_pose = batch["h_context_pose"]
                if h_context_dino.ndim != 5:
                    raise ValueError(f"h_context_dino must be [B,W,3,N,768], got {tuple(h_context_dino.shape)}")
                h_windows = int(h_context_dino.shape[1])
                h_flat_batch = {
                    "context_positions": torch.tensor(HLATENT_CONTEXT_POSITIONS, device=nodes.device, dtype=torch.long)
                    .view(1, 3)
                    .expand(batch_size * h_windows, -1),
                    "context_dino": h_context_dino.reshape(batch_size * h_windows, *h_context_dino.shape[2:]),
                    "context_cut3r": h_context_cut3r.reshape(batch_size * h_windows, *h_context_cut3r.shape[2:]),
                    "context_points": h_context_points.reshape(batch_size * h_windows, *h_context_points.shape[2:]),
                    "context_pose": h_context_pose.reshape(batch_size * h_windows, *h_context_pose.shape[2:]),
                }
                if self.spa3r_hlatent_frozen:
                    self.spa3r_hlatent.eval()
                    with torch.no_grad():
                        h_latent = self.spa3r_hlatent.encode_h_latent(h_flat_batch)
                else:
                    h_latent = self.spa3r_hlatent.encode_h_latent(h_flat_batch)
                h_latent = h_latent.reshape(batch_size, h_windows * h_latent.shape[1], h_latent.shape[2])
            h_nodes = self.spa3r_hlatent_proj(h_latent) + self.spa3r_hlatent_type_embedding
            if self.spa3r_hlatent_integration == "evidence_nodes":
                h_mask = torch.ones((batch_size, h_nodes.shape[1]), device=nodes.device, dtype=torch.bool)
                # 旧实验路径。H-latent 不替代原证据，只作为额外环境节点拼到 Flow 证据池末尾；
                # 后面的 cross-attention 自己决定每道题要不要读这些环境节点。
                nodes = torch.cat([nodes, h_nodes], dim=1)
                node_mask = torch.cat([node_mask, h_mask], dim=1)
        if not bool(node_mask.any(dim=1).all().item()):
            raise ValueError("each sample must contain at least one evidence node")

        qtype_hidden = self.qtype_embedding(qtype_idx)
        question_hidden = self.question_proj(question_hash)
        latent = self.bottleneck_tokens.expand(batch_size, -1, -1) + self.slot_embedding
        latent = latent + qtype_hidden.unsqueeze(1) + question_hidden.unsqueeze(1)
        h_gate_mean: Optional[torch.Tensor] = None
        if self.spa3r_hlatent is not None and self.spa3r_hlatent_integration == "latent_residual":
            if h_nodes is None:
                raise RuntimeError("latent_residual integration requires computed h_nodes")
            if (
                self.spa3r_hlatent_residual_attn is None
                or self.spa3r_hlatent_residual_norm is None
                or self.spa3r_hlatent_residual_proj is None
                or self.spa3r_hlatent_gate is None
            ):
                raise RuntimeError("latent_residual H-transform modules are not initialized")
            # 新版 H-transform 路径。这里不把 H token 当普通证据拼进去，
            # 而是让当前 Flow latent 主动读取 H-latent，再通过 soft gate 作为 residual 修正自身状态。
            h_context, _ = self.spa3r_hlatent_residual_attn(
                query=latent,
                key=h_nodes,
                value=h_nodes,
                need_weights=False,
            )
            h_delta = self.spa3r_hlatent_residual_proj(h_context)
            qtype_gate = qtype_hidden.unsqueeze(1).expand(-1, latent.shape[1], -1)
            question_gate = question_hidden.unsqueeze(1).expand(-1, latent.shape[1], -1)
            gate_input = torch.cat([latent, h_delta, qtype_gate, question_gate], dim=-1)
            h_gate = torch.sigmoid(self.spa3r_hlatent_gate(gate_input))
            latent = self.spa3r_hlatent_residual_norm(latent + h_gate * h_delta)
            h_gate_mean = h_gate.mean()
        last_attn = None
        for block in self.flow_blocks:
            latent, last_attn = block(latent, nodes, node_mask)
        if last_attn is None:
            raise RuntimeError("flow_blocks did not produce attention")
        updated_tokens = self.latent_norm(latent)
        memory = updated_tokens.mean(dim=1)

        candidate_text_hidden = self.candidate_text_proj(candidate_text_hash)
        candidate_numeric_hidden = self.candidate_numeric_proj(candidate_numeric)
        candidate_hidden = candidate_text_hidden + candidate_numeric_hidden + qtype_hidden.unsqueeze(1)
        memory_per_option = memory.unsqueeze(1).expand(-1, self.max_answer_options, -1)
        question_per_option = question_hidden.unsqueeze(1).expand(-1, self.max_answer_options, -1)
        scorer_input = torch.cat(
            [
                memory_per_option,
                candidate_hidden,
                question_per_option,
                memory_per_option * candidate_hidden,
                torch.abs(memory_per_option - candidate_hidden),
            ],
            dim=-1,
        )
        logits = self.answer_scorer(scorer_input).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        output = {
            "answer_logits": logits,
            "updated_evidence_tokens": updated_tokens,
            "last_cross_attention": last_attn.mean(dim=1),
            "node_mask": node_mask,
        }
        if h_gate_mean is not None:
            output["h_transform_gate_mean"] = h_gate_mean
        return output


class AllEvidenceCollator:
    """把变长 object nodes 和变长 coverage groups pad 成 batch。"""

    def __init__(
        self,
        *,
        qtype_to_index: Mapping[str, int],
        max_answer_options: int,
        include_hlatent_context: bool = False,
        hlatent_node_count: int = 0,
    ) -> None:
        self.qtype_to_index = dict(qtype_to_index)
        self.max_answer_options = int(max_answer_options)
        self.include_hlatent_context = bool(include_hlatent_context)
        self.hlatent_node_count = int(hlatent_node_count)

    def __call__(self, samples: Sequence[EvidenceSample]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("collate samples must not be empty")
        batch_size = len(samples)
        max_objects = max(max(1, int(sample.object_features.shape[0])) for sample in samples)
        node_count = FRAME_COUNT * 3 + max_objects + self.hlatent_node_count
        max_groups = max(1, max(len(build_prior_groups(sample, max_objects=max_objects)) for sample in samples))

        object_features = np.zeros((batch_size, max_objects, OBJECT_FEATURE_DIM), dtype=np.float32)
        object_mask = np.zeros((batch_size, max_objects), dtype=np.bool_)
        prior_group_mask = np.zeros((batch_size, max_groups, node_count), dtype=np.float32)
        prior_group_valid = np.zeros((batch_size, max_groups), dtype=np.float32)
        prior_group_count: List[int] = []

        for sample_i, sample in enumerate(samples):
            object_count = int(sample.object_features.shape[0])
            if object_count > 0:
                object_features[sample_i, :object_count, :] = sample.object_features
                object_mask[sample_i, :object_count] = True
            groups = build_prior_groups(sample, max_objects=max_objects)
            prior_group_count.append(len(groups))
            for group_i, node_indices in enumerate(groups):
                for node_index in node_indices:
                    if not 0 <= node_index < node_count:
                        raise IndexError(f"prior node index out of range: {node_index} vs node_count={node_count}")
                    prior_group_mask[sample_i, group_i, node_index] = 1.0
                prior_group_valid[sample_i, group_i] = 1.0

        batch = {
            "row_ids": [sample.row_id for sample in samples],
            "video_ids": [sample.video_id for sample in samples],
            "source_benchmarks": [sample.source_benchmark for sample in samples],
            "qtypes": [sample.qtype for sample in samples],
            "questions": [sample.question for sample in samples],
            "answer_texts": [sample.answer_texts for sample in samples],
            "prior_group_count": prior_group_count,
            "frame_features": torch.tensor(np.stack([sample.frame_features for sample in samples], axis=0), dtype=torch.float32),
            "pose_features": torch.tensor(np.stack([sample.pose_features for sample in samples], axis=0), dtype=torch.float32),
            "geo_features": torch.tensor(np.stack([sample.geo_features for sample in samples], axis=0), dtype=torch.float32),
            "object_features": torch.tensor(object_features, dtype=torch.float32),
            "object_mask": torch.tensor(object_mask, dtype=torch.bool),
            "qtype_idx": torch.tensor([self.qtype_to_index[sample.qtype] for sample in samples], dtype=torch.long),
            "answer_label": torch.tensor([sample.answer_label for sample in samples], dtype=torch.long),
            "candidate_mask": torch.tensor(np.stack([sample.answer_mask for sample in samples], axis=0), dtype=torch.bool),
            "candidate_text_hash": torch.tensor(np.stack([sample.candidate_text_hash for sample in samples], axis=0), dtype=torch.float32),
            "candidate_numeric": torch.tensor(np.stack([sample.candidate_numeric for sample in samples], axis=0), dtype=torch.float32),
            "question_hash": torch.tensor(np.stack([sample.question_hash for sample in samples], axis=0), dtype=torch.float32),
            "prior_group_mask": torch.tensor(prior_group_mask, dtype=torch.float32),
            "prior_group_valid": torch.tensor(prior_group_valid, dtype=torch.float32),
        }
        if self.include_hlatent_context:
            has_precomputed = [sample.h_precomputed_latent is not None for sample in samples]
            if any(has_precomputed):
                if not all(has_precomputed):
                    raise ValueError("H-latent collator received mixed precomputed/context samples")
                batch["h_precomputed_latent"] = torch.tensor(
                    np.stack([sample.h_precomputed_latent for sample in samples], axis=0),
                    dtype=torch.float32,
                )
                return batch
            for sample in samples:
                if (
                    sample.h_context_dino is None
                    or sample.h_context_cut3r is None
                    or sample.h_context_points is None
                    or sample.h_context_pose is None
                ):
                    raise ValueError("H-latent collator received sample without spa3r context tensors")
            batch["h_context_dino"] = torch.tensor(np.stack([sample.h_context_dino for sample in samples], axis=0), dtype=torch.float32)
            batch["h_context_cut3r"] = torch.tensor(np.stack([sample.h_context_cut3r for sample in samples], axis=0), dtype=torch.float32)
            batch["h_context_points"] = torch.tensor(np.stack([sample.h_context_points for sample in samples], axis=0), dtype=torch.float32)
            batch["h_context_pose"] = torch.tensor(np.stack([sample.h_context_pose for sample in samples], axis=0), dtype=torch.float32)
        return batch


def build_projector(input_dim: int, d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.LayerNorm(d_model),
    )


def compute_total_loss(
    *,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    prior_loss_weight: float,
    sparsity_loss_weight: float,
    diversity_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    logits = output["answer_logits"].float()
    labels = batch["answer_label"].to(device=logits.device, dtype=torch.long)
    candidate_mask = batch["candidate_mask"].to(device=logits.device, dtype=torch.bool)
    if logits.ndim != 2:
        raise ValueError(f"answer logits must be [B,C], got {tuple(logits.shape)}")
    if labels.shape != (logits.shape[0],):
        raise ValueError(f"answer labels must be [B], got {tuple(labels.shape)}")
    if candidate_mask.shape != logits.shape:
        raise ValueError(f"candidate_mask shape mismatch: {tuple(candidate_mask.shape)} vs {tuple(logits.shape)}")
    row_indices = torch.arange(logits.shape[0], device=logits.device)
    if not bool(candidate_mask[row_indices, labels].all().item()):
        raise ValueError("answer_label points to a padded/invalid candidate")
    answer_loss = F.cross_entropy(logits, labels)
    pred = logits.argmax(dim=-1)
    answer_acc = (pred == labels).to(torch.float32).mean()

    attn = output["last_cross_attention"].float()
    node_mask = output["node_mask"].to(device=attn.device, dtype=torch.bool)
    if attn.ndim != 3:
        raise ValueError(f"last_cross_attention must be [B,K,N], got {tuple(attn.shape)}")
    if node_mask.shape != attn.shape[:1] + attn.shape[2:]:
        raise ValueError(f"node_mask shape mismatch: {tuple(node_mask.shape)} vs attention {tuple(attn.shape)}")
    attn = attn.masked_fill(~node_mask.unsqueeze(1), 0.0)
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    node_mass = attn.mean(dim=1)

    prior_group_mask = batch["prior_group_mask"].to(device=attn.device, dtype=attn.dtype)
    prior_group_valid = batch["prior_group_valid"].to(device=attn.device, dtype=attn.dtype)
    if prior_group_mask.shape[:2] != prior_group_valid.shape:
        raise ValueError("prior group mask/valid shape mismatch")
    if int(prior_group_mask.shape[2]) != int(attn.shape[2]):
        raise ValueError(
            f"prior group node dim mismatch: {prior_group_mask.shape[2]} vs attention nodes {attn.shape[2]}"
        )
    group_mass = (node_mass.unsqueeze(1) * prior_group_mask).sum(dim=-1)
    prior_values = -torch.log(group_mass.clamp_min(1e-6))
    prior_den = prior_group_valid.sum().clamp_min(1.0)
    prior_loss = (prior_values * prior_group_valid).sum() / prior_den

    valid_node_mass = node_mass.masked_fill(~node_mask, 0.0)
    sparsity_entropy = -(valid_node_mass * torch.log(valid_node_mass.clamp_min(1e-8))).sum(dim=-1).mean()
    normalized_query_attn = attn / attn.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarity = torch.bmm(normalized_query_attn, normalized_query_attn.transpose(1, 2))
    k = int(similarity.shape[1])
    off_diag = similarity - torch.eye(k, device=similarity.device, dtype=similarity.dtype).unsqueeze(0)
    diversity_loss = off_diag.square().sum(dim=(1, 2)) / max(1, k * (k - 1))
    diversity_loss = diversity_loss.mean()

    total = answer_loss + float(prior_loss_weight) * prior_loss + float(sparsity_loss_weight) * sparsity_entropy + float(diversity_loss_weight) * diversity_loss
    metrics = {
        "answer_loss": answer_loss.detach(),
        "answer_acc": answer_acc.detach(),
        "prior_loss": prior_loss.detach(),
        "sparsity_entropy": sparsity_entropy.detach(),
        "diversity_loss": diversity_loss.detach(),
        "prior_groups_per_batch": prior_group_valid.sum(dim=1).mean().detach(),
        "attention_max_mass": node_mass.max(dim=-1).values.mean().detach(),
    }
    return total, metrics


@torch.no_grad()
def evaluate(*, model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    per_type_total: DefaultDict[str, int] = defaultdict(int)
    per_type_correct: DefaultDict[str, int] = defaultdict(int)
    per_source_total: DefaultDict[str, int] = defaultdict(int)
    per_source_correct: DefaultDict[str, int] = defaultdict(int)
    per_source_type_total: DefaultDict[str, int] = defaultdict(int)
    per_source_type_correct: DefaultDict[str, int] = defaultdict(int)
    loss_sums: DefaultDict[str, float] = defaultdict(float)
    batch_count = 0
    for batch in loader:
        batch_count += 1
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        loss, metrics = compute_total_loss(
            output=output,
            batch=batch,
            prior_loss_weight=0.0,
            sparsity_loss_weight=0.0,
            diversity_loss_weight=0.0,
        )
        logits = output["answer_logits"]
        labels = batch["answer_label"].to(device=logits.device)
        preds = logits.argmax(dim=-1)
        matches = (preds == labels).detach().cpu().numpy().astype(np.bool_)
        qtypes = list(batch["qtypes"])
        source_benchmarks = list(batch["source_benchmarks"])
        for matched, qtype, source_benchmark in zip(matches.tolist(), qtypes, source_benchmarks):
            total += 1
            correct += int(bool(matched))
            per_type_total[qtype] += 1
            per_type_correct[qtype] += int(bool(matched))
            per_source_total[source_benchmark] += 1
            per_source_correct[source_benchmark] += int(bool(matched))
            source_type_key = f"{source_benchmark}::{qtype}"
            per_source_type_total[source_type_key] += 1
            per_source_type_correct[source_type_key] += int(bool(matched))
        loss_sums["loss"] += float(loss.detach().cpu().item())
        for key, value in metrics.items():
            loss_sums[key] += float(value.detach().cpu().item())
        if max_batches > 0 and batch_count >= max_batches:
            break
    if total == 0:
        raise ValueError("evaluation loader produced zero rows")
    per_type = {}
    for qtype in sorted(per_type_total):
        q_total = per_type_total[qtype]
        per_type[qtype] = {
            "count": q_total,
            "correct": per_type_correct[qtype],
            "acc": per_type_correct[qtype] / q_total,
        }
    per_source = {}
    for source_benchmark in sorted(per_source_total):
        source_total = per_source_total[source_benchmark]
        per_source[source_benchmark] = {
            "count": source_total,
            "correct": per_source_correct[source_benchmark],
            "acc": per_source_correct[source_benchmark] / source_total,
        }
    per_source_type = {}
    for key in sorted(per_source_type_total):
        group_total = per_source_type_total[key]
        per_source_type[key] = {
            "count": group_total,
            "correct": per_source_type_correct[key],
            "acc": per_source_type_correct[key] / group_total,
        }
    averaged_metrics = {key: value / batch_count for key, value in loss_sums.items()}
    return {
        "overall_count": total,
        "overall_correct": correct,
        "overall_acc": correct / total,
        "batch_count": batch_count,
        "metrics": averaged_metrics,
        "per_source": per_source,
        "per_type": per_type,
        "per_source_type": per_source_type,
    }


def build_prior_groups(sample: EvidenceSample, *, max_objects: int) -> List[List[int]]:
    groups: List[List[int]] = []
    qtype = sample.qtype
    if qtype.startswith("vsti_camera_displacement") or qtype.startswith("vsti_camera_movement_direction"):
        # 动态相机题最少要覆盖题目提到的起止帧；没有显式帧号时退回首尾帧。
        frame_indices = parse_frame_indices_from_question(sample.question)
        if not frame_indices:
            frame_indices = [0, FRAME_COUNT - 1]
        for frame_index in frame_indices:
            groups.append(frame_pose_geo_nodes(frame_index))
        return dedupe_groups(groups)

    if qtype.startswith("vsti_camera_obj"):
        # camera-object 动态题通常问某一帧相机和物体的关系：同时覆盖目标帧和文本里提到的物体。
        frame_indices = parse_frame_indices_from_question(sample.question)
        for frame_index in frame_indices:
            groups.append(frame_pose_geo_nodes(frame_index))
        object_matches = matched_object_node_indices(sample, max_objects=max_objects)
        if object_matches:
            groups.append(object_matches)
        return dedupe_groups(groups)

    if qtype.startswith("vsti_obj_obj_relative_pos"):
        # object-object 动态题的关键是物体对；如果文本匹配不到 object bank，就至少让模型看首尾帧。
        object_matches = matched_object_node_indices(sample, max_objects=max_objects)
        if object_matches:
            groups.append(object_matches)
        else:
            groups.append(frame_pose_geo_nodes(0))
            groups.append(frame_pose_geo_nodes(FRAME_COUNT - 1))
        return dedupe_groups(groups)

    if qtype == "room_size_estimation":
        xyz = sample.pose_xyz
        extreme_indices = [
            int(np.argmin(xyz[:, 0])),
            int(np.argmax(xyz[:, 0])),
            int(np.argmin(xyz[:, 1])),
            int(np.argmax(xyz[:, 1])),
            0,
            FRAME_COUNT - 1,
        ]
        for frame_index in extreme_indices:
            groups.append(frame_pose_geo_nodes(frame_index))
        return dedupe_groups(groups)

    if qtype == "route_planning":
        groups.append(frame_pose_geo_nodes(0))
        groups.append(frame_pose_geo_nodes(FRAME_COUNT - 1))
        object_matches = matched_object_node_indices(sample, max_objects=max_objects)
        if object_matches:
            groups.append(object_matches)
        return dedupe_groups(groups)

    if qtype in {
        "object_abs_distance",
        "object_rel_distance",
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_size_estimation",
        "object_counting",
        "obj_appearance_order",
    }:
        object_matches = matched_object_node_indices(sample, max_objects=max_objects)
        if object_matches:
            groups.append(object_matches)
    return dedupe_groups(groups)


def parse_frame_indices_from_question(question: str) -> List[int]:
    """从 VSTI 问题文本里抽取 “frame 14” 这种 1-based 帧号，并转成 0-based slot。

    VSTI 动态题经常直接说“between frame 12 and frame 20 of 32”。
    这里不是用它算答案，只是给 coverage regularization 一个很弱的提示：
    这些帧至少应该被 bottleneck 看见。
    """

    indices: List[int] = []
    for match in re.finditer(r"\bframe\s+(\d+)\b", question.lower()):
        frame_number = int(match.group(1))
        if 1 <= frame_number <= FRAME_COUNT:
            indices.append(frame_number - 1)
    return sorted(set(indices))


def frame_pose_geo_nodes(frame_index: int) -> List[int]:
    if not 0 <= frame_index < FRAME_COUNT:
        raise IndexError(f"frame_index out of range: {frame_index}")
    return [frame_index, FRAME_COUNT + frame_index, FRAME_COUNT * 2 + frame_index]


def matched_object_node_indices(sample: EvidenceSample, *, max_objects: int) -> List[int]:
    text = " ".join([sample.question] + list(sample.answer_texts)).lower()
    node_indices: List[int] = []
    for object_i, prompt in enumerate(sample.object_prompts[:max_objects]):
        prompt_norm = normalize_space(prompt.lower())
        if not prompt_norm:
            continue
        words = [word for word in WORD_RE.findall(prompt_norm) if len(word) >= 3]
        if prompt_norm in text or any(word in text for word in words):
            node_indices.append(FRAME_COUNT * 3 + object_i)
    return node_indices


def dedupe_groups(groups: Sequence[Sequence[int]]) -> List[List[int]]:
    output: List[List[int]] = []
    seen = set()
    for group in groups:
        unique = tuple(sorted(set(int(value) for value in group)))
        if not unique or unique in seen:
            continue
        seen.add(unique)
        output.append(list(unique))
    return output


def build_answer_candidates(row: Mapping[str, Any], *, max_answer_options: int) -> Tuple[int, List[str], np.ndarray, np.ndarray]:
    row_id = require_str(row, "row_id")
    answer_kind = require_str(row, "answer_kind")
    answer_texts: List[str]
    label: int
    numeric_features = np.zeros((max_answer_options, 4), dtype=np.float32)
    if answer_kind == "option_letter":
        options = row.get("options")
        if not isinstance(options, list):
            raise ValueError(f"row_id={row_id} options must be list for option_letter answer")
        parsed_options = parse_options(options, row_id=row_id)
        if len(parsed_options) > max_answer_options:
            raise ValueError(f"row_id={row_id} has {len(parsed_options)} options > max_answer_options={max_answer_options}")
        answer_letter = require_str(row, "answer_option_letter")
        letters = [letter for letter, _ in parsed_options]
        if answer_letter not in letters:
            raise ValueError(f"row_id={row_id} answer_option_letter={answer_letter!r} not in options={letters}")
        label = letters.index(answer_letter)
        answer_texts = [text for _, text in parsed_options]
    else:
        if answer_kind not in {"numeric", "free_text"}:
            raise ValueError(f"row_id={row_id} unsupported answer_kind={answer_kind!r}")
        raw_value = row.get("answer_numeric_value")
        if raw_value is None:
            raise ValueError(f"row_id={row_id} numeric/free_text row missing answer_numeric_value")
        value = float(raw_value)
        qtype = require_str(row, "canonical_question_type")
        unit = str(row.get("answer_numeric_unit") or "none")
        candidates = make_numeric_candidates(value=value, unit=unit, qtype=qtype, row_id=row_id, max_options=max_answer_options)
        if len(candidates) != max_answer_options:
            raise ValueError(f"numeric candidate count must equal max_answer_options={max_answer_options}")
        answer_texts = [candidate.text for candidate in candidates]
        label = next(index for index, candidate in enumerate(candidates) if abs(candidate.value - value) <= 1e-6)
        unit_id = UNIT_TO_ID.get(unit, UNIT_TO_ID["none"])
        for index, candidate in enumerate(candidates):
            numeric_features[index, :] = np.asarray(
                [
                    1.0,
                    float(candidate.value),
                    math.log1p(abs(float(candidate.value))),
                    float(unit_id) / max(1, len(UNIT_TO_ID) - 1),
                ],
                dtype=np.float32,
            )
    answer_mask = np.zeros((max_answer_options,), dtype=np.bool_)
    answer_mask[: len(answer_texts)] = True
    while len(answer_texts) < max_answer_options:
        answer_texts.append("<PAD>")
    if not bool(answer_mask[label]):
        raise ValueError(f"row_id={row_id} answer label {label} is not valid")
    return label, answer_texts, answer_mask, numeric_features


def parse_options(options: Sequence[Any], *, row_id: str) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for option in options:
        if isinstance(option, Mapping):
            letter = str(option.get("letter", "")).strip()
            text = str(option.get("text", "")).strip()
        elif isinstance(option, str):
            match = re.match(r"^\s*([A-H])\s*[\.:\)]\s*(.*)$", option)
            if match is None:
                raise ValueError(f"row_id={row_id} option string lacks leading letter: {option!r}")
            letter = match.group(1)
            text = match.group(2).strip()
        else:
            raise TypeError(f"row_id={row_id} option must be dict or str, got {type(option).__name__}")
        if not letter:
            raise ValueError(f"row_id={row_id} option missing letter: {option!r}")
        if not text:
            raise ValueError(f"row_id={row_id} option missing text: {option!r}")
        parsed.append((letter, text))
    if len({letter for letter, _ in parsed}) != len(parsed):
        raise ValueError(f"row_id={row_id} option letters are not unique")
    return parsed


def make_numeric_candidates(*, value: float, unit: str, qtype: str, row_id: str, max_options: int) -> List[NumericCandidate]:
    if max_options <= 1:
        raise ValueError(f"max_options must be >1, got {max_options}")
    if qtype == "object_counting":
        base = int(round(value))
        raw_values = [value, max(0, base - 1), base + 1, max(0, base + 2)]
        deduped: List[float] = []
        candidate = 0
        while len(deduped) < max_options:
            proposed = float(raw_values[candidate] if candidate < len(raw_values) else base + candidate)
            if proposed not in deduped:
                deduped.append(proposed)
            candidate += 1
    else:
        magnitude = max(abs(value), 1.0)
        offsets = [0.0, -0.25 * magnitude, 0.35 * magnitude, 0.75 * magnitude]
        deduped = [float(max(0.0, value + offset)) for offset in offsets]
        rounded_seen = set()
        fixed: List[float] = []
        for candidate in deduped:
            rounded = round(candidate, 4)
            if rounded in rounded_seen:
                candidate = candidate + 0.13 * (len(fixed) + 1) * magnitude
                rounded = round(candidate, 4)
            rounded_seen.add(rounded)
            fixed.append(float(candidate))
        deduped = fixed
    rng = random.Random(stable_hash_int(row_id) % (2**32))
    rng.shuffle(deduped)
    if not any(abs(candidate - value) <= 1e-6 for candidate in deduped):
        raise ValueError(f"row_id={row_id} numeric candidates lost gold value")
    return tuple_to_numeric_candidates(deduped, unit=unit)


def tuple_to_numeric_candidates(values: Sequence[float], *, unit: str) -> List[NumericCandidate]:
    output: List[NumericCandidate] = []
    for value in values:
        if unit == "count":
            text = str(int(round(value)))
        elif unit == "centimeters":
            text = f"{value:.1f} centimeters"
        elif unit == "square_meters":
            text = f"{value:.2f} square meters"
        elif unit == "meters":
            text = f"{value:.2f} meters"
        else:
            text = f"{value:.3f}"
        output.append(NumericCandidate(text=text, value=float(value), unit=unit))
    return output


def hashed_bow(text: str, *, dim: int) -> np.ndarray:
    vector = np.zeros((dim,), dtype=np.float32)
    tokens = WORD_RE.findall(text.lower())
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="little", signed=False) % dim
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def stable_hash_int(text: str) -> int:
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], byteorder="little", signed=False)


def read_jsonl(
    path: Path,
    *,
    max_rows: int,
    max_rows_per_qtype: int,
    allowed_video_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"jsonl does not exist: {path}")
    if max_rows < 0 or max_rows_per_qtype < 0:
        raise ValueError("row limits must be non-negative")
    rows: List[Dict[str, Any]] = []
    per_type_count: Counter = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no} must be JSON object")
            if allowed_video_ids is not None and require_str(row, "video_id") not in allowed_video_ids:
                continue
            qtype = require_str(row, "canonical_question_type")
            if max_rows_per_qtype > 0 and per_type_count[qtype] >= max_rows_per_qtype:
                continue
            rows.append(row)
            per_type_count[qtype] += 1
            if max_rows > 0 and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"jsonl has no rows after limits: {path}")
    return rows


def load_object_manifest(path: Path) -> Dict[str, Path]:
    if not path.is_file():
        raise FileNotFoundError(f"object token manifest does not exist: {path}")
    mapping: Dict[str, Path] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no} must be JSON object")
            video_id = require_str(row, "video_id")
            object_path = require_existing_path(row, "object_token_npz_path")
            if video_id in mapping and mapping[video_id] != object_path:
                raise ValueError(f"duplicate object manifest video_id with different paths: {video_id}")
            mapping[video_id] = object_path
    if not mapping:
        raise ValueError(f"object token manifest is empty: {path}")
    return mapping


def build_qtype_to_index(*row_groups: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    qtypes = set()
    for rows in row_groups:
        for row in rows:
            qtypes.add(require_str(row, "canonical_question_type"))
    ordered = [qtype for qtype in CANONICAL_QTYPES if qtype in qtypes]
    ordered.extend(sorted(qtypes - set(ordered)))
    return {qtype: index for index, qtype in enumerate(ordered)}


def require_str(row: Mapping[str, Any], key: str) -> str:
    if key not in row:
        raise KeyError(f"row missing key={key}")
    value = row[key]
    if value is None:
        raise ValueError(f"row key={key} is None")
    text = str(value)
    if not text:
        raise ValueError(f"row key={key} is empty")
    return text


def require_int_list(row: Mapping[str, Any], key: str, *, expected_len: int) -> List[int]:
    if key not in row:
        raise KeyError(f"row missing key={key}")
    value = row[key]
    if not isinstance(value, list):
        raise ValueError(f"row key={key} must be list")
    if len(value) != expected_len:
        raise ValueError(f"row key={key} length mismatch: {len(value)} vs {expected_len}")
    return [int(item) for item in value]


def require_existing_path(row: Mapping[str, Any], key: str) -> Path:
    path = optional_path_value(row.get(key))
    if path is None:
        raise ValueError(f"row key={key} is missing/empty")
    if not path.is_file():
        raise FileNotFoundError(f"path for key={key} does not exist: {path}")
    return path


def optional_path_value(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(text)


def require_finite_np(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def tensor_metrics_to_json(metrics: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in metrics.items()}


def batch_shape_summary(batch: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            summary[key] = list(value.shape)
        elif isinstance(value, list):
            summary[key] = {"type": "list", "len": len(value)}
        else:
            summary[key] = str(type(value).__name__)
    return summary


def gradient_audit(model: nn.Module) -> Dict[str, Any]:
    groups = {
        "frame_proj": 0.0,
        "pose_proj": 0.0,
        "geo_proj": 0.0,
        "object_proj": 0.0,
        "bottleneck": 0.0,
        "spa3r_hlatent_bridge": 0.0,
        "flow_blocks": 0.0,
        "answer_scorer": 0.0,
    }
    nonzero_params = Counter()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().float().norm().cpu().item())
        if grad_norm <= 0.0:
            continue
        if name.startswith("frame_proj"):
            group = "frame_proj"
        elif name.startswith("pose_proj"):
            group = "pose_proj"
        elif name.startswith("geo_proj"):
            group = "geo_proj"
        elif name.startswith("object_proj"):
            group = "object_proj"
        elif name.startswith("bottleneck") or name.startswith("slot_embedding"):
            group = "bottleneck"
        elif (
            name.startswith("spa3r_hlatent_proj")
            or name.startswith("spa3r_hlatent_type_embedding")
            or name.startswith("spa3r_hlatent_residual")
            or name.startswith("spa3r_hlatent_gate")
        ):
            group = "spa3r_hlatent_bridge"
        elif name.startswith("spa3r_hlatent"):
            group = "spa3r_hlatent_encoder"
            groups.setdefault(group, 0.0)
        elif name.startswith("flow_blocks"):
            group = "flow_blocks"
        elif name.startswith("answer_scorer"):
            group = "answer_scorer"
        else:
            group = "other"
            groups.setdefault(group, 0.0)
        groups[group] += grad_norm
        nonzero_params[group] += 1
    missing = [group for group in ("frame_proj", "pose_proj", "geo_proj", "object_proj", "bottleneck", "flow_blocks", "answer_scorer") if groups[group] <= 0.0]
    return {
        "grad_norm_by_group": groups,
        "nonzero_param_count_by_group": dict(nonzero_params),
        "missing_required_gradient_groups": missing,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    qtype_to_index: Mapping[str, int],
    step: int,
    eval_summary: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "flow_as_picker_vsi_v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": normalize_args_for_json(args),
            "qtype_to_index": dict(qtype_to_index),
            "step": int(step),
            "eval_summary": dict(eval_summary),
        },
        path,
    )


def normalize_args_for_json(args: argparse.Namespace) -> Dict[str, Any]:
    def normalize_value(value: Any) -> Any:
        # argparse 里有些参数是 Path 列表（例如多个 visual cache root）。
        # 写 args.json 时必须递归转成普通 JSON 类型，否则训练还没开始就会因序列化失败退出。
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [normalize_value(item) for item in value]
        if isinstance(value, tuple):
            return [normalize_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): normalize_value(item) for key, item in value.items()}
        return value

    output: Dict[str, Any] = {}
    for key, value in vars(args).items():
        output[key] = normalize_value(value)
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
