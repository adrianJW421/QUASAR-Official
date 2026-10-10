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
"""Native-Qwen QUASAR spatial-bias + decoder-hidden auxiliary trainer.

这是一个隔离实验入口，不改动已有 safe richaux / metric-slot 结果。

实验核心：
1. 保留 safe MoE richaux 的 spatial bias 写回 Qwen visual feature；
2. 在 prompt 中加入少量 learnable spatial query tokens；
3. auxiliary heads 不再只接 spatial branch，而是接 Qwen decoder hidden states：
   - spatial query hidden states 做 sample-level metric / pose / BEV 监督；
   - visual token hidden states 做 patch/layout 监督。

以前是“旁边的小分支学会量尺子，再把一点 bias 写回图像 token”。
这次额外逼 Qwen/LoRA 自己的中间表示也必须会读尺子。这样 auxiliary loss
不是训练时的小测验，而是直接训练 answer path 怎么使用空间信息。
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, AutoProcessor



from quasar.train_flow_as_picker_vsi import (  # noqa: E402
    append_jsonl,
    normalize_args_for_json,
    set_seed,
    write_json,
)
from quasar.workspace_geometry_aux import (  # noqa: E402
    WorkspaceSigRegProjectionHead,
    compute_workspace_sigreg_loss,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial_moe_metric import (  # noqa: E402
    FORMAT_VERSION as BASE_METRIC_FORMAT_VERSION,
    NativeAnswerPatchSpatialDataset,
    NativeGlowPatchSpatialCollator as BasePatchSpatialCollator,
    NativeQwenGlowPatchSpatialModel as BasePatchSpatialModel,
    QWEN_VISION_KEYS,
    PROVIDED_POSE_EVAL_MODES,
    REFERENCE_SCHEMA_TO_MAPPING,
    SPATIAL_BIAS_EXPERTS,
    apply_lora_to_qwen_linears,
    audit_reference_schema_rows,
    audit_visual_cache_rows,
    compute_answer_span_ce,
    decode_label_audit,
    filter_native_rows,
    load_lora_state_dict,
    lora_state_dict,
    move_batch_to_device,
    normalize_visual_cache_roots,
    process_vision_inputs,
    read_native_jsonl,
    reference_type_mapping,
    resolve_checkpoint_file,
    trainable_parameters,
)


FORMAT_VERSION = "native_qwen3vl2b_glow_patch_spatial_decoder_aux_v1"
METRIC_VECTOR_DIM = 17
LOGGED_SCALAR_PREFIXES = (
    "loss_",
    "decoder_coordinate_",
    "spatial_query_sigreg_",
    "base_kl_",
    "metric_consistency_",
    "numeric_scaffold_",
    "global_3d_pe_",
)

# 这些 DecoderAux 分支读取“每个样本一个”的全局表示。正式 Full 从四个
# spatial-query hidden states 的均值读取；placement ablation 也允许从真实
# visual-token mask 的均值读取。Head、target 和 loss 都保持同一套。
TOKEN_DEPENDENT_DECODER_AUX_KEYS = (
    "reference_type",
    "query_pose",
    "bev",
    "object_center",
    "spatial_gate",
    "metric_scene_scale",
    "metric_room_extent",
    "metric_camera_displacement",
    "metric_object_size",
    "metric_object_pair_distance",
    "metric_camera_object_distance",
    "metric_object_count",
)
TOKEN_DEPENDENT_DECODER_AUX_MODULES = (
    "query_norm",
    "ref_type_head",
    "query_pose_head",
    "bev_head",
    "object_center_head",
    "spatial_gate_head",
    "metric_scene_scale_head",
    "metric_room_extent_head",
    "metric_camera_displacement_head",
    "metric_object_size_head",
    "metric_object_pair_distance_head",
    "metric_camera_object_distance_head",
    "metric_object_count_head",
)
TOKEN_INDEPENDENT_DECODER_AUX_MODULES = (
    "visual_norm",
    "patch_recon_head",
    "target_camera_coord_head",
    "semantic_query_coord_head",
    "bev_coord_head",
)
GLOBAL_READOUT_SOURCES = ("query_token_mean", "visual_masked_mean")


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    return value


def normalize_args_for_json(args: argparse.Namespace) -> Dict[str, Any]:
    """递归序列化 argparse 参数。

    full 训练会传入多个 cache root 和 monitor 配置。这里必须把 Path/list
    都写成普通 JSON，否则 checkpoint/args.json 会在半路写盘失败。
    """

    return {key: normalize_json_value(value) for key, value in vars(args).items()}


def masked_mean_visual_hidden_states(
    *,
    hidden_states: torch.Tensor,
    visual_token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只汇总每个样本真实的 visual-token hidden states。

    全局标签每个样本只算一次，所以先把该样本所有真实图像 token 做
    masked mean。文本、答案、padding 和 spatial-query token 都不在 mask 里。
    如果某个样本没有任何视觉 token，直接报错，不能静默造一个零向量。
    """

    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be [B,L,D], got {tuple(hidden_states.shape)}")
    if visual_token_mask.ndim != 2 or tuple(visual_token_mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError(
            "visual token mask shape mismatch: "
            f"hidden={tuple(hidden_states.shape)} mask={tuple(visual_token_mask.shape)}"
        )
    mask = visual_token_mask.to(device=hidden_states.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    empty_rows = (counts == 0).nonzero(as_tuple=False).flatten()
    if int(empty_rows.numel()) > 0:
        raise ValueError(
            "visual masked mean received samples with zero visual tokens: "
            f"batch_rows={empty_rows.detach().cpu().tolist()}"
        )
    weighted = hidden_states.float() * mask.unsqueeze(-1).to(dtype=torch.float32)
    pooled = weighted.sum(dim=1) / counts.to(dtype=torch.float32).unsqueeze(-1)
    if not bool(torch.isfinite(pooled.detach()).all().item()):
        raise ValueError("visual masked mean produced NaN or Inf")
    return pooled, counts


def is_logged_scalar(key: str, value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim == 0 and str(key).startswith(LOGGED_SCALAR_PREFIXES)


@contextmanager
def temporarily_disable_lora(module: nn.Module) -> Iterator[None]:
    states: list[tuple[nn.Module, bool, Any]] = []
    for submodule in module.modules():
        if hasattr(submodule, "lora_a") and hasattr(submodule, "lora_b"):
            had_attr = hasattr(submodule, "disable_lora")
            states.append((submodule, had_attr, getattr(submodule, "disable_lora", False)))
            setattr(submodule, "disable_lora", True)
    try:
        yield
    finally:
        for submodule, had_attr, value in states:
            if had_attr:
                setattr(submodule, "disable_lora", value)
            elif hasattr(submodule, "disable_lora"):
                delattr(submodule, "disable_lora")


@contextmanager
def temporarily_eval(module: nn.Module) -> Iterator[None]:
    was_training = bool(module.training)
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def compute_answer_span_current_to_base_kl(
    *,
    student_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if student_logits.shape != base_logits.shape:
        raise ValueError(f"student/base logits shape mismatch: {tuple(student_logits.shape)} vs {tuple(base_logits.shape)}")
    if student_logits.ndim != 3 or labels.ndim != 2 or student_logits.shape[:2] != labels.shape:
        raise ValueError(f"logits/labels shape mismatch for base KL: logits={tuple(student_logits.shape)} labels={tuple(labels.shape)}")
    shift_student = student_logits[:, :-1, :].contiguous().float()
    shift_base = base_logits[:, :-1, :].contiguous().float().detach()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    token_count = int(mask.sum().detach().cpu().item())
    if token_count <= 0:
        raise ValueError("base KL has zero supervised answer tokens after shift")
    student_selected = shift_student[mask]
    base_selected = shift_base[mask]
    student_log_probs = F.log_softmax(student_selected, dim=-1)
    base_log_probs = F.log_softmax(base_selected, dim=-1)
    student_probs = student_log_probs.exp()
    base_probs = base_log_probs.exp()
    kl = (student_probs * (student_log_probs - base_log_probs)).sum(dim=-1).mean()
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1).mean()
    base_entropy = -(base_probs * base_log_probs).sum(dim=-1).mean()
    return kl, {
        "base_kl_loss": kl.detach(),
        "base_kl_token_count": torch.tensor(float(token_count), device=student_logits.device),
        "base_kl_student_entropy": student_entropy.detach(),
        "base_kl_base_entropy": base_entropy.detach(),
    }


def build_spatial_query_tokens(count: int) -> tuple[str, ...]:
    if int(count) <= 0:
        raise ValueError(f"spatial_query_count must be positive, got {count}")
    return tuple(f"<|spatial_query_{idx}|>" for idx in range(int(count)))


class Global3DPositionEncoder(nn.Module):
    def __init__(
        self,
        *,
        qwen_hidden_size: int,
        hidden_dim: int,
        fourier_bands: int,
        coordinate_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(qwen_hidden_size) <= 0:
            raise ValueError(f"qwen_hidden_size must be positive, got {qwen_hidden_size}")
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if int(fourier_bands) <= 0:
            raise ValueError(f"fourier_bands must be positive, got {fourier_bands}")
        if float(coordinate_scale) <= 0.0:
            raise ValueError(f"coordinate_scale must be positive, got {coordinate_scale}")
        self.qwen_hidden_size = int(qwen_hidden_size)
        self.hidden_dim = int(hidden_dim)
        self.fourier_bands = int(fourier_bands)
        self.coordinate_scale = float(coordinate_scale)
        frequencies = (2.0 ** torch.arange(self.fourier_bands, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", frequencies)
        input_dim = 3 + 3 * 2 * self.fourier_bands + 1
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.qwen_hidden_size),
        )
        final = self.net[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Global3DPositionEncoder final layer must be nn.Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, *, points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if points.ndim != 2 or int(points.shape[-1]) != 3:
            raise ValueError(f"global 3D points must be [N,3], got {tuple(points.shape)}")
        if valid.shape != (points.shape[0],):
            raise ValueError(f"global 3D valid mask shape mismatch: valid={tuple(valid.shape)} points={tuple(points.shape)}")
        scaled = points.float() / float(self.coordinate_scale)
        angles = scaled.unsqueeze(-1) * self.frequencies.to(device=points.device, dtype=torch.float32).view(1, 1, -1)
        features = torch.cat(
            [
                scaled,
                torch.sin(angles).flatten(start_dim=1),
                torch.cos(angles).flatten(start_dim=1),
                valid.float().view(-1, 1),
            ],
            dim=-1,
        )
        delta = self.net(features)
        return delta * valid.float().view(-1, 1)


def strip_numeric_scaffold_tokens(text: Any) -> str:
    return re.sub(r"<\|num_[^>]+\|>", "", str(text or ""))


def parse_first_positive_number(text: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", strip_numeric_scaffold_tokens(text))
    if match is None:
        return None
    value = float(match.group(0))
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


NUMERIC_SCAFFOLD_QTYPES = {"object_abs_distance", "vsti_camera_displacement", "vsti_camera_obj_abs_dist"}
NUMERIC_SCAFFOLD_COARSE_BINS = 16
NUMERIC_SCAFFOLD_FINE_BINS = 16
NUMERIC_SCAFFOLD_TOTAL_BINS = NUMERIC_SCAFFOLD_COARSE_BINS * NUMERIC_SCAFFOLD_FINE_BINS
NUMERIC_SCAFFOLD_UNIT_TOKENS = {
    "meters": "<|num_unit_m|>",
    "meter": "<|num_unit_m|>",
    "m": "<|num_unit_m|>",
    "centimeters": "<|num_unit_cm|>",
    "centimeter": "<|num_unit_cm|>",
    "cm": "<|num_unit_cm|>",
    "count": "<|num_unit_count|>",
    "": "<|num_unit_value|>",
}


def numeric_scaffold_special_tokens() -> list[str]:
    return (
        sorted(set(NUMERIC_SCAFFOLD_UNIT_TOKENS.values()))
        + [f"<|num_c_{index:02d}|>" for index in range(NUMERIC_SCAFFOLD_COARSE_BINS)]
        + [f"<|num_f_{index:02d}|>" for index in range(NUMERIC_SCAFFOLD_FINE_BINS)]
    )


def row_canonical_qtype(row: Mapping[str, Any]) -> str:
    question = row.get("question") if isinstance(row.get("question"), Mapping) else {}
    return str(question.get("canonical_question_type") or question.get("question_type") or "").strip()


def row_answer(row: Mapping[str, Any]) -> Mapping[str, Any]:
    answer = row.get("answer")
    return answer if isinstance(answer, Mapping) else {}


def row_numeric_value(row: Mapping[str, Any]) -> float | None:
    answer = row_answer(row)
    for key in ("numeric_value", "value", "target_value"):
        value = answer.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return parse_first_positive_number(answer.get("text") or answer.get("raw") or "")


def row_numeric_unit(row: Mapping[str, Any]) -> str:
    unit = str(row_answer(row).get("numeric_unit") or "").strip().lower()
    return NUMERIC_SCAFFOLD_UNIT_TOKENS.get(unit, NUMERIC_SCAFFOLD_UNIT_TOKENS[""])


def build_numeric_scaffold_bin_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values_by_qtype: dict[str, list[float]] = {qtype: [] for qtype in sorted(NUMERIC_SCAFFOLD_QTYPES)}
    for row in rows:
        qtype = row_canonical_qtype(row)
        if qtype not in NUMERIC_SCAFFOLD_QTYPES or str(row_answer(row).get("type") or "") not in {"numeric", "count"}:
            continue
        value = row_numeric_value(row)
        if value is None or not math.isfinite(value) or value < 0.0:
            continue
        values_by_qtype[qtype].append(math.log1p(float(value)))
    stats: Dict[str, Any] = {
        "mode": "arnum_log1p_quantile",
        "coarse_bins": NUMERIC_SCAFFOLD_COARSE_BINS,
        "fine_bins": NUMERIC_SCAFFOLD_FINE_BINS,
        "total_bins": NUMERIC_SCAFFOLD_TOTAL_BINS,
        "qtypes": {},
    }
    for qtype, values in values_by_qtype.items():
        if not values:
            continue
        values = sorted(values)
        edges: list[float] = []
        for index in range(1, NUMERIC_SCAFFOLD_TOTAL_BINS):
            pos = int(round(index * (len(values) - 1) / NUMERIC_SCAFFOLD_TOTAL_BINS))
            edges.append(float(values[max(0, min(len(values) - 1, pos))]))
        stats["qtypes"][qtype] = {
            "count": len(values),
            "min_log1p": float(values[0]),
            "max_log1p": float(values[-1]),
            "edges_log1p": edges,
        }
    return stats


def numeric_bin_index(*, value: float, qtype_stats: Mapping[str, Any]) -> int:
    log_value = math.log1p(max(float(value), 0.0))
    edges = qtype_stats.get("edges_log1p")
    if not isinstance(edges, list):
        return 0
    finite_edges = [float(edge) for edge in edges if isinstance(edge, (int, float)) and math.isfinite(float(edge))]
    return max(0, min(NUMERIC_SCAFFOLD_TOTAL_BINS - 1, bisect.bisect_right(finite_edges, log_value)))


def build_numeric_scaffold_prefix(*, row: Mapping[str, Any], bin_stats: Mapping[str, Any]) -> str | None:
    qtype = row_canonical_qtype(row)
    qtype_stats = (bin_stats.get("qtypes") or {}).get(qtype) if isinstance(bin_stats.get("qtypes"), Mapping) else None
    if qtype not in NUMERIC_SCAFFOLD_QTYPES or not isinstance(qtype_stats, Mapping):
        return None
    value = row_numeric_value(row)
    if value is None or not math.isfinite(value) or value < 0.0:
        return None
    bin_index = numeric_bin_index(value=float(value), qtype_stats=qtype_stats)
    coarse = int(bin_index) // NUMERIC_SCAFFOLD_FINE_BINS
    fine = int(bin_index) % NUMERIC_SCAFFOLD_FINE_BINS
    return f"{row_numeric_unit(row)}<|num_c_{coarse:02d}|><|num_f_{fine:02d}|>"


def apply_numeric_scaffold_to_rows(*, rows: Sequence[Mapping[str, Any]], bin_stats: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], Dict[str, Any]]:
    rewritten: list[Mapping[str, Any]] = []
    changed_by_qtype: Counter[str] = Counter()
    skipped_by_reason: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        new_row = copy.deepcopy(row)
        qtype = row_canonical_qtype(new_row)
        answer = new_row.get("answer")
        if qtype not in NUMERIC_SCAFFOLD_QTYPES:
            skipped_by_reason["non_scaffold_qtype"] += 1
        elif not isinstance(answer, dict):
            skipped_by_reason["missing_answer"] += 1
        elif str(answer.get("type") or "") not in {"numeric", "count"}:
            skipped_by_reason["non_numeric_answer"] += 1
        else:
            prefix = build_numeric_scaffold_prefix(row=new_row, bin_stats=bin_stats)
            text = str(answer.get("text") or answer.get("raw") or "").strip()
            if prefix is None or not text:
                skipped_by_reason["unscaffoldable_numeric"] += 1
            else:
                answer["text"] = prefix + strip_numeric_scaffold_tokens(text).strip()
                answer["numeric_scaffold"] = {
                    "mode": "arnum",
                    "prefix": prefix,
                    "original_text": text,
                }
                changed_by_qtype[qtype] += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "sample_id": new_row.get("sample_id"),
                            "qtype": qtype,
                            "original_text": text,
                            "scaffolded_text": answer["text"],
                        }
                    )
        rewritten.append(new_row)
    audit = {
        "mode": "arnum",
        "input_rows": len(rows),
        "scaffolded_rows": int(sum(changed_by_qtype.values())),
        "changed_by_qtype": dict(changed_by_qtype),
        "skipped_by_reason": dict(skipped_by_reason),
        "examples": examples,
    }
    return rewritten, audit


def audit_numeric_scaffold_label_spans(label_audits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    unit_re = re.compile(r"<\|num_unit_[^>]+\|>")
    coarse_re = re.compile(r"<\|num_c_\d{2}\|>")
    fine_re = re.compile(r"<\|num_f_\d{2}\|>")
    records: list[dict[str, Any]] = []
    scaffold_rows = 0
    passed_rows = 0
    for index, row in enumerate(label_audits):
        target = str(row.get("target_text") or "")
        decoded = str(row.get("decoded_label_span") or "")
        target_unit = unit_re.search(target)
        target_coarse = coarse_re.search(target)
        target_fine = fine_re.search(target)
        target_has_scaffold = target_unit is not None or target_coarse is not None or target_fine is not None
        if not target_has_scaffold:
            records.append(
                {
                    "index": index,
                    "target_text": target,
                    "decoded_label_span": decoded,
                    "target_has_scaffold": False,
                    "passed": True,
                }
            )
            continue
        scaffold_rows += 1
        expected_prefix = "".join(
            match.group(0)
            for match in (target_unit, target_coarse, target_fine)
            if match is not None
        )
        literal = strip_numeric_scaffold_tokens(target).strip()
        prefix_start = decoded.find(expected_prefix) if expected_prefix else -1
        suffix_after_prefix = decoded[prefix_start + len(expected_prefix) :] if prefix_start >= 0 else ""
        literal_after_scaffold = bool(literal) and suffix_after_prefix.lstrip().startswith(literal)
        record = {
            "index": index,
            "target_text": target,
            "decoded_label_span": decoded,
            "target_has_scaffold": True,
            "expected_prefix": expected_prefix,
            "literal_without_scaffold": literal,
            "target_text_in_decoded": target in decoded,
            "prefix_in_decoded": prefix_start >= 0,
            "literal_after_scaffold": literal_after_scaffold,
            "decoded_unit_token_count": len(unit_re.findall(decoded)),
            "decoded_coarse_token_count": len(coarse_re.findall(decoded)),
            "decoded_fine_token_count": len(fine_re.findall(decoded)),
            "decoded_contains_im_end": "<|im_end|>" in decoded,
        }
        record["passed"] = bool(
            record["target_text_in_decoded"]
            and record["prefix_in_decoded"]
            and record["literal_after_scaffold"]
            and record["decoded_unit_token_count"] >= 1
            and record["decoded_coarse_token_count"] >= 1
            and record["decoded_fine_token_count"] >= 1
        )
        if bool(record["passed"]):
            passed_rows += 1
        records.append(record)
    return {
        "mode": "arnum_label_span",
        "rows": len(label_audits),
        "scaffold_target_rows": scaffold_rows,
        "passed_rows": passed_rows,
        "failed_rows": scaffold_rows - passed_rows,
        "passed": scaffold_rows == 0 or passed_rows == scaffold_rows,
        "records": records,
    }


def numeric_scaffold_bin_token_ids(tokenizer: Any, *, prefix: str, count: int) -> tuple[int, ...]:
    token_ids: list[int] = []
    for index in range(int(count)):
        token = f"<|num_{prefix}_{index:02d}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or int(token_id) < 0 or tokenizer.convert_ids_to_tokens(int(token_id)) != token:
            raise ValueError(f"numeric scaffold token is not registered as one token: {token!r}")
        token_ids.append(int(token_id))
    return tuple(token_ids)


def numeric_scaffold_ordinal_targets(*, targets: torch.Tensor, num_bins: int, sigma: float) -> torch.Tensor:
    if float(sigma) <= 0.0:
        raise ValueError(f"numeric scaffold ordinal sigma must be positive, got {sigma}")
    positions = torch.arange(int(num_bins), device=targets.device, dtype=torch.float32).unsqueeze(0)
    centers = targets.float().unsqueeze(1)
    weights = torch.exp(-0.5 * ((positions - centers) / float(sigma)) ** 2)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def metric_consistency_numeric_targets(batch: Mapping[str, Any], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_values: list[float] = []
    group_ids: list[int] = []
    valid_values: list[float] = []
    qtypes = list(batch.get("canonical_question_types", [""] * len(batch["target_texts"])))
    for qtype, answer_type, target_text in zip(qtypes, batch["answer_types"], batch["target_texts"]):
        value = parse_first_positive_number(target_text) if str(answer_type) in {"numeric", "count"} else None
        qtype = str(qtype)
        if value is None:
            target_values.append(0.0)
            group_ids.append(0)
            valid_values.append(0.0)
        elif qtype == "vsti_camera_displacement":
            target_values.append(float(value))
            group_ids.append(0)
            valid_values.append(1.0)
        elif qtype in {"vsti_camera_obj_abs_dist", "object_abs_distance", "camera_object_absolute_distance"}:
            target_values.append(float(value))
            group_ids.append(1)
            valid_values.append(1.0)
        else:
            target_values.append(0.0)
            group_ids.append(0)
            valid_values.append(0.0)
    targets = torch.tensor(target_values, device=device, dtype=torch.float32)
    valid = torch.tensor(valid_values, device=device, dtype=torch.float32)
    groups = torch.tensor(group_ids, device=device, dtype=torch.long)
    return targets, valid, groups


def gather_answer_numeric_hidden_states(
    *,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    answer_span_starts: torch.Tensor,
    answer_span_lengths: torch.Tensor,
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 3 or input_ids.ndim != 2 or hidden_states.shape[:2] != input_ids.shape:
        raise ValueError(f"hidden/input shape mismatch: hidden={tuple(hidden_states.shape)} input_ids={tuple(input_ids.shape)}")
    digit_token_ids: set[int] = set()
    for digit in "0123456789":
        for token_id in tokenizer(str(digit), add_special_tokens=False)["input_ids"]:
            digit_token_ids.add(int(token_id))
    vectors: list[torch.Tensor] = []
    valid: list[float] = []
    for row_i in range(int(input_ids.shape[0])):
        start = int(answer_span_starts[row_i].detach().cpu().item())
        length = int(answer_span_lengths[row_i].detach().cpu().item())
        end = min(start + max(length, 0), int(input_ids.shape[1]))
        positions = []
        for pos in range(start, end):
            token_id = int(input_ids[row_i, pos].detach().cpu().item())
            token_text = str(tokenizer.decode([token_id], skip_special_tokens=False))
            if token_id in digit_token_ids or re.search(r"\d", token_text):
                positions.append(max(pos - 1, 0))
        if positions:
            idx = torch.tensor(sorted(set(positions)), device=hidden_states.device, dtype=torch.long)
            vectors.append(hidden_states[row_i, idx, :].float().mean(dim=0))
            valid.append(1.0)
        else:
            vectors.append(hidden_states[row_i, max(start, 0), :].float() * 0.0)
            valid.append(0.0)
    return torch.stack(vectors, dim=0), torch.tensor(valid, device=hidden_states.device, dtype=torch.float32)


class NativeGlowPatchSpatialDecoderAuxCollator(BasePatchSpatialCollator):
    """给每个样本的 prompt 加 learnable spatial query token，并记录 token 位置。

    这些 token 不是把 metric 数值硬塞进去的信息槽，而是“查询空间信息的空白问题纸”。
    它们放在图像之后、问题之前；decoder 看到图像 token 后，再用这些 query token 汇总空间信息。
    """

    def __init__(
        self,
        *,
        processor: Any,
        max_pixels: int,
        spatial_merge_size: int,
        bev_grid_size: int,
        spatial_query_tokens: Sequence[str],
        reference_schema: str = "legacy5",
        spatial_query_tokens_enabled: bool = True,
    ) -> None:
        super().__init__(
            processor=processor,
            max_pixels=max_pixels,
            spatial_merge_size=spatial_merge_size,
            bev_grid_size=bev_grid_size,
            reference_schema=reference_schema,
        )
        self.spatial_query_tokens = tuple(str(token) for token in spatial_query_tokens)
        if not self.spatial_query_tokens:
            raise ValueError("spatial_query_tokens must not be empty")
        self.spatial_query_tokens_enabled = bool(spatial_query_tokens_enabled)
        self.spatial_query_token_ids: tuple[int, ...] = tuple(
            self._resolve_spatial_query_token_id(token) for token in self.spatial_query_tokens
        )

    def __call__(self, samples: Sequence[Any]) -> Dict[str, Any]:
        batch = super().__call__(samples)
        batch["canonical_question_types"] = [sample.canonical_question_type for sample in samples]
        return batch

    def collate_generation_prompts(self, samples: Sequence[Any]) -> Dict[str, Any]:
        batch = super().collate_generation_prompts(samples)
        batch["canonical_question_types"] = [sample.canonical_question_type for sample in samples]
        return batch

    def _resolve_spatial_query_token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or int(token_id) < 0:
            raise ValueError(f"spatial query token is not registered in tokenizer: {token!r}")
        if self.tokenizer.convert_ids_to_tokens(int(token_id)) != token:
            raise ValueError(f"spatial query token did not round-trip through tokenizer: {token!r} -> {token_id}")
        return int(token_id)

    def _prompt_with_spatial_queries(self, prompt: str) -> str:
        if not self.spatial_query_tokens_enabled:
            # 真正的 No-token 不给 decoder 留占位 token，也不保留一行假的 token 提示。
            return str(prompt)
        query_line = "Spatial query tokens: " + " ".join(self.spatial_query_tokens)
        return query_line + "\n" + str(prompt)

    def _attach_spatial_query_positions(self, encoded: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B,L] before locating spatial queries, got {tuple(input_ids.shape)}")
        if not self.spatial_query_tokens_enabled:
            for token, token_id in zip(self.spatial_query_tokens, self.spatial_query_token_ids):
                if bool((input_ids == int(token_id)).any().detach().cpu().item()):
                    raise ValueError(f"disabled spatial query token unexpectedly appeared in prompt: {token!r}")
            encoded["spatial_query_positions"] = torch.empty((int(input_ids.shape[0]), 0), dtype=torch.long)
            return encoded
        attention_mask = encoded.get("attention_mask")
        positions: list[list[int]] = []
        for batch_i in range(int(input_ids.shape[0])):
            row_positions: list[int] = []
            for token, token_id in zip(self.spatial_query_tokens, self.spatial_query_token_ids):
                hits = (input_ids[batch_i] == int(token_id)).nonzero(as_tuple=False).flatten()
                if int(hits.numel()) != 1:
                    raise ValueError(
                        f"expected exactly one spatial query token {token!r} id={token_id} in batch row {batch_i}, "
                        f"found {int(hits.numel())}"
                    )
                position = int(hits.item())
                if attention_mask is not None and int(attention_mask[batch_i, position].item()) != 1:
                    raise ValueError(f"spatial query token {token!r} landed on padded position {position} in batch row {batch_i}")
                row_positions.append(position)
            positions.append(row_positions)
        encoded["spatial_query_positions"] = torch.tensor(positions, dtype=torch.long)
        return encoded

    def _encode_prompt_batch(self, *, samples: Sequence[Any]) -> Dict[str, Any]:
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
            content.append({"type": "text", "text": self._prompt_with_spatial_queries(sample.prompt)})
            messages_batch.append([{"role": "user", "content": content}])
        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        image_inputs, video_inputs = process_vision_inputs(messages_batch)
        if video_inputs is not None:
            raise ValueError("Native-Qwen decoder-aux trainer expects image inputs only, got video_inputs")
        encoded = dict(self.processor(text=texts, images=image_inputs, padding=True, return_tensors="pt"))
        if "mm_token_type_ids" not in encoded:
            raise KeyError("Qwen3VL processor did not return mm_token_type_ids; cannot preserve native M-RoPE safely")
        if "image_grid_thw" not in encoded:
            raise KeyError("Qwen3VL processor did not return image_grid_thw")
        return self._attach_spatial_query_positions(encoded)

    def _encode_prompt_batch_from_visual_cache(self, *, samples: Sequence[Any]) -> Dict[str, Any]:
        texts: list[str] = []
        image_grid_chunks: list[torch.Tensor] = []
        for sample in samples:
            if sample.cached_image_grid_thw is None:
                raise ValueError(f"sample_id={sample.sample_id} missing cached_image_grid_thw")
            content = []
            for image_path in sample.image_paths:
                content.append({"type": "image", "image": image_path, "max_pixels": self.max_pixels})
            content.append({"type": "text", "text": self._prompt_with_spatial_queries(sample.prompt)})
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
        encoded = {
            "input_ids": input_ids,
            "attention_mask": text_inputs["attention_mask"],
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": image_grid_thw,
        }
        return self._attach_spatial_query_positions(encoded)


class NativeQwenGlowPatchSpatialDecoderAuxModel(BasePatchSpatialModel):
    """safe spatial bias + decoder hidden-state auxiliary supervision。

    spatial_flow 继续负责把 CUT3R/simple3 几何变成 visual bias；
    spatial_query_role 是一组可学习 query，不带任何 metric 数值；
    decoder_aux heads 读 Qwen decoder hidden states，直接把 auxiliary loss 反传进 LoRA。
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
        decoder_aux_weights: Mapping[str, float],
        num_spatial_bias_experts: int,
        bias_gate_temperature: float,
        safe_bias_residual_init: float,
        safe_bias_residual_max: float,
        bias_norm_max_ratio: float,
        spatial_query_count: int,
        spatial_query_tokens_enabled: bool = True,
        global_readout_source: str = "query_token_mean",
        preserve_inactive_parameters_in_optimizer: bool = False,
        spatial_query_sigreg_loss_weight: float = 0.0,
        spatial_query_sigreg_projection_dim: int = 128,
        spatial_query_sigreg_num_projections: int = 512,
        spatial_query_sigreg_num_integration_knots: int = 17,
        spatial_query_sigreg_seed: int = 20260609,
        base_kl_loss_weight: float = 0.0,
        metric_consistency_loss_weight: float = 0.0,
        metric_consistency_ar_loss_weight: float = 0.0,
        metric_consistency_query_loss_weight: float = 0.0,
        numeric_scaffold_token_ids: Sequence[int] = (),
        numeric_scaffold_bin_loss_weight: float = 0.0,
        numeric_scaffold_ordinal_loss_weight: float = 0.0,
        numeric_scaffold_ordinal_sigma: float = 1.0,
        global_3d_pe_mode: str = "none",
        global_3d_pe_hidden_dim: int = 512,
        global_3d_pe_fourier_bands: int = 8,
        global_3d_pe_coordinate_scale: float = 5.0,
        global_3d_pe_residual_scale: float = 0.05,
    ) -> None:
        super().__init__(
            qwen=qwen,
            tokenizer=tokenizer,
            qwen_hidden_size=qwen_hidden_size,
            flow_dim=flow_dim,
            num_flow_steps=num_flow_steps,
            dropout=dropout,
            bev_grid_size=bev_grid_size,
            reference_schema=reference_schema,
            aux_weights=aux_weights,
            num_spatial_bias_experts=num_spatial_bias_experts,
            bias_gate_temperature=bias_gate_temperature,
            safe_bias_residual_init=safe_bias_residual_init,
            safe_bias_residual_max=safe_bias_residual_max,
            bias_norm_max_ratio=bias_norm_max_ratio,
        )
        self.spatial_query_count = int(spatial_query_count)
        if self.spatial_query_count <= 0:
            raise ValueError(f"spatial_query_count must be positive, got {self.spatial_query_count}")
        self.spatial_query_tokens_enabled = bool(spatial_query_tokens_enabled)
        self.global_readout_source = str(global_readout_source)
        if self.global_readout_source not in GLOBAL_READOUT_SOURCES:
            raise ValueError(
                f"unsupported global_readout_source={self.global_readout_source!r}; "
                f"expected one of {GLOBAL_READOUT_SOURCES}"
            )
        self.preserve_inactive_parameters_in_optimizer = bool(preserve_inactive_parameters_in_optimizer)
        self.decoder_aux_weights = {key: float(value) for key, value in decoder_aux_weights.items()}
        if not self.spatial_query_tokens_enabled and self.global_readout_source == "query_token_mean":
            unexpected = {
                key: self.decoder_aux_weights.get(key, 0.0)
                for key in TOKEN_DEPENDENT_DECODER_AUX_KEYS
                if self.decoder_aux_weights.get(key, 0.0) != 0.0
            }
            if unexpected:
                raise ValueError(
                    "query-token readout without spatial-query tokens requires every global DecoderAux "
                    "loss weight to be zero; "
                    f"got {unexpected}"
                )
        self.spatial_query_role = nn.Parameter(torch.empty(self.spatial_query_count, int(qwen_hidden_size)))
        nn.init.normal_(self.spatial_query_role, mean=0.0, std=0.02)
        self.decoder_aux = nn.ModuleDict(
            {
                "query_norm": nn.LayerNorm(qwen_hidden_size),
                "visual_norm": nn.LayerNorm(qwen_hidden_size),
                "ref_type_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, len(self.reference_type_to_id))),
                "query_pose_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 12)),
                "bev_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, bev_grid_size * bev_grid_size)),
                "object_center_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, bev_grid_size * bev_grid_size)),
                "spatial_gate_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, len(SPATIAL_BIAS_EXPERTS))),
                "metric_scene_scale_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 1)),
                "metric_room_extent_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 3)),
                "metric_camera_displacement_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 3)),
                "metric_object_size_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 3)),
                "metric_object_pair_distance_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 3)),
                "metric_camera_object_distance_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 3)),
                "metric_object_count_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 1)),
                "patch_recon_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 3)),
                "target_camera_coord_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 3)),
                "semantic_query_coord_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 3)),
                "bev_coord_head": nn.Sequential(nn.Linear(qwen_hidden_size, flow_dim), nn.GELU(), nn.Linear(flow_dim, 2)),
            }
        )
        self.metric_consistency_query_head = nn.Sequential(
            nn.LayerNorm(qwen_hidden_size),
            nn.Linear(qwen_hidden_size, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, 2),
        )
        self.metric_consistency_ar_head = nn.Sequential(
            nn.LayerNorm(qwen_hidden_size),
            nn.Linear(qwen_hidden_size, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, 2),
        )
        self.spatial_query_sigreg_loss_weight = float(spatial_query_sigreg_loss_weight)
        self.spatial_query_sigreg_num_projections = int(spatial_query_sigreg_num_projections)
        self.spatial_query_sigreg_num_integration_knots = int(spatial_query_sigreg_num_integration_knots)
        self.spatial_query_sigreg_seed = int(spatial_query_sigreg_seed)
        self.base_kl_loss_weight = float(base_kl_loss_weight)
        self.metric_consistency_loss_weight = float(metric_consistency_loss_weight)
        self.metric_consistency_ar_loss_weight = float(metric_consistency_ar_loss_weight)
        self.metric_consistency_query_loss_weight = float(metric_consistency_query_loss_weight)
        self.numeric_scaffold_token_ids = tuple(int(token_id) for token_id in numeric_scaffold_token_ids)
        self.numeric_scaffold_token_id_to_row = {token_id: index for index, token_id in enumerate(self.numeric_scaffold_token_ids)}
        if self.numeric_scaffold_token_ids:
            self.numeric_scaffold_coarse_token_ids = numeric_scaffold_bin_token_ids(
                tokenizer, prefix="c", count=NUMERIC_SCAFFOLD_COARSE_BINS
            )
            self.numeric_scaffold_fine_token_ids = numeric_scaffold_bin_token_ids(
                tokenizer, prefix="f", count=NUMERIC_SCAFFOLD_FINE_BINS
            )
        else:
            self.numeric_scaffold_coarse_token_ids = tuple()
            self.numeric_scaffold_fine_token_ids = tuple()
        self.numeric_scaffold_bin_loss_weight = float(numeric_scaffold_bin_loss_weight)
        self.numeric_scaffold_ordinal_loss_weight = float(numeric_scaffold_ordinal_loss_weight)
        self.numeric_scaffold_ordinal_sigma = float(numeric_scaffold_ordinal_sigma)
        self.numeric_scaffold_input_delta: nn.Parameter | None = None
        self.numeric_scaffold_lm_head_delta: nn.Parameter | None = None
        if self.numeric_scaffold_token_ids:
            self.numeric_scaffold_input_delta = nn.Parameter(torch.zeros(len(self.numeric_scaffold_token_ids), int(qwen_hidden_size)))
            self.numeric_scaffold_lm_head_delta = nn.Parameter(torch.zeros(len(self.numeric_scaffold_token_ids), int(qwen_hidden_size)))
        self.global_3d_pe_mode = str(global_3d_pe_mode)
        if self.global_3d_pe_mode not in {"none", "fourier_mlp"}:
            raise ValueError(f"unsupported global_3d_pe_mode={self.global_3d_pe_mode!r}")
        self.global_3d_pe_residual_scale = float(global_3d_pe_residual_scale)
        if self.global_3d_pe_residual_scale < 0.0:
            raise ValueError("global_3d_pe_residual_scale must be non-negative")
        self.global_3d_pe: Global3DPositionEncoder | None = None
        if self.global_3d_pe_mode == "fourier_mlp":
            self.global_3d_pe = Global3DPositionEncoder(
                qwen_hidden_size=int(qwen_hidden_size),
                hidden_dim=int(global_3d_pe_hidden_dim),
                fourier_bands=int(global_3d_pe_fourier_bands),
                coordinate_scale=float(global_3d_pe_coordinate_scale),
                dropout=float(dropout),
            )
        if self.spatial_query_sigreg_loss_weight < 0.0:
            raise ValueError("spatial_query_sigreg_loss_weight must be non-negative")
        if self.base_kl_loss_weight < 0.0:
            raise ValueError("base_kl_loss_weight must be non-negative")
        if self.metric_consistency_loss_weight < 0.0 or self.metric_consistency_ar_loss_weight < 0.0 or self.metric_consistency_query_loss_weight < 0.0:
            raise ValueError("metric consistency loss weights must be non-negative")
        if self.numeric_scaffold_bin_loss_weight < 0.0 or self.numeric_scaffold_ordinal_loss_weight < 0.0:
            raise ValueError("numeric scaffold loss weights must be non-negative")
        if self.numeric_scaffold_ordinal_sigma <= 0.0:
            raise ValueError("numeric scaffold ordinal sigma must be positive")
        if not self.spatial_query_tokens_enabled and self.global_readout_source == "query_token_mean":
            if self.spatial_query_sigreg_loss_weight != 0.0:
                raise ValueError("query-token readout without tokens requires spatial_query_sigreg_loss_weight=0")
            if any(
                weight != 0.0
                for weight in (
                    self.metric_consistency_loss_weight,
                    self.metric_consistency_ar_loss_weight,
                    self.metric_consistency_query_loss_weight,
                )
            ):
                raise ValueError(
                    "query-token readout without tokens requires all query-coupled metric consistency weights to be zero"
                )
        self.spatial_query_sigreg_projection: WorkspaceSigRegProjectionHead | None = None
        if self.spatial_query_sigreg_loss_weight > 0.0 or (
            not self.spatial_query_tokens_enabled and self.global_readout_source == "query_token_mean"
        ):
            self.spatial_query_sigreg_projection = WorkspaceSigRegProjectionHead(
                hidden_size=int(qwen_hidden_size),
                projection_dim=int(spatial_query_sigreg_projection_dim),
            )
        if not self.spatial_query_tokens_enabled:
            # 四臂 shared-init 需要 optimizer parameter groups 逐项同构。此模式下保留
            # inactive 参数的 requires_grad，但 No-token forward 不读取它们，因此不会产生梯度。
            # 旧 matched-No-token 仍沿用冻结行为，避免改变已有实验语义。
            if not self.preserve_inactive_parameters_in_optimizer:
                self.spatial_query_role.requires_grad_(False)
                if self.global_readout_source == "query_token_mean":
                    for module_name in TOKEN_DEPENDENT_DECODER_AUX_MODULES:
                        for parameter in self.decoder_aux[module_name].parameters():
                            parameter.requires_grad_(False)
                    for parameter in self.metric_consistency_query_head.parameters():
                        parameter.requires_grad_(False)
                    if self.spatial_query_sigreg_projection is None:
                        raise ValueError(
                            "query-token readout without tokens must instantiate the frozen SIGReg projection "
                            "for state comparability"
                        )
                    for parameter in self.spatial_query_sigreg_projection.parameters():
                        parameter.requires_grad_(False)

    def decoder_aux_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {"spatial_query_role": self.spatial_query_role.detach().cpu()}
        if self.numeric_scaffold_token_ids:
            state["numeric_scaffold_token_ids"] = torch.tensor(self.numeric_scaffold_token_ids, dtype=torch.long)
            if self.numeric_scaffold_input_delta is None or self.numeric_scaffold_lm_head_delta is None:
                raise ValueError("numeric scaffold token ids exist but delta parameters are missing")
            state["numeric_scaffold_input_delta"] = self.numeric_scaffold_input_delta.detach().cpu()
            state["numeric_scaffold_lm_head_delta"] = self.numeric_scaffold_lm_head_delta.detach().cpu()
        for key, value in self.decoder_aux.state_dict().items():
            state[f"decoder_aux.{key}"] = value.detach().cpu()
        for key, value in self.metric_consistency_query_head.state_dict().items():
            state[f"metric_consistency_query_head.{key}"] = value.detach().cpu()
        for key, value in self.metric_consistency_ar_head.state_dict().items():
            state[f"metric_consistency_ar_head.{key}"] = value.detach().cpu()
        if self.spatial_query_sigreg_projection is not None:
            for key, value in self.spatial_query_sigreg_projection.state_dict().items():
                state[f"spatial_query_sigreg_projection.{key}"] = value.detach().cpu()
        if self.global_3d_pe is not None:
            state["global_3d_pe_residual_scale"] = torch.tensor(float(self.global_3d_pe_residual_scale), dtype=torch.float32)
            for key, value in self.global_3d_pe.state_dict().items():
                state[f"global_3d_pe.{key}"] = value.detach().cpu()
        return state

    def load_decoder_aux_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        # 已保存的读出参数必须完整恢复；关闭某项损失不等于可以忽略其权重。
        expected_keys = set(self.decoder_aux_state_dict())
        provided_keys = set(state)
        if expected_keys != provided_keys:
            raise KeyError(
                f"decoder aux checkpoint keys mismatch: missing={sorted(expected_keys - provided_keys)} "
                f"unexpected={sorted(provided_keys - expected_keys)}"
            )
        if "spatial_query_role" not in state:
            raise KeyError("decoder aux checkpoint missing spatial_query_role")
        if tuple(state["spatial_query_role"].shape) != tuple(self.spatial_query_role.shape):
            raise ValueError(
                f"spatial_query_role shape mismatch: ckpt={tuple(state['spatial_query_role'].shape)} "
                f"model={tuple(self.spatial_query_role.shape)}"
            )
        self.spatial_query_role.data.copy_(state["spatial_query_role"].to(device=self.spatial_query_role.device, dtype=self.spatial_query_role.dtype))
        if "numeric_scaffold_token_ids" in state:
            ckpt_ids = tuple(int(x) for x in state["numeric_scaffold_token_ids"].view(-1).tolist())
            if ckpt_ids != self.numeric_scaffold_token_ids:
                raise ValueError(f"numeric scaffold token ids mismatch: ckpt={ckpt_ids} model={self.numeric_scaffold_token_ids}")
            if self.numeric_scaffold_input_delta is None or self.numeric_scaffold_lm_head_delta is None:
                raise ValueError("checkpoint has numeric scaffold deltas but model was built without scaffold token ids")
            for key, parameter in (
                ("numeric_scaffold_input_delta", self.numeric_scaffold_input_delta),
                ("numeric_scaffold_lm_head_delta", self.numeric_scaffold_lm_head_delta),
            ):
                if key not in state:
                    raise KeyError(f"checkpoint missing {key}")
                if tuple(state[key].shape) != tuple(parameter.shape):
                    raise ValueError(f"{key} shape mismatch: ckpt={tuple(state[key].shape)} model={tuple(parameter.shape)}")
                parameter.data.copy_(state[key].to(device=parameter.device, dtype=parameter.dtype))
        aux_state: Dict[str, torch.Tensor] = {}
        metric_query_state: Dict[str, torch.Tensor] = {}
        metric_ar_state: Dict[str, torch.Tensor] = {}
        sigreg_state: Dict[str, torch.Tensor] = {}
        global_3d_pe_state: Dict[str, torch.Tensor] = {}
        ignored_training_only_keys: list[str] = []
        for key, value in state.items():
            if key in {"numeric_scaffold_token_ids", "numeric_scaffold_input_delta", "numeric_scaffold_lm_head_delta", "global_3d_pe_residual_scale"}:
                continue
            if key.startswith("decoder_aux."):
                aux_state[key[len("decoder_aux."):]] = value
            elif key.startswith("metric_consistency_query_head."):
                metric_query_state[key[len("metric_consistency_query_head."):]] = value
            elif key.startswith("metric_consistency_ar_head."):
                metric_ar_state[key[len("metric_consistency_ar_head."):]] = value
            elif key.startswith("spatial_query_sigreg_projection."):
                if self.spatial_query_sigreg_projection is not None:
                    sigreg_state[key[len("spatial_query_sigreg_projection."):]] = value
                else:
                    ignored_training_only_keys.append(str(key))
            elif key.startswith("global_3d_pe."):
                if self.global_3d_pe is not None:
                    global_3d_pe_state[key[len("global_3d_pe."):]] = value
                else:
                    ignored_training_only_keys.append(str(key))
        expected = set(self.decoder_aux.state_dict().keys())
        provided = set(aux_state.keys())
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            raise KeyError(f"decoder aux state mismatch: missing={missing[:20]} unexpected={unexpected[:20]}")
        self.decoder_aux.load_state_dict(aux_state)
        if metric_query_state:
            self.metric_consistency_query_head.load_state_dict(metric_query_state)
        if metric_ar_state:
            self.metric_consistency_ar_head.load_state_dict(metric_ar_state)
        if self.spatial_query_sigreg_projection is not None:
            expected_sigreg = set(self.spatial_query_sigreg_projection.state_dict().keys())
            provided_sigreg = set(sigreg_state.keys())
            if expected_sigreg and provided_sigreg:
                missing_sigreg = sorted(expected_sigreg - provided_sigreg)
                unexpected_sigreg = sorted(provided_sigreg - expected_sigreg)
                if missing_sigreg or unexpected_sigreg:
                    raise KeyError(
                        f"spatial query SIGReg projection state mismatch: "
                        f"missing={missing_sigreg[:20]} unexpected={unexpected_sigreg[:20]}"
                    )
                self.spatial_query_sigreg_projection.load_state_dict(sigreg_state)
        if self.global_3d_pe is None and global_3d_pe_state:
            raise ValueError("checkpoint contains global 3D PE weights but model was built with global_3d_pe_mode='none'")
        if self.global_3d_pe is not None and global_3d_pe_state:
            expected_global_3d = set(self.global_3d_pe.state_dict().keys())
            provided_global_3d = set(global_3d_pe_state.keys())
            missing_global_3d = sorted(expected_global_3d - provided_global_3d)
            unexpected_global_3d = sorted(provided_global_3d - expected_global_3d)
            if missing_global_3d or unexpected_global_3d:
                raise KeyError(
                    f"global 3D PE state mismatch: missing={missing_global_3d[:20]} unexpected={unexpected_global_3d[:20]}"
                )
            self.global_3d_pe.load_state_dict(global_3d_pe_state)
            if "global_3d_pe_residual_scale" in state:
                self.global_3d_pe_residual_scale = float(state["global_3d_pe_residual_scale"].view(()).item())

    def apply_numeric_scaffold_input_delta(self, *, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if not self.numeric_scaffold_token_ids or self.numeric_scaffold_input_delta is None:
            return inputs_embeds
        patched = inputs_embeds
        for token_id, row_i in self.numeric_scaffold_token_id_to_row.items():
            mask = input_ids == int(token_id)
            if bool(mask.any().detach().cpu().item()):
                patched = patched.clone() if patched is inputs_embeds else patched
                patched[mask] = patched[mask] + self.numeric_scaffold_input_delta[row_i].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return patched

    def apply_numeric_scaffold_logit_delta(self, *, logits: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.numeric_scaffold_token_ids or self.numeric_scaffold_lm_head_delta is None:
            return logits
        delta = torch.matmul(
            hidden_states.to(dtype=self.numeric_scaffold_lm_head_delta.dtype),
            self.numeric_scaffold_lm_head_delta.to(device=hidden_states.device).t(),
        ).to(dtype=logits.dtype)
        patched = logits.clone()
        token_ids = torch.tensor(self.numeric_scaffold_token_ids, device=logits.device, dtype=torch.long)
        patched[..., token_ids] = patched[..., token_ids] + delta.to(dtype=logits.dtype)
        return patched

    def _make_global_3d_pe_delta(
        self,
        *,
        image_embeds: torch.Tensor,
        points: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = image_embeds.new_zeros(image_embeds.shape)
        zero_scalar = image_embeds.detach().float().sum() * 0.0
        if self.global_3d_pe is None:
            return zero, {
                "global_3d_pe_enabled": torch.tensor(0, device=image_embeds.device),
                "global_3d_pe_valid_tokens": zero_scalar,
                "global_3d_pe_abs_mean": zero_scalar,
                "global_3d_pe_norm_ratio_mean": zero_scalar,
                "global_3d_pe_norm_ratio_max": zero_scalar,
                "global_3d_pe_residual_scale": torch.tensor(0.0, device=image_embeds.device),
            }
        if points is None or valid is None:
            raise ValueError("global 3D PE is enabled but batch is missing spatial_points or spatial_token_valid")
        points = points.to(device=image_embeds.device, dtype=torch.float32)
        valid = valid.to(device=image_embeds.device).bool()
        if tuple(points.shape) != (int(image_embeds.shape[0]), 3):
            raise ValueError(f"global 3D points must match visual token count, got points={tuple(points.shape)} image={tuple(image_embeds.shape)}")
        if tuple(valid.shape) != (int(image_embeds.shape[0]),):
            raise ValueError(f"global 3D valid mask must be [N], got valid={tuple(valid.shape)} image={tuple(image_embeds.shape)}")
        delta = self.global_3d_pe(points=points, valid=valid).to(device=image_embeds.device, dtype=torch.float32)
        delta = delta * float(self.global_3d_pe_residual_scale)
        visual_norm = torch.linalg.norm(image_embeds.detach().float(), dim=-1).clamp_min(1e-6)
        delta_norm = torch.linalg.norm(delta, dim=-1)
        valid_float = valid.float()
        valid_count = valid_float.sum().clamp_min(1.0)
        norm_ratio = delta_norm / visual_norm
        audit = {
            "global_3d_pe_enabled": torch.tensor(1, device=image_embeds.device),
            "global_3d_pe_valid_tokens": valid_float.sum().detach(),
            "global_3d_pe_abs_mean": ((delta.detach().abs().mean(dim=-1) * valid_float).sum() / valid_count),
            "global_3d_pe_norm_ratio_mean": ((norm_ratio.detach() * valid_float).sum() / valid_count),
            "global_3d_pe_norm_ratio_max": norm_ratio.detach().max(),
            "global_3d_pe_residual_scale": torch.tensor(float(self.global_3d_pe_residual_scale), device=image_embeds.device),
        }
        return delta.to(dtype=image_embeds.dtype), audit

    def _replace_spatial_query_embeddings(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_query_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if spatial_query_positions.ndim != 2:
            raise ValueError(f"spatial_query_positions must be [B,K], got {tuple(spatial_query_positions.shape)}")
        if not self.spatial_query_tokens_enabled:
            if tuple(spatial_query_positions.shape) != (int(inputs_embeds.shape[0]), 0):
                raise ValueError(
                    "matched No-token expects an empty [B,0] spatial_query_positions tensor, "
                    f"got {tuple(spatial_query_positions.shape)}"
                )
            zero = inputs_embeds.detach().float().sum() * 0.0
            return inputs_embeds, {
                "spatial_query_token_count": torch.tensor(0, device=inputs_embeds.device),
                "spatial_query_abs_mean": zero.to(device=inputs_embeds.device),
            }
        if int(spatial_query_positions.shape[1]) != self.spatial_query_count:
            raise ValueError(
                f"spatial_query_positions K mismatch: positions={tuple(spatial_query_positions.shape)} "
                f"expected K={self.spatial_query_count}"
            )
        if int(spatial_query_positions.shape[0]) != int(inputs_embeds.shape[0]):
            raise ValueError("spatial_query_positions batch size must match inputs_embeds")
        spatial_query_positions = spatial_query_positions.to(device=inputs_embeds.device, dtype=torch.long)
        if int(spatial_query_positions.min().detach().cpu().item()) < 0 or int(spatial_query_positions.max().detach().cpu().item()) >= int(inputs_embeds.shape[1]):
            raise ValueError(
                f"spatial_query_positions out of sequence bounds: shape={tuple(spatial_query_positions.shape)} "
                f"seq_len={int(inputs_embeds.shape[1])}"
            )
        batch_idx = torch.arange(int(inputs_embeds.shape[0]), device=inputs_embeds.device).unsqueeze(1).expand_as(spatial_query_positions)
        query_attention = attention_mask.to(device=inputs_embeds.device)[batch_idx, spatial_query_positions]
        if not bool((query_attention == 1).all().detach().cpu().item()):
            raise ValueError("spatial query positions must all point to attended prompt tokens")
        query_embeds = self.spatial_query_role.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).unsqueeze(0)
        query_embeds = query_embeds.expand(int(inputs_embeds.shape[0]), -1, -1)
        patched = inputs_embeds.clone()
        patched[batch_idx, spatial_query_positions, :] = query_embeds
        return patched, {
            "spatial_query_token_count": torch.tensor(spatial_query_positions.numel(), device=inputs_embeds.device),
            "spatial_query_abs_mean": query_embeds.detach().float().abs().mean().to(device=inputs_embeds.device),
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_audit_tensors: bool = False,
    ) -> Dict[str, torch.Tensor]:
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
            pose_input_mask=batch["effective_pose_input_mask"].float(),
        )
        spatial_bias = flow_out["spatial_bias"]
        qwen_out = self._qwen_forward_with_spatial_bias(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            spatial_bias=spatial_bias,
            spatial_query_positions=batch["spatial_query_positions"].long(),
            cached_image_embeds=batch.get("cached_image_embeds"),
            cached_deepstack_features=batch.get("cached_deepstack_features"),
            global_3d_points=batch.get("spatial_points"),
            global_3d_valid=batch.get("spatial_token_valid"),
        )
        logits = qwen_out["logits"]
        answer_ce = compute_answer_span_ce(logits=logits, labels=labels)
        flow_aux_losses = self._compute_aux_losses(batch=batch, flow_out=flow_out)
        decoder_aux_out = self._decoder_aux_predictions(batch=batch, qwen_out=qwen_out)
        decoder_coordinate_raw = self._compute_decoder_coordinate_raw_losses(batch=batch, aux_out=decoder_aux_out)
        decoder_aux_losses = self._compute_decoder_aux_losses(
            batch=batch,
            aux_out=decoder_aux_out,
            coordinate_raw_losses=decoder_coordinate_raw,
        )
        stabilization_losses = self._compute_stabilization_losses(
            batch=batch,
            qwen_out=qwen_out,
            spatial_bias=spatial_bias,
        )
        metric_consistency_losses = self._compute_metric_consistency_losses(
            batch=batch,
            qwen_out=qwen_out,
            aux_out=decoder_aux_out,
        )
        numeric_scaffold_losses = self._compute_numeric_scaffold_losses(logits=logits, labels=labels)
        total_loss = answer_ce
        for value in flow_aux_losses.values():
            total_loss = total_loss + value
        for value in decoder_aux_losses.values():
            total_loss = total_loss + value
        for key, value in stabilization_losses.items():
            if str(key).startswith("loss_"):
                total_loss = total_loss + value
        for key, value in metric_consistency_losses.items():
            if str(key).startswith("loss_"):
                total_loss = total_loss + value
        for key, value in numeric_scaffold_losses.items():
            if str(key).startswith("loss_"):
                total_loss = total_loss + value

        self.last_forward_audit = {
            "numeric_scaffold_token_count": len(self.numeric_scaffold_token_ids),
            "image_feature_tokens": int(qwen_out["image_feature_tokens"].detach().cpu().item()),
            "visual_cache_used": bool(int(qwen_out["visual_cache_used"].detach().cpu().item())),
            "spatial_bias_tokens": int(spatial_bias.shape[0]),
            "spatial_query_count": self.spatial_query_count,
            "spatial_query_tokens_enabled": self.spatial_query_tokens_enabled,
            "global_readout_source": self.global_readout_source,
            "spatial_query_token_count": int(qwen_out["spatial_query_token_count"].detach().cpu().item()),
            "spatial_query_abs_mean": float(qwen_out["spatial_query_abs_mean"].detach().cpu().item()),
            "raw_spatial_bias_abs_mean": float(spatial_bias.detach().float().abs().mean().cpu().item()),
            "safe_spatial_bias_abs_mean": float(qwen_out["safe_spatial_bias_abs_mean"].detach().cpu().item()),
            "safe_bias_residual_scale": float(qwen_out["safe_bias_residual_scale"].detach().cpu().item()),
            "safe_bias_norm_ratio_mean": float(qwen_out["safe_bias_norm_ratio_mean"].detach().cpu().item()),
            "safe_bias_norm_ratio_max": float(qwen_out["safe_bias_norm_ratio_max"].detach().cpu().item()),
            "bias_gate_mean": flow_out["bias_gate_probs"].detach().float().mean(dim=0).cpu().tolist(),
            "bias_gate_max_mean": float(flow_out["bias_gate_probs"].detach().float().amax(dim=-1).mean().cpu().item()),
            "answer_label_tokens": int((labels != -100).sum().detach().cpu().item()),
            "reference_schema": self.reference_schema,
            "target_camera_aux_tokens": int(batch["target_camera_token_mask"].detach().sum().cpu().item()),
            "semantic_query_aux_tokens": int(batch["semantic_query_token_mask"].detach().sum().cpu().item()),
            "canonical_bev_aux_tokens": int(batch["canonical_bev_token_mask"].detach().sum().cpu().item()),
            "object_center_aux_samples": int(batch["object_center_valid"].detach().sum().cpu().item()),
            "global_readout_input_shape": list(decoder_aux_out["_global_readout_input"].shape),
            "global_readout_hidden_shape": list(decoder_aux_out["_global_readout_hidden"].shape),
            "sequence_length": int(qwen_out["hidden_states"].shape[1]),
            "attention_token_counts_per_sample": batch["attention_mask"].detach().sum(dim=1).cpu().tolist(),
            "visual_token_counts_per_sample": decoder_aux_out["_visual_token_counts"].detach().cpu().tolist(),
            "padding_token_counts_per_sample": decoder_aux_out["_padding_token_counts"].detach().cpu().tolist(),
            "answer_token_counts_per_sample": decoder_aux_out["_answer_token_counts"].detach().cpu().tolist(),
            "spatial_query_token_counts_per_sample": decoder_aux_out[
                "_spatial_query_token_counts"
            ].detach().cpu().tolist(),
            "other_prompt_token_counts_per_sample": decoder_aux_out[
                "_other_prompt_token_counts"
            ].detach().cpu().tolist(),
            "empty_visual_mask_rows": [],
        }
        for key, value in stabilization_losses.items():
            if is_logged_scalar(key, value):
                self.last_forward_audit[key] = float(value.detach().cpu().item())
        for key, value in metric_consistency_losses.items():
            if is_logged_scalar(key, value):
                self.last_forward_audit[key] = float(value.detach().cpu().item())
        for key, value in numeric_scaffold_losses.items():
            if is_logged_scalar(key, value):
                self.last_forward_audit[key] = float(value.detach().cpu().item())
        for key, value in qwen_out.items():
            if is_logged_scalar(key, value):
                self.last_forward_audit[key] = float(value.detach().cpu().item())
        output = {
            "loss": total_loss,
            "answer_ce_loss": answer_ce.detach(),
            "logits": logits,
            "spatial_bias": spatial_bias,
        }
        for key, value in flow_aux_losses.items():
            output[key] = value.detach()
        for key, value in decoder_aux_losses.items():
            output[key] = value.detach()
        for key, value in stabilization_losses.items():
            if is_logged_scalar(key, value):
                output[key] = value.detach()
        for key, value in metric_consistency_losses.items():
            if is_logged_scalar(key, value):
                output[key] = value.detach()
        for key, value in numeric_scaffold_losses.items():
            if is_logged_scalar(key, value):
                output[key] = value.detach()
        output["decoder_coordinate_target_camera_raw_loss"] = decoder_coordinate_raw["target_camera"].detach()
        output["decoder_coordinate_semantic_query_raw_loss"] = decoder_coordinate_raw["semantic_query"].detach()
        output["decoder_coordinate_target_camera_valid_tokens"] = decoder_coordinate_raw["target_camera_valid_tokens"].detach()
        output["decoder_coordinate_semantic_query_valid_tokens"] = decoder_coordinate_raw["semantic_query_valid_tokens"].detach()
        if return_audit_tensors:
            # 这些张量只给启动前的同 checkpoint 因果/梯度硬门使用；正式训练默认不保留额外图引用。
            zero = logits.float().sum() * 0.0

            def audit_loss_sum(keys: Sequence[str]) -> torch.Tensor:
                result = zero
                for key in keys:
                    if key in decoder_aux_losses:
                        result = result + decoder_aux_losses[key]
                return result

            output["_audit_answer_ce_loss"] = answer_ce
            output["_audit_visual_hidden_states"] = qwen_out["visual_hidden_states"]
            output["_audit_visual_token_mask"] = qwen_out["visual_token_mask"]
            output["_audit_spatial_query_hidden_states"] = qwen_out["spatial_query_hidden_states"]
            output["_audit_global_readout_input"] = decoder_aux_out["_global_readout_input"]
            output["_audit_global_readout_hidden"] = decoder_aux_out["_global_readout_hidden"]
            global_head_output_keys = (
                "decoder_ref_type_logits",
                "decoder_query_pose_pred",
                "decoder_bev_logits",
                "decoder_object_center_logits",
                "decoder_spatial_gate_logits",
                "decoder_metric_scene_scale_pred",
                "decoder_metric_room_extent_pred",
                "decoder_metric_camera_displacement_pred",
                "decoder_metric_object_size_pred",
                "decoder_metric_object_pair_distance_pred",
                "decoder_metric_camera_object_distance_pred",
                "decoder_metric_object_count_pred",
            )
            if all(key in decoder_aux_out for key in global_head_output_keys):
                output["_audit_global_head_outputs"] = torch.cat(
                    [decoder_aux_out[key].float().reshape(batch_size, -1) for key in global_head_output_keys],
                    dim=1,
                )
            output["_audit_target_camera_raw_loss"] = decoder_coordinate_raw["target_camera"]
            output["_audit_semantic_query_raw_loss"] = decoder_coordinate_raw["semantic_query"]
            output["_audit_target_camera_pred"] = decoder_aux_out["decoder_target_camera_coord_pred"]
            output["_audit_semantic_query_pred"] = decoder_aux_out["decoder_semantic_query_coord_pred"]
            output["_audit_local_decoder_loss"] = audit_loss_sum(
                (
                    "loss_decoder_patch_consistency",
                    "loss_decoder_target_camera_coord",
                    "loss_decoder_semantic_query_coord",
                    "loss_decoder_bev_coord",
                )
            )
            output["_audit_global_spatial_decoder_loss"] = audit_loss_sum(
                (
                    "loss_decoder_bev",
                    "loss_decoder_object_center",
                    "loss_decoder_metric_scene_scale",
                    "loss_decoder_metric_room_extent",
                    "loss_decoder_metric_camera_displacement",
                    "loss_decoder_metric_object_size",
                    "loss_decoder_metric_object_pair_distance",
                    "loss_decoder_metric_camera_object_distance",
                    "loss_decoder_metric_object_count",
                )
            )
            output["_audit_condition_probe_decoder_loss"] = audit_loss_sum(
                (
                    "loss_decoder_ref_type",
                    "loss_decoder_query_pose",
                    "loss_decoder_spatial_gate",
                )
            )
        return output

    def _compute_numeric_scaffold_losses(self, *, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        hard_weight = float(self.numeric_scaffold_bin_loss_weight)
        ordinal_weight = float(self.numeric_scaffold_ordinal_loss_weight)
        if hard_weight <= 0.0 and ordinal_weight <= 0.0:
            return losses
        if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
            raise ValueError(f"numeric scaffold logits/labels mismatch: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
        if not self.numeric_scaffold_coarse_token_ids or not self.numeric_scaffold_fine_token_ids:
            raise ValueError("numeric scaffold bin loss requires --numeric-scaffold-mode arnum")
        zero = logits.float().sum() * 0.0
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous().long()

        def collect(token_ids_tuple: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            token_ids = torch.tensor(token_ids_tuple, device=logits.device, dtype=torch.long)
            class_logits = shift_logits.index_select(dim=-1, index=token_ids)
            target_indices = torch.full_like(shift_labels, -1)
            for class_i, token_id in enumerate(token_ids_tuple):
                target_indices = torch.where(
                    shift_labels == int(token_id),
                    torch.full_like(target_indices, int(class_i)),
                    target_indices,
                )
            mask = target_indices >= 0
            count = mask.sum().to(device=logits.device, dtype=shift_logits.dtype)
            if int(mask.sum().detach().cpu().item()) <= 0:
                return zero, zero, count, zero
            selected_logits = class_logits[mask]
            selected_targets = target_indices[mask]
            hard_sum = F.cross_entropy(selected_logits, selected_targets, reduction="sum")
            soft_targets = numeric_scaffold_ordinal_targets(
                targets=selected_targets,
                num_bins=int(selected_logits.shape[-1]),
                sigma=float(self.numeric_scaffold_ordinal_sigma),
            )
            ordinal_sum = (-(soft_targets * F.log_softmax(selected_logits, dim=-1)).sum(dim=-1)).sum()
            accuracy = (selected_logits.argmax(dim=-1) == selected_targets).float().mean()
            return hard_sum, ordinal_sum, count, accuracy

        coarse_hard_sum, coarse_ordinal_sum, coarse_count, coarse_acc = collect(self.numeric_scaffold_coarse_token_ids)
        fine_hard_sum, fine_ordinal_sum, fine_count, fine_acc = collect(self.numeric_scaffold_fine_token_ids)
        total_count = (coarse_count + fine_count).clamp_min(1.0)
        hard_loss = (coarse_hard_sum + fine_hard_sum) / total_count
        ordinal_loss = (coarse_ordinal_sum + fine_ordinal_sum) / total_count
        if hard_weight > 0.0:
            losses["loss_numeric_scaffold_bin_ce"] = hard_loss * hard_weight
        if ordinal_weight > 0.0:
            losses["loss_numeric_scaffold_ordinal"] = ordinal_loss * ordinal_weight
        losses["numeric_scaffold_bin_ce_loss"] = hard_loss.detach()
        losses["numeric_scaffold_ordinal_loss"] = ordinal_loss.detach()
        losses["numeric_scaffold_coarse_token_count"] = coarse_count.detach()
        losses["numeric_scaffold_fine_token_count"] = fine_count.detach()
        losses["numeric_scaffold_total_token_count"] = (coarse_count + fine_count).detach()
        losses["numeric_scaffold_coarse_accuracy"] = coarse_acc.detach()
        losses["numeric_scaffold_fine_accuracy"] = fine_acc.detach()
        losses["numeric_scaffold_bin_loss_weight"] = torch.tensor(hard_weight, device=logits.device, dtype=shift_logits.dtype)
        losses["numeric_scaffold_ordinal_loss_weight"] = torch.tensor(ordinal_weight, device=logits.device, dtype=shift_logits.dtype)
        return losses

    def _qwen_forward_with_spatial_bias(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        spatial_bias: torch.Tensor,
        spatial_query_positions: torch.Tensor,
        cached_image_embeds: Optional[torch.Tensor] = None,
        cached_deepstack_features: Optional[Sequence[torch.Tensor]] = None,
        global_3d_points: Optional[torch.Tensor] = None,
        global_3d_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base = self.base_qwen()
        qwen_model = base.model
        inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
        inputs_embeds = self.apply_numeric_scaffold_input_delta(input_ids=input_ids, inputs_embeds=inputs_embeds)
        inputs_embeds, query_audit = self._replace_spatial_query_embeddings(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            spatial_query_positions=spatial_query_positions,
        )
        if cached_image_embeds is None:
            if pixel_values is None:
                raise ValueError("pixel_values is required when cached_image_embeds is not provided")
            image_outputs = qwen_model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_source = list(getattr(image_outputs, "deepstack_features", ()) or ())
            visual_cache_used = False
        else:
            image_embeds = cached_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_source = list(cached_deepstack_features or ())
            visual_cache_used = True
        if image_embeds.shape != spatial_bias.shape:
            raise ValueError(
                f"spatial bias must match Qwen image embeds exactly, got bias={tuple(spatial_bias.shape)} "
                f"image_embeds={tuple(image_embeds.shape)}"
            )
        safe_spatial_bias, bias_audit = self._make_safe_spatial_bias(spatial_bias=spatial_bias, image_embeds=image_embeds)
        global_3d_pe_delta, global_3d_pe_audit = self._make_global_3d_pe_delta(
            image_embeds=image_embeds,
            points=global_3d_points,
            valid=global_3d_valid,
        )
        visual_delta = safe_spatial_bias.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) + global_3d_pe_delta.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_embeds = image_embeds + visual_delta
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
            deepstack_visual_embeds.append(
                deep.to(inputs_embeds.device, inputs_embeds.dtype)
                + visual_delta.to(inputs_embeds.device, inputs_embeds.dtype)
            )
        position_ids = qwen_model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        language_kwargs = {
            "input_ids": None,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "inputs_embeds": inputs_embeds,
            "use_cache": False,
        }
        if deepstack_visual_embeds:
            language_kwargs["visual_pos_masks"] = image_mask_1d
            language_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
        language_outputs = qwen_model.language_model(**language_kwargs)
        hidden_states = language_outputs[0]
        logits = self.apply_numeric_scaffold_logit_delta(logits=base.lm_head(hidden_states), hidden_states=hidden_states)
        spatial_query_positions = spatial_query_positions.to(device=hidden_states.device, dtype=torch.long)
        if self.spatial_query_tokens_enabled:
            batch_idx = torch.arange(int(hidden_states.shape[0]), device=hidden_states.device).unsqueeze(1).expand_as(spatial_query_positions)
            query_hidden = hidden_states[batch_idx, spatial_query_positions, :]
        else:
            if int(spatial_query_positions.numel()) != 0:
                raise ValueError("matched No-token reached decoder with non-empty spatial query positions")
            query_hidden = hidden_states.new_empty((int(hidden_states.shape[0]), 0, int(hidden_states.shape[-1])))
        visual_hidden = hidden_states[image_mask_1d]
        if visual_hidden.shape[0] != spatial_bias.shape[0]:
            raise ValueError(
                f"decoder visual hidden count must match spatial tokens: visual={tuple(visual_hidden.shape)} "
                f"bias={tuple(spatial_bias.shape)}"
            )
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "visual_hidden_states": visual_hidden,
            "visual_token_mask": image_mask_1d,
            "spatial_query_hidden_states": query_hidden,
            "image_feature_tokens": torch.tensor(image_embeds.shape[0], device=logits.device),
            "visual_cache_used": torch.tensor(1 if visual_cache_used else 0, device=logits.device),
            **query_audit,
            **bias_audit,
            **global_3d_pe_audit,
        }

    def _compute_stabilization_losses(
        self,
        *,
        batch: Mapping[str, Any],
        qwen_out: Mapping[str, torch.Tensor],
        spatial_bias: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        if self.spatial_query_sigreg_loss_weight > 0.0:
            if self.spatial_query_sigreg_projection is None:
                raise ValueError("spatial query SIGReg projection head is missing while SIGReg loss is enabled")
            if self.global_readout_source == "query_token_mean":
                sigreg_hidden = qwen_out["spatial_query_hidden_states"]
            else:
                # masked mean 每个样本只有一个向量，而正式 micro-batch=1；SIGReg 至少需要
                # 两个表示才能估计分布。这里对同一合法 visual mask 内的真实视觉 states
                # 做正则，不包含 padding/text/answer/query token，也不重复任何全局标签。
                sigreg_hidden = qwen_out["visual_hidden_states"].unsqueeze(0)
            projected = self.spatial_query_sigreg_projection(sigreg_hidden)
            sigreg_loss, sigreg_metrics = compute_workspace_sigreg_loss(
                projected,
                num_projections=self.spatial_query_sigreg_num_projections,
                num_integration_knots=self.spatial_query_sigreg_num_integration_knots,
                seed=self.spatial_query_sigreg_seed,
            )
            losses["loss_spatial_query_sigreg"] = sigreg_loss * self.spatial_query_sigreg_loss_weight
            losses["spatial_query_sigreg_loss"] = sigreg_loss.detach()
            losses["spatial_query_sigreg_weight"] = torch.tensor(
                float(self.spatial_query_sigreg_loss_weight),
                device=sigreg_loss.device,
                dtype=sigreg_loss.dtype,
            )
            metric_map = {
                "workspace_sigreg_projection_mean_abs": "spatial_query_sigreg_projection_mean_abs",
                "workspace_sigreg_projection_std_mean": "spatial_query_sigreg_projection_std_mean",
                "workspace_sigreg_projection_std_min": "spatial_query_sigreg_projection_std_min",
                "workspace_sigreg_offdiag_cosine": "spatial_query_sigreg_offdiag_cosine",
            }
            for source_key, target_key in metric_map.items():
                if source_key in sigreg_metrics:
                    losses[target_key] = sigreg_metrics[source_key].detach()
        if self.base_kl_loss_weight > 0.0:
            with torch.no_grad(), temporarily_disable_lora(self.qwen), temporarily_eval(self.qwen):
                base_out = self._qwen_forward_with_spatial_bias(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch["image_grid_thw"],
                    mm_token_type_ids=batch["mm_token_type_ids"],
                    spatial_bias=spatial_bias.detach(),
                    spatial_query_positions=batch["spatial_query_positions"].long(),
                    cached_image_embeds=batch.get("cached_image_embeds"),
                    cached_deepstack_features=batch.get("cached_deepstack_features"),
                    global_3d_points=batch.get("spatial_points"),
                    global_3d_valid=batch.get("spatial_token_valid"),
                )
            base_kl, base_kl_metrics = compute_answer_span_current_to_base_kl(
                student_logits=qwen_out["logits"],
                base_logits=base_out["logits"],
                labels=batch["labels"],
            )
            losses["loss_base_kl"] = base_kl * self.base_kl_loss_weight
            losses["base_kl_loss"] = base_kl.detach()
            losses["base_kl_weight"] = torch.tensor(
                float(self.base_kl_loss_weight),
                device=base_kl.device,
                dtype=base_kl.dtype,
            )
            for key, value in base_kl_metrics.items():
                losses[key] = value.detach()
        return losses

    def _compute_metric_consistency_losses(
        self,
        *,
        batch: Mapping[str, Any],
        qwen_out: Mapping[str, torch.Tensor],
        aux_out: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total_weight = float(self.metric_consistency_loss_weight)
        ar_weight = float(self.metric_consistency_ar_loss_weight)
        query_weight = float(self.metric_consistency_query_loss_weight)
        if total_weight <= 0.0 and ar_weight <= 0.0 and query_weight <= 0.0:
            return losses
        hidden_states = qwen_out["hidden_states"]
        target_values, numeric_valid, group_ids = metric_consistency_numeric_targets(batch, device=hidden_states.device)
        ar_hidden, digit_valid = gather_answer_numeric_hidden_states(
            hidden_states=hidden_states,
            input_ids=batch["input_ids"],
            answer_span_starts=batch["answer_span_starts"].long(),
            answer_span_lengths=batch["answer_span_lengths"].long(),
            tokenizer=self.tokenizer,
        )
        valid = numeric_valid * digit_valid
        valid_count = valid.sum().clamp_min(1.0)
        query_hidden = aux_out["_global_readout_hidden"].float()
        query_logits = self.metric_consistency_query_head(query_hidden).float()
        ar_logits = self.metric_consistency_ar_head(ar_hidden).float()
        pred_query_log = query_logits.gather(dim=1, index=group_ids.view(-1, 1)).squeeze(1)
        pred_ar_log = ar_logits.gather(dim=1, index=group_ids.view(-1, 1)).squeeze(1)
        target_log = torch.log1p(target_values.float().clamp_min(0.0))
        query_loss_raw = F.smooth_l1_loss(pred_query_log, target_log, reduction="none")
        ar_loss_raw = F.smooth_l1_loss(pred_ar_log, target_log, reduction="none")
        consistency_loss_raw = F.smooth_l1_loss(pred_ar_log, pred_query_log.detach(), reduction="none")
        query_loss = (query_loss_raw * valid).sum() / valid_count
        ar_loss = (ar_loss_raw * valid).sum() / valid_count
        consistency_loss = (consistency_loss_raw * valid).sum() / valid_count
        if query_weight > 0.0:
            losses["loss_metric_consistency_query"] = query_loss * query_weight
        if ar_weight > 0.0:
            losses["loss_metric_consistency_ar"] = ar_loss * ar_weight
        if total_weight > 0.0:
            losses["loss_metric_consistency"] = consistency_loss * total_weight
        losses["metric_consistency_query_raw_loss"] = query_loss.detach()
        losses["metric_consistency_ar_raw_loss"] = ar_loss.detach()
        losses["metric_consistency_raw_loss"] = consistency_loss.detach()
        losses["metric_consistency_valid_count"] = valid_count.detach()
        losses["metric_consistency_query_pred_mean"] = ((torch.expm1(pred_query_log).clamp_min(0.0) * valid).sum() / valid_count).detach()
        losses["metric_consistency_ar_pred_mean"] = ((torch.expm1(pred_ar_log).clamp_min(0.0) * valid).sum() / valid_count).detach()
        losses["metric_consistency_target_mean"] = ((target_values * valid).sum() / valid_count).detach()
        return losses

    def _decoder_aux_predictions(self, *, batch: Mapping[str, Any], qwen_out: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        visual_hidden = self.decoder_aux["visual_norm"](qwen_out["visual_hidden_states"].float())
        outputs = {
            "decoder_patch_recon": self.decoder_aux["patch_recon_head"](visual_hidden),
            "decoder_target_camera_coord_pred": self.decoder_aux["target_camera_coord_head"](visual_hidden),
            "decoder_semantic_query_coord_pred": self.decoder_aux["semantic_query_coord_head"](visual_hidden),
            "decoder_bev_coord_pred": self.decoder_aux["bev_coord_head"](visual_hidden),
        }
        visual_token_mask = qwen_out["visual_token_mask"].to(dtype=torch.bool)
        pooled_visual_hidden, visual_counts = masked_mean_visual_hidden_states(
            hidden_states=qwen_out["hidden_states"],
            visual_token_mask=visual_token_mask,
        )
        legacy_disabled_query_readout = (
            self.global_readout_source == "query_token_mean" and not self.spatial_query_tokens_enabled
        )
        if self.global_readout_source == "query_token_mean" and not legacy_disabled_query_readout:
            query_hidden = qwen_out["spatial_query_hidden_states"].float()
            global_readout_input = query_hidden.mean(dim=1)
        else:
            global_readout_input = pooled_visual_hidden
        sample_hidden = (
            global_readout_input.detach() * 0.0
            if legacy_disabled_query_readout
            else self.decoder_aux["query_norm"](global_readout_input)
        )
        batch_size = int(qwen_out["hidden_states"].shape[0])
        attention_mask = batch["attention_mask"].to(device=visual_token_mask.device, dtype=torch.bool)
        labels = batch["labels"].to(device=visual_token_mask.device)
        spatial_query_mask = torch.zeros_like(visual_token_mask)
        spatial_query_positions = batch["spatial_query_positions"].to(
            device=visual_token_mask.device,
            dtype=torch.long,
        )
        if int(spatial_query_positions.numel()) > 0:
            row_ids = torch.arange(batch_size, device=visual_token_mask.device).unsqueeze(1).expand_as(
                spatial_query_positions
            )
            spatial_query_mask[row_ids, spatial_query_positions] = True
        answer_mask = (labels != -100) & attention_mask
        padding_mask = ~attention_mask
        occupied = visual_token_mask | spatial_query_mask | answer_mask | padding_mask
        other_prompt_mask = ~occupied
        if bool((visual_token_mask & (spatial_query_mask | answer_mask | padding_mask)).any().detach().cpu().item()):
            raise ValueError("visual-token mask overlaps a non-visual token category")
        if bool((spatial_query_mask & (answer_mask | padding_mask)).any().detach().cpu().item()):
            raise ValueError("spatial-query token mask overlaps answer or padding positions")
        outputs.update({
            "_global_readout_input": global_readout_input,
            "_global_readout_hidden": sample_hidden,
            "_visual_token_counts": visual_counts,
            "_padding_token_counts": padding_mask.sum(dim=1),
            "_answer_token_counts": answer_mask.sum(dim=1),
            "_spatial_query_token_counts": spatial_query_mask.sum(dim=1),
            "_other_prompt_token_counts": other_prompt_mask.sum(dim=1),
        })
        if legacy_disabled_query_readout:
            return outputs
        outputs.update({
            "decoder_ref_type_logits": self.decoder_aux["ref_type_head"](sample_hidden),
            "decoder_query_pose_pred": self.decoder_aux["query_pose_head"](sample_hidden),
            "decoder_bev_logits": self.decoder_aux["bev_head"](sample_hidden).view(batch_size, self.spatial_flow.bev_grid_size, self.spatial_flow.bev_grid_size),
            "decoder_object_center_logits": self.decoder_aux["object_center_head"](sample_hidden).view(batch_size, self.spatial_flow.bev_grid_size, self.spatial_flow.bev_grid_size),
            "decoder_spatial_gate_logits": self.decoder_aux["spatial_gate_head"](sample_hidden),
            "decoder_metric_scene_scale_pred": self.decoder_aux["metric_scene_scale_head"](sample_hidden),
            "decoder_metric_room_extent_pred": self.decoder_aux["metric_room_extent_head"](sample_hidden),
            "decoder_metric_camera_displacement_pred": self.decoder_aux["metric_camera_displacement_head"](sample_hidden),
            "decoder_metric_object_size_pred": self.decoder_aux["metric_object_size_head"](sample_hidden),
            "decoder_metric_object_pair_distance_pred": self.decoder_aux["metric_object_pair_distance_head"](sample_hidden),
            "decoder_metric_camera_object_distance_pred": self.decoder_aux["metric_camera_object_distance_head"](sample_hidden),
            "decoder_metric_object_count_pred": self.decoder_aux["metric_object_count_head"](sample_hidden),
        })
        return outputs

    def _compute_decoder_coordinate_raw_losses(
        self,
        *,
        batch: Mapping[str, Any],
        aux_out: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """始终计算两项原始坐标误差，但不在这里决定它们是否进入总损失。

        Off 臂仍要跑同样的 coordinate heads，才能证明它和 Full 臂只是监督开关不同；
        但 raw 指标绝不能静默加进 total loss，否则所谓 Off 仍在训练这两个目标。
        """

        raw: Dict[str, torch.Tensor] = {}
        for name, pred_key, mask_key in (
            ("target_camera", "decoder_target_camera_coord_pred", "target_camera_token_mask"),
            ("semantic_query", "decoder_semantic_query_coord_pred", "semantic_query_token_mask"),
        ):
            pred = aux_out[pred_key].float()
            target = batch["query_frame_coord_targets"].float()
            valid = batch[mask_key].float().view(-1, 1)
            valid_tokens = valid.sum()
            if float(valid_tokens.detach().cpu().item()) > 0.0:
                raw[name] = ((pred - target).abs() * valid).sum() / valid_tokens / 3.0
            else:
                raw[name] = pred.sum() * 0.0
            raw[f"{name}_valid_tokens"] = valid_tokens
        return raw

    def _compute_decoder_aux_losses(
        self,
        *,
        batch: Mapping[str, Any],
        aux_out: Mapping[str, torch.Tensor],
        coordinate_raw_losses: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        weights = self.decoder_aux_weights
        if coordinate_raw_losses is None:
            coordinate_raw_losses = self._compute_decoder_coordinate_raw_losses(batch=batch, aux_out=aux_out)

        def masked_sample_l1(
            *,
            pred_key: str,
            target_key: str,
            valid_key: str,
            loss_key: str,
            weight_key: str,
        ) -> None:
            weight = float(weights.get(weight_key, 0.0))
            if weight <= 0.0:
                return
            valid = batch[valid_key].float().view(-1, 1)
            pred = aux_out[pred_key].float()
            target = batch[target_key].float()
            if float(valid.sum().detach().cpu().item()) > 0.0:
                l1 = (pred - target).abs() * valid
                losses[loss_key] = l1.sum() / valid.sum().clamp_min(1.0) / float(target.shape[-1]) * weight
            else:
                losses[loss_key] = pred.sum() * 0.0

        ref_weight = float(weights.get("reference_type", 0.0))
        if ref_weight > 0.0:
            losses["loss_decoder_ref_type"] = F.cross_entropy(
                aux_out["decoder_ref_type_logits"].float(), batch["reference_type_ids"].long()
            ) * ref_weight
        pose_weight = float(weights.get("query_pose", 0.0))
        if pose_weight > 0.0:
            valid = batch["query_pose_valid"].float().view(-1, 1)
            if float(valid.sum().detach().cpu().item()) > 0.0:
                pose_l1 = (aux_out["decoder_query_pose_pred"].float() - batch["query_pose_vec12"].float()).abs() * valid
                losses["loss_decoder_query_pose"] = pose_l1.sum() / valid.sum().clamp_min(1.0) / 12.0 * pose_weight
            else:
                losses["loss_decoder_query_pose"] = aux_out["decoder_query_pose_pred"].float().sum() * 0.0
        gate_weight = float(weights.get("spatial_gate", 0.0))
        if gate_weight > 0.0:
            gate_target = batch["spatial_gate_targets"].float()
            if gate_target.shape != aux_out["decoder_spatial_gate_logits"].shape:
                raise ValueError(
                    f"decoder spatial_gate_targets shape mismatch: target={tuple(gate_target.shape)} "
                    f"logits={tuple(aux_out['decoder_spatial_gate_logits'].shape)}"
                )
            gate_log_probs = F.log_softmax(aux_out["decoder_spatial_gate_logits"].float(), dim=-1)
            losses["loss_decoder_spatial_gate"] = (-(gate_target * gate_log_probs).sum(dim=-1).mean()) * gate_weight
        bev_weight = float(weights.get("bev", 0.0))
        if bev_weight > 0.0:
            losses["loss_decoder_bev"] = F.binary_cross_entropy_with_logits(
                aux_out["decoder_bev_logits"].float(), batch["bev_targets"].float()
            ) * bev_weight
        object_center_weight = float(weights.get("object_center", 0.0))
        if object_center_weight > 0.0:
            valid = batch["object_center_valid"].float().view(-1, 1, 1)
            if float(valid.sum().detach().cpu().item()) > 0.0:
                bce = F.binary_cross_entropy_with_logits(
                    aux_out["decoder_object_center_logits"].float(), batch["object_center_targets"].float(), reduction="none"
                )
                grid_area = float(batch["object_center_targets"].shape[-1] * batch["object_center_targets"].shape[-2])
                losses["loss_decoder_object_center"] = bce.mul(valid).sum() / valid.sum().clamp_min(1.0) / grid_area * object_center_weight
            else:
                losses["loss_decoder_object_center"] = aux_out["decoder_object_center_logits"].float().sum() * 0.0
        patch_weight = float(weights.get("patch_consistency", 0.0))
        if patch_weight > 0.0:
            target = batch["spatial_descriptors"].float()[:, 6:9]
            valid = batch["spatial_token_valid"].float().view(-1, 1)
            patch_l1 = (aux_out["decoder_patch_recon"].float() - target).abs() * valid
            losses["loss_decoder_patch_consistency"] = patch_l1.sum() / valid.sum().clamp_min(1.0) / 3.0 * patch_weight
        target_camera_weight = float(weights.get("target_camera_coord", 0.0))
        if target_camera_weight > 0.0:
            losses["loss_decoder_target_camera_coord"] = (
                coordinate_raw_losses["target_camera"] * target_camera_weight
            )
        semantic_query_weight = float(weights.get("semantic_query_coord", 0.0))
        if semantic_query_weight > 0.0:
            losses["loss_decoder_semantic_query_coord"] = (
                coordinate_raw_losses["semantic_query"] * semantic_query_weight
            )
        bev_coord_weight = float(weights.get("bev_coord", 0.0))
        if bev_coord_weight > 0.0:
            valid = batch["canonical_bev_token_mask"].float().view(-1, 1)
            if float(valid.sum().detach().cpu().item()) > 0.0:
                bev_l1 = (aux_out["decoder_bev_coord_pred"].float() - batch["bev_coord_targets"].float()).abs() * valid
                losses["loss_decoder_bev_coord"] = bev_l1.sum() / valid.sum().clamp_min(1.0) / 2.0 * bev_coord_weight
            else:
                losses["loss_decoder_bev_coord"] = aux_out["decoder_bev_coord_pred"].float().sum() * 0.0
        masked_sample_l1(
            pred_key="decoder_metric_scene_scale_pred",
            target_key="metric_scene_scale_targets",
            valid_key="metric_scene_scale_valid",
            loss_key="loss_decoder_metric_scene_scale",
            weight_key="metric_scene_scale",
        )
        masked_sample_l1(
            pred_key="decoder_metric_room_extent_pred",
            target_key="metric_room_extent_targets",
            valid_key="metric_room_extent_valid",
            loss_key="loss_decoder_metric_room_extent",
            weight_key="metric_room_extent",
        )
        masked_sample_l1(
            pred_key="decoder_metric_camera_displacement_pred",
            target_key="metric_camera_displacement_targets",
            valid_key="metric_camera_displacement_valid",
            loss_key="loss_decoder_metric_camera_displacement",
            weight_key="metric_camera_displacement",
        )
        masked_sample_l1(
            pred_key="decoder_metric_object_size_pred",
            target_key="metric_object_size_targets",
            valid_key="metric_object_size_valid",
            loss_key="loss_decoder_metric_object_size",
            weight_key="metric_object_size",
        )
        masked_sample_l1(
            pred_key="decoder_metric_object_pair_distance_pred",
            target_key="metric_object_pair_distance_targets",
            valid_key="metric_object_pair_distance_valid",
            loss_key="loss_decoder_metric_object_pair_distance",
            weight_key="metric_object_pair_distance",
        )
        masked_sample_l1(
            pred_key="decoder_metric_camera_object_distance_pred",
            target_key="metric_camera_object_distance_targets",
            valid_key="metric_camera_object_distance_valid",
            loss_key="loss_decoder_metric_camera_object_distance",
            weight_key="metric_camera_object_distance",
        )
        masked_sample_l1(
            pred_key="decoder_metric_object_count_pred",
            target_key="metric_object_count_targets",
            valid_key="metric_object_count_valid",
            loss_key="loss_decoder_metric_object_count",
            weight_key="metric_object_count",
        )
        return losses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the QUASAR spatial-bias model with decoder-hidden auxiliary objectives.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--qwen-cache-dir", type=Path, default=None)
    parser.add_argument("--attn-implementation", choices=("default", "sdpa", "flash_attention_2", "eager"), default="default")
    parser.add_argument("--visual-cache-root", type=Path, default=None)
    parser.add_argument(
        "--extra-visual-cache-root",
        type=Path,
        action="append",
        default=[],
        help="Additional visual cache root. When the full train cache and checkpoint-dev/monitor caches are separate, roots are searched in the order given.",
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
    parser.add_argument("--simple3-aux-required", action="store_true")
    parser.add_argument(
        "--query-pose-mode",
        choices=tuple(sorted(PROVIDED_POSE_EVAL_MODES)),
        default="legacy",
        help="Defaults to the historical legacy mode; controlled experiments must pass a value explicitly and can enforce this with --require-explicit-query-pose-mode.",
    )
    parser.add_argument(
        "--require-explicit-query-pose-mode",
        action="store_true",
        help="Fail before model loading if --query-pose-mode is not given explicitly on the command line.",
    )
    parser.add_argument("--flow-dim", type=int, default=512)
    parser.add_argument("--num-flow-steps", type=int, default=4)
    parser.add_argument("--num-spatial-bias-experts", type=int, default=len(SPATIAL_BIAS_EXPERTS))
    parser.add_argument("--bias-gate-temperature", type=float, default=1.0)
    parser.add_argument("--safe-bias-residual-init", type=float, default=0.05)
    parser.add_argument("--safe-bias-residual-max", type=float, default=0.25)
    parser.add_argument("--bias-norm-max-ratio", type=float, default=0.15)
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
    parser.add_argument("--spatial-query-count", type=int, default=4)
    parser.add_argument(
        "--spatial-query-tokens-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enabled by default; the matched No-token variant uses --no-spatial-query-tokens-enabled.",
    )
    parser.add_argument(
        "--global-readout-source",
        choices=GLOBAL_READOUT_SOURCES,
        default="query_token_mean",
        help=(
            'Single readout position for the 12 scene/query-level DecoderAux heads: the full model uses query_token_mean; the visual arms of the placement ablation use visual_masked_mean.'
        ),
    )
    parser.add_argument(
        "--preserve-inactive-parameters-in-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Four-arm shared initialization only: No-token/zero-weight parameters stay in the same optimizer groups but must receive no gradient when the forward pass does not read them.'
        ),
    )
    parser.add_argument("--reference-type-loss-weight", type=float, default=0.2)
    parser.add_argument("--query-pose-loss-weight", type=float, default=0.1)
    parser.add_argument("--bev-loss-weight", type=float, default=0.05)
    parser.add_argument("--patch-consistency-loss-weight", type=float, default=0.05)
    parser.add_argument("--target-camera-coord-loss-weight", type=float, default=0.1)
    parser.add_argument("--semantic-query-coord-loss-weight", type=float, default=0.1)
    parser.add_argument("--bev-coord-loss-weight", type=float, default=0.1)
    parser.add_argument("--object-center-loss-weight", type=float, default=0.05)
    parser.add_argument("--spatial-gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--metric-scene-scale-loss-weight", type=float, default=0.05)
    parser.add_argument("--metric-room-extent-loss-weight", type=float, default=0.05)
    parser.add_argument("--metric-camera-displacement-loss-weight", type=float, default=0.05)
    parser.add_argument("--metric-object-size-loss-weight", type=float, default=0.05)
    parser.add_argument("--metric-object-pair-distance-loss-weight", type=float, default=0.05)
    parser.add_argument("--metric-camera-object-distance-loss-weight", type=float, default=0.05)
    parser.add_argument("--metric-object-count-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-reference-type-loss-weight", type=float, default=0.2)
    parser.add_argument("--decoder-query-pose-loss-weight", type=float, default=0.1)
    parser.add_argument("--decoder-bev-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-patch-consistency-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-target-camera-coord-loss-weight", type=float, default=0.1)
    parser.add_argument("--decoder-semantic-query-coord-loss-weight", type=float, default=0.1)
    parser.add_argument("--decoder-bev-coord-loss-weight", type=float, default=0.1)
    parser.add_argument("--decoder-object-center-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-spatial-gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--decoder-metric-scene-scale-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-metric-room-extent-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-metric-camera-displacement-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-metric-object-size-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-metric-object-pair-distance-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-metric-camera-object-distance-loss-weight", type=float, default=0.05)
    parser.add_argument("--decoder-metric-object-count-loss-weight", type=float, default=0.05)
    parser.add_argument("--spatial-query-sigreg-loss-weight", "--spatial_query_sigreg_loss_weight", dest="spatial_query_sigreg_loss_weight", type=float, default=0.0)
    parser.add_argument("--spatial-query-sigreg-projection-dim", "--spatial_query_sigreg_projection_dim", dest="spatial_query_sigreg_projection_dim", type=int, default=128)
    parser.add_argument("--spatial-query-sigreg-num-projections", "--spatial_query_sigreg_num_projections", dest="spatial_query_sigreg_num_projections", type=int, default=512)
    parser.add_argument("--spatial-query-sigreg-num-integration-knots", "--spatial_query_sigreg_num_integration_knots", dest="spatial_query_sigreg_num_integration_knots", type=int, default=17)
    parser.add_argument("--spatial-query-sigreg-seed", "--spatial_query_sigreg_seed", dest="spatial_query_sigreg_seed", type=int, default=20260609)
    parser.add_argument("--base-kl-loss-weight", "--base_kl_loss_weight", dest="base_kl_loss_weight", type=float, default=0.0)
    parser.add_argument("--metric-consistency-loss-weight", "--metric_consistency_loss_weight", dest="metric_consistency_loss_weight", type=float, default=0.0)
    parser.add_argument("--metric-consistency-ar-loss-weight", "--metric_consistency_ar_loss_weight", dest="metric_consistency_ar_loss_weight", type=float, default=0.0)
    parser.add_argument("--metric-consistency-query-loss-weight", "--metric_consistency_query_loss_weight", dest="metric_consistency_query_loss_weight", type=float, default=0.0)
    parser.add_argument("--init-checkpoint-path", type=Path, default=None)
    parser.add_argument("--numeric-scaffold-mode", choices=("none", "arnum"), default="none")
    parser.add_argument("--numeric-scaffold-bin-stats-path", type=Path, default=None)
    parser.add_argument("--numeric-scaffold-bin-loss-weight", type=float, default=0.0)
    parser.add_argument("--numeric-scaffold-ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--numeric-scaffold-ordinal-sigma", type=float, default=1.0)
    parser.add_argument("--global-3d-pe-mode", choices=("none", "fourier_mlp"), default="none")
    parser.add_argument("--global-3d-pe-hidden-dim", type=int, default=512)
    parser.add_argument("--global-3d-pe-fourier-bands", type=int, default=8)
    parser.add_argument("--global-3d-pe-coordinate-scale", type=float, default=5.0)
    parser.add_argument("--global-3d-pe-residual-scale", type=float, default=0.05)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument(
        "--monitor-jsonl",
        action="append",
        default=[],
        help=(
            'Fast monitor set evaluated with each main validation pass, given as NAME=/path/to/file.jsonl. Monitors are only logged and do not affect best-checkpoint or early-stop decisions.'
        ),
    )
    parser.add_argument("--monitor-max-rows", type=int, default=0)
    parser.add_argument(
        "--monitor-max-batches",
        type=int,
        default=0,
        help="0 means use --eval-max-batches.",
    )
    parser.add_argument("--best-metric", default="loss")
    parser.add_argument("--best-mode", choices=("min", "max"), default="min")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--no-save-best-checkpoint", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument(
        "--startup-checkpoint-step",
        type=int,
        default=0,
        help="Save one extra checkpoint at this early step, in addition to the regular schedule, to inspect gradients after the zero-initialized projection opens.",
    )
    parser.add_argument("--save-optimizer-state", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--resume-allow-missing-optimizer-state",
        action="store_true",
        help="Only for earlier probe checkpoints; keep disabled for a full resume so that optimizer and RNG states are restored together.",
    )
    parser.add_argument(
        "--resume-reset-dataloader-position",
        action="store_true",
        help="Restore model/optimizer/step without skipping batches already consumed in the current epoch.",
    )
    parser.add_argument("--no-train-shuffle", action="store_true")
    parser.add_argument(
        "--write-initialization-manifest",
        action="store_true",
        help="Write the initial keys, shapes and per-tensor hashes of LoRA/Spatial Flow/DecoderAux.",
    )
    parser.add_argument(
        "--stop-after-initialization-manifest",
        action="store_true",
        help="Dry-run control only: build and audit the model, then exit without creating an optimizer or training.",
    )
    parser.add_argument(
        "--export-shared-init-dir",
        type=Path,
        default=None,
        help="Create a step-0 shared-initialization checkpoint with model/LoRA/optimizer/RNG state, then exit.",
    )
    return parser


def load_processor_and_qwen(args: argparse.Namespace) -> tuple[Any, nn.Module, int, int, Dict[str, Any]]:
    if not args.qwen_model_path.exists():
        raise FileNotFoundError(f"Qwen model path does not exist: {args.qwen_model_path}")
    config_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    model_kwargs: Dict[str, Any] = {"dtype": torch.bfloat16, "trust_remote_code": True}
    if args.qwen_cache_dir is not None:
        config_kwargs["cache_dir"] = str(args.qwen_cache_dir)
        model_kwargs["cache_dir"] = str(args.qwen_cache_dir)
    if str(args.attn_implementation) != "default":
        config_kwargs["attn_implementation"] = str(args.attn_implementation)
        model_kwargs["attn_implementation"] = str(args.attn_implementation)
    config = AutoConfig.from_pretrained(str(args.qwen_model_path), **config_kwargs)
    model_type = str(getattr(config, "model_type", ""))
    if model_type.startswith("qwen3_vl"):
        model_cls_name = "Qwen3VLForConditionalGeneration"
    elif model_type.startswith("qwen2_5_vl"):
        model_cls_name = "Qwen2_5_VLForConditionalGeneration"
    else:
        raise ValueError(f"expected Qwen3-VL or Qwen2.5-VL model_type, got {model_type!r}")
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or getattr(config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError(f"cannot resolve Qwen hidden size from config: {args.qwen_model_path}")
    spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
    if spatial_merge_size <= 0:
        raise ValueError("cannot resolve Qwen vision spatial_merge_size")
    model_cls = getattr(transformers, model_cls_name, None)
    if model_cls is None:
        raise ValueError(f"{model_cls_name} is missing in this transformers environment")
    processor = AutoProcessor.from_pretrained(str(args.qwen_model_path), **config_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    spatial_query_tokens = build_spatial_query_tokens(int(args.spatial_query_count))
    numeric_scaffold_tokens = tuple(numeric_scaffold_special_tokens()) if str(getattr(args, "numeric_scaffold_mode", "none")) == "arnum" else tuple()
    additional_tokens = list(spatial_query_tokens) + list(numeric_scaffold_tokens)
    try:
        added_special_tokens = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_tokens},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        existing_specials = list(getattr(processor.tokenizer, "additional_special_tokens", []) or [])
        merged_specials = existing_specials + [token for token in additional_tokens if token not in existing_specials]
        added_special_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": merged_specials})
    spatial_query_token_ids: list[int] = []
    for token in spatial_query_tokens:
        token_id = int(processor.tokenizer.convert_tokens_to_ids(token))
        if processor.tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"spatial query token did not register as a single tokenizer token: {token!r}")
        spatial_query_token_ids.append(token_id)
    numeric_scaffold_token_ids: list[int] = []
    for token in numeric_scaffold_tokens:
        token_id = int(processor.tokenizer.convert_tokens_to_ids(token))
        if processor.tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"numeric scaffold token did not register as a single tokenizer token: {token!r}")
        numeric_scaffold_token_ids.append(token_id)
    setattr(args, "numeric_scaffold_token_ids", numeric_scaffold_token_ids)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = int(args.max_pixels)
    qwen = model_cls.from_pretrained(str(args.qwen_model_path), **model_kwargs)
    if str(args.attn_implementation) != "default":
        qwen.config._attn_implementation = str(args.attn_implementation)
        text_config = getattr(qwen.config, "text_config", None)
        if text_config is not None:
            text_config._attn_implementation = str(args.attn_implementation)
    embedding_rows_before = int(qwen.get_input_embeddings().num_embeddings)
    if len(processor.tokenizer) > embedding_rows_before:
        qwen.resize_token_embeddings(len(processor.tokenizer))
    embedding_rows_after = int(qwen.get_input_embeddings().num_embeddings)
    qwen.config.use_cache = False
    if getattr(qwen, "generation_config", None) is not None:
        qwen.generation_config.use_cache = False
    if args.gradient_checkpointing:
        qwen.gradient_checkpointing_enable()
        if hasattr(qwen, "enable_input_require_grads"):
            qwen.enable_input_require_grads()
    for parameter in qwen.parameters():
        parameter.requires_grad = False

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
        "attn_implementation": str(args.attn_implementation),
        "resolved_attn_implementation": str(getattr(qwen.config, "_attn_implementation", None)),
        "resolved_text_attn_implementation": str(getattr(getattr(qwen.config, "text_config", None), "_attn_implementation", None)),
        "hidden_size": hidden_size,
        "spatial_merge_size": spatial_merge_size,
        "max_pixels": int(args.max_pixels),
        "spatial_queries": {
            "count": int(args.spatial_query_count),
            "enabled": bool(args.spatial_query_tokens_enabled),
            "tokens": list(spatial_query_tokens),
            "token_ids": spatial_query_token_ids,
            "added_special_tokens": int(added_special_tokens),
            "tokenizer_size": int(len(processor.tokenizer)),
            "embedding_rows_before_resize": embedding_rows_before,
            "embedding_rows_after_resize": embedding_rows_after,
            "input_embedding_policy": (
                "replace token embedding with trainable spatial_query_role in model forward"
                if bool(args.spatial_query_tokens_enabled)
                else "register parameters for matched initialization, but omit all spatial query tokens from the prompt and decoder"
            ),
        },
        "numeric_scaffold": {
            "mode": str(getattr(args, "numeric_scaffold_mode", "none")),
            "tokens": list(numeric_scaffold_tokens),
            "token_ids": numeric_scaffold_token_ids,
            "coarse_bins": NUMERIC_SCAFFOLD_COARSE_BINS,
            "fine_bins": NUMERIC_SCAFFOLD_FINE_BINS,
            "delta_policy": "train/save additive input embedding and LM-head rows for scaffold tokens only",
        },
        "lora": {
            "r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "target_modules": sorted(target_modules),
            "replaced_module_count": len(replaced),
            "first_replaced_modules": replaced[:20],
        },
    }
    return processor, qwen, hidden_size, spatial_merge_size, meta


def build_aux_weight_dict(args: argparse.Namespace, *, prefix: str = "") -> Dict[str, float]:
    p = f"{prefix}_" if prefix else ""
    return {
        "reference_type": float(getattr(args, f"{p}reference_type_loss_weight")),
        "query_pose": float(getattr(args, f"{p}query_pose_loss_weight")),
        "bev": float(getattr(args, f"{p}bev_loss_weight")),
        "patch_consistency": float(getattr(args, f"{p}patch_consistency_loss_weight")),
        "target_camera_coord": float(getattr(args, f"{p}target_camera_coord_loss_weight")),
        "semantic_query_coord": float(getattr(args, f"{p}semantic_query_coord_loss_weight")),
        "bev_coord": float(getattr(args, f"{p}bev_coord_loss_weight")),
        "object_center": float(getattr(args, f"{p}object_center_loss_weight")),
        "spatial_gate": float(getattr(args, f"{p}spatial_gate_loss_weight")),
        "metric_scene_scale": float(getattr(args, f"{p}metric_scene_scale_loss_weight")),
        "metric_room_extent": float(getattr(args, f"{p}metric_room_extent_loss_weight")),
        "metric_camera_displacement": float(getattr(args, f"{p}metric_camera_displacement_loss_weight")),
        "metric_object_size": float(getattr(args, f"{p}metric_object_size_loss_weight")),
        "metric_object_pair_distance": float(getattr(args, f"{p}metric_object_pair_distance_loss_weight")),
        "metric_camera_object_distance": float(getattr(args, f"{p}metric_camera_object_distance_loss_weight")),
        "metric_object_count": float(getattr(args, f"{p}metric_object_count_loss_weight")),
    }


def build_decoder_aux_model(
    *,
    args: argparse.Namespace,
    qwen: nn.Module,
    tokenizer: Any,
    hidden_size: int,
) -> NativeQwenGlowPatchSpatialDecoderAuxModel:
    """按同一套构造顺序建立 Full-on 或 Matched-No-token 模型。

    开关只在所有模块完成初始化后控制 forward/冻结状态，因此两臂共享参数可以逐位比较。
    """

    return NativeQwenGlowPatchSpatialDecoderAuxModel(
        qwen=qwen,
        tokenizer=tokenizer,
        qwen_hidden_size=hidden_size,
        flow_dim=int(args.flow_dim),
        num_flow_steps=int(args.num_flow_steps),
        dropout=float(args.dropout),
        bev_grid_size=int(args.bev_grid_size),
        reference_schema=str(args.reference_schema),
        aux_weights=build_aux_weight_dict(args),
        decoder_aux_weights=build_aux_weight_dict(args, prefix="decoder"),
        num_spatial_bias_experts=int(args.num_spatial_bias_experts),
        bias_gate_temperature=float(args.bias_gate_temperature),
        safe_bias_residual_init=float(args.safe_bias_residual_init),
        safe_bias_residual_max=float(args.safe_bias_residual_max),
        bias_norm_max_ratio=float(args.bias_norm_max_ratio),
        spatial_query_count=int(args.spatial_query_count),
        spatial_query_tokens_enabled=bool(args.spatial_query_tokens_enabled),
        global_readout_source=str(args.global_readout_source),
        preserve_inactive_parameters_in_optimizer=bool(args.preserve_inactive_parameters_in_optimizer),
        spatial_query_sigreg_loss_weight=float(getattr(args, "spatial_query_sigreg_loss_weight", 0.0)),
        spatial_query_sigreg_projection_dim=int(getattr(args, "spatial_query_sigreg_projection_dim", 128)),
        spatial_query_sigreg_num_projections=int(getattr(args, "spatial_query_sigreg_num_projections", 512)),
        spatial_query_sigreg_num_integration_knots=int(getattr(args, "spatial_query_sigreg_num_integration_knots", 17)),
        spatial_query_sigreg_seed=int(getattr(args, "spatial_query_sigreg_seed", 20260609)),
        base_kl_loss_weight=float(getattr(args, "base_kl_loss_weight", 0.0)),
        metric_consistency_loss_weight=float(getattr(args, "metric_consistency_loss_weight", 0.0)),
        metric_consistency_ar_loss_weight=float(getattr(args, "metric_consistency_ar_loss_weight", 0.0)),
        metric_consistency_query_loss_weight=float(getattr(args, "metric_consistency_query_loss_weight", 0.0)),
        numeric_scaffold_token_ids=list(getattr(args, "numeric_scaffold_token_ids", [])),
        numeric_scaffold_bin_loss_weight=float(getattr(args, "numeric_scaffold_bin_loss_weight", 0.0)),
        numeric_scaffold_ordinal_loss_weight=float(getattr(args, "numeric_scaffold_ordinal_loss_weight", 0.0)),
        numeric_scaffold_ordinal_sigma=float(getattr(args, "numeric_scaffold_ordinal_sigma", 1.0)),
        global_3d_pe_mode=str(getattr(args, "global_3d_pe_mode", "none")),
        global_3d_pe_hidden_dim=int(getattr(args, "global_3d_pe_hidden_dim", 512)),
        global_3d_pe_fourier_bands=int(getattr(args, "global_3d_pe_fourier_bands", 8)),
        global_3d_pe_coordinate_scale=float(getattr(args, "global_3d_pe_coordinate_scale", 5.0)),
        global_3d_pe_residual_scale=float(getattr(args, "global_3d_pe_residual_scale", 0.05)),
    )


def is_token_dependent_parameter_name(name: str, *, global_readout_source: str) -> bool:
    if name == "spatial_query_role":
        return True
    if global_readout_source != "query_token_mean":
        return False
    if name.startswith("metric_consistency_query_head.") or name.startswith("spatial_query_sigreg_projection."):
        return True
    return any(name.startswith(f"decoder_aux.{module_name}.") for module_name in TOKEN_DEPENDENT_DECODER_AUX_MODULES)


def build_initialization_manifest(model: NativeQwenGlowPatchSpatialDecoderAuxModel) -> Dict[str, Any]:
    """保存真实任务参数的 key/shape/value hash，不去重复哈希冻结的 8B base 权重。"""

    def state_manifest(state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
        combined = hashlib.sha256()
        entries: Dict[str, Any] = {}
        for key in sorted(state):
            tensor = state[key].detach().contiguous().cpu()
            # 先拉平成一维再按 byte 查看；标量 tensor 不能直接 view 成不同元素宽度的 dtype。
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            value_sha = hashlib.sha256(raw).hexdigest()
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
            combined.update(str(key).encode("utf-8"))
            combined.update(json.dumps(shape).encode("utf-8"))
            combined.update(dtype.encode("utf-8"))
            combined.update(raw)
            entries[str(key)] = {"shape": shape, "dtype": dtype, "sha256": value_sha}
        return {"combined_sha256": combined.hexdigest(), "key_count": len(entries), "entries": entries}

    model_state_shapes = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(model.state_dict().items())
    }
    named_parameters = list(model.named_parameters())
    trainable_names = sorted(name for name, parameter in named_parameters if parameter.requires_grad)
    global_decoder_weight_sum = sum(
        abs(float(model.decoder_aux_weights.get(key, 0.0))) for key in TOKEN_DEPENDENT_DECODER_AUX_KEYS
    )

    def inactive_for_configured_loss(name: str) -> bool:
        """标出为 shared-init 同构而保留、但当前 arm 不会收到 loss 梯度的参数。"""

        if name == "spatial_query_role" and not model.spatial_query_tokens_enabled:
            return True
        if (
            global_decoder_weight_sum == 0.0
            and any(name.startswith(f"decoder_aux.{module_name}.") for module_name in TOKEN_DEPENDENT_DECODER_AUX_MODULES)
        ):
            return True
        if name.startswith("metric_consistency_total_head.") and model.metric_consistency_loss_weight == 0.0:
            return True
        if name.startswith("metric_consistency_ar_head.") and model.metric_consistency_ar_loss_weight == 0.0:
            return True
        if name.startswith("metric_consistency_query_head.") and model.metric_consistency_query_loss_weight == 0.0:
            return True
        if name.startswith("spatial_query_sigreg_projection.") and model.spatial_query_sigreg_loss_weight == 0.0:
            return True
        return False

    inactive_trainable_names = sorted(
        name for name, parameter in named_parameters if parameter.requires_grad and inactive_for_configured_loss(name)
    )
    active_trainable_names = sorted(set(trainable_names) - set(inactive_trainable_names))
    frozen_token_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if is_token_dependent_parameter_name(name, global_readout_source=model.global_readout_source)
        and not parameter.requires_grad
    )
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "spatial_query_tokens_enabled": bool(model.spatial_query_tokens_enabled),
        "global_readout_source": str(model.global_readout_source),
        "preserve_inactive_parameters_in_optimizer": bool(model.preserve_inactive_parameters_in_optimizer),
        "model_state_key_count": len(model_state_shapes),
        "model_state_key_shapes": model_state_shapes,
        "parameter_counts": {
            "total_registered": sum(int(parameter.numel()) for _, parameter in named_parameters),
            "optimizer_trainable": sum(
                int(parameter.numel()) for _, parameter in named_parameters if parameter.requires_grad
            ),
            "configured_loss_active_trainable": sum(
                int(parameter.numel())
                for name, parameter in named_parameters
                if parameter.requires_grad and not inactive_for_configured_loss(name)
            ),
        },
        "trainable_parameter_names": trainable_names,
        "configured_loss_active_trainable_parameter_names": active_trainable_names,
        "configured_loss_inactive_trainable_parameter_names": inactive_trainable_names,
        "frozen_token_parameter_names": frozen_token_names,
        "task_state": {
            "qwen_lora": state_manifest(lora_state_dict(model.qwen)),
            "spatial_flow": state_manifest(model.spatial_flow.state_dict()),
            "decoder_aux": state_manifest(model.decoder_aux_state_dict()),
        },
    }


def gradient_audit(model: NativeQwenGlowPatchSpatialDecoderAuxModel) -> Dict[str, Any]:
    groups: Dict[str, float] = {"qwen_lora": 0.0, "spatial_flow": 0.0, "decoder_aux": 0.0, "numeric_scaffold": 0.0, "global_3d_pe": 0.0}
    nonzero: Counter[str] = Counter()
    trainable_total = 0
    unexpected_token_grad: list[str] = []
    token_package_nonzero_grad: list[str] = []
    frozen_token_parameters: list[str] = []
    visual_decoder_nonzero_grad: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_total += int(parameter.numel())
        elif is_token_dependent_parameter_name(name, global_readout_source=model.global_readout_source):
            frozen_token_parameters.append(name)
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().float().norm().cpu().item())
        if grad_norm <= 0.0:
            continue
        if is_token_dependent_parameter_name(name, global_readout_source=model.global_readout_source):
            token_package_nonzero_grad.append(name)
            if not model.spatial_query_tokens_enabled:
                unexpected_token_grad.append(name)
        if any(name.startswith(f"decoder_aux.{module_name}.") for module_name in TOKEN_INDEPENDENT_DECODER_AUX_MODULES):
            visual_decoder_nonzero_grad.append(name)
        if name.startswith("qwen.") and "lora_" in name:
            group = "qwen_lora"
        elif name.startswith("spatial_flow."):
            group = "spatial_flow"
        elif name in {"numeric_scaffold_input_delta", "numeric_scaffold_lm_head_delta"}:
            group = "numeric_scaffold"
        elif name.startswith("global_3d_pe."):
            group = "global_3d_pe"
        elif name.startswith("decoder_aux.") or name.startswith("spatial_query_sigreg_projection.") or name == "spatial_query_role":
            group = "decoder_aux"
        else:
            group = "other"
            groups.setdefault(group, 0.0)
        groups[group] += grad_norm
        nonzero[group] += 1
    required = ["qwen_lora", "spatial_flow", "decoder_aux"]
    if getattr(model, "global_3d_pe", None) is not None:
        required.append("global_3d_pe")
    missing = [group for group in required if groups.get(group, 0.0) <= 0.0]
    if not model.spatial_query_tokens_enabled and not visual_decoder_nonzero_grad:
        missing.append("decoder_aux_visual")
    if model.spatial_query_tokens_enabled and not token_package_nonzero_grad:
        missing.append("token_package")
    return {
        "grad_norm_by_group": groups,
        "nonzero_grad_params_by_group": dict(nonzero),
        "missing_required_grad_groups": missing,
        "unexpected_token_package_grad_params": sorted(unexpected_token_grad),
        "token_package_nonzero_grad_params": sorted(token_package_nonzero_grad),
        "frozen_token_package_params": sorted(frozen_token_parameters),
        "visual_decoder_nonzero_grad_params": sorted(visual_decoder_nonzero_grad),
        "trainable_parameters": trainable_total,
        "passed": not missing and not unexpected_token_grad,
    }


@torch.no_grad()
def evaluate_loss(model: NativeQwenGlowPatchSpatialDecoderAuxModel, loader: DataLoader, *, device: torch.device, max_batches: int) -> Dict[str, float]:
    model.eval()
    sums: Counter[str] = Counter()
    count = 0
    answer_token_total = 0
    answer_token_correct = 0
    answer_sequence_total = 0
    answer_sequence_exact = 0
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
            if is_logged_scalar(key, value):
                sums[key] += float(value.detach().cpu().item())
        logits = out["logits"]
        labels = batch["labels"]
        shift_logits = logits[:, :-1, :].detach()
        shift_labels = labels[:, 1:].detach()
        label_mask = shift_labels != -100
        if bool(label_mask.any().detach().cpu().item()):
            pred_ids = shift_logits.argmax(dim=-1)
            token_correct = (pred_ids == shift_labels) & label_mask
            answer_token_correct += int(token_correct.sum().detach().cpu().item())
            answer_token_total += int(label_mask.sum().detach().cpu().item())
            per_row_total = label_mask.sum(dim=1)
            per_row_correct = token_correct.sum(dim=1)
            valid_rows = per_row_total > 0
            answer_sequence_total += int(valid_rows.sum().detach().cpu().item())
            answer_sequence_exact += int(((per_row_total == per_row_correct) & valid_rows).sum().detach().cpu().item())
        count += 1
    if count == 0:
        raise ValueError("evaluate_loss saw zero batches")
    metrics = {key: value / count for key, value in sums.items()}
    metrics["answer_token_accuracy"] = (
        float(answer_token_correct) / float(answer_token_total) if answer_token_total > 0 else 0.0
    )
    metrics["answer_sequence_exact"] = (
        float(answer_sequence_exact) / float(answer_sequence_total) if answer_sequence_total > 0 else 0.0
    )
    metrics["answer_token_total"] = float(answer_token_total)
    metrics["answer_sequence_total"] = float(answer_sequence_total)
    return metrics


def parse_monitor_specs(specs: Sequence[str]) -> list[tuple[str, Path]]:
    monitors: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in specs:
        item = str(raw).strip()
        if not item:
            continue
        if "=" in item:
            label, path_text = item.split("=", 1)
            label = label.strip()
            path = Path(path_text.strip())
        else:
            path = Path(item)
            label = path.stem
        if not label:
            raise ValueError(f"empty monitor label in spec={raw!r}")
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
        if not safe_label:
            raise ValueError(f"monitor label cannot be sanitized: {label!r}")
        if safe_label in seen:
            raise ValueError(f"duplicate monitor label: {safe_label}")
        seen.add(safe_label)
        monitors.append((safe_label, path))
    return monitors


def build_eval_loader(
    *,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    num_workers: int,
    collator: NativeGlowPatchSpatialDecoderAuxCollator,
    max_visual_frames: int,
    scannet_frame_root: Optional[Path],
    depth_grid_hw: tuple[int, int],
    reference_schema: str,
    visual_cache_root: Optional[Path],
    visual_cache_roots: Optional[Sequence[Path]],
    qwen_model_path: Path,
    max_pixels: int,
    simple3_aux_required: bool,
    query_pose_mode: str,
) -> DataLoader:
    dataset = NativeAnswerPatchSpatialDataset(
        rows=rows,
        max_visual_frames=max_visual_frames,
        scannet_frame_root=scannet_frame_root,
        depth_grid_hw=depth_grid_hw,
        reference_schema=reference_schema,
        visual_cache_root=visual_cache_root,
        visual_cache_roots=visual_cache_roots,
        qwen_model_path=qwen_model_path,
        max_pixels=max_pixels,
        simple3_aux_required=simple3_aux_required,
        query_pose_mode=query_pose_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


def checkpoint_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state["torch_cuda_all"]))


def nested_state_equal(left: Any, right: Any) -> bool:
    """跨 CPU/GPU 严格比较 checkpoint 中的模型、optimizer 与 RNG 嵌套状态。"""

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left.keys()) == set(right.keys()) and all(nested_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(nested_state_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def is_metric_improved(*, value: float, best: Optional[float], mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise ValueError(f"unknown best mode: {mode}")


def write_training_state(path: Path, state: Mapping[str, Any]) -> None:
    write_json(path / "training_state.json", state)


def save_checkpoint(
    *,
    path: Path,
    model: NativeQwenGlowPatchSpatialDecoderAuxModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    batch_i: int,
    args: argparse.Namespace,
    save_optimizer_state: bool,
    best_state: Optional[Mapping[str, Any]] = None,
    eval_state: Optional[Mapping[str, Any]] = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "base_metric_format_version": BASE_METRIC_FORMAT_VERSION,
        "step": int(step),
        "epoch": int(epoch),
        "batch_i": int(batch_i),
        "spatial_flow_state_dict": model.spatial_flow.state_dict(),
        "decoder_aux_state_dict": model.decoder_aux_state_dict(),
        "qwen_lora_state_dict": lora_state_dict(model.qwen),
        "args": normalize_args_for_json(args),
        "rng_state": checkpoint_rng_state(),
    }
    if best_state is not None:
        payload["best_state"] = dict(best_state)
    if eval_state is not None:
        payload["eval_state"] = dict(eval_state)
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
            "spatial_query_count": int(args.spatial_query_count),
            "spatial_query_tokens_enabled": bool(args.spatial_query_tokens_enabled),
            "global_readout_source": str(getattr(args, "global_readout_source", "query_token_mean")),
            "preserve_inactive_parameters_in_optimizer": bool(
                getattr(args, "preserve_inactive_parameters_in_optimizer", False)
            ),
            "numeric_scaffold_mode": str(getattr(args, "numeric_scaffold_mode", "none")),
            "numeric_scaffold_token_ids": list(getattr(args, "numeric_scaffold_token_ids", [])),
            "global_3d_pe_mode": str(getattr(args, "global_3d_pe_mode", "none")),
            "note": "Project-local LoRA + spatial_flow + decoder_aux states are stored in glow_patch_spatial.pt.",
        },
    )


def load_patch_spatial_checkpoint(
    path: Path,
    model: NativeQwenGlowPatchSpatialDecoderAuxModel,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint_file = resolve_checkpoint_file(path)
    # 正式训练 checkpoint 里不只有 tensor，还有 optimizer/RNG/best_state。
    # 这是本项目训练脚本保存的文件，所以这里显式允许完整反序列化。
    payload = torch.load(checkpoint_file, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {checkpoint_file}")
    if str(payload.get("format_version")) != FORMAT_VERSION:
        raise ValueError(f"checkpoint format mismatch: {payload.get('format_version')!r} vs {FORMAT_VERSION!r}")
    for key in ("spatial_flow_state_dict", "decoder_aux_state_dict", "qwen_lora_state_dict"):
        if key not in payload:
            raise KeyError(f"checkpoint missing {key}: {checkpoint_file}")
    payload_args = payload.get("args") if isinstance(payload.get("args"), Mapping) else {}
    sigreg_keys = [key for key in payload["decoder_aux_state_dict"]
                   if key.startswith("spatial_query_sigreg_projection.")]
    if sigreg_keys and model.spatial_query_sigreg_projection is None:
        # SFT 父点包含这个读出头，GRPO 仍须完整装回它，但不启用对应损失。
        # 使用保存的结构设置，随后严格加载全部张量，不用随机参数补缺项。
        projection_dim = int(payload_args["spatial_query_sigreg_projection_dim"])
        model.spatial_query_sigreg_projection = WorkspaceSigRegProjectionHead(
            hidden_size=int(model.spatial_query_role.shape[-1]),
            projection_dim=projection_dim,
        ).to(device=model.spatial_query_role.device, dtype=model.spatial_query_role.dtype)
        model.spatial_query_sigreg_projection.requires_grad_(model.spatial_query_sigreg_loss_weight > 0.0)
    checkpoint_reference_schema = str(payload_args.get("reference_schema") or "legacy5")
    model_reference_schema = str(getattr(model, "reference_schema", "legacy5"))
    if checkpoint_reference_schema != model_reference_schema:
        raise ValueError(
            f"checkpoint reference_schema mismatch: ckpt={checkpoint_reference_schema!r} model={model_reference_schema!r}"
        )
    checkpoint_query_count = int(payload_args.get("spatial_query_count") or model.spatial_query_count)
    if checkpoint_query_count != int(model.spatial_query_count):
        raise ValueError(
            f"checkpoint spatial_query_count mismatch: ckpt={checkpoint_query_count} model={model.spatial_query_count}"
        )
    model.spatial_flow.load_state_dict(payload["spatial_flow_state_dict"])
    model.load_decoder_aux_state_dict(payload["decoder_aux_state_dict"])
    load_lora_state_dict(model.qwen, payload["qwen_lora_state_dict"])
    return dict(payload)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.init_checkpoint_path is not None and args.resume_from is not None:
        raise ValueError("--init-checkpoint-path and --resume-from cannot be combined: the former warm-starts, the latter resumes a run.")
    if bool(args.stop_after_initialization_manifest):
        args.write_initialization_manifest = True
    if bool(args.require_explicit_query_pose_mode) and "--query-pose-mode" not in sys.argv[1:]:
        raise ValueError(
            "--require-explicit-query-pose-mode was set, but --query-pose-mode was not written on the command line"
        )
    if int(args.startup_checkpoint_step) < 0:
        raise ValueError("--startup-checkpoint-step must be non-negative")
    if args.export_shared_init_dir is not None and (args.init_checkpoint_path is not None or args.resume_from is not None):
        raise ValueError("shared-init export must start fresh; it cannot be combined with init/resume checkpoints")
    if not bool(args.spatial_query_tokens_enabled) and str(args.global_readout_source) == "query_token_mean":
        decoder_weights = build_aux_weight_dict(args, prefix="decoder")
        unexpected = {
            key: decoder_weights[key]
            for key in TOKEN_DEPENDENT_DECODER_AUX_KEYS
            if decoder_weights[key] != 0.0
        }
        if unexpected:
            raise ValueError(f"query-token readout without tokens has nonzero global DecoderAux weights: {unexpected}")
        if float(args.spatial_query_sigreg_loss_weight) != 0.0:
            raise ValueError("query-token readout without tokens requires --spatial-query-sigreg-loss-weight 0")
        if any(
            float(weight) != 0.0
            for weight in (
                args.metric_consistency_loss_weight,
                args.metric_consistency_ar_loss_weight,
                args.metric_consistency_query_loss_weight,
            )
        ):
            raise ValueError("query-token readout without tokens requires all query-coupled metric consistency weights to be zero")
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "args.json", normalize_args_for_json(args))
    if not bool(args.spatial_query_tokens_enabled):
        write_json(
            args.output_dir / "matched_no_token_contract.json",
            {
                "format_version": "quasar_matched_no_token_contract_v1",
                "spatial_query_tokens_enabled": False,
                "global_readout_source": str(args.global_readout_source),
                "preserve_inactive_parameters_in_optimizer": bool(
                    args.preserve_inactive_parameters_in_optimizer
                ),
                "registered_spatial_query_parameter_count": int(args.spatial_query_count),
                "decoder_token_dependent_weights": {
                    key: build_aux_weight_dict(args, prefix="decoder")[key]
                    for key in TOKEN_DEPENDENT_DECODER_AUX_KEYS
                },
                "decoder_visual_state_weights": {
                    key: build_aux_weight_dict(args, prefix="decoder")[key]
                    for key in ("patch_consistency", "target_camera_coord", "semantic_query_coord", "bev_coord")
                },
                "spatial_query_sigreg_loss_weight": float(args.spatial_query_sigreg_loss_weight),
                "base_kl_loss_weight": float(args.base_kl_loss_weight),
                "policy": (
                    "Instantiate the Full model first, then omit spatial query tokens from every prompt/decoder pass; "
                    "global heads read the configured source; inactive token parameters stay gradient-free."
                ),
            },
        )
    visual_cache_roots = normalize_visual_cache_roots(args.visual_cache_root, args.extra_visual_cache_root)
    primary_visual_cache_root = visual_cache_roots[0] if visual_cache_roots else None

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
    monitor_specs = parse_monitor_specs(args.monitor_jsonl)
    monitor_rows: dict[str, list[dict[str, Any]]] = {}
    monitor_filter_audit: dict[str, Any] = {}
    for label, path in monitor_specs:
        rows_raw = read_native_jsonl(path)
        rows, audit = filter_native_rows(
            rows=rows_raw,
            max_rows=int(args.monitor_max_rows),
            require_patch_geometry=bool(args.require_patch_geometry),
            scannet_frame_root=args.scannet_frame_root,
            max_visual_frames=int(args.max_visual_frames),
        )
        monitor_rows[label] = rows
        monitor_filter_audit[label] = {"path": str(path), **audit}

    write_json(args.output_dir / "data_filter_audit.json", {"train": train_audit, "val": val_audit})
    if monitor_filter_audit:
        write_json(args.output_dir / "monitor_data_filter_audit.json", monitor_filter_audit)

    if str(args.numeric_scaffold_mode) == "arnum":
        if args.numeric_scaffold_bin_stats_path is not None and args.numeric_scaffold_bin_stats_path.is_file():
            with args.numeric_scaffold_bin_stats_path.open("r", encoding="utf-8") as handle:
                numeric_scaffold_bin_stats = json.load(handle)
        else:
            numeric_scaffold_bin_stats = build_numeric_scaffold_bin_stats(train_rows)
        write_json(args.output_dir / "numeric_scaffold_bin_stats.json", numeric_scaffold_bin_stats)
        train_rows, train_scaffold_audit = apply_numeric_scaffold_to_rows(rows=train_rows, bin_stats=numeric_scaffold_bin_stats)
        val_rows, val_scaffold_audit = apply_numeric_scaffold_to_rows(rows=val_rows, bin_stats=numeric_scaffold_bin_stats)
        monitor_scaffold_audit: dict[str, Any] = {}
        for label, rows in list(monitor_rows.items()):
            monitor_rows[label], audit = apply_numeric_scaffold_to_rows(rows=rows, bin_stats=numeric_scaffold_bin_stats)
            monitor_scaffold_audit[label] = audit
        write_json(
            args.output_dir / "numeric_scaffold_audit.json",
            {"train": train_scaffold_audit, "val": val_scaffold_audit, "monitors": monitor_scaffold_audit},
        )
        if int(train_scaffold_audit.get("scaffolded_rows", 0)) <= 0:
            raise ValueError(f"numeric scaffold mode produced zero train scaffold rows: {train_scaffold_audit}")

    write_json(
        args.output_dir / "reference_schema_audit.json",
        {
            "train": audit_reference_schema_rows(rows=train_rows, reference_schema=str(args.reference_schema), split_name="train"),
            "val": audit_reference_schema_rows(rows=val_rows, reference_schema=str(args.reference_schema), split_name="val"),
            "monitors": {
                label: audit_reference_schema_rows(rows=rows, reference_schema=str(args.reference_schema), split_name=label)
                for label, rows in monitor_rows.items()
            },
        },
    )
    if primary_visual_cache_root is not None:
        train_cache_audit = audit_visual_cache_rows(
            rows=train_rows,
            visual_cache_root=primary_visual_cache_root,
            visual_cache_roots=visual_cache_roots[1:],
            scannet_frame_root=args.scannet_frame_root,
            max_visual_frames=int(args.max_visual_frames),
            qwen_model_path=args.qwen_model_path,
            max_pixels=int(args.max_pixels),
            split_name="train",
        )
        val_cache_audit = audit_visual_cache_rows(
            rows=val_rows,
            visual_cache_root=primary_visual_cache_root,
            visual_cache_roots=visual_cache_roots[1:],
            scannet_frame_root=args.scannet_frame_root,
            max_visual_frames=int(args.max_visual_frames),
            qwen_model_path=args.qwen_model_path,
            max_pixels=int(args.max_pixels),
            split_name="val",
        )
        monitor_cache_audit = {
            label: audit_visual_cache_rows(
                rows=rows,
                visual_cache_root=primary_visual_cache_root,
                visual_cache_roots=visual_cache_roots[1:],
                scannet_frame_root=args.scannet_frame_root,
                max_visual_frames=int(args.max_visual_frames),
                qwen_model_path=args.qwen_model_path,
                max_pixels=int(args.max_pixels),
                split_name=f"monitor:{label}",
            )
            for label, rows in monitor_rows.items()
        }
        write_json(
            args.output_dir / "visual_cache_audit.json",
            {"train": train_cache_audit, "val": val_cache_audit, "monitors": monitor_cache_audit},
        )

    processor, qwen, hidden_size, spatial_merge_size, qwen_meta = load_processor_and_qwen(args)
    write_json(args.output_dir / "qwen_meta.json", qwen_meta)
    spatial_query_tokens = build_spatial_query_tokens(int(args.spatial_query_count))
    collator = NativeGlowPatchSpatialDecoderAuxCollator(
        processor=processor,
        max_pixels=int(args.max_pixels),
        spatial_merge_size=spatial_merge_size,
        bev_grid_size=int(args.bev_grid_size),
        spatial_query_tokens=spatial_query_tokens,
        reference_schema=str(args.reference_schema),
        spatial_query_tokens_enabled=bool(args.spatial_query_tokens_enabled),
    )
    train_dataset = NativeAnswerPatchSpatialDataset(
        rows=train_rows,
        max_visual_frames=int(args.max_visual_frames),
        scannet_frame_root=args.scannet_frame_root,
        depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
        reference_schema=str(args.reference_schema),
        visual_cache_root=primary_visual_cache_root,
        visual_cache_roots=visual_cache_roots[1:],
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
        simple3_aux_required=bool(args.simple3_aux_required),
        query_pose_mode=str(args.query_pose_mode),
    )
    val_loader = build_eval_loader(
        rows=val_rows,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        collator=collator,
        max_visual_frames=int(args.max_visual_frames),
        scannet_frame_root=args.scannet_frame_root,
        depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
        reference_schema=str(args.reference_schema),
        visual_cache_root=primary_visual_cache_root,
        visual_cache_roots=visual_cache_roots[1:],
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
        simple3_aux_required=bool(args.simple3_aux_required),
        query_pose_mode=str(args.query_pose_mode),
    )
    monitor_loaders = {
        label: build_eval_loader(
            rows=rows,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            collator=collator,
            max_visual_frames=int(args.max_visual_frames),
            scannet_frame_root=args.scannet_frame_root,
            depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
            reference_schema=str(args.reference_schema),
            visual_cache_root=primary_visual_cache_root,
            visual_cache_roots=visual_cache_roots[1:],
            qwen_model_path=args.qwen_model_path,
            max_pixels=int(args.max_pixels),
            simple3_aux_required=bool(args.simple3_aux_required),
            query_pose_mode=str(args.query_pose_mode),
        )
        for label, rows in monitor_rows.items()
    }

    def build_train_loader_for_epoch(epoch_index: int) -> DataLoader:
        generator = None
        shuffle = not bool(args.no_train_shuffle)
        if shuffle:
            generator = torch.Generator()
            # 每个 epoch 的 shuffle 只由 seed+epoch 决定。
            # resume 时重建同一 epoch loader，再跳过已消费 batch，可复现到 checkpoint 边界。
            generator.manual_seed(int(args.seed) + int(epoch_index))
        return DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=shuffle,
            generator=generator,
            num_workers=int(args.num_workers),
            collate_fn=collator,
            pin_memory=True,
        )

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    model = build_decoder_aux_model(
        args=args,
        qwen=qwen,
        tokenizer=processor.tokenizer,
        hidden_size=hidden_size,
    ).to(device)
    if bool(args.write_initialization_manifest):
        write_json(args.output_dir / "initialization_manifest.json", build_initialization_manifest(model))
    if bool(args.stop_after_initialization_manifest):
        write_json(
            args.output_dir / "run_summary.json",
            {
                "format_version": FORMAT_VERSION,
                "status": "initialization_manifest_only",
                "spatial_query_tokens_enabled": bool(args.spatial_query_tokens_enabled),
                "global_readout_source": str(args.global_readout_source),
            },
        )
        return
    if args.init_checkpoint_path is not None:
        init_payload = load_patch_spatial_checkpoint(args.init_checkpoint_path, model, map_location="cpu")
        write_json(
            args.output_dir / "init_checkpoint_audit.json",
            {
                "init_checkpoint_path": str(args.init_checkpoint_path),
                "init_step": int(init_payload.get("step", -1)),
                "init_epoch": int(init_payload.get("epoch", -1)),
            },
        )
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if args.export_shared_init_dir is not None:
        # 共享初态必须在任何 optimizer step 之前落盘，并把空 optimizer 与完整 RNG 一起保存。
        # 两个正式臂之后都走 --resume-from 读取这一份，而不是分别“用同一个 seed 再初始化一次”。
        save_checkpoint(
            path=args.export_shared_init_dir,
            model=model,
            optimizer=optimizer,
            step=0,
            epoch=0,
            batch_i=0,
            args=args,
            save_optimizer_state=True,
        )
        shared_init_file = args.export_shared_init_dir / "glow_patch_spatial.pt"
        shared_init_sha256 = hashlib.sha256()
        with shared_init_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                shared_init_sha256.update(chunk)
        write_json(
            args.output_dir / "shared_init_export_audit.json",
            {
                "format_version": "query_frame_coordinate_shared_init_v1",
                "checkpoint": str(shared_init_file),
                "sha256": shared_init_sha256.hexdigest(),
                "step": 0,
                "epoch": 0,
                "batch_i": 0,
                "optimizer_state_present": True,
                "optimizer_state_entries": len(optimizer.state_dict().get("state", {})),
                "scheduler": None,
                "rng_keys": sorted(checkpoint_rng_state()),
                "query_pose_mode": str(args.query_pose_mode),
            },
        )
        write_json(
            args.output_dir / "run_summary.json",
            {
                "format_version": FORMAT_VERSION,
                "status": "shared_init_exported",
                "shared_init_checkpoint": str(shared_init_file),
                "shared_init_sha256": shared_init_sha256.hexdigest(),
            },
        )
        return

    train_log_path = args.output_dir / "train_log.jsonl"
    first_batch_audit_path = args.output_dir / "first_batch_audit.json"
    global_step = 0
    start_epoch = 0
    resume_skip_batch_i = 0
    resume_rng_state: Optional[Mapping[str, Any]] = None
    best_state: dict[str, Any] = {
        "metric_name": str(args.best_metric),
        "mode": str(args.best_mode),
        "metric_value": None,
        "global_step": None,
        "epoch": None,
    }
    eval_state: dict[str, Any] = {
        "bad_eval_count": 0,
        "early_stopped": False,
        "early_stop_reason": None,
    }
    if args.resume_from is not None:
        resume_payload = load_patch_spatial_checkpoint(args.resume_from, model, map_location="cpu")
        if "optimizer_state_dict" not in resume_payload:
            if not bool(args.resume_allow_missing_optimizer_state):
                raise KeyError(
                    f"resume checkpoint has no optimizer_state_dict: {args.resume_from}. "
                    "Formal full resume requires checkpoints saved with --save-optimizer-state."
                )
        else:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        global_step = int(resume_payload.get("step", 0))
        start_epoch = int(resume_payload.get("epoch", 0))
        if not bool(args.resume_reset_dataloader_position):
            resume_skip_batch_i = int(resume_payload.get("batch_i", 0))
        if isinstance(resume_payload.get("best_state"), Mapping):
            best_state.update(dict(resume_payload["best_state"]))
        if isinstance(resume_payload.get("eval_state"), Mapping):
            eval_state.update(dict(resume_payload["eval_state"]))
        if isinstance(resume_payload.get("rng_state"), Mapping):
            resume_rng_state = resume_payload["rng_state"]
            restore_rng_state(resume_rng_state)
        resume_load_checks = {
            "spatial_flow_state_exact": nested_state_equal(
                model.spatial_flow.state_dict(), resume_payload["spatial_flow_state_dict"]
            ),
            "decoder_aux_state_exact": nested_state_equal(
                model.decoder_aux_state_dict(), resume_payload["decoder_aux_state_dict"]
            ),
            "qwen_lora_state_exact": nested_state_equal(
                lora_state_dict(model.qwen), resume_payload["qwen_lora_state_dict"]
            ),
            "optimizer_state_exact": (
                nested_state_equal(optimizer.state_dict(), resume_payload["optimizer_state_dict"])
                if "optimizer_state_dict" in resume_payload
                else None
            ),
            "rng_state_exact": (
                nested_state_equal(checkpoint_rng_state(), resume_rng_state)
                if resume_rng_state is not None
                else None
            ),
        }
        write_json(
            args.output_dir / "resume_audit.json",
            {
                "resume_from": str(args.resume_from),
                "global_step": global_step,
                "start_epoch": start_epoch,
                "resume_skip_batch_i": resume_skip_batch_i,
                "optimizer_state_loaded": "optimizer_state_dict" in resume_payload,
                "rng_state_loaded": isinstance(resume_payload.get("rng_state"), Mapping),
                "loaded_state_checks": resume_load_checks,
                "best_state": best_state,
                "eval_state": eval_state,
            },
        )
        failed_resume_checks = [key for key, value in resume_load_checks.items() if value is False]
        if failed_resume_checks:
            raise RuntimeError(f"resume state failed exact load audit: {failed_resume_checks}")
        write_json(
            args.output_dir / "resume_loaded_initialization_manifest.json",
            build_initialization_manifest(model),
        )

    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    stop_training = False
    monitor_max_batches = int(args.monitor_max_batches) if int(args.monitor_max_batches) > 0 else int(args.eval_max_batches)

    def run_eval_bundle(*, epoch: int, batch_i: int) -> dict[str, Any]:
        metrics = evaluate_loss(model, val_loader, device=device, max_batches=int(args.eval_max_batches))
        monitor_metrics = {
            label: evaluate_loss(model, loader, device=device, max_batches=monitor_max_batches)
            for label, loader in monitor_loaders.items()
        }
        metric_name = str(args.best_metric)
        if metric_name not in metrics:
            raise KeyError(f"--best-metric {metric_name!r} not found in val metrics keys={sorted(metrics)}")
        metric_value = float(metrics[metric_name])
        improved = is_metric_improved(
            value=metric_value,
            best=None if best_state.get("metric_value") is None else float(best_state["metric_value"]),
            mode=str(args.best_mode),
            min_delta=float(args.early_stop_min_delta),
        )
        if improved:
            best_state.update(
                {
                    "metric_name": metric_name,
                    "mode": str(args.best_mode),
                    "metric_value": metric_value,
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_i": batch_i,
                    "time": time.time(),
                }
            )
            eval_state["bad_eval_count"] = 0
            if not bool(args.no_save_best_checkpoint):
                save_checkpoint(
                    path=args.output_dir / "best_checkpoint",
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    epoch=epoch,
                    batch_i=batch_i,
                    args=args,
                    save_optimizer_state=bool(args.save_optimizer_state),
                    best_state=best_state,
                    eval_state=eval_state,
                )
        else:
            eval_state["bad_eval_count"] = int(eval_state.get("bad_eval_count", 0)) + 1
        if int(args.early_stop_patience) > 0 and int(eval_state.get("bad_eval_count", 0)) >= int(args.early_stop_patience):
            eval_state["early_stopped"] = True
            eval_state["early_stop_reason"] = (
                f"{eval_state['bad_eval_count']} evals without {metric_name} improvement "
                f"(mode={args.best_mode}, min_delta={args.early_stop_min_delta})"
            )
        record = {
            "global_step": global_step,
            "epoch": epoch,
            "batch_i": batch_i,
            "metrics": metrics,
            "monitor_metrics": monitor_metrics,
            "best_state": dict(best_state),
            "eval_state": dict(eval_state),
            "is_best": bool(improved),
            "time": time.time(),
        }
        append_jsonl(args.output_dir / "val_log.jsonl", record)
        write_training_state(
            args.output_dir,
            {
                "global_step": global_step,
                "epoch": epoch,
                "batch_i": batch_i,
                "best_state": best_state,
                "eval_state": eval_state,
            },
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        model.train()
        return record

    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        train_loader = build_train_loader_for_epoch(epoch)
        last_batch_i = 0
        processed_batches_in_epoch = 0
        for batch_i, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and resume_skip_batch_i > 0 and batch_i <= resume_skip_batch_i:
                if batch_i == resume_skip_batch_i and resume_rng_state is not None:
                    # DataLoader 为跳过旧 batch 仍会真实执行 dataset/collator，可能消耗 Python/NumPy/torch RNG。
                    # 在最后一个跳过 batch 后重放 checkpoint RNG，下一批才和未中断训练逐位一致。
                    restore_rng_state(resume_rng_state)
                    write_json(
                        args.output_dir / "resume_rng_replay_audit.json",
                        {
                            "epoch": epoch,
                            "replayed_after_batch_i": batch_i,
                            "next_batch_i": batch_i + 1,
                            "policy": "restore checkpoint RNG after materializing skipped batches",
                        },
                    )
                continue
            last_batch_i = batch_i
            processed_batches_in_epoch += 1
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
                audit["numeric_scaffold_label_span_audit"] = audit_numeric_scaffold_label_spans(audit["label_audit"])
                audit["sample_ids"] = batch["sample_ids"]
                expected_query_tokens = (
                    int(batch["spatial_query_positions"].numel())
                    if bool(args.spatial_query_tokens_enabled)
                    else 0
                )
                audit["passed"] = (
                    bool(audit["passed"])
                    and audit["spatial_bias_tokens"] == audit["image_feature_tokens"]
                    and audit["spatial_query_token_count"] == expected_query_tokens
                    and int(batch["spatial_query_positions"].numel()) == expected_query_tokens
                    and bool(audit["spatial_query_tokens_enabled"]) == bool(args.spatial_query_tokens_enabled)
                    and audit["answer_label_tokens"] > 0
                    and (str(args.numeric_scaffold_mode) != "arnum" or bool(audit["numeric_scaffold_label_span_audit"]["passed"]))
                )
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
                        "best_metric_value": best_state.get("metric_value"),
                        "best_metric_step": best_state.get("global_step"),
                        "raw_spatial_bias_abs_mean": float(model.last_forward_audit.get("raw_spatial_bias_abs_mean", 0.0)),
                        "safe_spatial_bias_abs_mean": float(model.last_forward_audit.get("safe_spatial_bias_abs_mean", 0.0)),
                    }
                    for key, value in out.items():
                        if is_logged_scalar(key, value):
                            record[key] = float(value.detach().cpu().item())
                    append_jsonl(train_log_path, record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                should_eval = int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0
                should_save = (
                    (int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0)
                    or (int(args.startup_checkpoint_step) > 0 and global_step == int(args.startup_checkpoint_step))
                )
                checkpoint_path = args.output_dir / "checkpoints" / f"step_{global_step:06d}"
                if should_eval and should_save:
                    # 长时间训练中，完整 val/monitor 可能跑很久，也可能因为数据问题中途失败。
                    # 所以当“评测步”和“保存步”重合时，先落一个可 resume 的安全点；
                    # 如果评测成功，下面会用带最新 best/eval_state 的版本覆盖同一路径。
                    save_checkpoint(
                        path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        epoch=epoch,
                        batch_i=batch_i,
                        args=args,
                        save_optimizer_state=bool(args.save_optimizer_state),
                        best_state=best_state,
                        eval_state=eval_state,
                    )
                if should_eval:
                    run_eval_bundle(epoch=epoch, batch_i=batch_i)
                if should_save:
                    save_checkpoint(
                        path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        epoch=epoch,
                        batch_i=batch_i,
                        args=args,
                        save_optimizer_state=bool(args.save_optimizer_state),
                        best_state=best_state,
                        eval_state=eval_state,
                    )
                if bool(eval_state.get("early_stopped")):
                    stop_training = True
                    break
                if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                    stop_training = True
                    break
        if stop_training:
            break
        # 梯度累积不足一个整步时，不静默丢掉最后一段梯度。
        if processed_batches_in_epoch > 0 and last_batch_i % int(args.gradient_accumulation_steps) != 0:
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), float(args.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            should_eval = int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0
            should_save = (
                (int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0)
                or (int(args.startup_checkpoint_step) > 0 and global_step == int(args.startup_checkpoint_step))
            )
            checkpoint_path = args.output_dir / "checkpoints" / f"step_{global_step:06d}"
            if should_eval and should_save:
                # 同上：先保护当前训练状态，再进入可能耗时很长的验证阶段。
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    epoch=epoch,
                    batch_i=last_batch_i,
                    args=args,
                    save_optimizer_state=bool(args.save_optimizer_state),
                    best_state=best_state,
                    eval_state=eval_state,
                )
            if should_eval:
                run_eval_bundle(epoch=epoch, batch_i=last_batch_i)
            if should_save:
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    epoch=epoch,
                    batch_i=last_batch_i,
                    args=args,
                    save_optimizer_state=bool(args.save_optimizer_state),
                    best_state=best_state,
                    eval_state=eval_state,
                )
        if bool(eval_state.get("early_stopped")):
            break
    final_metrics = evaluate_loss(model, val_loader, device=device, max_batches=int(args.eval_max_batches))
    final_monitor_metrics = {
        label: evaluate_loss(model, loader, device=device, max_batches=monitor_max_batches)
        for label, loader in monitor_loaders.items()
    }
    write_json(args.output_dir / "final_val_metrics.json", final_metrics)
    if final_monitor_metrics:
        write_json(args.output_dir / "final_monitor_metrics.json", final_monitor_metrics)
    save_checkpoint(
        path=args.output_dir / "final_checkpoint",
        model=model,
        optimizer=optimizer,
        step=global_step,
        epoch=int(args.epochs),
        batch_i=0,
        args=args,
        save_optimizer_state=bool(args.save_optimizer_state),
        best_state=best_state,
        eval_state=eval_state,
    )
    write_json(
        args.output_dir / "run_summary.json",
        {
            "format_version": FORMAT_VERSION,
            "reference_schema": str(args.reference_schema),
            "global_3d_pe_mode": str(getattr(args, "global_3d_pe_mode", "none")),
            "global_step": global_step,
            "elapsed_sec": time.time() - start_time,
            "final_val_metrics": final_metrics,
            "final_monitor_metrics": final_monitor_metrics,
            "best_state": best_state,
            "eval_state": eval_state,
            "first_batch_audit_path": str(first_batch_audit_path),
        },
    )


if __name__ == "__main__":
    main()
