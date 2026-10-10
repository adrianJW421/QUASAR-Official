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
"""Real generation eval for the QUASAR decoder-hidden auxiliary model.

这个脚本和训练入口成对存在：它加载 decoder-aux checkpoint，按同样的 prompt
加入 spatial query tokens，按同样的 safe spatial bias 写回 visual tokens，然后真实自回归生成答案。
本包的 VSI 计分沿用官方 VSI-Bench（thinking-in-space 中 lmms-eval vsibench task）的规则，
由 score_vsibench_official_like_predictions 和 quasar-reviewer 完成；历史答案用它重算与论文
未取整分数一致，见 docs/REPRODUCTION_RESULTS.md。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader



from quasar.eval_native_qwen3vl2b_glow_patch_spatial_moe_generate import (  # noqa: E402
    bench_from_source,
    build_parser as build_moe_eval_parser,
    score_native_answer,
    summarize_records,
    vsti_task_group,
)
from quasar.latency_benchmark_utils import (  # noqa: E402
    add_latency_args,
    latency_enabled,
    reset_peak_memory_if_needed,
    slice_rows_for_latency,
    summarize_latency_records,
    timed_generate_call,
    validate_latency_args,
    write_latency_outputs,
)
from quasar.score_vsibench_official_like_predictions import (  # noqa: E402
    summarize_official_like_vsi_from_prediction_rows,
    summarize_vsti_task_groups,
)
from quasar.train_flow_as_picker_vsi import (  # noqa: E402
    normalize_args_for_json,
    set_seed,
    write_json,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial_decoder_aux import (  # noqa: E402
    NativeGlowPatchSpatialDecoderAuxCollator,
    NativeQwenGlowPatchSpatialDecoderAuxModel,
    build_aux_weight_dict,
    build_spatial_query_tokens,
    load_patch_spatial_checkpoint,
    load_processor_and_qwen,
    resolve_checkpoint_file,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial_moe_metric import (  # noqa: E402
    NativeAnswerPatchSpatialDataset,
    audit_visual_cache_rows,
    filter_native_rows,
    move_batch_to_device,
    read_native_jsonl,
    reference_type_mapping,
)


FORMAT_VERSION = "native_qwen3vl2b_glow_patch_spatial_decoder_aux_generate_eval_v1"
GATE1_ARMS = (
    "none",
    "current",
    "hard_off",
    "zero",
    "permuted",
    "semantic_type_canonical",
)
GATE1_REFERENCE_FAMILIES = ("semantic", "target_camera", "all")


class _EvalZeroFlowBlock(torch.nn.Module):
    """Return a zero update for truncated eval prefixes without changing checkpoint structure."""

    def __init__(self, flow_dim: int) -> None:
        super().__init__()
        self.flow_dim = int(flow_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[..., : self.flow_dim] * 0.0


def apply_eval_flow_prefix(model: NativeQwenGlowPatchSpatialDecoderAuxModel, active_prefix: int) -> None:
    """Disable only the suffix blocks after a strict full-checkpoint load."""

    blocks = model.spatial_flow.flow_blocks
    if len(blocks) != 4:
        raise ValueError(f"prefix sensitivity requires exactly four loaded flow blocks, got {len(blocks)}")
    prefix = int(active_prefix)
    if prefix < 1 or prefix > len(blocks):
        raise ValueError(f"active flow prefix must be in [1,{len(blocks)}], got {prefix}")
    if prefix == len(blocks):
        return
    for index in range(prefix, len(blocks)):
        blocks[index] = _EvalZeroFlowBlock(int(model.spatial_flow.flow_dim))


def build_parser():
    parser = build_moe_eval_parser()
    parser.description = "Evaluate a QUASAR decoder-aux checkpoint by real generation."
    parser.add_argument("--resume-inference", action="store_true", help="Resume complete answer-free prediction lines in original order.")
    parser.add_argument("--verify-numeric-pose", action="store_true",
                        help="Check current/hard_off inputs and real-model pose invariance before generation.")
    parser.add_argument("--spatial-query-count", type=int, default=4)
    parser.add_argument("--attn-implementation", choices=("default", "sdpa", "flash_attention_2", "eager"), default="default")
    parser.add_argument("--simple3-aux-required", action="store_true", help="Require refs.simple3_aux when reference_schema=simple3.")
    parser.add_argument(
        "--query-pose-mode",
        choices=("legacy", "provided_full", "type_only", "shuffled_pose"),
        default="legacy",
        help=(
            "Provided-pose switch: provided_full consumes only audited pose inputs; type_only hard-blocks "
            "numerical query pose; shuffled_pose uses a deterministic same-gauge donor where available."
        ),
    )
    parser.add_argument(
        "--verify-missing-pose-gate",
        action="store_true",
        help=(
            "Before generation, run one real-Qwen missing-pose audit: two arbitrary dummy poses must yield "
            "spatial bias/logits invariant within measured CUDA numeric noise, and the selected logit gradient "
            "to the dummy pose must be exactly zero."
        ),
    )
    parser.add_argument(
        "--gate1-arm",
        choices=GATE1_ARMS,
        default="none",
        help=(
            "Numerical-reference control (named Gate-1 in the code; see docs/NUMERIC_POSE.md). The default 'none' leaves the evaluator unchanged. "
            "hard_off removes every numerical query-pose answer path; zero keeps the path enabled but "
            "zeros the exact pose-derived tensors consumed by generation; permuted consumes a frozen donor manifest; "
            "semantic_type_canonical "
            "changes only the learned reference-type id."
        ),
    )
    parser.add_argument(
        "--gate1-reference-family",
        choices=GATE1_REFERENCE_FAMILIES,
        default=None,
        help="Required for a non-none numerical-reference (Gate-1) arm; all applies current/hard_off to every reference type.",
    )
    parser.add_argument(
        "--gate1-permutation-manifest-jsonl",
        type=Path,
        default=None,
        help="Frozen one-to-one donor manifest. Required only for --gate1-arm permuted.",
    )
    parser.add_argument(
        "--gate1-expected-checkpoint-sha256",
        default=None,
        help="Expected SHA-256 of the checkpoint file for the numerical-reference (Gate-1) control. Required for every non-default arm.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--numeric-scaffold-mode", choices=("none", "arnum"), default="none")
    parser.add_argument("--global-3d-pe-mode", choices=("none", "fourier_mlp"), default="none")
    parser.add_argument("--global-3d-pe-hidden-dim", type=int, default=512)
    parser.add_argument("--global-3d-pe-fourier-bands", type=int, default=8)
    parser.add_argument("--global-3d-pe-coordinate-scale", type=float, default=5.0)
    parser.add_argument("--global-3d-pe-residual-scale", type=float, default=0.05)
    parser.add_argument("--spatial-bias-scale", type=float, default=1.0)
    parser.add_argument(
        "--active-flow-prefix",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
        help=(
            "Eval-only prefix truncation on a fully constructed checkpoint. "
            "Later flow blocks are replaced with zero-output modules while the trained /4 update scale is retained."
        ),
    )
    parser.add_argument(
        "--reference-type-override",
        default="none",
        help="Reference-type intervention (reference permutation in docs/EXPERIMENT_INDEX.md): none, permute, or fixed:<reference_type>. Applied before spatial descriptors are built.",
    )
    parser.add_argument(
        "--reference-type-override-map-jsonl",
        type=Path,
        default=None,
        help=(
            "Evaluation with predicted reference types: JSONL with sample_id and predicted_reference_type. "
            "Each eval sample must be covered; query pose is rebuilt from the predicted type."
        ),
    )
    parser.add_argument(
        "--gate-override",
        choices=("none", "uniform"),
        default="none",
        help="Branch-mixing intervention (uniform branch mixing in docs/EXPERIMENT_INDEX.md): force the spatial-flow branch gate to a fixed distribution.",
    )
    parser.add_argument(
        "--dump-decoder-aux",
        action="store_true",
        help="Also dump sample-level DecoderAux metric-head predictions/targets for diagnostic analysis.",
    )
    add_latency_args(parser)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_pose_tensor(value: Any, *, sample_id: str, field_name: str) -> torch.Tensor:
    pose = torch.as_tensor(value, dtype=torch.float32)
    if tuple(pose.shape) != (4, 4):
        raise ValueError(f"sample_id={sample_id} {field_name} must be 4x4, got {tuple(pose.shape)}")
    if not bool(torch.isfinite(pose).all().item()):
        raise ValueError(f"sample_id={sample_id} {field_name} contains non-finite values")
    return pose


def load_gate1_permutation_manifest(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    reference_family: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a frozen permutation and prove that it is a complete one-to-one intervention.

    这里不在评测时临时抽 donor。实验开始前就把每个样本换成谁写死，运行时只验收。
    这样三次 target permutation 和 semantic 跨样本 permutation 才能复现，也不会在分 shard
    后悄悄改变映射。
    """

    if not path.is_file():
        raise FileNotFoundError(f"Gate-1 permutation manifest does not exist: {path}")
    expected_ids = [str(row.get("sample_id") or "").strip() for row in rows]
    if any(not sample_id for sample_id in expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Gate-1 input rows must have non-empty unique sample_id values")
    expected_set = set(expected_ids)
    mapping: dict[str, dict[str, Any]] = {}
    donor_ids: list[str] = []
    seeds: set[int] = set()
    identical_pose_count = 0
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("sample_id") or "").strip()
        donor_id = str(record.get("donor_sample_id") or "").strip()
        family = str(record.get("reference_family") or "").strip()
        if not sample_id or not donor_id:
            raise ValueError(f"{path}:{line_index} requires sample_id and donor_sample_id")
        if sample_id in mapping:
            raise ValueError(f"{path}:{line_index} duplicate sample_id={sample_id}")
        if sample_id == donor_id:
            raise ValueError(f"{path}:{line_index} self permutation for sample_id={sample_id}")
        if family != reference_family:
            raise ValueError(
                f"{path}:{line_index} reference_family={family!r} does not match {reference_family!r}"
            )
        base_pose = _manifest_pose_tensor(
            record.get("base_query_pose_4x4"), sample_id=sample_id, field_name="base_query_pose_4x4"
        )
        donor_pose = _manifest_pose_tensor(
            record.get("query_pose_4x4"), sample_id=sample_id, field_name="query_pose_4x4"
        )
        base_gauge = str(record.get("geometry_gauge") or "").strip()
        donor_gauge = str(record.get("donor_geometry_gauge") or "").strip()
        if reference_family == "target_camera" and base_gauge != donor_gauge:
            raise ValueError(f"{path}:{line_index} target donor changed geometry gauge for {sample_id}")
        same_pose = bool(torch.equal(base_pose, donor_pose))
        declared_same_pose = bool(record.get("same_pose"))
        if same_pose != declared_same_pose:
            raise ValueError(
                f"{path}:{line_index} same_pose declaration mismatch for sample_id={sample_id}: "
                f"declared={declared_same_pose} measured={same_pose}"
            )
        identical_pose_count += int(same_pose)
        seed = int(record["seed"])
        seeds.add(seed)
        mapping[sample_id] = {
            **dict(record),
            "base_query_pose_tensor": base_pose,
            "query_pose_tensor": donor_pose,
        }
        donor_ids.append(donor_id)
    if set(mapping) != expected_set:
        missing = sorted(expected_set - set(mapping))
        extra = sorted(set(mapping) - expected_set)
        raise ValueError(
            f"Gate-1 permutation manifest/input id mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    if set(donor_ids) != expected_set or len(set(donor_ids)) != len(donor_ids):
        raise ValueError("Gate-1 donor ids must be a one-to-one permutation of the complete input id set")
    if len(seeds) != 1:
        raise ValueError(f"Gate-1 permutation manifest must contain exactly one seed, got {sorted(seeds)}")
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(mapping),
        "unique_donors": len(set(donor_ids)),
        "reference_family": reference_family,
        "seed": next(iter(seeds)),
        "self_mapping_count": 0,
        "identical_pose_count": identical_pose_count,
        "one_to_one": True,
        "built_from_unsharded_rows": True,
    }
    return mapping, audit


class Gate1PoseInterventionDataset:
    """Apply one explicit Gate-1 intervention after normal production sample loading.

    先让原 dataset 完整加载同一段文字、同一视觉缓存、同一 patch geometry 和原始 pose，
    然后只改本臂授权的那个量。这样 hard-off、zero、permutation 与 type 对照不会重建题目、
    不会重跑视觉塔，也不会把 reference type 和数值 pose 一起改掉。
    """

    def __init__(
        self,
        base_dataset: NativeAnswerPatchSpatialDataset,
        *,
        arm: str,
        reference_family: str,
        permutation_by_sample_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if arm == "none":
            raise ValueError("Gate1PoseInterventionDataset is unnecessary for arm=none")
        if arm not in GATE1_ARMS:
            raise ValueError(f"unsupported Gate-1 arm={arm!r}")
        if reference_family not in GATE1_REFERENCE_FAMILIES:
            raise ValueError(f"unsupported Gate-1 reference family={reference_family!r}")
        if reference_family == "all" and arm not in {"current", "hard_off"}:
            raise ValueError("all-reference comparison only supports current/hard_off")
        if arm == "semantic_type_canonical" and reference_family != "semantic":
            raise ValueError("semantic_type_canonical is only defined for the semantic family")
        if arm == "permuted" and not permutation_by_sample_id:
            raise ValueError("Gate-1 permuted arm requires a complete frozen permutation manifest")
        if arm != "permuted" and permutation_by_sample_id:
            raise ValueError("a Gate-1 permutation manifest is only valid for arm=permuted")
        self.base_dataset = base_dataset
        self.arm = arm
        self.reference_family = reference_family
        self.permutation_by_sample_id = {
            str(sample_id): dict(payload) for sample_id, payload in permutation_by_sample_id.items()
        }
        mapping = reference_type_mapping("simple3")
        self.expected_reference_type_id = mapping[
            "semantic_query_pose" if reference_family == "semantic" else "target_camera_pose"
        ]
        self.canonical_reference_type_id = mapping["canonical_bev_scene_pose"]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset[index]
        # 全题对照也包含原来已关闭的题；只切断数值通路，保留参照类型和观测。
        if self.reference_family == "all":
            if self.arm == "current":
                return sample
            return replace(sample, effective_pose_input_mask=False)
        if int(sample.reference_type_id) != int(self.expected_reference_type_id):
            raise ValueError(
                f"sample_id={sample.sample_id} Gate-1 family={self.reference_family} received "
                f"reference_type_id={sample.reference_type_id}, expected={self.expected_reference_type_id}"
            )
        if not bool(sample.effective_pose_input_mask):
            raise ValueError(
                f"sample_id={sample.sample_id} Gate-1 base sample must expose its current numerical pose; "
                f"mode={sample.query_pose_mode} status={sample.pose_status}"
            )
        if self.arm == "current":
            return sample
        mode = f"gate1::{self.reference_family}::{self.arm}::{sample.query_pose_mode}"
        if self.arm == "hard_off":
            return replace(sample, effective_pose_input_mask=False, query_pose_mode=mode)
        if self.arm == "zero":
            # zero 不能伪造一张全零 4x4 矩阵。那会把所有点改写成“相对世界原点”，
            # 导致 rel_query/cam_delta 仍非零。真正的 zero 在 generation collator 里直接
            # 清零模型消费的 vec12 和 descriptor[6:12]，这里保留原 pose 供审计。
            return replace(sample, query_pose_mode=mode)
        if self.arm == "semantic_type_canonical":
            # learned type id 在 generation collator 末端替换。dataset 先保持整条原始
            # semantic 样本不变，避免连监督张量或 pose resolver 一起改掉。
            return replace(sample, query_pose_mode=mode)
        if self.arm == "permuted":
            payload = self.permutation_by_sample_id.get(str(sample.sample_id))
            if payload is None:
                raise KeyError(f"sample_id={sample.sample_id} missing from Gate-1 permutation manifest")
            base_pose = payload["base_query_pose_tensor"]
            if not torch.equal(sample.query_pose.detach().cpu().float(), base_pose.detach().cpu().float()):
                max_error = float(
                    (sample.query_pose.detach().cpu().float() - base_pose.detach().cpu().float()).abs().max().item()
                )
                raise ValueError(
                    f"sample_id={sample.sample_id} current pose differs from frozen permutation base; "
                    f"max_abs_error={max_error}"
                )
            base_gauge = str(payload.get("geometry_gauge") or "")
            if base_gauge and str(sample.geometry_gauge) != base_gauge:
                raise ValueError(f"sample_id={sample.sample_id} geometry gauge differs from frozen manifest")
            return replace(
                sample,
                query_pose=payload["query_pose_tensor"].clone(),
                effective_pose_input_mask=True,
                query_pose_mode=mode,
                shuffled_pose_donor_sample_id=str(payload["donor_sample_id"]),
                shuffled_pose_applied=True,
            )
        raise AssertionError(f"unhandled Gate-1 arm={self.arm}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class Gate1GenerationCollator:
    """Make zero/type interventions on the exact tensors consumed by answer generation."""

    def __init__(self, base_collator: NativeGlowPatchSpatialDecoderAuxCollator, *, arm: str) -> None:
        self.base_collator = base_collator
        self.arm = str(arm)
        self.canonical_reference_type_id = reference_type_mapping("simple3")["canonical_bev_scene_pose"]

    def collate_generation_prompts(self, samples: Sequence[Any]) -> Dict[str, Any]:
        batch = self.base_collator.collate_generation_prompts(samples)
        base_pose_vec = batch["query_pose_vec12"].clone()
        base_descriptor_query = batch["spatial_descriptors"][:, 6:12].clone()
        base_reference_ids = batch["reference_type_ids"].clone()
        if self.arm == "hard_off":
            # 上游读取器应已关闭全部派生数值；这里检查，不补零掩盖漏接的输入。
            for value in (batch["effective_pose_input_mask"], batch["query_pose_vec12"],
                          batch["spatial_descriptors"][:, 6:12]):
                if torch.count_nonzero(value).item():
                    raise AssertionError("hard_off retains numerical reference input")
        if self.arm == "zero":
            # zero 与 hard-off 都让所有数值 pose 特征为 0；唯一差别是 zero 的 mask=1，
            # 因而保留 query_pose_proj(0) 的 learned bias，hard-off 会在投影后再乘 0。
            batch["query_pose_vec12"] = torch.zeros_like(batch["query_pose_vec12"])
            batch["spatial_descriptors"] = batch["spatial_descriptors"].clone()
            batch["spatial_descriptors"][:, 6:12] = 0.0
            if not bool((batch["effective_pose_input_mask"] == 1.0).all().item()):
                raise AssertionError("Gate-1 zero requires effective_pose_input_mask=1")
        elif self.arm == "semantic_type_canonical":
            batch["reference_type_ids"] = torch.full_like(
                batch["reference_type_ids"], int(self.canonical_reference_type_id)
            )
        batch["gate1_base_query_pose_vec12_hashes"] = [tensor_sha256(base_pose_vec[index]) for index in range(base_pose_vec.shape[0])]
        batch["gate1_effective_query_pose_vec12_hashes"] = [
            tensor_sha256(batch["query_pose_vec12"][index]) for index in range(batch["query_pose_vec12"].shape[0])
        ]
        batch["gate1_base_descriptor_query_hash"] = tensor_sha256(base_descriptor_query)
        batch["gate1_effective_descriptor_query_hash"] = tensor_sha256(batch["spatial_descriptors"][:, 6:12])
        batch["gate1_base_reference_type_ids"] = base_reference_ids
        return batch


def load_reference_type_override_map(path: Path, *, reference_schema: str) -> dict[str, str]:
    """读取 text-only router 预测的参考类型，变成 sample_id -> reference type。

    这里故意不做缺省规则。预测参考类型评测要检验 LLM router 的端到端效果，
    如果某个样本缺预测或输出非法，就应该直接失败，而不是静默退回规则标签。
    """

    if str(reference_schema) != "simple3":
        raise ValueError("reference_type_override_map_jsonl is only defined for reference_schema=simple3")
    if not path.is_file():
        raise FileNotFoundError(f"reference type override map does not exist: {path}")
    valid_references = set(reference_type_mapping(reference_schema))
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            record = json.loads(line)
            sample_id = str(record.get("sample_id") or "").strip()
            predicted = str(record.get("predicted_reference_type") or "").strip()
            parse_status = str(record.get("parse_status") or "").strip()
            if not sample_id:
                raise ValueError(f"{path}:{line_index} missing sample_id")
            if sample_id in mapping:
                raise ValueError(f"{path}:{line_index} duplicate sample_id={sample_id}")
            if parse_status and parse_status != "ok":
                raise ValueError(f"{path}:{line_index} sample_id={sample_id} parse_status={parse_status!r}")
            if predicted not in valid_references:
                raise ValueError(
                    f"{path}:{line_index} sample_id={sample_id} predicted_reference_type={predicted!r} "
                    f"must be one of {sorted(valid_references)}"
                )
            mapping[sample_id] = predicted
    if not mapping:
        raise ValueError(f"reference type override map is empty: {path}")
    return mapping


def select_shard(rows: list[Mapping[str, Any]], *, num_shards: int, shard_index: int) -> list[Mapping[str, Any]]:
    if int(num_shards) <= 0:
        raise ValueError(f"--num-shards must be positive, got {num_shards}")
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError(f"--shard-index must be in [0,num_shards), got {shard_index}/{num_shards}")
    if int(num_shards) == 1:
        return rows
    selected = [row for index, row in enumerate(rows) if index % int(num_shards) == int(shard_index)]
    if not selected:
        raise ValueError(f"shard {shard_index}/{num_shards} selected zero rows from {len(rows)} rows")
    return selected


def build_shuffled_query_pose_map(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """从完整、未分 shard 的 repaired rows 构造确定性的同 gauge 错 pose 对照。

    不同视频的 CUT3R 坐标系不能互相交换，所以 donor 只能来自相同 gauge 和
    相同 reference kind。canonical 是全场景固定参考系，交换它没有反事实意义，因此不参与。
    若某个 gauge 只有一个独特 pose，就明确记为 ineligible，不静默跨场景凑 donor。
    """

    groups: dict[tuple[str, str], list[tuple[str, Any, str]]] = defaultdict(list)
    eligible_rows = 0
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        refs = row.get("refs") if isinstance(row.get("refs"), Mapping) else {}
        aux = refs.get("simple3_aux") if isinstance(refs.get("simple3_aux"), Mapping) else None
        if not isinstance(aux, Mapping):
            raise KeyError(f"sample_id={sample_id} shuffled_pose requires repaired refs.simple3_aux")
        if str(aux.get("provided_pose_protocol_version") or "") != "quasar_provided_pose_eval_repair_v1":
            raise ValueError(f"sample_id={sample_id} shuffled_pose requires repaired protocol v1")
        if str(aux.get("pose_status") or "") != "ok" or not bool(aux.get("pose_input_mask")):
            continue
        reference_kind = str(aux.get("reference_kind") or "")
        if reference_kind not in {"target_camera", "semantic_query"}:
            continue
        pose = aux.get("query_pose_4x4")
        if pose is None:
            raise ValueError(f"sample_id={sample_id} status=ok but query_pose_4x4 is null")
        gauge = str(aux.get("geometry_gauge") or "")
        pose_key = json.dumps(pose, sort_keys=True, separators=(",", ":"))
        groups[(gauge, reference_kind)].append((sample_id, pose, pose_key))
        eligible_rows += 1

    donor_map: dict[str, dict[str, Any]] = {}
    ineligible: list[str] = []
    for (gauge, reference_kind), items in sorted(groups.items()):
        ordered = sorted(items, key=lambda item: item[0])
        for item_index, (sample_id, _pose, pose_key) in enumerate(ordered):
            donor = None
            for offset in range(1, len(ordered)):
                candidate = ordered[(item_index + offset) % len(ordered)]
                if candidate[2] != pose_key:
                    donor = candidate
                    break
            if donor is None:
                ineligible.append(sample_id)
                continue
            donor_map[sample_id] = {
                "donor_sample_id": donor[0],
                "query_pose_4x4": donor[1],
                "geometry_gauge": gauge,
                "reference_kind": reference_kind,
            }
    audit = {
        "eligible_query_pose_rows": eligible_rows,
        "shuffled_rows": len(donor_map),
        "ineligible_same_gauge_no_distinct_pose": len(ineligible),
        "first_ineligible_sample_ids": ineligible[:20],
        "group_count": len(groups),
        "built_from_unsharded_rows": True,
    }
    return donor_map, audit


def sample_next_token(*, next_logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0.0:
        return next_logits.argmax(dim=-1)
    scaled = next_logits / float(temperature)
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        picked = torch.multinomial(sorted_probs, num_samples=1).squeeze(1)
        return sorted_idx.gather(1, picked.view(1, 1)).squeeze(1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def generation_stop_ids(tokenizer: Any) -> set[int]:
    stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        stop_ids.add(int(im_end_id))
    return stop_ids


DECODER_AUX_METRIC_KEYS = {
    "scene_scale": ("decoder_metric_scene_scale_pred", "metric_scene_scale_targets", "metric_scene_scale_valid"),
    "room_extent": ("decoder_metric_room_extent_pred", "metric_room_extent_targets", "metric_room_extent_valid"),
    "camera_displacement": (
        "decoder_metric_camera_displacement_pred",
        "metric_camera_displacement_targets",
        "metric_camera_displacement_valid",
    ),
    "object_size": ("decoder_metric_object_size_pred", "metric_object_size_targets", "metric_object_size_valid"),
    "object_pair_distance": (
        "decoder_metric_object_pair_distance_pred",
        "metric_object_pair_distance_targets",
        "metric_object_pair_distance_valid",
    ),
    "camera_object_distance": (
        "decoder_metric_camera_object_distance_pred",
        "metric_camera_object_distance_targets",
        "metric_camera_object_distance_valid",
    ),
    "object_count": ("decoder_metric_object_count_pred", "metric_object_count_targets", "metric_object_count_valid"),
}

PRIMARY_DECODER_AUX_METRIC_BY_QTYPE = {
    "object_abs_distance": "camera_object_distance",
    "object_counting": "object_count",
    "object_size_estimation": "object_size",
    "room_size_estimation": "room_extent",
    "vsti_camera_displacement": "camera_displacement",
    "vsti_camera_obj_abs_dist": "camera_object_distance",
}


def detached_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().float().cpu().view(-1).tolist()]


def expm1_clamped(values: Sequence[float]) -> list[float]:
    # 这些 metric target/pred 在训练里大多用 log1p 尺度；这里额外给一个人能读懂的近似原尺度。
    return [max(0.0, float(torch.expm1(torch.tensor(float(value))).item())) for value in values]


def l1_mean(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) == 0:
        return None
    return sum(abs(float(a) - float(b)) for a, b in zip(left, right)) / float(len(left))


@torch.no_grad()
def compute_decoder_aux_metric_payload(
    *,
    model: NativeQwenGlowPatchSpatialDecoderAuxModel,
    batch: Mapping[str, Any],
    qtype: str,
) -> dict[str, Any]:
    """跑一次 prompt-only 前向，把 DecoderAux metric heads 的输出写成 JSON 友好的诊断信息。

    LM 最终答案是“嘴上说出来的数字”，metric head 是模型内部另一个“小仪表盘”。
    这个函数就是把小仪表盘读出来，后面才能判断：仪表盘本身准不准、以及它有没有帮到最终回答。
    """

    batch_size = int(batch["input_ids"].shape[0])
    if batch_size != 1:
        raise ValueError("decoder aux diagnostic currently expects batch_size=1")
    flow_out = model.spatial_flow(
        descriptors=batch["spatial_descriptors"].float(),
        token_sample_ids=batch["spatial_token_sample_ids"].long(),
        reference_type_ids=batch["reference_type_ids"].long(),
        query_pose_vec12=batch["query_pose_vec12"].float(),
        batch_size=batch_size,
        pose_input_mask=batch["effective_pose_input_mask"].float(),
    )
    spatial_bias = flow_out["spatial_bias"] * float(getattr(model, "eval_spatial_bias_scale", 1.0))
    qwen_out = model._qwen_forward_with_spatial_bias(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch["image_grid_thw"],
        mm_token_type_ids=batch["mm_token_type_ids"],
        spatial_bias=spatial_bias,
        spatial_query_positions=batch["spatial_query_positions"].long(),
        cached_image_embeds=batch.get("cached_image_embeds"),
        cached_deepstack_features=batch.get("cached_deepstack_features"),
        global_3d_points=batch.get("spatial_points"),
        global_3d_valid=batch.get("spatial_token_valid"),
    )
    aux_out = model._decoder_aux_predictions(batch=batch, qwen_out=qwen_out)

    metrics: dict[str, Any] = {}
    for metric_name, (pred_key, target_key, valid_key) in DECODER_AUX_METRIC_KEYS.items():
        pred_raw = detached_float_list(aux_out[pred_key])
        target_raw = detached_float_list(batch[target_key])
        valid = float(batch[valid_key].detach().float().cpu().view(-1)[0].item())
        pred_value = expm1_clamped(pred_raw)
        target_value = expm1_clamped(target_raw)
        metrics[metric_name] = {
            "valid": valid,
            "pred_raw": pred_raw,
            "target_raw": target_raw,
            "l1_raw": l1_mean(pred_raw, target_raw) if valid > 0.0 else None,
            "pred_expm1": pred_value,
            "target_expm1": target_value,
            "l1_expm1": l1_mean(pred_value, target_value) if valid > 0.0 else None,
        }

    primary_name = PRIMARY_DECODER_AUX_METRIC_BY_QTYPE.get(str(qtype))
    primary = None
    if primary_name is not None:
        primary = {"name": primary_name, **metrics[primary_name]}
    return {
        "metrics": metrics,
        "primary": primary,
        "spatial_query_abs_mean": float(qwen_out["spatial_query_abs_mean"].detach().cpu().item()),
        "safe_spatial_bias_abs_mean": float(qwen_out["safe_spatial_bias_abs_mean"].detach().cpu().item()),
        "safe_bias_norm_ratio_mean": float(qwen_out["safe_bias_norm_ratio_mean"].detach().cpu().item()),
        "safe_bias_norm_ratio_max": float(qwen_out["safe_bias_norm_ratio_max"].detach().cpu().item()),
    }


@torch.no_grad()
def generate_one_no_cache(
    *,
    model: NativeQwenGlowPatchSpatialDecoderAuxModel,
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    spatial_query_positions: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_embeds: torch.Tensor,
    deepstack_visual_embeds: Sequence[torch.Tensor],
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str, list[int]]:
    stop_ids = generation_stop_ids(tokenizer)
    base = model.base_qwen()
    qwen_model = base.model
    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        if generated:
            generated_ids = torch.tensor([generated], device=input_ids.device, dtype=input_ids.dtype)
            current_input_ids = torch.cat([input_ids, generated_ids], dim=1)
            generated_mm = torch.zeros_like(generated_ids)
            current_mm_token_type_ids = torch.cat([mm_token_type_ids, generated_mm], dim=1)
        else:
            current_input_ids = input_ids
            current_mm_token_type_ids = mm_token_type_ids
        current_attention_mask = torch.ones_like(current_input_ids)
        inputs_embeds = qwen_model.get_input_embeddings()(current_input_ids)
        inputs_embeds = model.apply_numeric_scaffold_input_delta(input_ids=current_input_ids, inputs_embeds=inputs_embeds)
        inputs_embeds, _query_audit = model._replace_spatial_query_embeddings(
            inputs_embeds=inputs_embeds,
            attention_mask=current_attention_mask,
            spatial_query_positions=spatial_query_positions,
        )
        image_mask, _ = qwen_model.get_placeholder_mask(
            current_input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_mask_1d = image_mask[..., 0]
        position_ids = qwen_model.compute_3d_position_ids(
            input_ids=current_input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=current_attention_mask,
            past_key_values=None,
            mm_token_type_ids=current_mm_token_type_ids,
        )
        language_kwargs = {
            "input_ids": None,
            "position_ids": position_ids,
            "attention_mask": current_attention_mask,
            "past_key_values": None,
            "inputs_embeds": inputs_embeds,
            "use_cache": False,
        }
        if deepstack_visual_embeds:
            language_kwargs["visual_pos_masks"] = image_mask_1d
            language_kwargs["deepstack_visual_embeds"] = list(deepstack_visual_embeds)
        language_outputs = qwen_model.language_model(**language_kwargs)
        hidden_states = language_outputs[0]
        next_logits = model.apply_numeric_scaffold_logit_delta(logits=base.lm_head(hidden_states), hidden_states=hidden_states)[:, -1, :].float()
        next_token = sample_next_token(next_logits=next_logits, temperature=temperature, top_p=top_p)
        token_id = int(next_token.item())
        if token_id in stop_ids:
            break
        generated.append(token_id)
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    raw_text = tokenizer.decode(generated, skip_special_tokens=False).strip()
    return text, raw_text, generated


@torch.no_grad()
def generate_one(
    *,
    model: NativeQwenGlowPatchSpatialDecoderAuxModel,
    batch: Mapping[str, Any],
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str, list[int]]:
    if int(batch["input_ids"].shape[0]) != 1:
        raise ValueError("generate_one requires batch_size=1 so right-padding cannot corrupt autoregressive positions")

    flow_out = model.spatial_flow(
        descriptors=batch["spatial_descriptors"].float(),
        token_sample_ids=batch["spatial_token_sample_ids"].long(),
        reference_type_ids=batch["reference_type_ids"].long(),
        query_pose_vec12=batch["query_pose_vec12"].float(),
        batch_size=1,
        pose_input_mask=batch["effective_pose_input_mask"].float(),
    )
    spatial_bias = flow_out["spatial_bias"] * float(getattr(model, "eval_spatial_bias_scale", 1.0))

    prompt_len = int(batch["attention_mask"][0].sum().item())
    input_ids = batch["input_ids"][:, :prompt_len]
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = batch["mm_token_type_ids"][:, :prompt_len]
    spatial_query_positions = batch["spatial_query_positions"].long()
    if int(spatial_query_positions.max().detach().cpu().item()) >= prompt_len:
        raise ValueError(
            f"spatial query position points outside prompt: max={int(spatial_query_positions.max().item())} prompt_len={prompt_len}"
        )

    stop_ids = generation_stop_ids(tokenizer)

    base = model.base_qwen()
    qwen_model = base.model
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    inputs_embeds = model.apply_numeric_scaffold_input_delta(input_ids=input_ids, inputs_embeds=inputs_embeds)
    inputs_embeds, _query_audit = model._replace_spatial_query_embeddings(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        spatial_query_positions=spatial_query_positions,
    )

    if batch.get("cached_image_embeds") is None:
        image_outputs = qwen_model.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True)
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        deepstack_source = list(getattr(image_outputs, "deepstack_features", ()) or ())
    else:
        image_embeds = batch["cached_image_embeds"].to(inputs_embeds.device, inputs_embeds.dtype)
        deepstack_source = list(batch.get("cached_deepstack_features") or ())
    if image_embeds.shape != spatial_bias.shape:
        raise ValueError(
            f"spatial bias must match Qwen image embeds exactly during generation, "
            f"bias={tuple(spatial_bias.shape)} image_embeds={tuple(image_embeds.shape)}"
        )

    safe_spatial_bias, _bias_audit = model._make_safe_spatial_bias(spatial_bias=spatial_bias, image_embeds=image_embeds)
    model.poseoff_generation_audit = {
        "pose_projection_after_mask_max_abs": float(model.spatial_flow.poseoff_projection_max_abs),
        "visual_writeback_max_abs": float(safe_spatial_bias.abs().max().item()),
        "deepstack_layers": len(deepstack_source),
        "spatial_query_count": int(spatial_query_positions.shape[1]),
    }
    global_3d_pe_delta, _global_3d_pe_audit = model._make_global_3d_pe_delta(
        image_embeds=image_embeds,
        points=batch.get("spatial_points"),
        valid=batch.get("spatial_token_valid"),
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
                f"deepstack layer {layer_i} shape must match spatial bias during generation, "
                f"deep={tuple(deep.shape)} bias={tuple(spatial_bias.shape)}"
            )
        deepstack_visual_embeds.append(
            deep.to(inputs_embeds.device, inputs_embeds.dtype)
            + visual_delta.to(inputs_embeds.device, inputs_embeds.dtype)
        )

    model_type = str(getattr(getattr(base, "config", None), "model_type", ""))
    if model_type.startswith("qwen2_5_vl"):
        return generate_one_no_cache(
            model=model,
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            spatial_query_positions=spatial_query_positions,
            image_grid_thw=batch["image_grid_thw"],
            image_embeds=image_embeds,
            deepstack_visual_embeds=deepstack_visual_embeds,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    position_ids = qwen_model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=batch["image_grid_thw"],
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
        "use_cache": True,
    }
    if deepstack_visual_embeds:
        language_kwargs["visual_pos_masks"] = image_mask_1d
        language_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
    language_outputs = qwen_model.language_model(**language_kwargs)
    past_key_values = language_outputs.past_key_values
    hidden_states = language_outputs[0]
    next_logits = model.apply_numeric_scaffold_logit_delta(logits=base.lm_head(hidden_states), hidden_states=hidden_states)[:, -1, :].float()

    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        next_token = sample_next_token(next_logits=next_logits, temperature=temperature, top_p=top_p)
        token_id = int(next_token.item())
        if token_id in stop_ids:
            break
        generated.append(token_id)

        next_input_ids = next_token.view(1, 1)
        next_inputs_embeds = qwen_model.get_input_embeddings()(next_input_ids)
        next_inputs_embeds = model.apply_numeric_scaffold_input_delta(input_ids=next_input_ids, inputs_embeds=next_inputs_embeds)
        next_position_ids = qwen_model.compute_3d_position_ids(
            input_ids=next_input_ids,
            image_grid_thw=None,
            video_grid_thw=None,
            inputs_embeds=next_inputs_embeds,
            attention_mask=None,
            past_key_values=past_key_values,
            mm_token_type_ids=None,
        )
        language_outputs = qwen_model.language_model(
            input_ids=None,
            position_ids=next_position_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=next_inputs_embeds,
            use_cache=True,
        )
        past_key_values = language_outputs.past_key_values
        hidden_states = language_outputs[0]
        next_logits = model.apply_numeric_scaffold_logit_delta(logits=base.lm_head(hidden_states), hidden_states=hidden_states)[:, -1, :].float()

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    raw_text = tokenizer.decode(generated, skip_special_tokens=False).strip()
    return text, raw_text, generated


def verify_missing_pose_gate_with_real_qwen(
    *,
    model: NativeQwenGlowPatchSpatialDecoderAuxModel,
    dataset: Any,
    collator: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """用真实 Qwen 前向验收 missing 门控，不只验一个小型空间模块。

    这里故意把 missing 样本的 dummy pose 改成两组完全不同的数。如果 descriptor 之后
    仍有任何一条暗路读取 pose，spatial bias 或下一 token 的完整 logits 就会超出同输入重跑的
    CUDA 数值噪声。最后再对一个 logit 反传，确认 dummy pose 的梯度严格为 0。
    """

    missing_sample = None
    for index in range(len(dataset)):
        candidate = dataset[index]
        if not bool(candidate.effective_pose_input_mask):
            missing_sample = candidate
            break
    if missing_sample is None:
        raise ValueError("--verify-missing-pose-gate requires at least one effectively masked sample")

    batch = collator.collate_generation_prompts([missing_sample])
    batch = move_batch_to_device(batch, device)
    if bool(float(batch["effective_pose_input_mask"].view(-1)[0].item()) > 0.5):
        raise AssertionError("selected missing sample unexpectedly has effective_pose_input_mask=true")

    pose_a = torch.linspace(-3.0, 3.0, steps=12, device=device, dtype=torch.float32).view(1, 12)
    pose_b = torch.linspace(19.0, -11.0, steps=12, device=device, dtype=torch.float32).view(1, 12)

    def forward_from_pose(query_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flow_out = model.spatial_flow(
            descriptors=batch["spatial_descriptors"].float(),
            token_sample_ids=batch["spatial_token_sample_ids"].long(),
            reference_type_ids=batch["reference_type_ids"].long(),
            query_pose_vec12=query_pose,
            batch_size=1,
            pose_input_mask=batch["effective_pose_input_mask"].float(),
        )
        spatial_bias = flow_out["spatial_bias"] * float(getattr(model, "eval_spatial_bias_scale", 1.0))
        qwen_out = model._qwen_forward_with_spatial_bias(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch["image_grid_thw"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            spatial_bias=spatial_bias,
            spatial_query_positions=batch["spatial_query_positions"].long(),
            cached_image_embeds=batch.get("cached_image_embeds"),
            cached_deepstack_features=batch.get("cached_deepstack_features"),
            global_3d_points=batch.get("spatial_points"),
            global_3d_valid=batch.get("spatial_token_valid"),
        )
        # 下一 token 的整张词表 logits 已足够决定真实生成，并避免保留整段序列的大张量。
        return spatial_bias, qwen_out["logits"][:, -1, :]

    with torch.no_grad():
        bias_a, logits_a = forward_from_pose(pose_a)
        bias_dtype = bias_a.dtype
        logits_dtype = logits_a.dtype
        bias_a_cpu = bias_a.detach().float().cpu()
        logits_a_cpu = logits_a.detach().float().cpu()
        del bias_a, logits_a
        # 同一个 pose 重跑一次，用它量出 CUDA atomic/GEMM 自身的数值抖动。
        # 之后 A-vs-B 的差异不能超过这个重复运行噪声加上按 dtype 推导的舍入下限。
        bias_repeat, logits_repeat = forward_from_pose(pose_a)
        bias_repeat_cpu = bias_repeat.detach().float().cpu()
        logits_repeat_cpu = logits_repeat.detach().float().cpu()
        del bias_repeat, logits_repeat
        bias_b, logits_b = forward_from_pose(pose_b)
        bias_b_cpu = bias_b.detach().float().cpu()
        logits_b_cpu = logits_b.detach().float().cpu()
        del bias_b, logits_b

    bias_equal = torch.equal(bias_a_cpu, bias_b_cpu)
    logits_equal = torch.equal(logits_a_cpu, logits_b_cpu)
    bias_repeat_max_abs_diff = float((bias_a_cpu - bias_repeat_cpu).abs().max().item())
    logits_repeat_max_abs_diff = float((logits_a_cpu - logits_repeat_cpu).abs().max().item())
    bias_max_abs_diff = float((bias_a_cpu - bias_b_cpu).abs().max().item())
    logits_max_abs_diff = float((logits_a_cpu - logits_b_cpu).abs().max().item())
    bias_scale = max(1.0, float(bias_a_cpu.abs().max().item()), float(bias_b_cpu.abs().max().item()))
    logits_scale = max(1.0, float(logits_a_cpu.abs().max().item()), float(logits_b_cpu.abs().max().item()))
    bias_roundoff_floor = 8.0 * float(torch.finfo(bias_dtype).eps) * bias_scale
    logits_roundoff_floor = 2.0 * float(torch.finfo(logits_dtype).eps) * logits_scale
    bias_allowed_max_abs_diff = 2.0 * bias_repeat_max_abs_diff + bias_roundoff_floor
    logits_allowed_max_abs_diff = 2.0 * logits_repeat_max_abs_diff + logits_roundoff_floor
    bias_invariant_within_numeric_noise = bool(bias_max_abs_diff <= bias_allowed_max_abs_diff)
    logits_invariant_within_numeric_noise = bool(logits_max_abs_diff <= logits_allowed_max_abs_diff)
    next_token_argmax_equal = bool(int(logits_a_cpu.argmax(dim=-1).item()) == int(logits_b_cpu.argmax(dim=-1).item()))

    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        pose_for_grad = pose_a.detach().clone().requires_grad_(True)
        _bias_grad, logits_grad = forward_from_pose(pose_for_grad)
        selected_logit = logits_grad[0, -1]
        pose_grad = torch.autograd.grad(selected_logit, pose_for_grad, allow_unused=False)[0]
        pose_grad_nonzero = int(torch.count_nonzero(pose_grad.detach()).item())
        pose_grad_max_abs = float(pose_grad.detach().abs().max().item())
        del _bias_grad, logits_grad, selected_logit
    finally:
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)

    passed = bool(
        bias_invariant_within_numeric_noise
        and logits_invariant_within_numeric_noise
        and next_token_argmax_equal
        and pose_grad_nonzero == 0
        and pose_grad_max_abs == 0.0
    )
    audit = {
        "sample_id": str(missing_sample.sample_id),
        "query_pose_mode": str(missing_sample.query_pose_mode),
        "pose_status": str(missing_sample.pose_status),
        "base_pose_input_mask": bool(missing_sample.pose_input_mask),
        "effective_pose_input_mask": bool(missing_sample.effective_pose_input_mask),
        "spatial_bias_dtype": str(bias_dtype),
        "spatial_bias_exactly_equal": bias_equal,
        "spatial_bias_repeat_same_pose_max_abs_diff": bias_repeat_max_abs_diff,
        "spatial_bias_max_abs_diff": bias_max_abs_diff,
        "spatial_bias_allowed_max_abs_diff": bias_allowed_max_abs_diff,
        "spatial_bias_invariant_within_numeric_noise": bias_invariant_within_numeric_noise,
        "next_token_logits_dtype": str(logits_dtype),
        "next_token_logits_exactly_equal": logits_equal,
        "next_token_logits_repeat_same_pose_max_abs_diff": logits_repeat_max_abs_diff,
        "next_token_logits_max_abs_diff": logits_max_abs_diff,
        "next_token_logits_allowed_max_abs_diff": logits_allowed_max_abs_diff,
        "next_token_logits_invariant_within_numeric_noise": logits_invariant_within_numeric_noise,
        "next_token_argmax_equal": next_token_argmax_equal,
        "dummy_pose_gradient_nonzero_count": pose_grad_nonzero,
        "dummy_pose_gradient_max_abs": pose_grad_max_abs,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(f"real-Qwen missing-pose hard-gate verification failed: {audit}")
    return audit


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise ValueError("real generation eval currently requires --batch-size 1")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")
    gate1_active = str(args.gate1_arm) != "none"
    if args.verify_numeric_pose and (args.gate1_reference_family != 'all' or args.gate1_arm not in {'current', 'hard_off'}):
        raise ValueError('verify-numeric-pose requires current/hard_off over all reference types')
    if not gate1_active:
        if (
            args.gate1_reference_family is not None
            or args.gate1_permutation_manifest_jsonl is not None
            or args.gate1_expected_checkpoint_sha256 is not None
        ):
            raise ValueError("Gate-1 family/manifest/checkpoint-hash arguments require a non-none --gate1-arm")
    else:
        if args.gate1_reference_family is None:
            raise ValueError("a non-none --gate1-arm requires --gate1-reference-family")
        if args.gate1_reference_family == "all" and args.gate1_arm not in {"current", "hard_off"}:
            raise ValueError("all-reference comparison only supports current/hard_off")
        if str(args.gate1_arm) == "permuted" and args.gate1_permutation_manifest_jsonl is None:
            raise ValueError("--gate1-arm permuted requires --gate1-permutation-manifest-jsonl")
        if str(args.gate1_arm) != "permuted" and args.gate1_permutation_manifest_jsonl is not None:
            raise ValueError("--gate1-permutation-manifest-jsonl is only valid for --gate1-arm permuted")
        expected_checkpoint_sha256 = str(args.gate1_expected_checkpoint_sha256 or "").strip().lower()
        if len(expected_checkpoint_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_checkpoint_sha256):
            raise ValueError("Gate-1 requires a lowercase 64-character --gate1-expected-checkpoint-sha256")
        if str(args.gate1_reference_family) == "semantic" and str(args.query_pose_mode) != "legacy":
            raise ValueError("semantic Gate-1 arms must use the legacy production sidecar and query_pose_mode=legacy")
        if str(args.gate1_reference_family) == "target_camera" and str(args.query_pose_mode) != "provided_full":
            raise ValueError("target-camera Gate-1 arms must use repaired parser-valid rows and query_pose_mode=provided_full")
        if str(args.gate1_arm) == "semantic_type_canonical" and str(args.gate1_reference_family) != "semantic":
            raise ValueError("semantic_type_canonical is only valid for the semantic family")
        # Gate-1 必须是单变量因果诊断。下面这些其他消融若同时开启，结果就无法解释。
        if (
            str(args.reference_type_override) != "none"
            or args.reference_type_override_map_jsonl is not None
            or str(args.gate_override) != "none"
            or int(args.active_flow_prefix) != 4
            or float(args.spatial_bias_scale) != 1.0
            or str(args.numeric_scaffold_mode) != "none"
            or str(args.global_3d_pe_mode) != "none"
            or bool(args.dump_decoder_aux)
        ):
            raise ValueError("Gate-1 cannot be combined with routing/prefix/bias/scaffold/3D-PE/decoder-aux interventions")
    if str(args.query_pose_mode) != "legacy":
        if str(args.reference_schema) != "simple3" or not bool(args.simple3_aux_required):
            raise ValueError("repaired query-pose modes require --reference-schema simple3 --simple3-aux-required")
        if str(args.reference_type_override) != "none" or args.reference_type_override_map_jsonl is not None:
            raise ValueError("repaired query-pose modes cannot be combined with reference-type interventions")
    validate_latency_args(args)
    set_seed(int(args.seed))
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 生成端必须服从 checkpoint 的真实 token/readout 架构。旧 checkpoint 没有这些
    # 字段时保持历史 tokens-on/query-mean；新四臂则不能把 C/D 静默评成 tokens-on。
    checkpoint_file = resolve_checkpoint_file(args.checkpoint_path)
    checkpoint_file_sha256 = None
    if gate1_active:
        checkpoint_file_sha256 = sha256_file(checkpoint_file)
        if checkpoint_file_sha256 != str(args.gate1_expected_checkpoint_sha256).strip().lower():
            raise ValueError(
                "Gate-1 checkpoint trust-boundary mismatch: "
                f"expected={args.gate1_expected_checkpoint_sha256} actual={checkpoint_file_sha256} "
                f"path={checkpoint_file}"
            )
    checkpoint_contract_payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    checkpoint_train_args = (
        checkpoint_contract_payload.get("args")
        if isinstance(checkpoint_contract_payload.get("args"), Mapping)
        else {}
    )
    checkpoint_spatial_query_tokens_enabled = bool(
        checkpoint_train_args.get("spatial_query_tokens_enabled", True)
    )
    checkpoint_global_readout_source = str(
        checkpoint_train_args.get("global_readout_source", "query_token_mean")
    )
    checkpoint_spatial_query_sigreg_weight = float(
        checkpoint_train_args.get("spatial_query_sigreg_loss_weight", 0.0)
    )
    checkpoint_spatial_query_sigreg_projection_dim = int(
        checkpoint_train_args.get("spatial_query_sigreg_projection_dim", 128)
    )
    # load_processor_and_qwen 会按这个开关登记空间查询 token。评测端不能另设一套默认值，
    # 必须在加载基础模型前就使用 checkpoint 自己保存的结构契约。
    args.spatial_query_tokens_enabled = checkpoint_spatial_query_tokens_enabled
    del checkpoint_contract_payload

    raw_rows = read_native_jsonl(args.eval_jsonl)
    rows_all, data_audit = filter_native_rows(
        rows=raw_rows,
        max_rows=int(args.max_rows),
        require_patch_geometry=bool(args.require_patch_geometry),
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
        allow_legacy_mca=bool(args.allow_legacy_mca),
    )
    unsharded_eval_rows = len(rows_all)
    shuffled_query_pose_by_sample_id: dict[str, dict[str, Any]] = {}
    shuffled_pose_audit = None
    if str(args.query_pose_mode) == "shuffled_pose":
        shuffled_query_pose_by_sample_id, shuffled_pose_audit = build_shuffled_query_pose_map(rows_all)
    gate1_permutation_by_sample_id: dict[str, dict[str, Any]] = {}
    gate1_permutation_audit = None
    if str(args.gate1_arm) == "permuted":
        gate1_permutation_by_sample_id, gate1_permutation_audit = load_gate1_permutation_manifest(
            Path(args.gate1_permutation_manifest_jsonl),
            rows=rows_all,
            reference_family=str(args.gate1_reference_family),
        )
    rows = select_shard(rows_all, num_shards=int(args.num_shards), shard_index=int(args.shard_index))
    rows = slice_rows_for_latency(rows, args)
    existing_records = []
    if args.resume_inference:
        if not args.inference_only or latency_enabled(args):
            raise ValueError('Resume requires answer-free generation without latency measurement')
        from quasar.reviewer_data import read_complete_predictions
        existing_records = read_complete_predictions(args.output_dir / 'predictions.jsonl',
                            [str(row['sample_id']) for row in rows], require_complete=False)
        rows = rows[len(existing_records):]
        if not rows:
            print(json.dumps({'status': 'already_complete', 'count': len(existing_records)}))
            return
    row_by_sample_id = {str(row["sample_id"]): row for row in rows}

    reference_type_override_by_sample_id = None
    if args.reference_type_override_map_jsonl is not None:
        if str(args.reference_type_override) != "none":
            raise ValueError("--reference-type-override-map-jsonl cannot be combined with --reference-type-override")
        reference_type_override_by_sample_id = load_reference_type_override_map(
            Path(args.reference_type_override_map_jsonl),
            reference_schema=str(args.reference_schema),
        )
        missing_sample_ids = [str(row["sample_id"]) for row in rows if str(row["sample_id"]) not in reference_type_override_by_sample_id]
        if missing_sample_ids:
            raise KeyError(
                f"reference type override map missing {len(missing_sample_ids)} eval rows; "
                f"first_missing={missing_sample_ids[:5]}"
            )

    visual_cache_audit = None
    if args.visual_cache_root is not None:
        visual_cache_audit = audit_visual_cache_rows(
            rows=rows,
            visual_cache_root=args.visual_cache_root,
            scannet_frame_root=args.scannet_frame_root,
            max_visual_frames=int(args.max_visual_frames),
            qwen_model_path=args.qwen_model_path,
            max_pixels=int(args.max_pixels),
            split_name=f"eval_shard_{int(args.shard_index):03d}_of_{int(args.num_shards):03d}",
        )

    processor, qwen, hidden_size, spatial_merge_size, qwen_meta = load_processor_and_qwen(args)
    spatial_query_tokens = build_spatial_query_tokens(int(args.spatial_query_count))
    base_collator = NativeGlowPatchSpatialDecoderAuxCollator(
        processor=processor,
        max_pixels=int(args.max_pixels),
        spatial_merge_size=spatial_merge_size,
        bev_grid_size=int(args.bev_grid_size),
        spatial_query_tokens=spatial_query_tokens,
        reference_schema=str(args.reference_schema),
        spatial_query_tokens_enabled=checkpoint_spatial_query_tokens_enabled,
    )
    base_dataset = NativeAnswerPatchSpatialDataset(
        rows=rows,
        max_visual_frames=int(args.max_visual_frames),
        scannet_frame_root=args.scannet_frame_root,
        depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
        reference_schema=str(args.reference_schema),
        visual_cache_root=args.visual_cache_root,
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
        simple3_aux_required=bool(args.simple3_aux_required),
        reference_type_override=str(args.reference_type_override),
        reference_type_override_by_sample_id=reference_type_override_by_sample_id,
        query_pose_mode=str(args.query_pose_mode),
        shuffled_query_pose_by_sample_id=shuffled_query_pose_by_sample_id,
        inference_only=bool(args.inference_only),
    )
    dataset: Any = base_dataset
    if gate1_active:
        dataset = Gate1PoseInterventionDataset(
            base_dataset,
            arm=str(args.gate1_arm),
            reference_family=str(args.gate1_reference_family),
            permutation_by_sample_id=gate1_permutation_by_sample_id,
        )
    collator: Any = base_collator
    if gate1_active:
        # zero/type 两个臂必须在生产 collator 完成正常构造后才动最终模型输入。
        # 这样不会重建 prompt、视觉缓存或 patch geometry，也不会误改训练监督张量。
        collator = Gate1GenerationCollator(base_collator, arm=str(args.gate1_arm))
    aux_weights = build_aux_weight_dict(args)
    model = NativeQwenGlowPatchSpatialDecoderAuxModel(
        qwen=qwen,
        tokenizer=processor.tokenizer,
        qwen_hidden_size=hidden_size,
        flow_dim=int(args.flow_dim),
        num_flow_steps=int(args.num_flow_steps),
        dropout=float(args.dropout),
        bev_grid_size=int(args.bev_grid_size),
        reference_schema=str(args.reference_schema),
        aux_weights=aux_weights,
        decoder_aux_weights={},
        num_spatial_bias_experts=int(args.num_spatial_bias_experts),
        bias_gate_temperature=float(args.bias_gate_temperature),
        safe_bias_residual_init=float(args.safe_bias_residual_init),
        safe_bias_residual_max=float(args.safe_bias_residual_max),
        bias_norm_max_ratio=float(args.bias_norm_max_ratio),
        spatial_query_count=int(args.spatial_query_count),
        spatial_query_tokens_enabled=checkpoint_spatial_query_tokens_enabled,
        global_readout_source=checkpoint_global_readout_source,
        spatial_query_sigreg_loss_weight=checkpoint_spatial_query_sigreg_weight,
        spatial_query_sigreg_projection_dim=checkpoint_spatial_query_sigreg_projection_dim,
        numeric_scaffold_token_ids=list(getattr(args, "numeric_scaffold_token_ids", [])),
        global_3d_pe_mode=str(getattr(args, "global_3d_pe_mode", "none")),
        global_3d_pe_hidden_dim=int(getattr(args, "global_3d_pe_hidden_dim", 512)),
        global_3d_pe_fourier_bands=int(getattr(args, "global_3d_pe_fourier_bands", 8)),
        global_3d_pe_coordinate_scale=float(getattr(args, "global_3d_pe_coordinate_scale", 5.0)),
        global_3d_pe_residual_scale=float(getattr(args, "global_3d_pe_residual_scale", 0.05)),
    )
    setattr(model, "eval_spatial_bias_scale", float(args.spatial_bias_scale))
    # gate_override 是纯评测干预，不写进 checkpoint。模型加载完后显式挂到 spatial_flow 上，
    # 让所有 generation/diagnostic forward 都走同一个被干预的 gate 行为。
    model.spatial_flow.gate_override = str(args.gate_override)
    checkpoint_payload = load_patch_spatial_checkpoint(args.checkpoint_path, model, map_location="cpu")
    apply_eval_flow_prefix(model, int(args.active_flow_prefix))
    # 评测消融需要覆盖 checkpoint 里保存的安全残差上限，而不是重新训练。
    with torch.no_grad():
        model.spatial_flow.bias_norm_max_ratio.fill_(float(args.bias_norm_max_ratio))
        model.spatial_flow.safe_bias_residual_max.fill_(float(args.safe_bias_residual_max))
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    model.to(device)
    model.eval()

    missing_pose_gate_verification = None
    if bool(args.verify_missing_pose_gate):
        missing_pose_gate_verification = verify_missing_pose_gate_with_real_qwen(
            model=model,
            dataset=dataset,
            collator=collator,
            device=device,
        )
        write_json(args.output_dir / "missing_pose_gate_verification.json", missing_pose_gate_verification)
        print(json.dumps({"event": "missing_pose_gate_verification", **missing_pose_gate_verification}, ensure_ascii=False), flush=True)

    run_config = {
        "format": FORMAT_VERSION,
        "args": normalize_args_for_json(args),
        "raw_rows": len(raw_rows),
        "unsharded_eval_rows": unsharded_eval_rows,
        "eval_rows": len(rows),
        "shard": {"index": int(args.shard_index), "num_shards": int(args.num_shards)},
        "data_audit": data_audit,
        "visual_cache_audit": visual_cache_audit,
        "qwen_meta": qwen_meta,
        "checkpoint": {
            "path": str(args.checkpoint_path),
            "resolved_file": str(checkpoint_file),
            "sha256": checkpoint_file_sha256,
            "step": checkpoint_payload.get("step"),
            "epoch": checkpoint_payload.get("epoch"),
        },
        "checkpoint_spatial_query_contract": {
            "spatial_query_tokens_enabled": checkpoint_spatial_query_tokens_enabled,
            "tokens_inserted_per_sample": int(args.spatial_query_count)
            if checkpoint_spatial_query_tokens_enabled
            else 0,
            "global_readout_source": checkpoint_global_readout_source,
            "spatial_query_sigreg_loss_weight": checkpoint_spatial_query_sigreg_weight,
        },
        "query_pose_protocol": {
            "mode": str(args.query_pose_mode),
            "shuffled_pose_audit": shuffled_pose_audit,
            "missing_pose_gate_verification": missing_pose_gate_verification,
        },
        "gate1": {
            "active": gate1_active,
            "arm": str(args.gate1_arm),
            "reference_family": args.gate1_reference_family,
            "permutation_manifest": gate1_permutation_audit,
            "default_path_unchanged_when_arm_none": True,
            "intervention_contract": {
                "hard_off": "zero pose-derived tensors and set effective_pose_input_mask=0",
                "zero": "zero query_pose_vec12 and descriptor columns 6:12 while keeping effective_pose_input_mask=1",
                "permuted": "replace the full numerical pose from a frozen one-to-one donor manifest",
                "semantic_type_canonical": "change only the learned reference_type_id after normal collation",
            },
        },
        "flow_prefix": {
            "active_flow_prefix": int(args.active_flow_prefix),
            "loaded_flow_blocks": len(model.spatial_flow.flow_blocks),
            "update_scale_denominator": int(model.spatial_flow.num_flow_steps),
        },
        "source_counts": dict(Counter(str(row["source"]["tag"]) for row in rows)),
        "question_type_counts": dict(Counter(str(row["question"]["canonical_question_type"]) for row in rows)),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps({"event": "run_config", **run_config}, ensure_ascii=False), flush=True)

    if args.verify_numeric_pose:
        from quasar.numeric_pose_check import verify_real_inputs
        verify_real_inputs(model, base_dataset, dataset, base_collator, collator,
                           device, str(args.gate1_arm), args.output_dir)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(args.num_workers), collate_fn=collator.collate_generation_prompts)
    records: list[Dict[str, Any]] = list(existing_records)
    latency_records: list[Dict[str, Any]] = []
    pred_path = args.output_dir / "predictions.jsonl"
    with pred_path.open("a" if args.resume_inference else "w", encoding="utf-8", buffering=1) as handle:
        for index, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            is_timed_latency_row = latency_enabled(args) and index > int(args.latency_warmup_rows)
            if latency_enabled(args) and index == int(args.latency_warmup_rows) + 1:
                # warmup 阶段会触发 CUDA kernel 初始化；清空峰值后，显存统计才对应正式测速样本。
                reset_peak_memory_if_needed(device)
            if is_timed_latency_row:
                latency_sec, generate_result = timed_generate_call(
                    device,
                    lambda: generate_one(
                        model=model,
                        batch=batch,
                        tokenizer=processor.tokenizer,
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                    ),
                )
                prediction, raw_prediction, generated_ids = generate_result
            else:
                prediction, raw_prediction, generated_ids = generate_one(
                    model=model,
                    batch=batch,
                    tokenizer=processor.tokenizer,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
            sample_id = str(batch["sample_ids"][0])
            row = row_by_sample_id[sample_id]
            source_tag = str(row["source"]["tag"])
            qtype = str(row["question"]["canonical_question_type"])
            bench = bench_from_source(source_tag)
            decoder_aux_payload = None
            if bool(args.dump_decoder_aux):
                decoder_aux_payload = compute_decoder_aux_metric_payload(model=model, batch=batch, qtype=qtype)
            metric_payload = {} if args.inference_only else score_native_answer(prediction=prediction, row=row)
            source_simple3_aux = row.get("refs", {}).get("simple3_aux", {})
            source_pose_4x4 = (
                source_simple3_aux.get("query_pose_4x4")
                if isinstance(source_simple3_aux, Mapping)
                else None
            )
            source_pose_4x4_sha256 = hashlib.sha256(
                json.dumps(source_pose_4x4, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            gate1_permutation_payload = gate1_permutation_by_sample_id.get(sample_id)
            record = {
                "sample_id": sample_id,
                "bench": bench,
                "source_tag": source_tag,
                "canonical_question_type": qtype,
                "vsti_task_group": vsti_task_group(qtype) if bench == "VSTI" else None,
                "answer_type": str(row["answer"]["type"]),
                "target_text": str(row["answer"].get("text") or ""),
                "prediction_text": prediction,
                "prediction_text_raw": raw_prediction,
                "generated_token_count": len(generated_ids),
                "intervention_audit": model.poseoff_generation_audit,
                "query_pose_protocol": {
                    "mode": str(batch["query_pose_modes"][0]),
                    "reference_kind": str(batch["reference_kinds"][0]),
                    "pose_source": str(batch["pose_sources"][0]),
                    "pose_status": str(batch["pose_statuses"][0]),
                    "base_pose_input_mask": bool(float(batch["pose_input_mask"].view(-1)[0].item()) > 0.5),
                    "effective_pose_input_mask": bool(float(batch["effective_pose_input_mask"].view(-1)[0].item()) > 0.5),
                    "geometry_gauge": str(batch["geometry_gauges"][0]),
                    "provenance": str(batch["pose_provenances"][0]),
                    "failure_reason": str(batch["pose_failure_reasons"][0]),
                    "effective_query_pose_hash": str(batch["effective_query_pose_hashes"][0]),
                    "shuffled_pose_applied": bool(batch["shuffled_pose_applied"][0]),
                    "shuffled_pose_donor_sample_id": batch["shuffled_pose_donor_sample_ids"][0],
                },
                "gate1": {
                    "arm": str(args.gate1_arm),
                    "reference_family": args.gate1_reference_family,
                    "base_reference_type_id": int(batch["gate1_base_reference_type_ids"].view(-1)[0].item())
                    if "gate1_base_reference_type_ids" in batch
                    else int(batch["reference_type_ids"].view(-1)[0].item()),
                    "effective_reference_type_id": int(batch["reference_type_ids"].view(-1)[0].item()),
                    "base_query_pose_vec12_hash": batch.get("gate1_base_query_pose_vec12_hashes", [None])[0],
                    "effective_query_pose_vec12_hash": batch.get(
                        "gate1_effective_query_pose_vec12_hashes", [None]
                    )[0],
                    "base_descriptor_query_hash": batch.get("gate1_base_descriptor_query_hash"),
                    "effective_descriptor_query_hash": batch.get("gate1_effective_descriptor_query_hash"),
                    "source_query_pose_4x4_sha256": source_pose_4x4_sha256,
                    "query_pose_vec12_all_zero": bool(
                        torch.count_nonzero(batch["query_pose_vec12"].detach()).item() == 0
                    ),
                    "descriptor_query_channels_all_zero": bool(
                        torch.count_nonzero(batch["spatial_descriptors"][:, 6:12].detach()).item() == 0
                    ),
                    "permutation_manifest_sha256": (
                        gate1_permutation_audit["sha256"] if gate1_permutation_audit is not None else None
                    ),
                    "permutation_donor_sample_id": (
                        str(gate1_permutation_payload["donor_sample_id"])
                        if gate1_permutation_payload is not None
                        else None
                    ),
                    "permutation_geometry_gauge": (
                        str(gate1_permutation_payload.get("geometry_gauge") or "")
                        if gate1_permutation_payload is not None
                        else None
                    ),
                    "permutation_donor_geometry_gauge": (
                        str(gate1_permutation_payload.get("donor_geometry_gauge") or "")
                        if gate1_permutation_payload is not None
                        else None
                    ),
                    "permutation_same_pose": (
                        bool(gate1_permutation_by_sample_id[sample_id]["same_pose"])
                        if sample_id in gate1_permutation_by_sample_id
                        else None
                    ),
                },
                **metric_payload,
            }
            if decoder_aux_payload is not None:
                record["decoder_aux_metric"] = decoder_aux_payload
            if is_timed_latency_row:
                latency_records.append(
                    {
                        "sample_index": index,
                        "timed_index": len(latency_records) + 1,
                        "sample_id": sample_id,
                        "bench": bench,
                        "source_tag": source_tag,
                        "canonical_question_type": qtype,
                        "answer_type": str(row["answer"]["type"]),
                        "latency_sec": latency_sec,
                        "generated_token_count": len(generated_ids),
                    }
                )
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if int(args.log_every) > 0 and (index % int(args.log_every) == 0 or index == 1):
                partial = {} if args.inference_only else summarize_records(records)
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "seen": index,
                            "total": len(rows),
                            "shard_index": int(args.shard_index),
                            "num_shards": int(args.num_shards),
                            "elapsed_sec": round(time.time() - started_at, 3),
                            "vsi_score": partial.get("vsi_overall", {}).get("mean_score"),
                            "vsti_score": partial.get("vsti_overall", {}).get("mean_score"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {"generated_rows": len(records)} if args.inference_only else summarize_records(records)
    if not args.inference_only:
        summary["official_like_vsi"] = summarize_official_like_vsi_from_prediction_rows(records, row_by_sample_id)
        summary["vsti_task_group_macro"] = summarize_vsti_task_groups([record for record in records if record["bench"] == "VSTI"])
    summary_payload = {
        "status": "completed",
        "format": FORMAT_VERSION,
        "elapsed_sec": round(time.time() - started_at, 3),
        "eval_rows": len(records),
        "unsharded_eval_rows": unsharded_eval_rows,
        "shard": {"index": int(args.shard_index), "num_shards": int(args.num_shards)},
        "summary": summary,
    }
    if latency_enabled(args):
        latency_summary = summarize_latency_records(
            records=latency_records,
            args=args,
            device=device,
            total_elapsed_sec=time.time() - started_at,
        )
        summary_payload["latency_summary"] = latency_summary
        write_latency_outputs(args=args, records=latency_records, summary=latency_summary)
    write_json(args.output_dir / "eval_summary.json", summary)
    write_json(args.output_dir / "summary.json", summary_payload)
    print(json.dumps({"event": "completed", **summary_payload}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
