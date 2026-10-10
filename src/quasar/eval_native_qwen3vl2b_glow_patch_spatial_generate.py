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
"""Real generate eval for Native-Qwen QUASAR patch-level spatial injection.

这个脚本不做 MCA logits shortcut，也不只看 loss。
它加载 SFT/RL checkpoint，让模型在原生 Qwen 自回归路径上真正生成答案，
然后按 native-answer schema 逐条打分，并按 VSI/VSTI 官方题型汇总。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, DefaultDict, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader



from quasar.build_native_answer_spatial_qa_pack import (  # noqa: E402
    parse_count_answer,
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
from quasar.train_native_qwen3vl2b_glow_patch_spatial import (  # noqa: E402
    FORMAT_VERSION as PATCH_TRAIN_FORMAT_VERSION,
    NativeAnswerPatchSpatialDataset,
    NativeGlowPatchSpatialCollator,
    NativeQwenGlowPatchSpatialModel,
    audit_visual_cache_rows,
    filter_native_rows,
    load_lora_state_dict,
    load_processor_and_qwen,
    move_batch_to_device,
    read_native_jsonl,
    resolve_checkpoint_file,
)


FORMAT_VERSION = "native_qwen3vl2b_glow_patch_spatial_generate_eval_v1"
VSI_SOURCE_TAGS = {"vlm3r_canonical_vsi"}
VSTI_SOURCE_TAGS = {"vsti_official_row32", "vlm3r_unified_vsti"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a QUASAR patch-spatial checkpoint by real generation.")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--qwen-cache-dir", type=Path, default=None)
    parser.add_argument("--visual-cache-root", type=Path, default=None)
    parser.add_argument("--scannet-frame-root", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--require-patch-geometry", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-visual-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--depth-grid-h", type=int, default=24)
    parser.add_argument("--depth-grid-w", type=int, default=32)
    parser.add_argument("--reference-schema", choices=("legacy5", "simple3"), default="legacy5")
    parser.add_argument("--flow-dim", type=int, default=512)
    parser.add_argument("--num-flow-steps", type=int, default=4)
    parser.add_argument(
        "--disable-spatial-flow-bias",
        action="store_true",
        help="Ablation: generate without injecting spatial_flow bias into Qwen visual tokens.",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--bev-grid-size", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--reference-type-loss-weight", type=float, default=0.2)
    parser.add_argument("--query-pose-loss-weight", type=float, default=0.1)
    parser.add_argument("--bev-loss-weight", type=float, default=0.05)
    parser.add_argument("--patch-consistency-loss-weight", type=float, default=0.05)
    parser.add_argument("--target-camera-coord-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-query-coord-loss-weight", type=float, default=0.0)
    parser.add_argument("--bev-coord-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\.\-\+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_first_float(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text.replace(",", ""))
    if match is None:
        return None
    return float(match.group(0))


def parse_choice_prediction(prediction: str, answer: Mapping[str, Any]) -> str | None:
    options = answer.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError("choice metric requires non-empty answer.options")
    allowed = {str(option["letter"]).strip().upper(): str(option["text"]).strip() for option in options}
    stripped = prediction.strip()
    match = re.match(r"^\s*([A-Z])(?:[\.|\)|:：]|\s|$)", stripped, flags=re.IGNORECASE)
    if match is not None:
        letter = match.group(1).upper()
        if letter in allowed:
            return letter
    normalized_prediction = normalize_text(prediction)
    for letter, text in allowed.items():
        norm_text = normalize_text(text)
        if norm_text and (norm_text == normalized_prediction or norm_text in normalized_prediction):
            return letter
    # 开放式输出有时会先说一句话再给字母；解析允许找第一个独立选项字母。
    tokens = re.findall(r"\b[A-Z]\b", prediction.upper())
    for token in tokens:
        if token in allowed:
            return token
    return None


def score_native_answer(*, prediction: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    answer = row["answer"]
    answer_type = str(answer["type"])
    target_text = str(answer.get("text") or "")
    eval_payload = row.get("eval") if isinstance(row.get("eval"), Mapping) else {}
    metric = str(eval_payload.get("metric") or "")
    tolerance_value = eval_payload.get("tolerance")
    tolerance = float(tolerance_value) if tolerance_value is not None else None

    if answer_type == "multiple_choice":
        pred_letter = parse_choice_prediction(prediction, answer)
        gold_letter = str(answer.get("option_letter") or "").strip().upper()
        if not gold_letter:
            raise ValueError(f"sample_id={row['sample_id']} multiple_choice missing option_letter")
        correct = pred_letter == gold_letter
        return {"score": 1.0 if correct else 0.0, "correct": correct, "metric": metric or "choice_accuracy", "parsed": pred_letter}

    if answer_type == "count":
        pred_count = parse_count_answer(prediction)
        gold_value = answer.get("numeric_value")
        if gold_value is None:
            gold_value = parse_count_answer(target_text)
        if gold_value is None:
            raise ValueError(f"sample_id={row['sample_id']} count answer missing numeric_value and unparseable text={target_text!r}")
        correct = pred_count is not None and int(pred_count) == int(gold_value)
        return {"score": 1.0 if correct else 0.0, "correct": correct, "metric": metric or "count_exact", "parsed": pred_count, "target_numeric": int(gold_value)}

    if answer_type == "numeric":
        pred_value = parse_first_float(prediction)
        gold_raw = answer.get("numeric_value")
        if gold_raw is None:
            gold_raw = parse_first_float(target_text)
        if gold_raw is None:
            raise ValueError(f"sample_id={row['sample_id']} numeric answer missing numeric_value and unparseable text={target_text!r}")
        gold = float(gold_raw)
        abs_error = None if pred_value is None else abs(float(pred_value) - gold)
        if pred_value is None:
            score = 0.0
            correct = False
        elif tolerance is not None:
            correct = abs_error <= tolerance
            score = 1.0 if correct else 0.0
        else:
            # 没有官方 tolerance 的连续值不能假装成分类准确率。
            # 这里用 [0,1] 的软分数参与 overall，同时单独报告 MAE。
            correct = False
            score = max(0.0, 1.0 - abs_error / max(abs(gold), 1.0))
        return {
            "score": float(score),
            "correct": bool(correct),
            "metric": metric or "numeric_value",
            "parsed": pred_value,
            "target_numeric": gold,
            "abs_error": abs_error,
            "tolerance": tolerance,
        }

    if answer_type == "yes_no":
        norm = normalize_text(prediction)
        pred = None
        if re.search(r"\byes\b|\btrue\b", norm):
            pred = "yes"
        elif re.search(r"\bno\b|\bfalse\b", norm):
            pred = "no"
        gold_norm = "yes" if normalize_text(target_text) in {"yes", "true"} else "no"
        correct = pred == gold_norm
        return {"score": 1.0 if correct else 0.0, "correct": correct, "metric": metric or "yes_no_exact", "parsed": pred}

    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(target_text)
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    if not pred_tokens and not gold_tokens:
        f1 = 1.0
    elif not pred_tokens or not gold_tokens or sum(overlap.values()) == 0:
        f1 = 0.0
    else:
        precision = sum(overlap.values()) / len(pred_tokens)
        recall = sum(overlap.values()) / len(gold_tokens)
        f1 = 2.0 * precision * recall / (precision + recall)
    exact = pred_norm == gold_norm
    return {"score": 1.0 if exact else float(f1), "correct": bool(exact), "metric": metric or "short_text_normalized", "parsed": pred_norm, "token_f1": float(f1)}


def bench_from_source(source_tag: str) -> str:
    if source_tag in VSI_SOURCE_TAGS:
        return "VSI"
    if source_tag in VSTI_SOURCE_TAGS:
        return "VSTI"
    return source_tag


def vsti_task_group(qtype: str) -> str:
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
    return qtype


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero eval records")
    buckets: dict[str, DefaultDict[str, list[Mapping[str, Any]]]] = {
        "by_bench": defaultdict(list),
        "by_bench_question_type": defaultdict(list),
        "by_vsti_task_group": defaultdict(list),
        "debug_by_answer_type": defaultdict(list),
    }
    for record in records:
        bench = str(record["bench"])
        qtype = str(record["canonical_question_type"])
        buckets["by_bench"][bench].append(record)
        buckets["by_bench_question_type"][f"{bench}::{qtype}"].append(record)
        if bench == "VSTI":
            buckets["by_vsti_task_group"][str(record["vsti_task_group"])].append(record)
        buckets["debug_by_answer_type"][str(record["answer_type"])].append(record)

    def pack(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"count": 0, "mean_score": None}
        scores = [float(row["score"]) for row in rows]
        exact_capable = [row for row in rows if row.get("correct") is not None]
        numeric_errors = [float(row["abs_error"]) for row in rows if row.get("abs_error") is not None]
        payload: Dict[str, Any] = {
            "count": len(rows),
            "mean_score": sum(scores) / len(scores),
        }
        if exact_capable:
            payload["exact_correct_or_tolerance_correct"] = sum(1 for row in exact_capable if bool(row.get("correct")))
            payload["exact_or_tolerance_acc"] = payload["exact_correct_or_tolerance_correct"] / len(exact_capable)
        if numeric_errors:
            payload["numeric_mae"] = sum(numeric_errors) / len(numeric_errors)
        return payload

    summary = {
        "overall": pack(records),
        "vsi_overall": pack([record for record in records if record["bench"] == "VSI"]),
        "vsti_overall": pack([record for record in records if record["bench"] == "VSTI"]),
    }
    for name, bucket in buckets.items():
        summary[name] = {key: pack(value) for key, value in sorted(bucket.items())}
    return summary


def load_patch_spatial_generation_checkpoint(
    path: Path,
    model: NativeQwenGlowPatchSpatialModel,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """加载 patch-spatial checkpoint，专门服务真实生成评测。

    后期版本给 spatial_flow 新增了 object-center / query-frame / BEV coordinate
    这几组辅助 head。较早的 GRPO checkpoint 没有这些 head，但真实生成只用
    spatial_bias，不读这些辅助 head。为了能横向评测老 checkpoint，这里只允许缺失
    这几组“生成时完全不用”的新辅助 head；其它任何 key 缺失或多出来，都继续报错。
    """

    checkpoint_file = resolve_checkpoint_file(path)
    payload = torch.load(checkpoint_file, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {checkpoint_file}")
    if str(payload.get("format_version")) != PATCH_TRAIN_FORMAT_VERSION:
        raise ValueError(f"checkpoint format mismatch: {payload.get('format_version')!r} vs {PATCH_TRAIN_FORMAT_VERSION!r}")
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

    flow_state = payload["spatial_flow_state_dict"]
    if not isinstance(flow_state, Mapping):
        raise ValueError(f"spatial_flow_state_dict must be a mapping: {checkpoint_file}")
    current_flow_state = model.spatial_flow.state_dict()
    provided_keys = {str(key) for key in flow_state.keys()}
    expected_keys = {str(key) for key in current_flow_state.keys()}
    missing = sorted(expected_keys - provided_keys)
    unexpected = sorted(provided_keys - expected_keys)
    allowed_missing_prefixes = (
        "object_center_head.",
        "query_frame_coord_head.",
        "bev_coord_head.",
    )
    disallowed_missing = [
        key for key in missing if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected or disallowed_missing:
        raise KeyError(
            "spatial_flow checkpoint key mismatch during generation eval: "
            f"missing={disallowed_missing[:20]} unexpected={unexpected[:20]}"
        )
    if missing:
        print(
            json.dumps(
                {
                    "event": "generation_checkpoint_compat",
                    "message": "old checkpoint lacks eval-unused auxiliary heads; loading spatial_flow with explicit allow-list",
                    "allowed_missing_count": len(missing),
                    "allowed_missing_first_keys": missing[:20],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    model.spatial_flow.load_state_dict(flow_state, strict=False)
    load_lora_state_dict(model.qwen, payload["qwen_lora_state_dict"])
    return dict(payload)


@torch.no_grad()
def generate_one(
    *,
    model: NativeQwenGlowPatchSpatialModel,
    batch: Mapping[str, Any],
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, list[int]]:
    if int(batch["input_ids"].shape[0]) != 1:
        raise ValueError("generate_one requires batch_size=1 so right-padding cannot corrupt autoregressive positions")
    flow_out = model.spatial_flow(
        descriptors=batch["spatial_descriptors"].float(),
        token_sample_ids=batch["spatial_token_sample_ids"].long(),
        reference_type_ids=batch["reference_type_ids"].long(),
        query_pose_vec12=batch["query_pose_vec12"].float(),
        batch_size=1,
    )
    spatial_bias = model.spatial_bias_for_qwen(flow_out["spatial_bias"])
    prompt_len = int(batch["attention_mask"][0].sum().item())
    input_ids = batch["input_ids"][:, :prompt_len]
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = batch["mm_token_type_ids"][:, :prompt_len]
    stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        stop_ids.add(int(im_end_id))

    # 评测必须是真生成，但不能每生成一个 token 都重新跑一遍视觉塔。
    # 第一步把 32 帧视觉 token + spatial bias 放进 Qwen，并打开 KV cache；
    # 后续 token 只走文本 decoder cache，这和原生自回归生成是一致的。
    base = model.base_qwen()
    qwen_model = base.model
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    if batch.get("cached_image_embeds") is None:
        image_outputs = qwen_model.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True)
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        deepstack_source = list(image_outputs.deepstack_features)
    else:
        if batch.get("cached_deepstack_features") is None:
            raise ValueError("cached_deepstack_features is required when cached_image_embeds is provided")
        image_embeds = batch["cached_image_embeds"].to(inputs_embeds.device, inputs_embeds.dtype)
        deepstack_source = list(batch["cached_deepstack_features"])
    if image_embeds.shape != spatial_bias.shape:
        raise ValueError(
            f"spatial bias must match Qwen image embeds exactly during generation, "
            f"bias={tuple(spatial_bias.shape)} image_embeds={tuple(image_embeds.shape)}"
        )
    image_embeds = image_embeds + spatial_bias.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
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
            + spatial_bias.to(inputs_embeds.device, inputs_embeds.dtype)
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
    language_outputs = qwen_model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=image_mask_1d,
        deepstack_visual_embeds=deepstack_visual_embeds,
        use_cache=True,
    )
    past_key_values = language_outputs.past_key_values
    next_logits = base.lm_head(language_outputs[0])[:, -1, :].float()

    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        if temperature <= 0.0:
            next_token = next_logits.argmax(dim=-1)
        else:
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
                next_token = sorted_idx.gather(1, picked.view(1, 1)).squeeze(1)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        token_id = int(next_token.item())
        if token_id in stop_ids:
            break
        generated.append(token_id)
        next_input_ids = next_token.view(1, 1)
        next_inputs_embeds = qwen_model.get_input_embeddings()(next_input_ids)
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
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            use_cache=True,
        )
        past_key_values = language_outputs.past_key_values
        next_logits = base.lm_head(language_outputs[0])[:, -1, :].float()
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, generated


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise ValueError("real generation eval currently requires --batch-size 1 for exact M-RoPE/right-padding handling")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")
    set_seed(int(args.seed))
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_native_jsonl(args.eval_jsonl)
    rows, data_audit = filter_native_rows(
        rows=raw_rows,
        max_rows=int(args.max_rows),
        require_patch_geometry=bool(args.require_patch_geometry),
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
    )
    row_by_sample_id = {str(row["sample_id"]): row for row in rows}
    visual_cache_audit = None
    if args.visual_cache_root is not None:
        # 如果指定了视觉 cache，评测就必须全程使用 cache。
        # 先做覆盖率审计，避免跑到一半才发现某条样本缺 cache。
        visual_cache_audit = audit_visual_cache_rows(
            rows=rows,
            visual_cache_root=args.visual_cache_root,
            scannet_frame_root=args.scannet_frame_root,
            max_visual_frames=int(args.max_visual_frames),
            qwen_model_path=args.qwen_model_path,
            max_pixels=int(args.max_pixels),
            split_name="eval",
        )
    processor, qwen, hidden_size, spatial_merge_size, qwen_meta = load_processor_and_qwen(args)
    collator = NativeGlowPatchSpatialCollator(
        processor=processor,
        max_pixels=int(args.max_pixels),
        spatial_merge_size=spatial_merge_size,
        bev_grid_size=int(args.bev_grid_size),
        reference_schema=str(args.reference_schema),
    )
    dataset = NativeAnswerPatchSpatialDataset(
        rows=rows,
        max_visual_frames=int(args.max_visual_frames),
        scannet_frame_root=args.scannet_frame_root,
        depth_grid_hw=(int(args.depth_grid_h), int(args.depth_grid_w)),
        reference_schema=str(args.reference_schema),
        visual_cache_root=args.visual_cache_root,
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
    )
    aux_weights = {
        "reference_type": float(args.reference_type_loss_weight),
        "query_pose": float(args.query_pose_loss_weight),
        "bev": float(args.bev_loss_weight),
        "patch_consistency": float(args.patch_consistency_loss_weight),
        "target_camera_coord": float(args.target_camera_coord_loss_weight),
        "semantic_query_coord": float(args.semantic_query_coord_loss_weight),
        "bev_coord": float(args.bev_coord_loss_weight),
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
    )
    checkpoint_payload = load_patch_spatial_generation_checkpoint(args.checkpoint_path, model, map_location="cpu")
    checkpoint_args = checkpoint_payload.get("args") if isinstance(checkpoint_payload.get("args"), Mapping) else {}
    if bool(checkpoint_args.get("disable_spatial_flow_bias")) and not bool(args.disable_spatial_flow_bias):
        # NoFlow checkpoint 如果评测时忘了传开关，不能静默恢复 flow bias；
        # 否则训练和评测就不是同一个实验口径了。
        model.disable_spatial_flow_bias = True
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    model.to(device)
    model.eval()

    run_config = {
        "format": FORMAT_VERSION,
        "args": normalize_args_for_json(args),
        "eval_rows": len(rows),
        "data_audit": data_audit,
        "visual_cache_audit": visual_cache_audit,
        "qwen_meta": qwen_meta,
        "checkpoint": {
            "path": str(args.checkpoint_path),
            "step": checkpoint_payload.get("step"),
            "epoch": checkpoint_payload.get("epoch"),
        },
        "source_counts": dict(Counter(str(row["source"]["tag"]) for row in rows)),
        "question_type_counts": dict(Counter(str(row["question"]["canonical_question_type"]) for row in rows)),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps({"event": "run_config", **run_config}, ensure_ascii=False), flush=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(args.num_workers), collate_fn=collator.collate_generation_prompts)
    records: list[Dict[str, Any]] = []
    pred_path = args.output_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for index, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            prediction, generated_ids = generate_one(
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
            metric_payload = score_native_answer(prediction=prediction, row=row)
            record = {
                "sample_id": sample_id,
                "bench": bench,
                "source_tag": source_tag,
                "canonical_question_type": qtype,
                "vsti_task_group": vsti_task_group(qtype) if bench == "VSTI" else None,
                "answer_type": str(row["answer"]["type"]),
                "target_text": str(row["answer"].get("text") or ""),
                "prediction_text": prediction,
                "generated_token_count": len(generated_ids),
                **metric_payload,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if int(args.log_every) > 0 and (index % int(args.log_every) == 0 or index == 1):
                partial = summarize_records(records)
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "seen": index,
                            "elapsed_sec": round(time.time() - started_at, 3),
                            "vsi_score": partial.get("vsi_overall", {}).get("mean_score"),
                            "vsti_score": partial.get("vsti_overall", {}).get("mean_score"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = summarize_records(records)
    summary["official_like_vsi"] = summarize_official_like_vsi_from_prediction_rows(records, row_by_sample_id)
    summary["vsti_task_group_macro"] = summarize_vsti_task_groups([record for record in records if record["bench"] == "VSTI"])
    summary_payload = {
        "status": "completed",
        "format": FORMAT_VERSION,
        "elapsed_sec": round(time.time() - started_at, 3),
        "eval_rows": len(records),
        "summary": summary,
    }
    write_json(args.output_dir / "eval_summary.json", summary)
    write_json(args.output_dir / "summary.json", summary_payload)
    print(json.dumps({"event": "completed", **summary_payload}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
