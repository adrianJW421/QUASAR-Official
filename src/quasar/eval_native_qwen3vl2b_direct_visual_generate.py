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
"""Real generate eval for the native-answer direct Qwen3VL visual SFT baseline.

这个脚本评测“只用原生 QwenVL + LoRA”的 direct SFT baseline。
它不注入 patch 3D descriptor，也不走 latent flow；但它必须复用训练时同一套
Qwen visual cache，然后让模型真正自回归生成答案，再按 native-answer schema 打分。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader



from quasar.eval_native_qwen3vl2b_glow_patch_spatial_generate import (  # noqa: E402
    bench_from_source,
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
from quasar.train_native_qwen3vl2b_direct_visual_sft_baseline import (  # noqa: E402
    FORMAT_VERSION as DIRECT_TRAIN_FORMAT_VERSION,
    NativeDirectVisualCachedCollator,
    NativeDirectVisualCachedDataset,
    NativeQwenDirectVisualModel,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial import (  # noqa: E402
    audit_visual_cache_rows,
    filter_native_rows,
    load_lora_state_dict,
    move_batch_to_device,
    read_native_jsonl,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial_decoder_aux import (  # noqa: E402
    load_processor_and_qwen,
)


FORMAT_VERSION = "native_qwen3vl2b_direct_visual_generate_eval_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate native direct Qwen3VL visual SFT checkpoint by generation.")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--no-sft", action="store_true", help="Evaluate base Qwen3-VL without direct-visual LoRA/SFT weights.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--qwen-cache-dir", type=Path, default=None)
    parser.add_argument("--visual-cache-root", type=Path, required=True)
    parser.add_argument("--scannet-frame-root", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-visual-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", choices=("default", "sdpa", "flash_attention_2", "eager"), default="default")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--inference-only", action="store_true", help="Generate from answer-free input; score separately.")
    parser.add_argument("--resume-inference", action="store_true", help="Resume complete answer-free prediction lines in original order.")
    add_latency_args(parser)
    return parser


def resolve_direct_checkpoint_file(path: Path) -> Path:
    """把 checkpoint 目录或文件统一解析成 direct_qwen_sft.pt。

    direct baseline 的 checkpoint 文件名和 QUASAR patch-spatial 不同。
    这里不做模糊兜底；找不到标准文件就直接报错，避免误加载别的实验。
    """

    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "direct_qwen_sft.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve direct Qwen SFT checkpoint from: {path}")


def load_direct_checkpoint(path: Path, model: NativeQwenDirectVisualModel, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """严格加载 direct Qwen visual SFT checkpoint。

    这个 baseline 只有 Qwen LoRA，没有 spatial_flow。LoRA 没加载成功，
    评测就会退化成未微调 Qwen，所以这里必须逐 key 检查。
    """

    checkpoint_file = resolve_direct_checkpoint_file(path)
    # 历史训练原件带有随机数状态；仅加载使用者明确提供且信任的权重。
    payload = torch.load(checkpoint_file, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {checkpoint_file}")
    allowed_format_versions = {DIRECT_TRAIN_FORMAT_VERSION, "native_qwen25vl_direct_visual_sft_v1"}
    if str(payload.get("format_version")) not in allowed_format_versions:
        raise ValueError(
            f"direct checkpoint format mismatch: {payload.get('format_version')!r} "
            f"vs {sorted(allowed_format_versions)!r}"
        )
    if "qwen_lora_state_dict" not in payload:
        raise KeyError(f"direct checkpoint missing qwen_lora_state_dict: {checkpoint_file}")
    load_lora_state_dict(model.qwen, payload["qwen_lora_state_dict"])
    return dict(payload)


def collate_direct_generation_prompts(
    collator: NativeDirectVisualCachedCollator,
    samples: Sequence[Any],
) -> Dict[str, Any]:
    """只打包 prompt，不拼接 gold answer。

    训练 collator 会把标准答案接到输入后面算 loss；真实评测不能读取答案。
    所以这里只复用它的 prompt/cache 打包逻辑，并补上后续汇总需要的元信息。
    """

    if not samples:
        raise ValueError("direct generation collator samples must not be empty")
    batch = collator._encode_prompt_batch_from_visual_cache(samples=samples)
    collator._append_cached_visual_payload(batch=batch, samples=samples)
    batch["sample_ids"] = [sample.sample_id for sample in samples]
    batch["source_tags"] = [sample.source_tag for sample in samples]
    batch["answer_types"] = [sample.answer_type for sample in samples]
    batch["target_texts"] = [sample.target_text for sample in samples]
    batch["prompt_texts"] = [sample.prompt for sample in samples]
    batch["visual_cache_keys"] = [sample.visual_cache_key for sample in samples]
    return batch


@torch.no_grad()
def generate_one(
    *,
    model: NativeQwenDirectVisualModel,
    batch: Mapping[str, Any],
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, list[int]]:
    if int(batch["input_ids"].shape[0]) != 1:
        raise ValueError("generate_one requires batch_size=1 so right-padding cannot corrupt autoregressive positions")
    if batch.get("cached_deepstack_features") is None:
        raise ValueError("direct generation requires cached_deepstack_features")

    prompt_len = int(batch["attention_mask"][0].sum().item())
    input_ids = batch["input_ids"][:, :prompt_len]
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = batch["mm_token_type_ids"][:, :prompt_len]
    stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        stop_ids.add(int(im_end_id))

    # 第一步把缓存里的视觉 token 塞回 Qwen 原生 embedding 位置；
    # 后续生成 token 只走 Qwen language_model 的 KV cache，不重新处理视觉信息。
    base = model.base_qwen()
    qwen_model = base.model
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    image_embeds = batch["cached_image_embeds"].to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    image_mask_1d = image_mask[..., 0]
    deepstack_visual_embeds = []
    for layer_i, deep in enumerate(batch["cached_deepstack_features"]):
        if tuple(deep.shape) != tuple(batch["cached_image_embeds"].shape):
            raise ValueError(
                f"deepstack layer {layer_i} shape must match cached image embeds during generation, "
                f"deep={tuple(deep.shape)} embeds={tuple(batch['cached_image_embeds'].shape)}"
            )
        deepstack_visual_embeds.append(deep.to(inputs_embeds.device, inputs_embeds.dtype))

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
    if bool(args.no_sft) and args.checkpoint_path is not None:
        raise ValueError("--no-sft cannot be combined with --checkpoint-path")
    if not bool(args.no_sft) and args.checkpoint_path is None:
        raise ValueError("--checkpoint-path is required unless --no-sft is set")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")
    validate_latency_args(args)
    set_seed(int(args.seed))
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_native_jsonl(args.eval_jsonl)
    rows, data_audit = filter_native_rows(
        rows=raw_rows,
        max_rows=int(args.max_rows),
        require_patch_geometry=False,
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
    )
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
    visual_cache_audit = audit_visual_cache_rows(
        rows=rows,
        visual_cache_root=args.visual_cache_root,
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
        split_name="eval",
    )

    args.no_lora = bool(args.no_sft)
    setattr(args, "spatial_query_count", 1)
    # 纯视觉回答不插入空间提问token；与共享加载器明确约定这一点。
    setattr(args, "spatial_query_tokens_enabled", False)
    setattr(args, "numeric_scaffold_mode", "none")
    processor, qwen, _hidden_size, spatial_merge_size, qwen_meta = load_processor_and_qwen(args)
    collator = NativeDirectVisualCachedCollator(
        processor=processor,
        max_pixels=int(args.max_pixels),
        spatial_merge_size=spatial_merge_size,
    )
    dataset = NativeDirectVisualCachedDataset(
        rows=rows,
        visual_cache_root=args.visual_cache_root,
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
        inference_only=bool(args.inference_only),
    )
    model = NativeQwenDirectVisualModel(qwen=qwen)
    checkpoint_payload: Dict[str, Any] = {"mode": "base_qwen_no_sft"}
    if not bool(args.no_sft):
        assert args.checkpoint_path is not None
        checkpoint_payload = load_direct_checkpoint(args.checkpoint_path, model, map_location="cpu")
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
            "mode": "base_qwen_no_sft" if bool(args.no_sft) else "direct_visual_sft",
            "path": None if args.checkpoint_path is None else str(args.checkpoint_path),
            "step": checkpoint_payload.get("step"),
            "epoch": checkpoint_payload.get("epoch"),
        },
        "source_counts": dict(Counter(str(row["source"]["tag"]) for row in rows)),
        "question_type_counts": dict(Counter(str(row["question"]["canonical_question_type"]) for row in rows)),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps({"event": "run_config", **run_config}, ensure_ascii=False), flush=True)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=lambda samples: collate_direct_generation_prompts(collator, samples),
    )
    records: list[Dict[str, Any]] = list(existing_records)
    latency_records: list[Dict[str, Any]] = []
    pred_path = args.output_dir / "predictions.jsonl"
    with pred_path.open("a" if args.resume_inference else "w", encoding="utf-8", buffering=1) as handle:
        for index, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            is_timed_latency_row = latency_enabled(args) and index > int(args.latency_warmup_rows)
            if latency_enabled(args) and index == int(args.latency_warmup_rows) + 1:
                # warmup 结束后再清空 peak memory，这样表里的显存峰值只对应正式计时阶段。
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
                prediction, generated_ids = generate_result
            else:
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
            metric_payload = {} if args.inference_only else score_native_answer(prediction=prediction, row=row)
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
