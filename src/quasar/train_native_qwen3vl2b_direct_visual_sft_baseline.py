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
"""Native-answer direct Qwen3VL-2B visual LoRA SFT baseline.

这个脚本是当前 native-answer 实验口径下的“原生 QwenVL 直接 SFT”对照：
- 输入：同一批 32 帧 Qwen visual tokens + question text；
- 不注入 patch 3D descriptor；
- 不使用 QUASAR latent flow；
- 不改 Qwen 的语言模型自回归过程；
- 只训练 Qwen3VL-2B 的 LoRA，让它直接生成原始 answer.text。

这是为了回答一个最朴素的问题——如果完全不用我们的空间注入和 latent flow，
只让 QwenVL 看图、看题、按原始答案文本做 SFT，它在同一个数据 setting 下能到哪里。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


from quasar.train_flow_as_picker_vsi import (  # noqa: E402
    append_jsonl,
    normalize_args_for_json,
    set_seed,
    write_json,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial import (  # noqa: E402
    ANSWER_TYPES,
    CACHE_FORMAT_VERSION,
    LETTERS,
    as_str_list,
    audit_visual_cache_rows,
    build_target_text,
    cache_identity_from_row,
    compute_answer_span_ce,
    filter_native_rows,
    find_visual_cache_path,
    lora_state_dict,
    move_batch_to_device,
    normalize_visual_cache_roots,
    read_native_jsonl,
    require_mapping,
    require_str,
    scene_dir_has_min_frames,
    trainable_parameters,
    visual_cache_path_for_key,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial_decoder_aux import (  # noqa: E402
    load_processor_and_qwen,
)


FORMAT_VERSION = "native_qwen3vl2b_direct_visual_sft_v1"


@dataclass(frozen=True)
class NativeDirectVisualSample:
    sample_id: str
    source_tag: str
    scene_id: str
    video_id: str
    canonical_question_type: str
    answer_type: str
    prompt: str
    target_text: str
    image_paths: tuple[str, ...]
    visual_cache_key: str
    cached_image_grid_thw: torch.Tensor
    cached_image_embeds: torch.Tensor
    cached_deepstack_features: tuple[torch.Tensor, ...]


def build_direct_native_prompt(*, question_payload: Mapping[str, Any], answer: Mapping[str, Any], answer_type: str) -> str:
    """构造 direct-Qwen baseline 的 native-answer prompt。

    这里刻意不提“injected 3D evidence”，因为这个 baseline 不看我们的 3D/flow 信息。
    它只看视频帧和题目，然后按原始 answer.text 学会自回归输出答案。
    """

    question = require_str(question_payload, "text")
    qtype = str(question_payload.get("canonical_question_type") or "unknown")
    reference = str(question_payload.get("reference_condition") or "unknown")
    situation = str(question_payload.get("situation") or "").strip()
    lines = [
        "You are a spatial reasoning assistant.",
        "Use the video frames to answer the question.",
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


class NativeDirectVisualCachedDataset(Dataset):
    """只读取 Qwen 视觉 cache 的 direct baseline 数据集。

    这个 baseline 不需要 3D 点云和 pose，但必须严格复用同一批视觉 cache。
    这样比较的差异才是“有没有空间注入/latent flow”，而不是“有没有重复跑视觉塔”。
    """

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        visual_cache_root: Path,
        scannet_frame_root: Optional[Path],
        max_visual_frames: int,
        qwen_model_path: Path,
        max_pixels: int,
        visual_cache_roots: Optional[Sequence[Path]] = None,
        inference_only: bool = False,
    ) -> None:
        if not rows:
            raise ValueError("NativeDirectVisualCachedDataset rows must not be empty")
        if max_visual_frames <= 0:
            raise ValueError(f"max_visual_frames must be positive, got {max_visual_frames}")
        self.rows = list(rows)
        self.visual_cache_roots = normalize_visual_cache_roots(visual_cache_root, visual_cache_roots)
        self.visual_cache_root = self.visual_cache_roots[0] if self.visual_cache_roots else None
        self.scannet_frame_root = Path(scannet_frame_root) if scannet_frame_root is not None else None
        self.max_visual_frames = int(max_visual_frames)
        self.qwen_model_path = Path(qwen_model_path)
        self.max_pixels = int(max_pixels)
        self.inference_only = bool(inference_only)
        if not self.visual_cache_roots:
            raise ValueError("NativeDirectVisualCachedDataset requires at least one visual cache root")
        for root in self.visual_cache_roots:
            if not root.is_dir():
                raise FileNotFoundError(f"visual cache root does not exist: {root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> NativeDirectVisualSample:
        row = self.rows[index]
        sample_id = require_str(row, "sample_id")
        source = require_mapping(row, "source")
        scene = require_mapping(row, "scene")
        question_payload = require_mapping(row, "question")
        answer = require_mapping(row, "answer")
        if self.inference_only and set(answer) - {"type", "options", "numeric_unit"}:
            raise ValueError(f"sample_id={sample_id} inference input contains answer supervision")

        answer_type = str(answer.get("type") or "").strip()
        if answer_type not in ANSWER_TYPES:
            raise ValueError(f"sample_id={sample_id} unsupported answer type={answer_type!r}")
        if bool(answer.get("legacy_mca")):
            raise ValueError(f"sample_id={sample_id} legacy_mca reached direct native baseline")

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
        # 训练集 cache 和 dev/val cache 可能分目录保存，这里只在显式给定的 root 中查找。
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
        if int(cache_payload.get("max_pixels")) != self.max_pixels:
            raise ValueError(f"sample_id={sample_id} visual cache max_pixels mismatch: {cache_path}")
        if int(cache_payload.get("max_visual_frames")) != self.max_visual_frames:
            raise ValueError(f"sample_id={sample_id} visual cache max_visual_frames mismatch: {cache_path}")

        qwen_cache = cache_payload.get("qwen")
        spatial_cache = cache_payload.get("spatial")
        if not isinstance(qwen_cache, Mapping):
            raise ValueError(f"sample_id={sample_id} visual cache missing qwen payload: {cache_path}")
        if not isinstance(spatial_cache, Mapping):
            raise ValueError(f"sample_id={sample_id} visual cache missing spatial metadata: {cache_path}")
        cached_image_grid_thw = qwen_cache["image_grid_thw"].to(torch.long)
        cached_image_embeds = qwen_cache["image_embeds"].to(torch.bfloat16)
        cached_deepstack_features = tuple(tensor.to(torch.bfloat16) for tensor in qwen_cache["deepstack_features"])
        image_paths = as_str_list(spatial_cache.get("image_paths"))
        if len(image_paths) != self.max_visual_frames:
            raise ValueError(
                f"sample_id={sample_id} cached spatial image_paths must have {self.max_visual_frames} frames, "
                f"got {len(image_paths)}"
            )
        if cached_image_grid_thw.ndim != 2 or cached_image_grid_thw.shape != (self.max_visual_frames, 3):
            raise ValueError(
                f"sample_id={sample_id} cached image_grid_thw must be [{self.max_visual_frames},3], "
                f"got {tuple(cached_image_grid_thw.shape)}"
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

        # 回答时只读题目和选项；训练仍沿原答案构造，不用空答案伪装训练输入。
        target_text = "" if self.inference_only else build_target_text(answer=answer, answer_type=answer_type, sample_id=sample_id)
        prompt = build_direct_native_prompt(question_payload=question_payload, answer=answer, answer_type=answer_type)
        return NativeDirectVisualSample(
            sample_id=sample_id,
            source_tag=str(source.get("tag") or "unknown"),
            scene_id=str(scene.get("scene_id") or scene.get("scene_name") or ""),
            video_id=str(scene.get("video_id") or scene.get("scene_id") or ""),
            canonical_question_type=str(question_payload.get("canonical_question_type") or "unknown"),
            answer_type=answer_type,
            prompt=prompt,
            target_text=target_text,
            image_paths=tuple(image_paths),
            visual_cache_key=visual_cache_key,
            cached_image_grid_thw=cached_image_grid_thw,
            cached_image_embeds=cached_image_embeds,
            cached_deepstack_features=cached_deepstack_features,
        )


class NativeDirectVisualCachedCollator:
    """把 visual cache + native prompt 拼成 Qwen 自回归 SFT batch。"""

    def __init__(self, *, processor: Any, max_pixels: int, spatial_merge_size: int) -> None:
        if max_pixels <= 0:
            raise ValueError(f"max_pixels must be positive, got {max_pixels}")
        if spatial_merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_pixels = int(max_pixels)
        self.spatial_merge_size = int(spatial_merge_size)

    def __call__(self, samples: Sequence[NativeDirectVisualSample]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("direct visual collator samples must not be empty")
        encoded = self._encode_prompt_batch_from_visual_cache(samples=samples)
        batch = self._append_answer_span(encoded=encoded, samples=samples)
        self._append_cached_visual_payload(batch=batch, samples=samples)
        batch["sample_ids"] = [sample.sample_id for sample in samples]
        batch["source_tags"] = [sample.source_tag for sample in samples]
        batch["answer_types"] = [sample.answer_type for sample in samples]
        batch["canonical_question_types"] = [sample.canonical_question_type for sample in samples]
        batch["target_texts"] = [sample.target_text for sample in samples]
        batch["prompt_texts"] = [sample.prompt for sample in samples]
        batch["visual_cache_keys"] = [sample.visual_cache_key for sample in samples]
        return batch

    def _encode_prompt_batch_from_visual_cache(self, *, samples: Sequence[NativeDirectVisualSample]) -> Dict[str, Any]:
        """按 cache 的 image_grid_thw 手工展开 Qwen image token。

        我们不重新读图，也不重新跑 image processor。cache 已经记录每帧会产生多少视觉 token，
        所以这里只做 Qwen chat template 和占位符展开，保证语言模型看到的 token 排布和原生 Qwen 一致。
        """

        texts: list[str] = []
        image_grid_chunks: list[torch.Tensor] = []
        for sample in samples:
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

    def _append_answer_span(
        self,
        *,
        encoded: Mapping[str, torch.Tensor],
        samples: Sequence[NativeDirectVisualSample],
    ) -> Dict[str, Any]:
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

    @staticmethod
    def _append_cached_visual_payload(*, batch: Dict[str, Any], samples: Sequence[NativeDirectVisualSample]) -> None:
        batch["cached_image_embeds"] = torch.cat([sample.cached_image_embeds for sample in samples], dim=0)
        deep_layers: list[torch.Tensor] = []
        layer_count = len(samples[0].cached_deepstack_features)
        for sample in samples:
            if len(sample.cached_deepstack_features) != layer_count:
                raise ValueError(f"sample_id={sample.sample_id} cached deepstack layer count mismatch")
        for layer_i in range(layer_count):
            deep_layers.append(torch.cat([sample.cached_deepstack_features[layer_i] for sample in samples], dim=0))
        batch["cached_deepstack_features"] = deep_layers


class NativeQwenDirectVisualModel(nn.Module):
    """只含 QwenVL + LoRA 的 direct visual baseline。

    这里复刻 Qwen3VL forward 中“把 image_embeds 塞回 text embedding”的必要步骤，
    但不加任何 spatial bias。这样可以复用视觉 cache，同时不破坏 Qwen 原生 language_model。
    """

    def __init__(self, *, qwen: nn.Module) -> None:
        super().__init__()
        self.qwen = qwen
        self.last_forward_audit: Dict[str, Any] = {}

    def base_qwen(self) -> nn.Module:
        if hasattr(self.qwen, "get_base_model"):
            return self.qwen.get_base_model()
        return self.qwen

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        qwen_out = self._qwen_forward_from_cached_visual(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image_grid_thw=batch["image_grid_thw"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            cached_image_embeds=batch["cached_image_embeds"],
            cached_deepstack_features=batch["cached_deepstack_features"],
        )
        logits = qwen_out["logits"]
        labels = batch["labels"]
        loss = compute_answer_span_ce(logits=logits, labels=labels)
        self.last_forward_audit = {
            "image_feature_tokens": int(qwen_out["image_feature_tokens"].detach().cpu().item()),
            "visual_cache_used": bool(int(qwen_out["visual_cache_used"].detach().cpu().item())),
            "answer_label_tokens": int((labels != -100).sum().detach().cpu().item()),
        }
        return {"loss": loss, "answer_ce_loss": loss.detach(), "logits": logits}

    def _qwen_forward_from_cached_visual(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        cached_image_embeds: torch.Tensor,
        cached_deepstack_features: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        base = self.base_qwen()
        qwen_model = base.model
        inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
        image_embeds = cached_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        if not cached_deepstack_features:
            raise ValueError("cached_deepstack_features must not be empty")
        image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_mask_1d = image_mask[..., 0]
        deepstack_visual_embeds = []
        for layer_i, deep in enumerate(cached_deepstack_features):
            if tuple(deep.shape) != tuple(cached_image_embeds.shape):
                raise ValueError(
                    f"deepstack layer {layer_i} shape must match cached image embeds, "
                    f"got deep={tuple(deep.shape)} embeds={tuple(cached_image_embeds.shape)}"
                )
            deepstack_visual_embeds.append(deep.to(inputs_embeds.device, inputs_embeds.dtype))
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
            "visual_cache_used": torch.tensor(1, device=logits.device),
        }


def gradient_audit(model: NativeQwenDirectVisualModel) -> Dict[str, Any]:
    grad_norm = 0.0
    nonzero = 0
    trainable_total = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_total += int(parameter.numel())
        if parameter.grad is None:
            continue
        if name.startswith("qwen.") and "lora_" in name:
            value = float(parameter.grad.detach().float().norm().cpu().item())
            if value > 0.0:
                grad_norm += value
                nonzero += 1
    missing = [] if grad_norm > 0.0 else ["qwen_lora"]
    return {
        "qwen_lora_grad_norm": grad_norm,
        "qwen_lora_nonzero_param_count": nonzero,
        "trainable_parameters": trainable_total,
        "missing_required_grad_groups": missing,
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
def evaluate_loss(model: NativeQwenDirectVisualModel, loader: DataLoader, *, device: torch.device, max_batches: int) -> Dict[str, float]:
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
        count += 1
    if count == 0:
        raise ValueError("evaluate_loss saw zero batches")
    return {key: value / count for key, value in sums.items()}


def save_checkpoint(
    *,
    path: Path,
    model: NativeQwenDirectVisualModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    save_optimizer_state: bool,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "epoch": int(epoch),
        "qwen_lora_state_dict": lora_state_dict(model.qwen),
        "args": normalize_args_for_json(args),
    }
    if save_optimizer_state:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path / "direct_qwen_sft.pt")
    write_json(
        path / "qwen_lora_config.json",
        {
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_target_modules": str(args.lora_target_modules),
            "note": "Project-local LoRA Linear state is stored in direct_qwen_sft.pt::qwen_lora_state_dict.",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train native-answer direct Qwen3VL visual LoRA SFT baseline.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--qwen-cache-dir", type=Path, default=None)
    parser.add_argument("--attn-implementation", choices=("default", "sdpa", "flash_attention_2", "eager"), default="default")
    parser.add_argument("--visual-cache-root", type=Path, required=True)
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
    parser.add_argument("--max-visual-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-optimizer-state", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "args.json", normalize_args_for_json(args))

    train_rows_raw = read_native_jsonl(args.train_jsonl)
    val_rows_raw = read_native_jsonl(args.val_jsonl)
    train_rows, train_audit = filter_native_rows(
        rows=train_rows_raw,
        max_rows=int(args.max_train_rows),
        require_patch_geometry=False,
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
    )
    val_rows, val_audit = filter_native_rows(
        rows=val_rows_raw,
        max_rows=int(args.max_val_rows),
        require_patch_geometry=False,
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
    )
    write_json(args.output_dir / "data_filter_audit.json", {"train": train_audit, "val": val_audit})
    visual_cache_roots = normalize_visual_cache_roots(args.visual_cache_root, args.extra_visual_cache_root)
    primary_visual_cache_root = visual_cache_roots[0]
    extra_visual_cache_roots = visual_cache_roots[1:]
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

    setattr(args, "spatial_query_count", 1)
    # 共享加载器登记词表信息，纯视觉训练仍只使用原视觉和文本输入。
    setattr(args, "spatial_query_tokens_enabled", False)
    setattr(args, "numeric_scaffold_mode", "none")
    processor, qwen, _hidden_size, spatial_merge_size, qwen_meta = load_processor_and_qwen(args)
    write_json(args.output_dir / "qwen_meta.json", qwen_meta)
    collator = NativeDirectVisualCachedCollator(
        processor=processor,
        max_pixels=int(args.max_pixels),
        spatial_merge_size=spatial_merge_size,
    )
    train_dataset = NativeDirectVisualCachedDataset(
        rows=train_rows,
        visual_cache_root=primary_visual_cache_root,
        visual_cache_roots=extra_visual_cache_roots,
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
    )
    val_dataset = NativeDirectVisualCachedDataset(
        rows=val_rows,
        visual_cache_root=primary_visual_cache_root,
        visual_cache_roots=extra_visual_cache_roots,
        scannet_frame_root=args.scannet_frame_root,
        max_visual_frames=int(args.max_visual_frames),
        qwen_model_path=args.qwen_model_path,
        max_pixels=int(args.max_pixels),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
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
    model = NativeQwenDirectVisualModel(qwen=qwen).to(device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_log_path = args.output_dir / "train_log.jsonl"
    first_batch_audit_path = args.output_dir / "first_batch_audit.json"
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        for batch_i, batch in enumerate(train_loader, start=1):
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
                audit["passed"] = bool(audit["passed"]) and bool(audit["visual_cache_used"]) and audit["answer_label_tokens"] > 0
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
                        args=args,
                        save_optimizer_state=bool(args.save_optimizer_state),
                    )
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
        args=args,
        save_optimizer_state=bool(args.save_optimizer_state),
    )
    write_json(
        args.output_dir / "run_summary.json",
        {
            "format_version": FORMAT_VERSION,
            "global_step": global_step,
            "elapsed_sec": time.time() - start_time,
            "final_val_metrics": final_metrics,
            "first_batch_audit_path": str(first_batch_audit_path),
        },
    )


if __name__ == "__main__":
    main()
