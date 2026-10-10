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
"""历史 DirectVisual 自回归路径，直接接收官方原始 Qwen 模型。"""
from typing import Mapping, Any
import torch

@torch.no_grad()
def generate_one(
    *,
    model: Any,
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
    base = model
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
