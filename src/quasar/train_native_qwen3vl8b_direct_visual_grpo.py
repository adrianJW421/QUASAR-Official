#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from quasar.eval_native_qwen3vl2b_direct_visual_generate import (  # noqa: E402
    collate_direct_generation_prompts,
    generate_one,
    load_direct_checkpoint,
)
from quasar.eval_native_qwen3vl2b_glow_patch_spatial_generate import parse_first_float, score_native_answer  # noqa: E402
from quasar.score_vsibench_official_like_predictions import (  # noqa: E402
    MCA_QUESTION_TYPES,
    NA_QUESTION_TYPES,
    fuzzy_matching,
    mean_relative_accuracy,
    to_float,
)
from quasar.train_flow_as_picker_vsi import append_jsonl, normalize_args_for_json, set_seed, write_json  # noqa: E402
from quasar.train_native_qwen3vl2b_direct_visual_sft_baseline import (  # noqa: E402
    NativeDirectVisualCachedCollator,
    NativeDirectVisualCachedDataset,
    NativeQwenDirectVisualModel,
    save_checkpoint,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial_decoder_aux import temporarily_disable_lora  # noqa: E402
from quasar.train_native_qwen3vl2b_glow_patch_spatial import (  # noqa: E402
    audit_visual_cache_rows,
    filter_native_rows,
    load_processor_and_qwen,
    move_batch_to_device,
    normalize_visual_cache_roots,
    read_native_jsonl,
    trainable_parameters,
)

FORMAT_VERSION = 'native_qwen3vl8b_direct_visual_grpo_v1'
NUMERIC_QTYPES = {'vsti_camera_displacement', 'vsti_camera_obj_abs_dist', 'object_abs_distance'}
VSTI_NUMERIC_QTYPES = {'vsti_camera_displacement', 'vsti_camera_obj_abs_dist'}
VSTI_MCA_QTYPES = {'vsti_camera_movement_direction'}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='GRPO for DirectVisual Qwen3-VL baseline.')
    parser.add_argument('--train-jsonl', type=Path, required=True)
    parser.add_argument('--val-jsonl', type=Path, required=True)
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--qwen-model-path', type=Path, required=True)
    parser.add_argument('--qwen-cache-dir', type=Path, default=None)
    parser.add_argument('--visual-cache-root', type=Path, required=True)
    parser.add_argument('--extra-visual-cache-root', type=Path, action='append', default=[])
    parser.add_argument('--scannet-frame-root', type=Path, default=None)
    parser.add_argument('--max-train-rows', type=int, default=0)
    parser.add_argument('--max-val-rows', type=int, default=0)
    parser.add_argument('--max-train-steps', type=int, default=0)
    parser.add_argument('--max-groups', type=int, default=0)
    parser.add_argument('--qtype-balanced-train', action='store_true')
    parser.add_argument('--grpo-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-visual-frames', type=int, default=32)
    parser.add_argument('--max-pixels', type=int, default=50176)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lora-target-modules', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--num-generations', type=int, default=4)
    parser.add_argument('--rl-scope', choices=('numeric', 'all'), default='all')
    parser.add_argument('--reward-mode', choices=('official_mra_numeric', 'official_mra_mixed', 'dense_numeric', 'dense_log_numeric'), default='official_mra_mixed')
    parser.add_argument('--dense-numeric-tau', type=float, default=0.5)
    parser.add_argument('--dense-numeric-min-scale', type=float, default=1.0)
    parser.add_argument('--parse-fail-penalty', type=float, default=0.0)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--kl-beta', type=float, default=0.02)
    parser.add_argument('--advantage-eps', type=float, default=1e-6)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--save-every-steps', type=int, default=0)
    parser.add_argument('--save-optimizer-state', action='store_true')
    return parser


def canonical_qtype(row: Mapping[str, Any]) -> str:
    question = row.get('question') if isinstance(row.get('question'), Mapping) else {}
    return str(question.get('canonical_question_type') or row.get('canonical_question_type') or 'unknown')


def is_vsti_official_qtype(qtype: str) -> bool:
    return qtype in VSTI_NUMERIC_QTYPES or qtype in VSTI_MCA_QTYPES or qtype.startswith('vsti_camera_obj_rel_dist') or qtype.startswith('vsti_obj_obj_relative_pos')


def is_numeric_rl_row(row: Mapping[str, Any]) -> bool:
    answer = row.get('answer') if isinstance(row.get('answer'), Mapping) else {}
    return str(answer.get('type')) == 'numeric' and canonical_qtype(row) in NUMERIC_QTYPES


def is_all_task_rl_row(row: Mapping[str, Any]) -> bool:
    answer = row.get('answer') if isinstance(row.get('answer'), Mapping) else {}
    answer_type = str(answer.get('type') or '')
    qtype = canonical_qtype(row)
    if answer_type not in {'numeric', 'count', 'multiple_choice'}:
        return False
    return qtype in MCA_QUESTION_TYPES or qtype in NA_QUESTION_TYPES or is_vsti_official_qtype(qtype)


def is_rl_row(row: Mapping[str, Any], args: argparse.Namespace) -> bool:
    if str(args.rl_scope) == 'numeric':
        return is_numeric_rl_row(row)
    return is_all_task_rl_row(row)


def select_qtype_balanced_rows(rows: Sequence[Mapping[str, Any]], *, max_groups: int, seed: int) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if max_groups <= 0:
        raise ValueError('qtype balanced training requires --max-groups > 0')
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(canonical_qtype(row), []).append(row)
    rng = random.Random(int(seed))
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    selected: list[Mapping[str, Any]] = []
    sorted_qtypes = sorted(groups)
    cursor = {qtype: 0 for qtype in sorted_qtypes}
    while len(selected) < max_groups:
        progressed = False
        for qtype in sorted_qtypes:
            index = cursor[qtype]
            if index >= len(groups[qtype]):
                continue
            selected.append(groups[qtype][index])
            cursor[qtype] = index + 1
            progressed = True
            if len(selected) >= max_groups:
                break
        if not progressed:
            break
    if not selected:
        raise ValueError('qtype balanced selection produced zero rows')
    rng.shuffle(selected)
    return selected, {
        'enabled': True,
        'requested_max_groups': int(max_groups),
        'selected_rows': len(selected),
        'source_qtype_counts': dict(Counter(canonical_qtype(row) for row in rows)),
        'selected_qtype_counts': dict(Counter(canonical_qtype(row) for row in selected)),
    }


def gold_number(row: Mapping[str, Any]) -> float:
    answer = row['answer']
    raw = answer.get('numeric_value')
    if raw is None:
        raw = parse_first_float(str(answer.get('text') or ''))
    if raw is None:
        raise ValueError(f"missing gold number for sample_id={row.get('sample_id')}")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"invalid gold number for sample_id={row.get('sample_id')}: {value}")
    return value


def compute_mra_numeric_score(prediction: str, gold: float, args: argparse.Namespace) -> dict[str, Any]:
    pred_value = to_float(fuzzy_matching(prediction))
    if pred_value is None:
        pred_value = parse_first_float(prediction)
    if pred_value is None or not math.isfinite(float(pred_value)):
        return {'score': float(args.parse_fail_penalty), 'parsed': pred_value, 'target_numeric': float(gold), 'abs_error': None, 'relative_error': None, 'log_error': None, 'official_mra_score': 0.0}
    pred = float(pred_value)
    abs_error = abs(pred - float(gold))
    scale = max(abs(float(gold)), float(args.dense_numeric_min_scale))
    relative_error = abs_error / scale
    log_error = abs(math.log1p(max(pred, 0.0)) - math.log1p(max(float(gold), 0.0)))
    if float(gold) == 0.0:
        official_mra_score = 1.0 if abs_error == 0.0 else 0.0
    else:
        official_mra_score = mean_relative_accuracy(pred, float(gold))
    if args.reward_mode in {'official_mra_numeric', 'official_mra_mixed'}:
        reward = official_mra_score
    elif args.reward_mode == 'dense_log_numeric':
        reward = math.exp(-log_error / float(args.dense_numeric_tau))
    else:
        reward = math.exp(-relative_error / float(args.dense_numeric_tau))
    return {'score': float(reward), 'parsed': pred_value, 'target_numeric': float(gold), 'abs_error': abs_error, 'relative_error': relative_error, 'log_error': log_error, 'official_mra_score': float(official_mra_score)}


def compute_mixed_reward(*, prediction: str, row: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    qtype = canonical_qtype(row)
    answer = row['answer']
    answer_type = str(answer['type'])
    if qtype in NA_QUESTION_TYPES or qtype in VSTI_NUMERIC_QTYPES or answer_type in {'numeric', 'count'}:
        metric = compute_mra_numeric_score(prediction, gold_number(row), args)
        native_payload = score_native_answer(prediction=prediction, row=row)
        return {'score': float(metric['score']), 'reward_mode': str(args.reward_mode), 'official_mra_score': float(metric['official_mra_score']), 'native_score': float(native_payload.get('score', 0.0)), 'native_metric': native_payload.get('metric'), 'metric': 'MRA:.5:.95:.05', 'parsed': metric['parsed'], 'target_numeric': metric['target_numeric'], 'abs_error': metric['abs_error'], 'relative_error': metric['relative_error'], 'log_error': metric['log_error']}
    if qtype in MCA_QUESTION_TYPES:
        first_token = fuzzy_matching(prediction)
        gold_letter = str(answer.get('option_letter') or answer.get('raw') or '').strip()
        if not gold_letter:
            raise ValueError(f"sample_id={row.get('sample_id')} MCA row missing answer.option_letter/raw")
        native_payload = score_native_answer(prediction=prediction, row=row)
        return {'score': 1.0 if first_token.lower() == gold_letter.lower() else 0.0, 'reward_mode': str(args.reward_mode), 'native_score': float(native_payload.get('score', 0.0)), 'native_metric': native_payload.get('metric'), 'metric': 'accuracy', 'parsed': first_token, 'target': gold_letter}
    if is_vsti_official_qtype(qtype) and answer_type == 'multiple_choice':
        native_payload = score_native_answer(prediction=prediction, row=row)
        return {'score': float(native_payload.get('score', 0.0)), 'reward_mode': str(args.reward_mode), 'native_score': float(native_payload.get('score', 0.0)), 'native_metric': native_payload.get('metric'), 'metric': native_payload.get('metric') or 'choice_accuracy', 'parsed': native_payload.get('parsed'), 'target': str(answer.get('option_letter') or answer.get('text') or '')}
    raise ValueError(f"unsupported all-task GRPO row sample_id={row.get('sample_id')} qtype={qtype!r} answer_type={answer_type!r}")


def compute_reward(*, prediction: str, row: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if str(args.rl_scope) == 'numeric':
        metric = compute_mra_numeric_score(prediction, gold_number(row), args)
        native_payload = score_native_answer(prediction=prediction, row=row)
        return {'score': float(metric['score']), 'reward_mode': str(args.reward_mode), 'official_mra_score': float(metric['official_mra_score']), 'native_score': float(native_payload.get('score', 0.0)), 'native_metric': native_payload.get('metric'), 'parsed': metric['parsed'], 'target_numeric': metric['target_numeric'], 'abs_error': metric['abs_error'], 'relative_error': metric['relative_error'], 'log_error': metric['log_error']}
    return compute_mixed_reward(prediction=prediction, row=row, args=args)


def token_logprobs_for_generated_tokens(*, model: NativeQwenDirectVisualModel, batch: Mapping[str, Any], generated_token_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if int(batch['input_ids'].shape[0]) != 1:
        raise ValueError('GRPO logprob currently requires batch_size=1')
    if not generated_token_ids:
        raise ValueError('cannot compute logprob for empty generated_token_ids')
    prompt_len = int(batch['attention_mask'][0].sum().item())
    prompt_ids = batch['input_ids'][:, :prompt_len]
    prompt_mm = batch['mm_token_type_ids'][:, :prompt_len]
    gen = torch.tensor([list(int(x) for x in generated_token_ids)], dtype=torch.long, device=prompt_ids.device)
    input_ids = torch.cat([prompt_ids, gen], dim=1)
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.cat([prompt_mm, torch.zeros((1, gen.shape[1]), dtype=prompt_mm.dtype, device=prompt_mm.device)], dim=1)
    qwen_out = model._qwen_forward_from_cached_visual(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=batch['image_grid_thw'],
        mm_token_type_ids=mm_token_type_ids,
        cached_image_embeds=batch['cached_image_embeds'],
        cached_deepstack_features=batch['cached_deepstack_features'],
    )
    logits = qwen_out['logits'][:, :-1, :].float()
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_len:] = gen
    shift_labels = labels[:, 1:]
    label_mask = shift_labels.ne(-100)
    if int(label_mask.sum().detach().cpu().item()) != len(generated_token_ids):
        raise RuntimeError('GRPO label mask does not match generated token count')
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    selected = token_log_probs[label_mask]
    return selected, selected.mean()


def build_model(args: argparse.Namespace, *, device: torch.device) -> tuple[Any, NativeQwenDirectVisualModel, Dict[str, Any], Dict[str, Any], int]:
    processor, qwen, _hidden_size, spatial_merge_size, qwen_meta = load_processor_and_qwen(args)
    model = NativeQwenDirectVisualModel(qwen=qwen)
    checkpoint_payload = load_direct_checkpoint(args.checkpoint_path, model, map_location='cpu')
    model.to(device)
    return processor, model, qwen_meta, checkpoint_payload, int(spatial_merge_size)


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise ValueError('GRPO currently requires --batch-size 1')
    if args.gradient_accumulation_steps <= 0:
        raise ValueError('gradient_accumulation_steps must be positive')
    if args.num_generations < 2:
        raise ValueError('GRPO requires --num-generations >= 2')
    if str(args.rl_scope) == 'all' and str(args.reward_mode) != 'official_mra_mixed':
        raise ValueError('all-task GRPO requires --reward-mode official_mra_mixed')
    if str(args.rl_scope) == 'numeric' and str(args.reward_mode) == 'official_mra_mixed':
        raise ValueError('numeric GRPO cannot use --reward-mode official_mra_mixed')

    set_seed(int(args.seed))
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'args.json', normalize_args_for_json(args))

    train_rows_raw = [row for row in read_native_jsonl(args.train_jsonl) if is_rl_row(row, args)]
    val_rows_raw = [row for row in read_native_jsonl(args.val_jsonl) if is_rl_row(row, args)]
    train_rows, train_audit = filter_native_rows(rows=train_rows_raw, max_rows=int(args.max_train_rows), require_patch_geometry=False, scannet_frame_root=args.scannet_frame_root, max_visual_frames=int(args.max_visual_frames))
    val_rows, val_audit = filter_native_rows(rows=val_rows_raw, max_rows=int(args.max_val_rows), require_patch_geometry=False, scannet_frame_root=args.scannet_frame_root, max_visual_frames=int(args.max_visual_frames))
    if not train_rows:
        raise ValueError(f'no {args.rl_scope} GRPO train rows after filtering')
    qtype_balance_audit: dict[str, Any] = {'enabled': False}
    if bool(args.qtype_balanced_train):
        train_rows, qtype_balance_audit = select_qtype_balanced_rows(rows=train_rows, max_groups=int(args.max_groups), seed=int(args.seed))
    visual_cache_roots = normalize_visual_cache_roots(args.visual_cache_root, args.extra_visual_cache_root)
    if not visual_cache_roots:
        raise ValueError('DirectVisual GRPO requires visual cache roots')
    primary_visual_cache_root = visual_cache_roots[0]
    extra_visual_cache_roots = visual_cache_roots[1:]
    cache_audit = {
        'train': audit_visual_cache_rows(rows=train_rows, visual_cache_root=primary_visual_cache_root, visual_cache_roots=extra_visual_cache_roots, scannet_frame_root=args.scannet_frame_root, max_visual_frames=int(args.max_visual_frames), qwen_model_path=args.qwen_model_path, max_pixels=int(args.max_pixels), split_name='train'),
        'val': audit_visual_cache_rows(rows=val_rows, visual_cache_root=primary_visual_cache_root, visual_cache_roots=extra_visual_cache_roots, scannet_frame_root=args.scannet_frame_root, max_visual_frames=int(args.max_visual_frames), qwen_model_path=args.qwen_model_path, max_pixels=int(args.max_pixels), split_name='val'),
    }
    write_json(args.output_dir / 'data_filter_audit.json', {'train': train_audit, 'val': val_audit, 'visual_cache': cache_audit, 'qtype_balance': qtype_balance_audit, 'rl_scope': str(args.rl_scope), 'numeric_qtypes': sorted(NUMERIC_QTYPES)})
    row_by_sample_id = {str(row['sample_id']): row for row in train_rows}

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == 'cpu' else 'cpu')
    processor, model, qwen_meta, checkpoint_payload, spatial_merge_size = build_model(args, device=device)
    params = list(trainable_parameters(model))
    if not params:
        raise ValueError('DirectVisual GRPO selected zero trainable parameters')
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    collator = NativeDirectVisualCachedCollator(processor=processor, max_pixels=int(args.max_pixels), spatial_merge_size=spatial_merge_size)
    train_dataset = NativeDirectVisualCachedDataset(rows=train_rows, visual_cache_root=primary_visual_cache_root, visual_cache_roots=extra_visual_cache_roots, scannet_frame_root=args.scannet_frame_root, max_visual_frames=int(args.max_visual_frames), qwen_model_path=args.qwen_model_path, max_pixels=int(args.max_pixels))
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=int(args.num_workers), collate_fn=lambda samples: collate_direct_generation_prompts(collator, samples), pin_memory=True)

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    run_config = {
        'format': FORMAT_VERSION,
        'args': normalize_args_for_json(args),
        'checkpoint': {'path': str(args.checkpoint_path), 'step': checkpoint_payload.get('step'), 'epoch': checkpoint_payload.get('epoch')},
        'train_rows': len(train_rows),
        'val_rows_for_record': len(val_rows),
        'train_qtype_counts': dict(Counter(canonical_qtype(row) for row in train_rows)),
        'trainable_parameter_count': sum(int(p.numel()) for p in params),
        'trainable_name_count': len(trainable_names),
        'trainable_first_names': trainable_names[:30],
        'qwen_meta': qwen_meta,
        'visual_cache_roots': [str(root) for root in visual_cache_roots],
    }
    write_json(args.output_dir / 'run_config.json', run_config)
    print(json.dumps({'event': 'run_config', **run_config}, ensure_ascii=False), flush=True)

    log_path = args.output_dir / 'grpo_train_log.jsonl'
    rollout_path = args.output_dir / 'grpo_rollouts.jsonl'
    global_step = 0
    seen_groups = 0
    skipped_groups = 0
    pending_backward = 0
    stop_training = False
    optimizer.zero_grad(set_to_none=True)

    with rollout_path.open('w', encoding='utf-8') as rollout_handle:
        for epoch in range(int(args.grpo_epochs)):
            for batch in loader:
                seen_groups += 1
                batch = move_batch_to_device(batch, device)
                sample_id = str(batch['sample_ids'][0])
                row = row_by_sample_id[sample_id]

                model.eval()
                generations: list[dict[str, Any]] = []
                for gen_i in range(int(args.num_generations)):
                    with torch.no_grad():
                        prediction, generated_ids = generate_one(model=model, batch=batch, tokenizer=processor.tokenizer, max_new_tokens=int(args.max_new_tokens), temperature=float(args.temperature), top_p=float(args.top_p))
                    metric_payload = compute_reward(prediction=prediction, row=row, args=args)
                    generations.append({'gen_i': gen_i, 'prediction_text': prediction, 'generated_ids': generated_ids, 'reward': float(metric_payload['score']), 'metric': metric_payload})
                valid_generations = [item for item in generations if item['generated_ids']]
                if len(valid_generations) < 2:
                    skipped_groups += 1
                    if int(args.max_groups) > 0 and seen_groups >= int(args.max_groups):
                        stop_training = True
                        break
                    continue

                rewards = torch.tensor([float(item['reward']) for item in valid_generations], device=device, dtype=torch.float32)
                reward_mean = rewards.mean()
                reward_std = rewards.std(unbiased=False)
                advantages = rewards - reward_mean if float(reward_std.detach().cpu().item()) <= 0.0 else (rewards - reward_mean) / reward_std.clamp_min(float(args.advantage_eps))
                old_logprob_list: list[torch.Tensor] = []
                ref_logprob_list: list[torch.Tensor] = []
                with torch.no_grad():
                    for item in valid_generations:
                        old_logps, _old_mean = token_logprobs_for_generated_tokens(model=model, batch=batch, generated_token_ids=item['generated_ids'])
                        if float(args.kl_beta) > 0.0:
                            with temporarily_disable_lora(model.qwen):
                                ref_logps, _ref_mean = token_logprobs_for_generated_tokens(model=model, batch=batch, generated_token_ids=item['generated_ids'])
                        else:
                            ref_logps = old_logps.detach()
                        old_logprob_list.append(old_logps.detach())
                        ref_logprob_list.append(ref_logps.detach())

                model.train()
                candidate_loss_values: list[float] = []
                candidate_logprob_means: list[float] = []
                candidate_kl_means: list[float] = []
                for cand_i, item in enumerate(valid_generations):
                    cur_logps, cur_mean = token_logprobs_for_generated_tokens(model=model, batch=batch, generated_token_ids=item['generated_ids'])
                    old_logps = old_logprob_list[cand_i].to(device=cur_logps.device, dtype=cur_logps.dtype)
                    ref_logps = ref_logprob_list[cand_i].to(device=cur_logps.device, dtype=cur_logps.dtype)
                    ratio = torch.exp(cur_logps - old_logps)
                    advantage = advantages[cand_i].to(device=cur_logps.device, dtype=cur_logps.dtype)
                    unclipped = ratio * advantage
                    clipped = ratio.clamp(1.0 - float(args.clip_epsilon), 1.0 + float(args.clip_epsilon)) * advantage
                    ref_minus_cur = ref_logps - cur_logps
                    per_token_kl = torch.exp(ref_minus_cur) - ref_minus_cur - 1.0
                    candidate_loss = (-torch.minimum(unclipped, clipped) + float(args.kl_beta) * per_token_kl).mean()
                    if not bool(torch.isfinite(candidate_loss).detach().cpu().item()):
                        raise RuntimeError(f'non-finite GRPO candidate loss at sample_id={sample_id} cand_i={cand_i}')
                    (candidate_loss / float(len(valid_generations)) / int(args.gradient_accumulation_steps)).backward()
                    candidate_loss_values.append(float(candidate_loss.detach().cpu().item()))
                    candidate_logprob_means.append(float(cur_mean.detach().cpu().item()))
                    candidate_kl_means.append(float(per_token_kl.detach().float().mean().cpu().item()))
                policy_loss_value = sum(candidate_loss_values) / len(candidate_loss_values)
                pending_backward += 1

                rollout_handle.write(json.dumps({'time': time.time(), 'epoch': epoch, 'seen_groups': seen_groups, 'sample_id': sample_id, 'target_text': str(row['answer'].get('text') or ''), 'answer_type': str(row['answer']['type']), 'canonical_question_type': canonical_qtype(row), 'reward_mean': float(reward_mean.detach().cpu().item()), 'reward_std': float(reward_std.detach().cpu().item()), 'advantages': [float(x) for x in advantages.detach().cpu().tolist()], 'generations': [{key: value for key, value in item.items() if key != 'generated_ids'} | {'generated_token_count': len(item['generated_ids'])} for item in valid_generations]}, ensure_ascii=False) + '\n')
                rollout_handle.flush()

                if pending_backward % int(args.gradient_accumulation_steps) == 0:
                    if float(args.grad_clip) > 0.0:
                        torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % int(args.log_every) == 0 or global_step == 1:
                        record = {'time': time.time(), 'elapsed_sec': time.time() - started_at, 'epoch': epoch, 'global_step': global_step, 'seen_groups': seen_groups, 'skipped_groups': skipped_groups, 'valid_generations': len(valid_generations), 'reward_mean': float(reward_mean.detach().cpu().item()), 'reward_std': float(reward_std.detach().cpu().item()), 'policy_loss': float(policy_loss_value), 'aux_loss': 0.0, 'kl_mean': sum(candidate_kl_means) / len(candidate_kl_means), 'logprob_mean': sum(candidate_logprob_means) / len(candidate_logprob_means)}
                        append_jsonl(log_path, record)
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                    if int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
                        save_checkpoint(path=args.output_dir / 'checkpoints' / f'step_{global_step:06d}', model=model, optimizer=optimizer, step=global_step, epoch=epoch, args=args, save_optimizer_state=bool(args.save_optimizer_state))
                    if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                        stop_training = True
                        break
                if int(args.max_groups) > 0 and seen_groups >= int(args.max_groups):
                    stop_training = True
                    break
            if stop_training:
                break

    if pending_backward > 0 and pending_backward % int(args.gradient_accumulation_steps) != 0:
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    save_checkpoint(path=args.output_dir / 'final_checkpoint', model=model, optimizer=optimizer, step=global_step, epoch=int(args.grpo_epochs), args=args, save_optimizer_state=bool(args.save_optimizer_state))
    summary = {'status': 'completed', 'format': FORMAT_VERSION, 'elapsed_sec': round(time.time() - started_at, 3), 'seen_groups': seen_groups, 'global_step': global_step, 'skipped_groups': skipped_groups, 'final_checkpoint': str(args.output_dir / 'final_checkpoint' / 'direct_qwen_sft.pt')}
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({'event': 'completed', **summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
