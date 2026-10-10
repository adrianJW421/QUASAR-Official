"""官方原始模型的只读推理；不导入训练模块，不接收答案文件。"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from .common import read_rows, write_json, index_rows
from .cached_generate import generate_one


def cached_batch(row, processor, device):
    """用冻结的真实网格展开图像占位符，只注入 RGB 视觉张量。"""
    visual = torch.load(row['feature_path'], map_location='cpu', weights_only=True)
    if set(visual) != {'image_grid_thw', 'image_embeds', 'deepstack_features'}:
        raise ValueError('inference cache contains non-RGB fields')
    grid = visual['image_grid_thw']
    content = [dict(type='image', image=p, max_pixels=50176) for p in row['image_paths']]
    content.append(dict(type='text', text=row['prompt']))
    text = processor.apply_chat_template([dict(role='user', content=content)],
                                         tokenize=False, add_generation_prompt=True)
    for g in grid:
        text = text.replace(processor.image_token, '<|placeholder|>' * int(g.prod() // 4), 1)
    text = text.replace('<|placeholder|>', processor.image_token)
    encoded = processor.tokenizer([text], padding=True, return_attention_mask=True, return_tensors='pt')
    mm = (encoded['input_ids'] == processor.image_token_id).long()
    if int(mm.sum()) != int((grid.prod(-1) // 4).sum()):
        raise ValueError('image token/grid mismatch')
    return dict(input_ids=encoded['input_ids'].to(device),
                attention_mask=encoded['attention_mask'].to(device), mm_token_type_ids=mm.to(device),
                image_grid_thw=grid.to(device), cached_image_embeds=visual['image_embeds'].to(device),
                cached_deepstack_features=[x.to(device) for x in visual['deepstack_features']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True, help='Authorized frozen inputs, protocol and acceptance directory.')
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--acceptance', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    if args.shards <= 0 or not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard selection')
    R = args.run_dir
    started = time.time()
    args.attempt.mkdir(parents=True, exist_ok=True)
    write_json(args.attempt/'PID.json', dict(pid=os.getpid(), started_unix=started,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), shard=args.shard, shards=args.shards))
    protocol = json.loads((R/'PROTOCOL.json').read_text())
    public = read_rows(R/'inputs/public.jsonl')
    if list(index_rows(public)) != protocol['sample_order']:
        raise ValueError('frozen public input order changed')
    if any(set(row) != {'sample_id','prompt','image_paths','cache_key','feature_path'} for row in public):
        raise ValueError('generation manifest has unexpected fields')
    if not args.acceptance:
        acceptance = json.loads((R/'ACCEPTANCE.json').read_text())
        if acceptance['status'] != 'PASS':
            raise ValueError('acceptance required before full generation')
        rows = public[args.shard::args.shards]
        ledger = R/'predictions'/f'shard_{args.shard}.jsonl'
        # 先验证已保存回答，再加载模型；完整分片不应再次占用显卡。
        from quasar.reviewer_data import read_complete_predictions
        saved = read_complete_predictions(ledger, [row['sample_id'] for row in rows], require_complete=False)
        done = index_rows(saved)
        if len(done) == len(rows):
            write_json(args.attempt/'COMPLETE.json', dict(status='PASS',count=len(done),generated_this_attempt=0))
            return
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = protocol['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = protocol['cudnn_allow_tf32']
    model_path = protocol['model_path']
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    processor.image_processor.max_pixels = protocol['max_pixels']
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
                local_files_only=True, dtype=torch.bfloat16, attn_implementation=protocol['attention'],
                output_loading_info=True)
    if any(info.get(key) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise ValueError(f'official weights incomplete: {info}')
    if any('lora' in name.lower() or 'spatial_query' in name for name, _ in model.named_parameters()):
        raise ValueError('trained/spatial parameters present in Base')
    if any('spatial' in token.lower() for token in processor.tokenizer.get_added_vocab()):
        raise ValueError('extra spatial tokenizer entries present')
    model.eval().requires_grad_(False).to('cuda')
    load = dict(status='PASS', model_path=model_path, processor_path=model_path,
                model_class=type(model).__name__, torch=torch.__version__, transformers=transformers.__version__,
                loading_info={key:list(info.get(key,[])) for key in
                    ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']},
                tokenizer_size=len(processor.tokenizer),
                embedding_rows=model.get_input_embeddings().num_embeddings,
                dtype=str(next(model.parameters()).dtype), attention=model.config._attn_implementation,
                language_attention=model.config.text_config._attn_implementation,
                no_training_weights=True, no_added_spatial_tokens=True, trainable_parameters=0,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    write_json(args.attempt/'LOAD.json', load)
    if args.acceptance:
        ids = json.loads((R/'ASSET_AUDIT.json').read_text())['acceptance_ids']
        receipts = []
        stop_ids = {processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids('<|im_end|>')}
        for sample_id in ids:
            row = index_rows(public)[sample_id]
            batch = cached_batch(row, processor, 'cuda')
            content = [dict(type='image', image=p, max_pixels=50176) for p in row['image_paths']]
            content.append(dict(type='text', text=row['prompt']))
            messages = [dict(role='user', content=content)]
            raw_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            if videos is not None:
                raise ValueError('expected 32 independent images')
            raw = processor(text=[raw_text], images=images, padding=True, return_tensors='pt').to('cuda')
            for key in ['input_ids','attention_mask','image_grid_thw','mm_token_type_ids']:
                if not torch.equal(raw[key], batch[key]):
                    raise ValueError(f'official processor/cache input mismatch: {sample_id}/{key}')
            with torch.no_grad():
                features = model.model.get_image_features(raw['pixel_values'],raw['image_grid_thw'],return_dict=True)
                native_features = [torch.cat(features.pooler_output), *features.deepstack_features]
                cached_features = [batch['cached_image_embeds'], *batch['cached_deepstack_features']]
                diffs = [dict(exact=torch.equal(x,y), max_abs=float((x.float()-y.float()).abs().max()))
                         for x,y in zip(native_features,cached_features)]
                cached_text, cached_tokens = generate_one(model=model,batch=batch,tokenizer=processor.tokenizer,
                                                          max_new_tokens=24,temperature=0.0,top_p=1.0)
                output = model.generate(**raw, do_sample=False, max_new_tokens=24,
                                        use_cache=True, eos_token_id=sorted(stop_ids),
                                        pad_token_id=processor.tokenizer.pad_token_id)
                official_tokens = output[0,raw['input_ids'].shape[1]:].tolist()
                official_tokens = official_tokens[:next((i for i,t in enumerate(official_tokens) if t in stop_ids),len(official_tokens))]
            receipt = dict(sample_id=sample_id, image_grids=batch['image_grid_thw'].tolist(),
                           rgb_feature_differences=diffs, cached_tokens=cached_tokens, official_tokens=official_tokens,
                           cached_text=cached_text, official_text=processor.tokenizer.decode(official_tokens,skip_special_tokens=True).strip(),
                           tokens_equal=cached_tokens==official_tokens)
            receipts.append(receipt)
            write_json(args.attempt/'PARITY_CASES.json',receipts)
            if cached_tokens != official_tokens:
                raise ValueError(f'cache/native generation differs: {sample_id}')
        write_json(args.attempt/'ACCEPTANCE.json', dict(status='PASS', cases=len(receipts),
                    raw_rgb_vs_cached_generated_tokens_exact=True, actual_grid_exact=True,
                    public_inputs_only=True, model_load=load, elapsed_sec=time.time()-started))
        return
    # 每个分片只有一个写入者；恢复时逐题续接，不能覆盖已经生成的答案。
    ledger.parent.mkdir(parents=True,exist_ok=True)
    generated = 0
    generation_started = time.time()
    with ledger.open('a', buffering=1) as handle:
        for row in rows:
            if row['sample_id'] in done:
                continue
            tick = time.time()
            batch = cached_batch(row,processor,'cuda')
            text, tokens = generate_one(model=model,batch=batch,tokenizer=processor.tokenizer,
                                        max_new_tokens=24,temperature=0.0,top_p=1.0)
            record = dict(sample_id=row['sample_id'],prediction_text=text,token_ids=tokens,
                          generated_token_count=len(tokens),elapsed_sec=time.time()-tick,
                          attempt=str(args.attempt),completed_unix=time.time())
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            done[row['sample_id']] = record
            generated += 1
            if generated == 1 or generated % 50 == 0 or len(done) == len(rows):
                elapsed = time.time()-generation_started
                progress = dict(status='RUNNING',pid=os.getpid(),shard=args.shard,completed=len(done),total=len(rows),
                    generated_this_attempt=generated,seconds_per_question=elapsed/generated,
                    eta_seconds=(len(rows)-len(done))*elapsed/generated,updated_unix=time.time())
                write_json(args.attempt/'PROGRESS.json',progress)
                print(json.dumps(progress),flush=True)
    write_json(args.attempt/'COMPLETE.json',dict(status='PASS',count=len(done),elapsed_sec=time.time()-started))


if __name__ == '__main__':
    main()
