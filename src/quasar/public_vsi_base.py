"""Public VSI Base: the official model reads only the original question text, options and 32 RGB frames."""
import argparse
import json
from pathlib import Path
import random
import time

from quasar.base_vsti.prompt import build_direct_native_prompt
from quasar.reviewer_data import read_complete_predictions, verify_backbone

REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
SETTINGS = dict(protocol='public-vsi-v1', revision=REVISION, frames=32, max_pixels=50176,
                dtype='bfloat16', attention='sdpa', seed=0, do_sample=False, max_new_tokens=64)


def public_messages(row):
    """显式检查模型入口；计分答案、几何和训练目标不能混入Base。"""
    if set(row) != {'format_version', 'sample_id', 'source', 'scene', 'question', 'answer', 'inputs', 'refs', 'flags'}:
        raise ValueError('Unexpected Base input fields')
    if set(row['answer']) != {'type', 'options', 'numeric_unit'}:
        raise ValueError('Gold answer in Base input')
    if set(row['question']) != {'text', 'canonical_question_type', 'raw_question_type',
                               'reference_condition', 'query_pose_policy', 'situation'}:
        raise ValueError('Unexpected question fields')
    if row['refs'] != {'geometry': {}, 'pose': {}} or set(row['inputs']) != {'image_paths'}:
        raise ValueError('Base takes RGB only')
    paths = row['inputs']['image_paths']
    if len(paths) != 32 or any(not Path(p).is_file() for p in paths):
        raise ValueError('Base needs the registered 32 RGB files')
    prompt = build_direct_native_prompt(question_payload=row['question'], answer=row['answer'],
                                        answer_type=row['answer']['type'])
    content = [dict(type='image', image=p, max_pixels=50176) for p in paths]
    content.append(dict(type='text', text=prompt))
    return [dict(role='user', content=content)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl', type=Path, required=True)
    p.add_argument('--qwen-model-path', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--inspect-only', action='store_true', help='Verify real processor inputs without generating answers.')
    a = p.parse_args()
    rows = [json.loads(l) for l in a.input_jsonl.read_text().splitlines()]
    ids = [r['sample_id'] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('Expected nonempty unique ordered Base inputs')
    messages = [public_messages(r) for r in rows]
    a.output_root.mkdir(parents=True, exist_ok=True)
    contract = dict(SETTINGS, model_path=str(a.qwen_model_path.resolve()), inputs=rows)
    freeze = a.output_root/'BASE_CONTRACT.json'
    if freeze.exists() and json.loads(freeze.read_text()) != contract:
        raise ValueError('Base run changed; use a new output-root')
    freeze.write_text(json.dumps(contract, indent=2)+'\n')
    ledger = a.output_root/'predictions.jsonl'
    saved = read_complete_predictions(ledger, ids, require_complete=False)
    if not a.inspect_only and len(saved) == len(rows):
        return
    verify_backbone(a.qwen_model_path, REVISION)
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    processor = AutoProcessor.from_pretrained(a.qwen_model_path, local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    processor.image_processor.max_pixels = 50176
    if not a.inspect_only:
        model, info = Qwen3VLForConditionalGeneration.from_pretrained(a.qwen_model_path,
            local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
        if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise ValueError(f'Incomplete official backbone: {info}')
        model.eval().requires_grad_(False).to('cuda:0')
        (a.output_root/'MODEL_LOAD.json').write_text(json.dumps(dict(settings=SETTINGS,
            model_class=type(model).__name__, loading_info=info), indent=2)+'\n')
    receipts = []
    for index in range(0 if a.inspect_only else len(saved), len(rows)):
        row, message = rows[index], messages[index]
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(message)
        if videos is not None or len(images) != 32:
            raise ValueError('Processor did not receive 32 independent RGB images')
        batch = processor(text=[prompt], images=images, return_tensors='pt', padding=True)
        if batch['image_grid_thw'].shape[0] != 32:
            raise ValueError('Processor image-grid count changed')
        receipts.append(dict(sample_id=row['sample_id'], prompt=message[0]['content'][-1]['text'],
            image_paths=row['inputs']['image_paths'], image_grid_thw=batch['image_grid_thw'].tolist(),
            input_token_count=int(batch['input_ids'].shape[1]), no_geometry=True, no_gold=True))
        if a.inspect_only:
            continue
        # 每条回答完整写一行，恢复时从已经核过的连续前缀之后接着作答。
        tick = time.time()
        with torch.inference_mode():
            output = model.generate(**batch.to('cuda:0'), do_sample=False, max_new_tokens=64, use_cache=True,
                eos_token_id=sorted({processor.tokenizer.eos_token_id,
                    processor.tokenizer.convert_tokens_to_ids('<|im_end|>')}),
                pad_token_id=processor.tokenizer.pad_token_id)
        tokens = output[0, batch['input_ids'].shape[1]:].tolist()
        record = dict(sample_id=row['sample_id'], prediction_text=processor.tokenizer.decode(tokens,
            skip_special_tokens=True).strip(), token_ids=tokens, elapsed_sec=time.time()-tick, protocol='public-vsi-v1')
        with ledger.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps(dict(completed=index+1, total=len(rows))), flush=True)
    (a.output_root/('READER_CHECK.json' if a.inspect_only else 'GENERATION_INPUTS.json')).write_text(
        json.dumps(dict(settings=SETTINGS, cases=receipts), indent=2)+'\n')


if __name__ == '__main__':
    main()
