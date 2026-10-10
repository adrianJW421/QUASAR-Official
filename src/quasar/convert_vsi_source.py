"""Convert the fields of the provider's VSI test.jsonl, keeping the original question order, IDs, options and answers."""
import argparse
import json
from pathlib import Path

from quasar.build_native_answer_spatial_qa_pack import (
    build_answer_payload, canonical_qtype_from_row, normalize_options,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial_moe_metric import infer_simple_reference_condition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl', type=Path, required=True)
    p.add_argument('--output-jsonl', type=Path, required=True)
    a = p.parse_args()
    if a.output_jsonl.exists():
        raise FileExistsError(a.output_jsonl)
    rows = []
    ids = set()
    for order, line in enumerate(a.input_jsonl.read_text().splitlines(), 1):
        raw = json.loads(line)
        identifier = raw['id']
        if identifier in ids:
            raise ValueError(f'Duplicate upstream VSI ID: {identifier}')
        ids.add(identifier)
        row = dict(raw, answer=raw['ground_truth'])
        tag = 'official_vsi'
        qtype = canonical_qtype_from_row(row, dataset=raw['dataset'], source_tag=tag)
        options = normalize_options(raw['options'] or [], context=f'VSI id={identifier}')
        answer = build_answer_payload(row, canonical_qtype=qtype, options=options,
                                      legacy_mca=False, context=f'VSI id={identifier}')
        question = {'text': raw['question'], 'canonical_question_type': qtype,
                    'raw_question_type': raw['question_type']}
        question['reference_condition'] = infer_simple_reference_condition(
            row={'source': {'tag': tag}}, question_payload=question)
        # 原方ID可能不连续；保留原ID，并单独记录实际行号，不拿列表下标顶替ID。
        rows.append({'source_row': order, 'sample_id': f'official_vsi::{identifier}',
            'source_tag': tag, 'source_dataset': 'VSI-Bench', 'upstream_id': identifier,
            'video_id': raw['dataset'] + '__' + raw['scene_name'],
            'question': question, 'answer': answer})
    if not rows:
        raise ValueError('Empty upstream VSI input')
    a.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    a.output_jsonl.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    print(json.dumps({'converted_rows': len(rows), 'original_ids_preserved': True}))


if __name__ == '__main__':
    main()
