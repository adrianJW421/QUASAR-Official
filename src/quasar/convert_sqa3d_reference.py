"""Build reference positions for checked members from the original SQA3D annotations; answers stay in a separate QA file."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def build_references(qa, plans, questions, annotations, split):
    """沿原Loc3R转换的xy/yaw和情境顺序，核对原题后只导出站位定义。"""
    q_by_id = {x['question_id']: x for x in questions['questions']}
    a_by_id = {x['question_id']: x for x in annotations['annotations']}
    if len(q_by_id) != len(questions['questions']) or len(a_by_id) != len(annotations['annotations']):
        raise ValueError('Duplicate SQA3D question_id')
    result = []
    for plan in plans:
        row = qa[plan['source_row']]
        if row['sample_id'] != plan['sample_id'] or row['source_tag'] != 'sqa3d_loc3r':
            raise ValueError('SQA3D source occurrence mismatch')
        if plan['geometry_source'] != 'scannet_depth_pose':
            raise ValueError('SQA3D xy/yaw requires original ScanNet world coordinates')
        prefix, qid, situation_index = row['sample_id'].split('::')[1].rsplit('_', 2)
        if prefix != 'sqa3d':
            raise ValueError('Invalid SQA3D member id')
        question, annotation = q_by_id[int(qid)], a_by_id[int(qid)]
        situations = question.get('alternative_situation', []) + [question['situation']] if split == 'train' else [question['situation']]
        if not 0 <= int(situation_index) < len(situations):
            raise ValueError('SQA3D situation index outside original annotation')
        if (row['question']['text'] != question['question']
                or row['question']['situation'] != situations[int(situation_index)]
                or plan['scene_id'] != annotation['scene_id']
                or row['video_id'] != 'sqa3d__' + annotation['scene_id']):
            raise ValueError('SQA3D question, situation or scene version mismatch')
        position, rotation = annotation['position'], annotation['rotation']
        xy = [float(position['x']), float(position['y'])]
        x, y, z, w = [float(rotation[k]) for k in ['_x', '_y', '_z', '_w']]
        # 原训练转换直接取四元数的yaw，不引入重新标定、轴交换或高度估计。
        yaw = math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
        if not all(math.isfinite(v) for v in [*xy, x, y, z, w, yaw]):
            raise ValueError('Nonfinite SQA3D reference annotation')
        result.append({'source_row': plan['source_row'], 'sample_id': row['sample_id'],
                       'source': 'sqa3d_original_xy_yaw',
                       'geometry_gauge': 'scannet_world:' + annotation['scene_id'],
                       'loc3r_position_xy': xy, 'loc3r_orientation_rad': yaw})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-jsonl', type=Path, required=True)
    parser.add_argument('--input-plan', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plans = [json.loads(line) for line in args.input_plan.read_text().splitlines() if line.strip()]
    wanted = [p['source_row'] for p in plans]
    if not wanted or wanted != sorted(set(wanted)):
        raise ValueError('Keep distinct source rows in original order')
    qa = {}
    with args.qa_jsonl.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['source_row'] in wanted:
                if row['source_row'] in qa:
                    raise ValueError('Duplicate QA source_row')
                qa[row['source_row']] = row
    references = build_references(qa, plans, json.loads(args.questions.read_text()),
                                  json.loads(args.annotations.read_text()), args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        for reference in references:
            handle.write(json.dumps(reference, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'references': len(references), 'coordinate_system': 'original_scannet_world',
                      'answer_fields_exported': False}))


if __name__ == '__main__':
    main()
