"""把固定版本的官方题集转成主表输入，答案只交给单独的评分步骤。"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import re

from quasar.build_native_answer_spatial_qa_pack import (
    canonical_qtype_from_row, infer_numeric_unit, infer_reference_condition,
    infer_query_pose_policy, normalize_options,
)


def verify_backbone(path: Path, revision: str) -> None:
    """从官方下载记录确认底座版本，并在加载模型前指出缺失的权重分片。"""
    receipt = path / '.cache/huggingface/download/config.json.metadata'
    actual = path.name if path.name == revision else (
        receipt.read_text().splitlines()[0] if receipt.is_file() else None)
    if actual != revision:
        raise ValueError(f'Backbone revision is not {revision}; use the pinned hf download command: {path}')
    for name in ['config.json', 'model.safetensors.index.json']:
        if not (path/name).is_file():
            raise FileNotFoundError(path/name)
    index = json.loads((path/'model.safetensors.index.json').read_text())
    for name in sorted(set(index['weight_map'].values())):
        if not (path/name).is_file():
            raise FileNotFoundError(path/name)


def load_official_questions(path: Path, dataset: str, contract: dict, *, vsi_base=False) -> list[dict]:
    """只接受清单绑定的官方文件；同名的新版本不能混进历史主表。"""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != contract['qa_sha256']:
        raise ValueError(f'{dataset}: wrong official QA version; download revision {contract["revision"]}')
    raw = json.loads(data) if dataset == 'vsti' else [json.loads(x) for x in data.splitlines()]
    if len(raw) != contract['count'] or len({x['id'] for x in raw}) != len(raw):
        raise ValueError(f'{dataset}: changed question membership')
    rows = []
    for order, item in enumerate(raw, 1):
        tag = 'vsti_official_row32' if dataset == 'vsti' else 'official_eval_legacy_vsi'
        qtype = canonical_qtype_from_row(item, dataset=dataset, source_tag=tag)
        text = re.sub(r'\s+', ' ', item['question']).strip() if dataset == 'vsi' and not vsi_base else item['question'].strip()
        options = normalize_options(item['options'] or [], context=f'{dataset}/{item["id"]}')
        if dataset == 'vsi' and not vsi_base and options and len(options) < 4:
            # 历史训练模型把方向/路线题补为四项。只使用题目选项构造干扰项，不读取正确答案。
            texts = [x['text'] for x in options]
            if qtype.startswith('object_rel_direction'):
                candidates = ['front', 'back', 'left', 'right', 'front-left', 'front-right', 'back-left', 'back-right']
            elif qtype == 'route_planning':
                sequences = [tuple(re.sub(r'\s+', ' ', part).strip().title() for part in x.split(',') if part.strip()) for x in texts]
                lengths = {len(x) for x in sequences}
                if len(lengths) != 1:
                    raise ValueError('Historical route options have inconsistent lengths')
                candidates = [', '.join(x) for x in itertools.product(
                    ('Turn Left', 'Turn Right', 'Turn Back'), repeat=next(iter(lengths)))]
                candidates += ['Go Forward', 'Turn Around', 'Stop']
            else:
                raise ValueError(f'No verified option expansion for {qtype}')
            present = {x.casefold() for x in texts}
            for candidate in candidates:
                if len(texts) == 4:
                    break
                if candidate.casefold() not in present:
                    texts.append(candidate)
                    present.add(candidate.casefold())
            if len(texts) != 4:
                raise ValueError('Unable to reproduce four-option input')
            options = [{'letter': chr(65+i), 'text': x} for i, x in enumerate(texts)]
        reference = infer_reference_condition(canonical_qtype=qtype, question=text, dataset=dataset, row=item)
        question = dict(text=text, canonical_question_type=qtype, raw_question_type=item['question_type'],
                        reference_condition=reference, query_pose_policy=infer_query_pose_policy(reference), situation=None)
        kind = 'multiple_choice' if options else ('count' if qtype == 'object_counting' else 'numeric')
        unit = infer_numeric_unit(qtype, text)
        if dataset == 'vsti':
            sample_id = f'{tag}::vsti_test__{item["id"]}'
            scene = Path(item['video_path']).stem
            video = 'vsti_bench__scannet__' + scene
            media = item['video_path']
        else:
            sample_id = f'{tag}::vsi_all_mca__{item["id"]}'
            scene = item['scene_name']
            video = item['dataset'] + '__' + scene
            media = item['dataset'] + '/' + scene + '.mp4'
        answer_input = dict(type=kind, options=options, numeric_unit=unit)
        # 这里明确分开模型可读字段和评分字段；模型进程只接收 input。
        letter = str(item['mc_answer'] if dataset == 'vsti' else item['ground_truth']).strip() if options else None
        if options:
            answer_text = next(x['text'] for x in options if x['letter'] == letter)
            numeric = None
        else:
            answer_text = str(item['ground_truth']).strip()
            number = re.search(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)', answer_text.replace(',', ''))
            if number is None:
                raise ValueError('Official numeric label has no number')
            numeric = float(number.group())
        row = {'format_version': 'glow_native_answer_spatial_qa_v1', 'sample_id': sample_id,
               'source': {'tag': tag, 'dataset': item.get('dataset', 'vsti_bench'), 'row_id': item['id']},
               'scene': {'scene_id': scene, 'scene_name': scene, 'video_id': video},
               'question': question, 'answer': answer_input, 'inputs': {'image_paths': []},
               'refs': {'geometry': {}, 'pose': {}}, 'flags': {'is_legacy_mca': False}}
        gold = dict(row, answer=dict(answer_input, text=answer_text, raw=item['ground_truth'],
                                    numeric_value=numeric, option_letter=letter))
        rows.append({'source_row': order, 'input': row, 'label': gold, 'media_relative_path': media})
    return rows


def read_complete_predictions(path: Path, expected_ids: list[str], *, require_complete: bool) -> list[dict]:
    """续跑只认已经完整写出的逐题回答；缺题、重复题和截断文件都明确报错。"""
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError('Duplicate expected question IDs')
    if not path.exists():
        if require_complete:
            raise FileNotFoundError(path)
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError(f'Incomplete final prediction line: {path}; preserve it and repair this attempt before resuming')
    records = [json.loads(line) for line in raw.splitlines()]
    ids = [x['sample_id'] for x in records]
    if len(ids) != len(set(ids)) or not set(ids) <= set(expected_ids):
        raise ValueError('Duplicate or unexpected predictions')
    if ids != expected_ids[:len(ids)]:
        raise ValueError('Prediction order changed')
    if any('error' in x or not isinstance(x.get('prediction_text'), str) for x in records):
        raise ValueError('Failed prediction cannot count as a completed answer')
    if require_complete and ids != expected_ids:
        raise ValueError(f'Missing answers: got {len(ids)} of {len(expected_ids)}')
    return records
