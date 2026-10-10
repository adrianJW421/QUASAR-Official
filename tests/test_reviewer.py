import hashlib
import json

import pytest

from quasar.reviewer import write_jsonl
from quasar.reviewer_data import load_official_questions, read_complete_predictions, verify_backbone


def test_backbone_version_and_missing_shard(tmp_path):
    receipt = tmp_path/'.cache/huggingface/download/config.json.metadata'
    receipt.parent.mkdir(parents=True)
    receipt.write_text('wrong_revision\n')
    with pytest.raises(ValueError, match='Backbone revision'):
        verify_backbone(tmp_path, 'expected_revision')
    receipt.write_text('expected_revision\n')
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'part.safetensors'}}))
    with pytest.raises(FileNotFoundError, match='part.safetensors'):
        verify_backbone(tmp_path, 'expected_revision')
    (tmp_path/'part.safetensors').write_bytes(b'fixture')
    verify_backbone(tmp_path, 'expected_revision')


def test_official_version_and_answer_separation(tmp_path):
    path = tmp_path / 'test.jsonl'
    item = dict(id=17, dataset='scannet', scene_name='scene0', question_type='object_counting',
                question='How many chairs?', options=[], ground_truth='4')
    path.write_text(json.dumps(item)+'\n')
    contract = dict(qa_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), revision='fixed', count=1)
    row = load_official_questions(path, 'vsi', contract)[0]
    assert row['label']['answer']['numeric_value'] == 4
    assert set(row['input']['answer']) == {'type', 'options', 'numeric_unit'}
    item['ground_truth'] = '5'
    path.write_text(json.dumps(item)+'\n')
    with pytest.raises(ValueError, match='wrong official QA version'):
        load_official_questions(path, 'vsi', contract)
    contract['qa_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    changed_label = load_official_questions(path, 'vsi', contract)[0]
    assert changed_label['input'] == row['input']
    assert changed_label['label']['answer']['numeric_value'] == 5


def test_resume_accepts_only_complete_ordered_prefix(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    a = dict(sample_id='a', prediction_text='')
    b = dict(sample_id='b', prediction_text='B')
    write_jsonl(path, [a])
    assert read_complete_predictions(path, ['a', 'b'], require_complete=False) == [a]
    with pytest.raises(ValueError, match='Missing answers'):
        read_complete_predictions(path, ['a', 'b'], require_complete=True)
    with path.open('a') as f:
        f.write(json.dumps(b)+'\n')
    assert read_complete_predictions(path, ['a', 'b'], require_complete=True) == [a, b]
    assert path.read_text().count('sample_id') == 2


@pytest.mark.parametrize('records,message', [
    ([{'sample_id': 'a', 'prediction_text': 'A'}]*2, 'Duplicate'),
    ([{'sample_id': 'b', 'prediction_text': 'B'}], 'order'),
    ([{'sample_id': 'a', 'prediction_text': 'A', 'error': 'failed'}], 'Failed'),
])
def test_invalid_saved_answers_are_not_completed(tmp_path, records, message):
    path = tmp_path / 'answers.jsonl'
    write_jsonl(path, records)
    with pytest.raises(ValueError, match=message):
        read_complete_predictions(path, ['a', 'b'], require_complete=False)


def test_truncated_record_and_changed_frozen_input_fail(tmp_path):
    path = tmp_path / 'answers.jsonl'
    path.write_text('{"sample_id":"a"')
    with pytest.raises(ValueError, match='Incomplete final'):
        read_complete_predictions(path, ['a'], require_complete=False)
    with pytest.raises(ValueError, match='Frozen run input changed'):
        write_jsonl(path, [{'sample_id': 'a'}])


def test_base_completed_resume_without_cuda_override_or_model_load(tmp_path, monkeypatch):
    from quasar.base_vsti.inference import main
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr('sys.argv', ['base', '--run-dir', str(tmp_path), '--attempt', str(tmp_path/'attempt'),
                                   '--shard', '0', '--shards', '1'])
    (tmp_path/'PROTOCOL.json').write_text(json.dumps({'sample_order': ['a']}))
    (tmp_path/'ACCEPTANCE.json').write_text('{"status":"PASS"}')
    write_jsonl(tmp_path/'inputs/public.jsonl', [dict(sample_id='a', prompt='question', image_paths=[],
                                                    cache_key='a', feature_path='not-needed')])
    ledger = tmp_path/'predictions/shard_0.jsonl'
    write_jsonl(ledger, [dict(sample_id='a', prediction_text='A')])
    before = ledger.read_bytes()
    main()
    assert ledger.read_bytes() == before
    assert json.loads((tmp_path/'attempt/COMPLETE.json').read_text())['generated_this_attempt'] == 0
    assert not (tmp_path/'attempt/LOAD.json').exists()
