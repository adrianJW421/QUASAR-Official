"""公开输入生成必须拒绝缺参照、错帧和把训练答案带进推理的输入。"""
import copy
import json
import sys

import numpy as np
from PIL import Image
import pytest

from quasar import prepare_public_inputs as public
from quasar import extract_public_frames as frames
from quasar import convert_vsi_source as vsi


@pytest.fixture
def raw_input(tmp_path):
    image = tmp_path / 'frame.jpg'
    Image.new('RGB', (4, 4)).save(image)
    geometry = tmp_path / 'geometry/video_npz'
    geometry.mkdir(parents=True)
    np.savez(geometry / 'video.npz', video_id=np.asarray('video'),
        frame_candidate_ids=np.asarray(['frame0']), frame_image_paths=np.asarray([str(image)]),
        target_points=np.arange(12, dtype=np.float32).reshape(1, 4, 3),
        camera_pose=np.eye(4, dtype=np.float32)[None], patch_grid_hw=np.array([[2, 2]]),
        token_mask=np.ones((1, 4), dtype=bool), token_confidence=np.ones((1, 4), dtype=np.float32))
    qa = {'source_row': 1, 'sample_id': 'source::first', 'source_tag': 'vlm3r',
        'source_dataset': 'VLM-3R-DATA', 'video_id': 'video',
        'question': {'text': 'How far at frame 1 of 32?', 'canonical_question_type': 'camera_displacement',
                     'reference_condition': 'target_camera_pose'},
        'answer': {'type': 'numeric', 'text': '123456', 'numeric_value': 123456., 'options': []}}
    plan = {'source_row': 1, 'sample_id': qa['sample_id'], 'video_id': 'video', 'scene_id': 'scene',
            'geometry_source': 'cut3r', 'target_frame_index': 0}
    manifest = {'video_id': 'video', 'frame_candidate_id': 'frame0', 'frame_image_path': str(image),
                'frame_id': 0, 'sample_order': 0, 'embedding_row_index': 0}
    for name, row in [('qa', qa), ('plan', plan), ('frames', manifest)]:
        (tmp_path / (name + '.jsonl')).write_text(json.dumps(row) + '\n')
    return tmp_path, qa, plan, manifest


def invoke(root, monkeypatch, extra=()):
    monkeypatch.setattr(sys, 'argv', ['public', '--qa-jsonl', str(root / 'qa.jsonl'),
        '--input-plan', str(root / 'plan.jsonl'), '--frame-manifest', str(root / 'frames.jsonl'),
        '--geometry-root', str(root / 'geometry'), '--output-root', str(root / 'prepared'),
        '--max-visual-frames', '1', *extra])
    public.main()


def test_inference_separates_gold(raw_input, monkeypatch):
    root, _, _, _ = raw_input
    invoke(root, monkeypatch)
    row = json.loads((root / 'prepared/inference.jsonl').read_text())
    assert set(row['answer']) == {'type', 'options'}
    assert '123456' not in (root / 'prepared/inference.jsonl').read_text()
    assert json.loads((root / 'prepared/labels.jsonl').read_text())['answer']['text'] == '123456'
    assert row['refs']['simple3_aux']['query_pose_4x4'] == np.eye(4).tolist()


def test_canonical_reference_matches_same_geometry(raw_input, monkeypatch):
    root, qa, _, _ = raw_input
    qa['question']['reference_condition'] = 'canonical_bev_scene_pose'
    (root / 'qa.jsonl').write_text(json.dumps(qa) + '\n')
    invoke(root, monkeypatch)
    aux = json.loads((root / 'prepared/inference.jsonl').read_text())['refs']['simple3_aux']
    assert aux['canonical_bev']['scene_center'] == [4.5, 5.5, 6.5]
    assert aux['query_pose_valid'] is False


def test_missing_semantic_annotation_stops_without_placeholder(raw_input, monkeypatch):
    root, qa, _, _ = raw_input
    qa['question']['reference_condition'] = 'semantic_query_pose'
    (root / 'qa.jsonl').write_text(json.dumps(qa) + '\n')
    with pytest.raises(ValueError, match='Missing sourced semantic reference'):
        invoke(root, monkeypatch)
    assert not (root / 'prepared').exists()


@pytest.mark.parametrize('problem', ['old_path', 'wrong_candidate', 'wrong_video'])
def test_geometry_frame_identity_is_checked(raw_input, monkeypatch, problem):
    root, _, _, _ = raw_input
    path = root / 'geometry/video_npz/video.npz'
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    if problem == 'old_path':
        data['frame_image_paths'] = np.asarray([str(root / 'historical_frame.jpg')])
    elif problem == 'wrong_candidate':
        data['frame_candidate_ids'] = np.asarray(['another_frame'])
    else:
        data['video_id'] = np.asarray('another_video')
    np.savez(path, **data)
    with pytest.raises(ValueError, match='order|another video'):
        invoke(root, monkeypatch)


def test_gold_cannot_be_reference_definition(raw_input, monkeypatch):
    root, _, _, _ = raw_input
    ref = root / 'refs.jsonl'
    ref.write_text(json.dumps({'source_row': 1, 'sample_id': 'source::first', 'answer': '123456'}) + '\n')
    with pytest.raises(ValueError, match='supervision fields'):
        invoke(root, monkeypatch, ['--reference-jsonl', str(ref)])


def test_wrong_coordinate_gauge_is_rejected(raw_input, monkeypatch):
    root, qa, _, _ = raw_input
    qa['question']['reference_condition'] = 'semantic_query_pose'
    (root / 'qa.jsonl').write_text(json.dumps(qa) + '\n')
    ref = root / 'refs.jsonl'
    ref.write_text(json.dumps({'source_row': 1, 'sample_id': qa['sample_id'],
        'source': 'original_scene_annotation', 'geometry_gauge': 'different_world', 'pose_4x4': np.eye(4).tolist()}) + '\n')
    with pytest.raises(ValueError, match='coordinate gauge mismatch'):
        invoke(root, monkeypatch, ['--reference-jsonl', str(ref)])


def test_missing_target_index_is_rejected(raw_input, monkeypatch):
    root, _, plan, _ = raw_input
    plan['target_frame_index'] = None
    (root / 'plan.jsonl').write_text(json.dumps(plan) + '\n')
    with pytest.raises(ValueError, match='target_frame_index'):
        invoke(root, monkeypatch)


def test_duplicate_ids_are_joined_by_source_row(raw_input, monkeypatch):
    root, qa, plan, _ = raw_input
    qa2, plan2 = copy.deepcopy(qa), copy.deepcopy(plan)
    qa2['source_row'] = plan2['source_row'] = 2
    qa2['question']['text'] = 'How far in frame 1 of 32?'
    (root / 'qa.jsonl').write_text('\n'.join(json.dumps(x) for x in [qa, qa2]) + '\n')
    (root / 'plan.jsonl').write_text('\n'.join(json.dumps(x) for x in [plan, plan2]) + '\n')
    invoke(root, monkeypatch)
    rows = [json.loads(x) for x in (root / 'prepared/inference.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['question'] != rows[1]['question']


def test_raw_video_wrong_frame_count_stops(raw_input, monkeypatch):
    root, _, _, _ = raw_input
    video = root / 'video.mp4'
    video.touch()
    plan = root / 'video_plan.jsonl'
    plan.write_text(json.dumps({'video_id': 'video', 'video_path': str(video),
        'expected_frame_ids': [0, 8], 'image_long_edge': 720}) + '\n')
    def probe(command, **kwargs):
        assert command[0] == 'ffprobe'
        return type('Probe', (), {'stdout': json.dumps({'streams': [{'nb_frames': '10'}]})})()
    monkeypatch.setattr(frames.subprocess, 'run', probe)
    monkeypatch.setattr(sys, 'argv', ['extract', '--video-plan', str(plan), '--num-frames', '2',
                                    '--output-root', str(root / 'raw')])
    with pytest.raises(ValueError, match='Frame selection disagrees'):
        frames.main()


def test_original_vsi_ids_and_choice_order_are_preserved(tmp_path, monkeypatch):
    raw = {'id': 9, 'dataset': 'scannet', 'scene_name': 'scene', 'question_type': 'object_rel_direction_easy',
           'question': 'Standing at the table, which direction is the chair?',
           'ground_truth': 'B', 'options': ['A. left', 'B. right']}
    inp = tmp_path / 'test.jsonl'
    out = tmp_path / 'qa.jsonl'
    inp.write_text(json.dumps(raw) + '\n')
    monkeypatch.setattr(sys, 'argv', ['convert', '--input-jsonl', str(inp), '--output-jsonl', str(out)])
    vsi.main()
    row = json.loads(out.read_text())
    assert row['sample_id'] == 'official_vsi::9' and row['source_row'] == 1
    assert row['question']['text'] == raw['question']
    assert row['answer']['raw'] == 'B'
    assert row['answer']['options'] == [{'letter': 'A', 'text': 'left'}, {'letter': 'B', 'text': 'right'}]
