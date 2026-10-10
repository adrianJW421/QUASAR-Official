import hashlib
import json

import numpy as np
import pytest

from quasar.reviewer_data import load_official_questions
from quasar.vsi_reference import parse_station, compute_reference, validate_pose, ReferenceInputBlocked


def test_base_preserves_two_options_and_newlines(tmp_path):
    item = dict(id=2, dataset='scannet', scene_name='room', question_type='object_rel_direction_easy',
        question='If I am standing by the stove and facing the sofa,\nis the tv left or right?',
        options=['A. left', 'B. right'], ground_truth='A')
    path = tmp_path/'qa.jsonl'
    path.write_text(json.dumps(item)+'\n')
    contract = dict(qa_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=1, revision='fixture')
    base = load_official_questions(path, 'vsi', contract, vsi_base=True)[0]
    trained = load_official_questions(path, 'vsi', contract)[0]
    assert base['input']['question']['text'] == item['question']
    assert [x['text'] for x in base['input']['answer']['options']] == ['left', 'right']
    assert len(trained['input']['answer']['options']) == 4
    assert base['label']['answer']['option_letter'] == trained['label']['answer']['option_letter'] == 'A'
    item['ground_truth'] = 'B'
    path.write_text(json.dumps(item)+'\n')
    contract['qa_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_official_questions(path, 'vsi', contract, vsi_base=True)[0]['input'] == base['input']


@pytest.mark.parametrize('text,qtype,mode,start,end', [
    ('If I am standing by the stove and facing the sofa, is the tv left?', 'object_rel_direction_easy', 'facing', 'the stove', 'the sofa'),
    ('You are a robot beginning at the bed facing the tv. You want to navigate.', 'route_planning', 'facing', 'the bed', 'the tv'),
    ('You are a robot beginning by the chair and facing the window. You want to navigate.', 'route_planning', 'facing', 'the chair', 'the window'),
    ('You are a robot beginning at the refrigerator, with your back to the fridge. You want to navigate.', 'route_planning', 'back_to', 'the refrigerator', 'the fridge'),
])
def test_station_drops_target_clause(text, qtype, mode, start, end):
    assert parse_station(text, qtype) == dict(mode=mode, station=start, orientation_object=end)


def test_unknown_station_is_not_defaulted():
    with pytest.raises(ReferenceInputBlocked, match='Unrecognized'):
        parse_station('Stand somewhere and look forward.', 'route_planning')


def geometry_fixture():
    x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
    points = np.stack([x, y, np.zeros_like(x)], axis=-1).reshape(-1, 3)
    points = np.tile(points, (32, 1, 1))
    points[1, :, 2] = 2
    poses = np.tile(np.eye(4), (32, 1, 1))
    poses[:, 2, 3] = -1
    return dict(target_points=points, token_mask=np.ones((32, 64), bool),
                patch_grid_hw=np.tile([8, 8], (32, 1)), camera_pose=poses)


def test_pose_heading_and_translation_in_rebuilt_coordinates():
    g = geometry_fixture()
    located = dict(status='unique', reason='fixture', station=dict(frame=0, bbox=[0, 0, 1000, 1000]),
                   orientation=dict(frame=1, bbox=[0, 0, 1000, 1000]))
    pose, evidence = compute_reference(dict(mode='facing'), located, g, np.arange(32))
    # 世界z方向的对象应落在参照前方；不通过复算同一函数来定义预期答案。
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(np.linalg.inv(pose) @ [0, 0, 2, 1], [0, 0, 2, 1], atol=1e-10)
    assert len(evidence['station']['patch_indices']) == 16
    g['token_mask'][0] = False
    with pytest.raises(ReferenceInputBlocked, match='Insufficient'):
        compute_reference(dict(mode='facing'), located, g, np.arange(32))


def test_back_to_face_normal_is_outward():
    g = geometry_fixture()
    located = dict(status='unique', reason='fixture', station=dict(frame=0, bbox=[0, 0, 1000, 1000]),
                   orientation=dict(frame=0, bbox=[0, 0, 1000, 1000]))
    pose, _ = compute_reference(dict(mode='back_to'), located, g, np.arange(32))
    np.testing.assert_allclose(np.array(pose)[:3, 2], [0, 0, -1], atol=1e-10)


def test_ambiguous_objects_and_wrong_coordinate_matrix_rejected():
    with pytest.raises(ReferenceInputBlocked, match='unique'):
        compute_reference(dict(mode='facing'), dict(status='ambiguous', reason='two chairs', station=None, orientation=None),
                          geometry_fixture(), np.arange(32))
    pose = np.eye(4)
    pose[0, 0] = 2
    with pytest.raises(ValueError, match='rigid rotation'):
        validate_pose(pose)
    pose = np.eye(4)
    pose[3, 0] = 1
    with pytest.raises(ValueError, match='homogeneous'):
        validate_pose(pose)


def test_public_base_complete_resume_never_loads_model(tmp_path, monkeypatch):
    from quasar.public_vsi_base import main
    # 完整输入必须仍合法；没有模型路径也能证明已完成续跑没有加载模型。
    frame = tmp_path/'frame.png'
    frame.write_bytes(b'not_read_on_complete_resume')
    row = dict(format_version='glow_native_answer_spatial_qa_v1',sample_id='a',source={},scene={},flags={},
        question=dict(text='How many chairs?', canonical_question_type='object_counting',raw_question_type='object_counting',
            reference_condition='none',query_pose_policy=None,situation=None),
        answer=dict(type='count',options=[],numeric_unit=None),inputs=dict(image_paths=[str(frame)]*32),
        refs=dict(geometry={},pose={}))
    inp=tmp_path/'input.jsonl'
    inp.write_text(json.dumps(row)+'\n')
    out=tmp_path/'out';out.mkdir()
    saved=json.dumps(dict(sample_id='a',prediction_text='1'))+'\n'
    (out/'predictions.jsonl').write_text(saved)
    monkeypatch.setattr('sys.argv',['base','--input-jsonl',str(inp),'--output-root',str(out),
                                  '--qwen-model-path',str(tmp_path/'no_model')])
    main()
    assert (out/'predictions.jsonl').read_text()==saved
    assert not (out/'MODEL_LOAD.json').exists()
    row['answer']['text']='secret_label';inp.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Gold answer'):
        main()
