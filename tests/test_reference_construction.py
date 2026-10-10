"""检查坐标转换、同源帧绑定及缺失来源拒绝；教学数值不代表真实题目。"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from quasar.construct_reference import construct_reference


def fixture():
    definition = json.loads((Path(__file__).parents[1]/'data/examples/reference_definition.synthetic.json').read_text())
    frames = [dict(video_id='teaching', frame_id=10, frame_candidate_id='teaching_10', frame_image_path='frame.jpg')]
    geometry = dict(video_id=np.asarray('teaching'), frame_candidate_ids=np.asarray(['teaching_10']),
                    frame_image_paths=np.asarray(['frame.jpg']))
    return definition, frames, geometry


def test_station_offset_and_axes():
    definition, frames, geometry = fixture()
    ref, audit = construct_reference(definition, frames, geometry)
    pose = np.asarray(ref['pose_4x4'])
    # 物体在(1,0,0)，人向前偏移1；目标在人的正前方3个单位。
    np.testing.assert_allclose(pose[:3, 3], [1, 0, 1])
    np.testing.assert_allclose(np.linalg.inv(pose) @ [1, 0, 4, 1], [0, 0, 3, 1])
    assert audit['kind'] == 'synthetic_teaching'
    assert set(ref) == {'source_row', 'sample_id', 'geometry_gauge', 'source', 'pose_4x4'}


def test_sourced_rotation_translation_and_units():
    definition, frames, geometry = fixture()
    definition['source_gauge'] = 'original_scan:teaching'
    definition['source_units'] = 'metres'
    definition['source_to_geometry'] = dict(scale=2, rotation=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        translation=[4, 5, 6], source='synthetic known alignment')
    ref, _ = construct_reference(definition, frames, geometry)
    pose = np.asarray(ref['pose_4x4'])
    np.testing.assert_allclose(pose[:3, 3], [6, 5, 4])
    np.testing.assert_allclose(np.linalg.inv(pose) @ [12, 5, 4, 1], [0, 0, 6, 1])


@pytest.mark.parametrize('problem', ['answer', 'missing_offset', 'missing_source', 'wrong_frames',
    'wrong_geometry', 'wrong_paths', 'reflection', 'scale', 'nan', 'parallel', 'same_gauge_transform'])
def test_invalid_definitions_are_rejected(problem):
    definition, frames, geometry = fixture()
    if problem == 'answer': definition['answer'] = 'A'
    elif problem == 'missing_offset': del definition['station_offset']
    elif problem == 'missing_source': definition['source_to_geometry']['source'] = ''
    elif problem == 'wrong_frames': definition['frame_ids'] = [11]
    elif problem == 'wrong_geometry': geometry['video_id'] = np.asarray('other')
    elif problem == 'wrong_paths': geometry['frame_image_paths'] = np.asarray(['old.jpg'])
    elif problem == 'reflection': definition['source_to_geometry']['rotation'][0][0] = -1
    elif problem == 'scale': definition['source_to_geometry']['scale'] = 0
    elif problem == 'nan': definition['station_point'][0] = float('nan')
    elif problem == 'parallel': definition['facing_point'] = [1, 3, 1]
    elif problem == 'same_gauge_transform': definition['source_to_geometry']['translation'] = [1, 0, 0]
    with pytest.raises(ValueError):
        construct_reference(definition, frames, geometry)
