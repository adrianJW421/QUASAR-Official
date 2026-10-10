import copy
import math

import pytest

from quasar.convert_sqa3d_reference import build_references


def inputs():
    qa = {4: {'sample_id': 'sqa3d_loc3r::sqa3d_17_0', 'source_tag': 'sqa3d_loc3r',
              'video_id': 'sqa3d__scene0010_00', 'question': {'text': 'What color?', 'situation': 'Looking east'},
              'answer': {'text': 'secret answer'}}}
    plan = [{'source_row': 4, 'sample_id': qa[4]['sample_id'],
             'geometry_source': 'scannet_depth_pose', 'scene_id': 'scene0010_00'}]
    questions = {'questions': [{'question_id': 17, 'question': 'What color?',
                                'situation': 'Main situation', 'alternative_situation': ['Looking east']}]}
    annotations = {'annotations': [{'question_id': 17, 'scene_id': 'scene0010_00',
                    'position': {'x': 1.5, 'y': -2., 'z': 9.},
                    'rotation': {'_x': 0., '_y': 0., '_z': math.sqrt(.5), '_w': math.sqrt(.5)},
                    'answers': [{'answer': 'another secret'}]}]}
    return qa, plan, questions, annotations


def test_sqa_reference_original_order_and_answer_separation():
    qa, plan, questions, annotations = inputs()
    reference = build_references(qa, plan, questions, annotations, 'train')[0]
    assert reference['loc3r_position_xy'] == [1.5, -2.]
    assert reference['loc3r_orientation_rad'] == pytest.approx(math.pi/2)
    assert set(reference) == {'source_row', 'sample_id', 'source', 'geometry_gauge',
                             'loc3r_position_xy', 'loc3r_orientation_rad'}
    altered = copy.deepcopy(annotations)
    altered['annotations'][0]['answers'] = [{'answer': 'changed'}]
    assert build_references(qa, plan, questions, altered, 'train')[0] == reference


@pytest.mark.parametrize('changed', ['scene', 'situation', 'coordinates'])
def test_sqa_reference_rejects_wrong_binding(changed):
    qa, plan, questions, annotations = inputs()
    if changed == 'scene': plan[0]['scene_id'] = 'scene0000_00'
    if changed == 'situation': qa[4]['question']['situation'] = 'Another version'
    if changed == 'coordinates': plan[0]['geometry_source'] = 'cut3r'
    with pytest.raises(ValueError):
        build_references(qa, plan, questions, annotations, 'train')
