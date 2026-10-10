"""检查数据迁移的真实风险：错接缓存、改变目标、以及推理依赖金标。"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from quasar import prepare_release_data as prep
from quasar import prepare_rebuilt_reference as rebuilt
from quasar import train_native_qwen3vl2b_glow_patch_spatial as basic
from quasar import train_native_qwen3vl2b_glow_patch_spatial_moe_metric as spatial
from quasar.train_native_qwen3vl2b_direct_visual_sft_baseline import NativeDirectVisualCachedDataset


@pytest.fixture
def original(tmp_path):
    root = tmp_path / 'original'
    root.mkdir()
    image = root / 'frame.png'
    Image.new('RGB', (4, 4)).save(image)
    geometry = root / 'geometry.npz'
    np.savez(geometry, frame_image_paths=np.asarray([str(image)]),
             target_points=np.arange(12, dtype=np.float32).reshape(1, 4, 3),
             camera_pose=np.eye(4, dtype=np.float32)[None], patch_grid_hw=np.array([[2, 2]]),
             token_mask=np.asarray([[True, False, True, True]]), token_confidence=np.ones((1, 4), dtype=np.float32))
    row = {'format_version': 'glow_native_answer_spatial_qa_v1', 'sample_id': 'source::first',
           'source': {'tag': 'synthetic', 'dataset': 'synthetic'},
           'scene': {'scene_id': 'scene', 'video_id': 'video'},
           'question': {'text': 'What is the size?', 'canonical_question_type': 'object_size_estimation'},
           'answer': {'type': 'numeric', 'text': '7', 'numeric_value': 7., 'options': []},
           'inputs': {'image_paths': [str(image)]}, 'flags': {'is_legacy_mca': False},
           'refs': {'geometry': {'cut3r_npz_path': str(geometry)}, 'pose': {}, 'simple3_aux': {
               'reference_schema': 'simple3', 'reference_type': 'target_camera_pose', 'reference_type_id': 0,
               'query_pose_valid': True, 'query_pose_4x4': np.eye(4).tolist(),
               'target_camera': {'enabled': True}, 'query_pose_source': 'observation',
               'metric_aux': {'enabled': True, 'scene': {'log1p_scene_scale': 9., 'log1p_room_extent_xyz': [8., 7., 6.]},
                              'camera': {'valid': True, 'log1p_primary_first_max': [5., 4., 3.]},
                              'objects': {'valid': False, 'failure_reason': 'missing_object_bank'}}}}}
    return root, row


@pytest.mark.parametrize('module', [basic, spatial])
@pytest.mark.parametrize('reference', ['target_camera_pose', 'semantic_query_pose', 'canonical_bev_scene_pose'])
def test_answering_reads_reference_without_training_labels(original, module, reference):
    _, row = original
    row = copy.deepcopy(row)
    aux = row['refs']['simple3_aux']
    aux['reference_type'] = reference
    aux['reference_type_id'] = module.reference_type_mapping('simple3')[reference]
    if reference == 'semantic_query_pose':
        aux['semantic_query'] = {'enabled': True}
    if reference == 'canonical_bev_scene_pose':
        aux['query_pose_valid'] = False
        aux['canonical_bev'] = {'enabled': True, 'scene_center': [1., 2., 3.], 'scene_scale': 4.}
    projected = prep.inference_record(row)
    assert set(projected['answer']) == {'type', 'options'}
    assert 'metric_aux' not in projected['refs']['simple3_aux']
    dataset = module.NativeAnswerPatchSpatialDataset(rows=[projected], max_visual_frames=1,
        scannet_frame_root=None, depth_grid_hw=(2, 2), reference_schema='simple3',
        simple3_aux_required=True, inference_only=True)
    sample = dataset[0]
    assert sample.target_text == ''
    assert sample.reference_type_id == aux['reference_type_id']
    torch.testing.assert_close(sample.query_pose, torch.eye(4))
    assert sample.aux_object_centers is None
    if module is spatial:
        assert sample.aux_metric_scene_scale is None
    # 训练入口仍要求真实答案；新增推理开关不降低训练数据的原检查。
    dataset.inference_only = False
    with pytest.raises((KeyError, ValueError)):
        dataset[0]


def test_direct_answering_without_training_labels(original):
    root, row = original
    projected = prep.inference_record(row)
    model = root / 'model'
    key, identity = basic.cache_identity_from_row(projected, scannet_frame_root=None,
        max_visual_frames=1, qwen_model_path=model, max_pixels=50176)
    cache_root = root / 'visual'
    cache = basic.visual_cache_path_for_key(cache_root, key)
    cache.parent.mkdir(parents=True)
    torch.save({'format_version': spatial.CACHE_FORMAT_VERSION, 'cache_key': key,
        'identity_text': identity, 'qwen_model_path': str(model), 'max_pixels': 50176,
        'max_visual_frames': 1, 'spatial_merge_size': 2,
        'qwen': {'image_grid_thw': torch.tensor([[1, 2, 2]]),
                 'image_embeds': torch.ones(1, 8), 'deepstack_features': [torch.ones(1, 8)]},
        'spatial': {'image_paths': projected['inputs']['image_paths']}}, cache)
    dataset = NativeDirectVisualCachedDataset(rows=[projected], visual_cache_root=cache_root,
        scannet_frame_root=None, max_visual_frames=1, qwen_model_path=model, max_pixels=50176,
        inference_only=True)
    sample = dataset[0]
    assert sample.target_text == ''
    assert sample.cached_image_embeds.shape == (1, 8)
    assert 'What is the size?' in sample.prompt
    dataset.inference_only = False
    with pytest.raises((KeyError, ValueError)):
        dataset[0]
    dataset.inference_only = True
    dataset.rows = [row]
    with pytest.raises(ValueError, match='answer supervision'):
        dataset[0]


@pytest.mark.parametrize('wrong_cache', [False, True])
def test_migrate_checks_old_identity_and_preserves_values(original, tmp_path, monkeypatch, wrong_cache):
    root, row = original
    old_model = root / 'model'
    old_key, old_text = spatial.cache_identity_from_row(row, scannet_frame_root=None,
        max_visual_frames=1, qwen_model_path=old_model, max_pixels=50176)
    cache_root = root / 'visual'
    cache = spatial.visual_cache_path_for_key(cache_root, old_key)
    cache.parent.mkdir(parents=True)
    payload = {'format_version': spatial.CACHE_FORMAT_VERSION, 'cache_key': old_key,
               'identity_text': 'wrong' if wrong_cache else old_text, 'qwen_model_path': str(old_model),
               'max_pixels': 50176, 'max_visual_frames': 1, 'spatial_merge_size': 1,
               'qwen': {'image_grid_thw': torch.tensor([[1, 2, 2]]),
                        'image_embeds': torch.arange(32).reshape(4, 8).to(torch.bfloat16),
                        'deepstack_features': [torch.arange(32).reshape(4, 8).to(torch.bfloat16)]},
               'spatial': {'image_paths': row['inputs']['image_paths'], 'points': torch.arange(12).reshape(4, 3).float()}}
    torch.save(payload, cache)
    source = root / 'records.jsonl'
    # 重复成员原样保留，不能在迁移时静默去重。
    source.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
    bindings = tmp_path / 'bindings.json'
    bindings.write_text(json.dumps([{'source_root': str(root), 'logical_root': 'source'}]))
    output = tmp_path / 'prepared'
    model = tmp_path / 'model'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--input-jsonl', str(source), '--output-root', str(output),
        '--bindings', str(bindings), '--historical-model-path', str(old_model), '--model-path', str(model),
        '--visual-cache-root', str(cache_root), '--max-visual-frames', '1', '--max-pixels', '50176'])
    if wrong_cache:
        with pytest.raises(ValueError, match='identity_text mismatch'):
            prep.main()
        assert not (output / 'PREPARATION.json').exists()
        return
    prep.main()
    rows = [json.loads(x) for x in (output / 'records.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[0] == rows[1]
    assert rows[0]['answer'] == row['answer']
    assert rows[0]['refs']['simple3_aux'] == row['refs']['simple3_aux']
    data = spatial.NativeAnswerPatchSpatialDataset(rows=rows, max_visual_frames=1,
        scannet_frame_root=None, depth_grid_hw=(2, 2), reference_schema='simple3', simple3_aux_required=True,
        visual_cache_root=output / 'visual_cache', qwen_model_path=model, max_pixels=50176)
    loaded = data[0]
    assert loaded.target_text == '7' and loaded.aux_metric_scene_scale.item() == 9.
    assert not loaded.patch_valid[0, 0, 1]
    assert torch.equal(loaded.cached_image_embeds, payload['qwen']['image_embeds'])
    assert (output / 'ID_MAP.tsv').read_text().endswith('2\tsource::first\t2\n')


@pytest.mark.parametrize('reverse_frames', [False, True])
def test_rebuilt_reference_uses_observation_pose_and_exact_rgb_order(original, tmp_path, monkeypatch, reverse_frames):
    root, row = original
    image2 = root / 'second.png'
    Image.new('RGB', (4, 4)).save(image2)
    row['inputs']['image_paths'].append(str(image2))
    row['refs']['simple3_aux']['target_camera']['target_frame_index'] = 1
    source = root / 'inference.jsonl'
    source.write_text(json.dumps(prep.inference_record(row)) + '\n')
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, :3, 3] = [3., 4., 5.]
    frames = list(row['inputs']['image_paths'])
    if reverse_frames:
        frames.reverse()
    geometry = root / 'rebuilt.npz'
    np.savez(geometry, frame_image_paths=np.asarray(frames), camera_pose=poses,
             target_points=np.arange(24, dtype=np.float32).reshape(2, 4, 3),
             patch_grid_hw=np.array([[2, 2], [2, 2]]), token_mask=np.ones((2, 4), dtype=bool))
    output = tmp_path / 'rebuilt.jsonl'
    monkeypatch.setattr(sys, 'argv', ['bind', '--input-jsonl', str(source), '--geometry-npz', str(geometry),
        '--output-jsonl', str(output), '--max-visual-frames', '2'])
    if reverse_frames:
        with pytest.raises(ValueError, match='exact original RGB frame order'):
            rebuilt.main()
        assert not output.exists()
        return
    rebuilt.main()
    data = spatial.NativeAnswerPatchSpatialDataset(rows=[json.loads(output.read_text())], max_visual_frames=2,
        scannet_frame_root=None, depth_grid_hw=(2, 2), reference_schema='simple3',
        simple3_aux_required=True, inference_only=True)
    item = data[0]
    torch.testing.assert_close(item.query_pose, item.source_poses[1])
    assert item.query_pose[:3, 3].tolist() == [3., 4., 5.]
    assert item.aux_object_centers is None and item.target_text == ''


def test_scene_directory_migration_preserves_finite_frame_selection(original, tmp_path, monkeypatch):
    root, row = original
    frames_root = root / 'scannet'
    scene_dir = frames_root / 'scene'
    for name in ['color', 'depth', 'pose', 'intrinsic']:
        (scene_dir / name).mkdir(parents=True)
    np.savetxt(scene_dir / 'intrinsic/intrinsic_depth.txt', np.eye(4))
    for n in range(4):
        Image.new('RGB', (4, 4)).save(scene_dir / 'color' / f'{n}.jpg')
        Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(scene_dir / 'depth' / f'{n}.png')
        pose = np.eye(4)
        if n == 1:
            pose[0, 0] = np.nan
        np.savetxt(scene_dir / 'pose' / f'{n}.txt', pose)
    row['refs']['geometry'].pop('cut3r_npz_path')
    old_model = root / 'model'
    old_key, old_text = spatial.cache_identity_from_row(row, scannet_frame_root=frames_root,
        max_visual_frames=2, qwen_model_path=old_model, max_pixels=50176)
    cache_root = root / 'visual'
    cache = spatial.visual_cache_path_for_key(cache_root, old_key)
    cache.parent.mkdir(parents=True)
    payload = {'format_version': spatial.CACHE_FORMAT_VERSION, 'cache_key': old_key,
        'identity_text': old_text, 'qwen_model_path': str(old_model), 'max_pixels': 50176,
        'max_visual_frames': 2, 'spatial': {'image_paths': [str(scene_dir / 'color/0.jpg'), str(scene_dir / 'color/3.jpg')]}}
    torch.save(payload, cache)
    source = root / 'records.jsonl'
    source.write_text(json.dumps(row) + '\n')
    bindings = tmp_path / 'bindings.json'
    bindings.write_text(json.dumps([{'source_root': str(root), 'logical_root': 'source'}]))
    output = tmp_path / 'prepared'
    mapped_frames = output / 'assets/source/scannet'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--input-jsonl', str(source), '--output-root', str(output),
        '--bindings', str(bindings), '--historical-model-path', str(old_model), '--model-path', str(tmp_path / 'model'),
        '--visual-cache-root', str(cache_root), '--max-visual-frames', '2', '--max-pixels', '50176',
        '--historical-scannet-frame-root', str(frames_root), '--scannet-frame-root', str(mapped_frames)])
    prep.main()
    data = spatial.NativeAnswerPatchSpatialDataset(rows=[json.loads((output / 'records.jsonl').read_text())],
        max_visual_frames=2, scannet_frame_root=mapped_frames, depth_grid_hw=(2, 2),
        reference_schema='simple3', simple3_aux_required=True)
    item = data[0]
    assert [Path(p).name for p in item.image_paths] == ['0.jpg', '3.jpg']
    assert len(list((mapped_frames / 'scene/color').glob('*.jpg'))) == 4
    assert (mapped_frames / 'scene/pose/1.txt').read_bytes() == (scene_dir / 'pose/1.txt').read_bytes()
