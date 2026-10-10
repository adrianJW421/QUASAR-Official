"""Combine public QA, original frames/depth and this run's reconstruction into answer-free batch inputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from quasar.export_observation_geometry import load_frame_rows_by_video, sanitize_filename
from quasar.train_native_qwen3vl2b_glow_patch_spatial import (
    compute_scene_center_scale, parse_target_frame_index, pose_from_xy_yaw,
)
from quasar.train_native_qwen3vl2b_glow_patch_spatial_moe_metric import (
    NativeAnswerPatchSpatialDataset, reference_type_mapping, infer_simple_reference_condition,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qa-jsonl', type=Path, required=True, help='The included licensed QA projection.')
    p.add_argument('--input-plan', type=Path, required=True, help='One source_row/sample_id and original-input binding per row.')
    p.add_argument('--frame-manifest', type=Path)
    p.add_argument('--geometry-root', type=Path, help='Output of this run of quasar-export-geometry.')
    p.add_argument('--scannet-frame-root', type=Path, help='Original-provider color/depth/pose/intrinsic export.')
    p.add_argument('--reference-jsonl', type=Path, help='Explicit semantic reference definitions with a documented source; never gold answers or object targets.')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--max-visual-frames', type=int, required=True)
    p.add_argument('--depth-grid-h', type=int, default=24)
    p.add_argument('--depth-grid-w', type=int, default=32)
    a = p.parse_args()
    if a.max_visual_frames <= 0:
        raise ValueError('max-visual-frames must be positive')
    plans = [json.loads(x) for x in a.input_plan.read_text().splitlines() if x.strip()]
    orders = [x['source_row'] for x in plans]
    if not orders or any(type(x) is not int or x <= 0 for x in orders) or orders != sorted(set(orders)):
        raise ValueError('Select distinct QA source rows in original member order; IDs alone are insufficient')
    wanted = set(orders)
    qa = {}
    with a.qa_jsonl.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['source_row'] in wanted:
                if row['source_row'] in qa:
                    raise ValueError('QA source_row must be unique')
                qa[row['source_row']] = row
    if set(qa) != wanted:
        raise ValueError('Missing selected QA rows')
    frames = load_frame_rows_by_video(a.frame_manifest) if a.frame_manifest is not None else {}
    references = {}
    if a.reference_jsonl is not None:
        for line in a.reference_jsonl.read_text().splitlines():
            ref = json.loads(line)
            # 参照定义是推理输入，不能夹带答案、对象库或训练目标。
            allowed = {'source_row', 'sample_id', 'geometry_gauge', 'source', 'pose_4x4',
                       'loc3r_position_xy', 'loc3r_orientation_rad'}
            if set(ref) - allowed:
                raise ValueError('Semantic reference contains unsupported or supervision fields')
            key = (ref['source_row'], ref['sample_id'])
            if key in references:
                raise ValueError('Duplicate semantic reference occurrence')
            references[key] = ref
    if a.output_root.exists():
        raise FileExistsError(a.output_root)
    loader = NativeAnswerPatchSpatialDataset(rows=[qa[x] for x in orders], max_visual_frames=a.max_visual_frames,
        scannet_frame_root=None, depth_grid_hw=(a.depth_grid_h, a.depth_grid_w),
        reference_schema='simple3', simple3_aux_required=True, inference_only=True)
    records = []
    labels = []
    for plan in plans:
        source = qa[plan['source_row']]
        sample_id = source['sample_id']
        if sample_id != plan['sample_id'] or source['video_id'] != plan['video_id']:
            raise ValueError('Input plan disagrees with QA source row, ID or video')
        reference_type = source['question']['reference_condition']
        if source['source_tag'] in {'vsti_official_row32', 'official_eval_legacy_vsi'}:
            # 主表题目保留原来的五类文字标记；空间模块沿历史规则映射为三类。
            reference_type = infer_simple_reference_condition(
                row={'source': {'tag': source['source_tag']}}, question_payload=source['question'])
        mapping = reference_type_mapping('simple3')
        if reference_type not in mapping:
            raise ValueError(f'Unsupported reference definition: {reference_type}')
        scene = {'scene_id': plan['scene_id'], 'scene_name': plan['scene_id'], 'video_id': plan['video_id']}
        geometry = {}
        poses = {}
        route = plan['geometry_source']
        if route == 'cut3r':
            if a.geometry_root is None or plan['video_id'] not in frames:
                raise ValueError('CUT3R input requires this run\'s frame manifest and geometry root')
            path = a.geometry_root / 'video_npz' / (sanitize_filename(plan['video_id']) + '.npz')
            image_paths, points, valid, confidence, camera_poses = loader._load_cut3r_npz(sample_id=sample_id, path=path)
            expected = frames[plan['video_id']]
            with np.load(path, allow_pickle=False) as z:
                if str(z['video_id'].tolist()) != plan['video_id']:
                    raise ValueError('Rebuilt geometry belongs to another video')
                if z['frame_candidate_ids'].tolist() != [x['frame_candidate_id'] for x in expected]:
                    raise ValueError('Rebuilt geometry candidate order mismatch')
            if image_paths != [x['frame_image_path'] for x in expected]:
                raise ValueError('Rebuilt geometry RGB order mismatch or old paths remain')
            if len(image_paths) > a.max_visual_frames:
                # 几何先沿完整历史序列计算，再按原加载器规则取模型的32帧。
                # 提前把几何输入缩短为32帧会改变CUT3R状态和坐标。
                keep = torch.linspace(0, len(image_paths)-1, steps=a.max_visual_frames).round().long()
                image_paths = [image_paths[i] for i in keep.tolist()]
                points, valid, confidence, camera_poses = points[keep], valid[keep], confidence[keep], camera_poses[keep]
            if 'model_frame_ids' in plan:
                import re
                actual = [int(re.search(r'frame_(\d+)', x).group(1)) for x in image_paths]
                if actual != plan['model_frame_ids']:
                    raise ValueError('Model-frame order differs from the historical record')
            geometry['cut3r_npz_path'] = str(path.resolve())
            gauge = 'cut3r_reconstruction:' + plan['video_id']
        elif route == 'scannet_depth_pose':
            if a.scannet_frame_root is None:
                raise ValueError('Depth/pose route requires the original ScanNet frame export')
            folder = a.scannet_frame_root / plan['scene_id']
            image_paths, points, valid, confidence, camera_poses = loader._load_scannet_scene_frames(
                sample_id=sample_id, scene_dir=folder)
            geometry['candidate_depth_paths'] = [str(folder / 'depth' / (Path(x).stem + '.png')) for x in image_paths]
            geometry['candidate_intrinsic_depth_paths'] = [str(folder / 'intrinsic/intrinsic_depth.txt')] * len(image_paths)
            poses['candidate_pose_paths'] = [str(folder / 'pose' / (Path(x).stem + '.txt')) for x in image_paths]
            gauge = 'scannet_world:' + plan['scene_id']
        else:
            raise ValueError(f'Unsupported raw geometry source: {route}')
        if len(image_paths) != a.max_visual_frames:
            raise ValueError('Public input requires the exact frozen frame count')
        if not bool(torch.isfinite(camera_poses).all()):
            raise ValueError('Source camera poses must be finite')
        if route == 'scannet_depth_pose' and [int(Path(x).stem) for x in image_paths] != plan['expected_frame_ids']:
            raise ValueError('ScanNet finite-pose frame selection differs from the member record')
        aux = {'reference_schema': 'simple3', 'reference_type': reference_type,
               'reference_type_id': mapping[reference_type]}
        if reference_type == 'target_camera_pose':
            index = plan['target_frame_index']
            if type(index) is not int or not 0 <= index < len(image_paths):
                raise ValueError('Missing or invalid target_frame_index')
            if index != parse_target_frame_index(source['question']['text'], max_frames=len(image_paths)):
                raise ValueError('Target frame index disagrees with the original question')
            pose = camera_poses[index].tolist()
            aux.update(query_pose_4x4=pose, query_pose_valid=True, query_pose_source='rebuilt_' + route + '_target_frame')
            aux['target_camera'] = {'enabled': True, 'pose_valid': True, 'pose_4x4': pose,
                                    'target_frame_index': index, 'source': aux['query_pose_source']}
        elif reference_type == 'canonical_bev_scene_pose':
            # 沿现有场景中心/尺度定义；这里的姿态是定义出的场景参照，不是估计一个人的站位。
            center, scale = compute_scene_center_scale(points=points, valid=valid)
            pose = torch.eye(4, dtype=torch.float32)
            pose[:3, 3] = center
            aux.update(query_pose_4x4=pose.tolist(), query_pose_valid=False,
                       query_pose_source='rebuilt_scene_points_center_scale')
            aux['canonical_bev'] = {'enabled': True, 'pose_valid': False,
                'canonical_pose_4x4': pose.tolist(), 'scene_center': center.tolist(), 'scene_scale': float(scale)}
        else:
            ref = references.get((source['source_row'], sample_id))
            if ref is None:
                raise ValueError(f'Missing sourced semantic reference definition: {sample_id}')
            if not ref['source'] or ref['geometry_gauge'] != gauge:
                raise ValueError('Semantic reference provenance or coordinate gauge mismatch')
            if 'pose_4x4' in ref:
                pose = torch.tensor(ref['pose_4x4'], dtype=torch.float32)
            else:
                if source['source_tag'] != 'sqa3d_loc3r' or route != 'scannet_depth_pose':
                    raise ValueError('Loc3R XY/yaw definition is only supported with original ScanNet poses')
                xy = ref['loc3r_position_xy']
                if len(xy) != 2:
                    raise ValueError('Loc3R position requires two coordinates')
                pose = pose_from_xy_yaw(float(xy[0]), float(xy[1]), float(ref['loc3r_orientation_rad']))
            if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all()):
                raise ValueError('Semantic reference pose must be finite [4,4]')
            from quasar.vsi_reference import validate_pose
            validate_pose(pose.numpy())
            aux.update(query_pose_4x4=pose.tolist(), query_pose_valid=True, query_pose_source=ref['source'])
            aux['semantic_query'] = {'enabled': True, 'pose_valid': True, 'pose_4x4': pose.tolist(), 'source': ref['source']}
        # 从QA投影取字段，不从私有native记录搬运；金标只写独立评分文件。
        records.append({'format_version': 'glow_native_answer_spatial_qa_v1', 'sample_id': sample_id,
            'source': {'tag': source['source_tag'], 'dataset': source['source_dataset'], 'row_id': source['source_row']},
            'scene': scene, 'question': copy.deepcopy(source['question']),
            'answer': {k: copy.deepcopy(source['answer'][k]) for k in ['type', 'options', 'numeric_unit'] if k in source['answer']},
            'inputs': {'image_paths': image_paths}, 'flags': {'is_legacy_mca': False},
            'refs': {'geometry': geometry, 'pose': poses, 'simple3_aux': aux}})
        labels.append({'source_row': source['source_row'], 'sample_id': sample_id, 'answer': source['answer']})
    # 在写入前用实际读取器回读。缺参照、缺画面或字段不合规格都不能产出看似成功的数据。
    check = NativeAnswerPatchSpatialDataset(rows=records, max_visual_frames=a.max_visual_frames,
        scannet_frame_root=None, depth_grid_hw=(a.depth_grid_h, a.depth_grid_w),
        reference_schema='simple3', simple3_aux_required=True, inference_only=True)
    for i in range(len(check)):
        sample = check[i]
        if sample.target_text or sample.aux_metric_scene_scale is not None or sample.aux_object_centers is not None:
            raise ValueError('Gold or object supervision reached inference')
    a.output_root.mkdir(parents=True)
    for name, rows in [('inference.jsonl', records), ('labels.jsonl', labels)]:
        # 完整写好后再公开文件名，避免把中断留下的半份输入当成可续跑结果。
        partial = a.output_root / (name+'.partial')
        partial.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
        partial.replace(a.output_root / name)
    print(json.dumps({'records': len(records), 'supervision_separate': True, 'historical_targets_regenerated': False}))


if __name__ == '__main__':
    main()
