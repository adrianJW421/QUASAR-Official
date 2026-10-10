"""Convert a sourced station position and heading into the coordinates of the same video reconstruction and write an answer-free reference file."""
import argparse
import json
from pathlib import Path

import numpy as np

from quasar.export_observation_geometry import load_frame_rows_by_video
from quasar.vsi_reference import validate_pose


def construct_reference(definition, frame_rows, geometry):
    """只做明确给定的坐标转换；物体是谁、人与物体的偏移和坐标对齐必须由来源提供。"""
    required = {'source_row', 'sample_id', 'video_id', 'source', 'kind', 'source_gauge',
                'geometry_gauge', 'source_units', 'geometry_units', 'frame_ids',
                'station_point', 'station_offset', 'facing_point', 'up', 'source_to_geometry'}
    if set(definition) != required:
        raise ValueError('Reference definition has missing or unsupported fields; answers are not accepted')
    if definition['kind'] not in {'inference_reference', 'synthetic_teaching'}:
        raise ValueError('Reference kind must state inference_reference or synthetic_teaching')
    if type(definition['source_row']) is not int or definition['source_row'] <= 0:
        raise ValueError('source_row must be the positive original QA occurrence')
    for key in ['sample_id', 'video_id', 'source', 'source_gauge', 'source_units', 'geometry_units']:
        if not isinstance(definition[key], str) or not definition[key].strip():
            raise ValueError('Missing reference provenance: ' + key)
    video = definition['video_id']
    if definition['geometry_gauge'] != 'cut3r_reconstruction:' + video:
        raise ValueError('Reference must target this video reconstruction coordinate gauge')
    if not frame_rows or any(row['video_id'] != video for row in frame_rows):
        raise ValueError('Reference frame manifest belongs to another video')
    if definition['frame_ids'] != [row['frame_id'] for row in frame_rows]:
        raise ValueError('Reference frame order differs from the observation manifest')
    if str(geometry['video_id'].item()) != video:
        raise ValueError('Reference geometry belongs to another video')
    for key, field in [('frame_candidate_ids', 'frame_candidate_id'), ('frame_image_paths', 'frame_image_path')]:
        if geometry[key].tolist() != [str(row[field]) for row in frame_rows]:
            raise ValueError('Geometry frame binding differs from the observation manifest')
    transform = definition['source_to_geometry']
    if not isinstance(transform, dict) or set(transform) != {'scale', 'rotation', 'translation', 'source'}:
        raise ValueError('A sourced similarity transform is required, including an explicit identity transform')
    if not isinstance(transform['source'], str) or not transform['source'].strip():
        raise ValueError('Missing coordinate alignment source')
    scale = float(transform['scale'])
    rotation = np.asarray(transform['rotation'], dtype=float)
    translation = np.asarray(transform['translation'], dtype=float)
    if not np.isfinite(scale) or scale <= 0 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError('Invalid coordinate transform shape or scale')
    rigid = np.eye(4)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = translation
    validate_pose(rigid)
    if definition['source_gauge'] == definition['geometry_gauge']:
        if scale != 1 or not np.array_equal(rigid, np.eye(4)) or definition['source_units'] != definition['geometry_units']:
            raise ValueError('Same coordinate gauge requires identical units and an identity transform')
    vectors = {}
    for key in ['station_point', 'station_offset', 'facing_point', 'up']:
        vector = np.asarray(definition[key], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError('Expected a finite three-coordinate vector: ' + key)
        vectors[key] = vector
    # 点要旋转、缩放、平移；方向只旋转。偏移不能默认为零，否则会把物体中心当作人的站位。
    position = scale * (rotation @ (vectors['station_point'] + vectors['station_offset'])) + translation
    target = scale * (rotation @ vectors['facing_point']) + translation
    up = rotation @ vectors['up']
    if np.linalg.norm(up) <= 1e-8:
        raise ValueError('Up direction is zero')
    up /= np.linalg.norm(up)
    forward = target - position
    forward -= up * np.dot(forward, up)
    if np.linalg.norm(forward) <= 1e-8:
        raise ValueError('Facing point cannot define a horizontal forward direction')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = position
    validate_pose(pose)
    reference = {key: definition[key] for key in ['source_row', 'sample_id', 'geometry_gauge']}
    reference.update(source=definition['kind'] + ': ' + definition['source'], pose_4x4=pose.tolist())
    audit = dict(kind=definition['kind'], source_gauge=definition['source_gauge'],
                 geometry_gauge=definition['geometry_gauge'], geometry_units=definition['geometry_units'],
                 frame_ids=definition['frame_ids'], source_to_geometry=transform,
                 position=position.tolist(), facing_point=target.tolist(), up=up.tolist(),
                 axes='pose columns: right, down, forward; camera-to-observation transform',
                 establishes_real_object_identity=False)
    return reference, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--definition', type=Path, required=True)
    parser.add_argument('--frame-manifest', type=Path, required=True)
    parser.add_argument('--geometry-npz', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.output_jsonl.with_suffix('.audit.json')
    if args.output_jsonl.exists() or audit_path.exists():
        raise FileExistsError('Reference output already exists')
    definition = json.loads(args.definition.read_text())
    frames = load_frame_rows_by_video(args.frame_manifest)[definition['video_id']]
    with np.load(args.geometry_npz, allow_pickle=False) as geometry:
        reference, audit = construct_reference(definition, frames, geometry)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(json.dumps(reference, ensure_ascii=False) + '\n')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'reference': str(args.output_jsonl), 'kind': definition['kind']}))


if __name__ == '__main__':
    main()
