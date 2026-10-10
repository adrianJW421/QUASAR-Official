"""Bind a new CUT3R estimate of the same RGB frames to the target frame originally named by the question."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quasar.prepare_release_data import inference_record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl', type=Path, required=True)
    p.add_argument('--geometry-npz', type=Path, required=True)
    p.add_argument('--output-jsonl', type=Path, required=True)
    p.add_argument('--max-visual-frames', type=int, required=True)
    a = p.parse_args()
    if a.output_jsonl.exists():
        raise FileExistsError(a.output_jsonl)
    rows = [json.loads(line) for line in a.input_jsonl.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError('Rebuild input must contain the frozen single target-camera item')
    row = inference_record(rows[0])
    aux = row['refs']['simple3_aux']
    if aux['reference_type'] != 'target_camera_pose':
        raise ValueError('Rebuild requires an explicit target_camera_pose reference')
    index = aux['target_camera'].get('target_frame_index')
    if not isinstance(index, int) or not 0 <= index < a.max_visual_frames:
        raise ValueError('Missing or invalid original target_frame_index')
    with np.load(a.geometry_npz, allow_pickle=False) as geometry:
        frames = [str(x) for x in geometry['frame_image_paths'].tolist()]
        poses = np.asarray(geometry['camera_pose'], dtype=np.float32)
    if len(frames) != a.max_visual_frames or frames != row['inputs']['image_paths']:
        raise ValueError('Rebuilt geometry must use the exact original RGB frame order')
    if poses.shape != (a.max_visual_frames, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('Rebuilt camera poses must be finite [F,4,4]')
    # 目标帧号不变；参照姿态改用该帧的新估计，让观测与参照共用同一坐标系。
    # 旧姿态、金标、对象库和训练目标均不参与这一步。
    pose = poses[index].tolist()
    aux.update(query_pose_4x4=pose, query_pose_valid=True, query_pose_source='rebuilt_cut3r_target_frame')
    aux['target_camera'].update(pose_4x4=pose, pose_valid=True, source='rebuilt_cut3r_target_frame')
    row['refs']['geometry']['cut3r_npz_path'] = str(a.geometry_npz.resolve())
    a.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    a.output_jsonl.write_text(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'sample_id': row['sample_id'], 'target_frame_index': index, 'frames': len(frames)}))


if __name__ == '__main__':
    main()
