"""Prepare relocatable data from the original local records, keeping order, targets and cached numerical values."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def inference_record(row: dict[str, Any]) -> dict[str, Any]:
    """只取作答需要的字段；答案、对象库及训练目标另存，不交给推理进程。"""
    aux = row['refs']['simple3_aux']
    reference = {k: copy.deepcopy(aux[k]) for k in [
        'format_version', 'reference_schema', 'reference_type', 'reference_type_id',
        'query_pose_valid', 'query_pose_4x4', 'query_pose_source',
    ] if k in aux}
    for name, keys in {
        'target_camera': ['enabled', 'pose_valid', 'pose_4x4', 'target_frame_index', 'frame_index', 'frame_id', 'source'],
        'semantic_query': ['enabled', 'pose_valid', 'pose_4x4', 'source', 'failure_reason'],
        'canonical_bev': ['enabled', 'pose_valid', 'canonical_pose_4x4', 'scene_center', 'scene_scale'],
    }.items():
        if name in aux:
            reference[name] = {k: copy.deepcopy(aux[name][k]) for k in keys if k in aux[name]}
    for k in ['provided_pose_protocol_version', 'reference_kind', 'pose_source', 'pose_status',
              'pose_input_mask', 'geometry_gauge', 'provenance', 'failure_reason']:
        if k in aux:
            reference[k] = copy.deepcopy(aux[k])
    answer = {k: copy.deepcopy(row['answer'][k]) for k in ['type', 'options', 'numeric_unit'] if k in row['answer']}
    return {
        'format_version': row['format_version'], 'sample_id': row['sample_id'],
        'source': {k: row['source'][k] for k in ['tag', 'dataset', 'benchmark', 'split', 'row_id'] if k in row['source']},
        'scene': copy.deepcopy(row['scene']), 'question': copy.deepcopy(row['question']),
        'answer': answer, 'inputs': copy.deepcopy(row['inputs']),
        'refs': {'geometry': copy.deepcopy(row['refs']['geometry']),
                 'pose': copy.deepcopy(row['refs']['pose']), 'simple3_aux': reference},
        'flags': {'is_legacy_mca': False},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--bindings', type=Path, required=True, help='JSON list of source_root and logical_root.')
    p.add_argument('--assets-mode', choices=['copy', 'link'], default='copy')
    p.add_argument('--historical-model-path', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--historical-scannet-frame-root', type=Path)
    p.add_argument('--scannet-frame-root', type=Path)
    p.add_argument('--visual-cache-root', type=Path, action='append', required=True)
    p.add_argument('--max-visual-frames', type=int, required=True)
    p.add_argument('--max-pixels', type=int, required=True)
    p.add_argument('--inference-only', action='store_true')
    a = p.parse_args()
    from quasar.train_native_qwen3vl2b_glow_patch_spatial_moe_metric import (
        CACHE_FORMAT_VERSION, cache_identity_from_row, find_visual_cache_path, visual_cache_path_for_key,
        frame_path_sort_key, scene_dir_has_min_frames,
    )
    import torch

    bindings = sorted(json.loads(a.bindings.read_text()), key=lambda x: len(x['source_root']), reverse=True)
    root = a.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_jsonl = root / ('inference.jsonl' if a.inference_only else 'records.jsonl')
    if output_jsonl.exists():
        raise FileExistsError(f'Use a new output version: {output_jsonl}')
    moved_files = {}
    moved_caches = {}
    occurrences = Counter()

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [relocate(v) for v in value]
        if not isinstance(value, str) or not Path(value).is_absolute():
            return value
        old = Path(value)
        for binding in bindings:
            source_root = Path(binding['source_root'])
            if old.is_relative_to(source_root):
                return str(root / 'assets' / binding['logical_root'] / old.relative_to(source_root))
        raise ValueError(f'No path binding for {value}')

    def materialize(text, *, rewrite_npz=False):
        if text in moved_files:
            return moved_files[text]
        source = Path(text)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = Path(relocate(text))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        if rewrite_npz:
            # NPZ内的帧路径也要迁移；点、姿态、mask及所有数值逐数组原样回读。
            with np.load(source, allow_pickle=True) as z:
                arrays = {k: z[k] for k in z.files}
            for k, arr in list(arrays.items()):
                if k == 'frame_image_paths':
                    for x in arr.tolist():
                        materialize(str(x))
                    arrays[k] = np.asarray([relocate(str(x)) for x in arr.tolist()])
                elif arr.dtype.kind in ['U', 'S']:
                    arrays[k] = np.asarray([relocate(str(x)) for x in arr.reshape(-1)]).reshape(arr.shape)
            np.savez_compressed(destination, **arrays)
            with np.load(source, allow_pickle=True) as before, np.load(destination, allow_pickle=True) as after:
                if before.files != after.files:
                    raise ValueError(f'NPZ fields changed: {source}')
                for k in before.files:
                    if before[k].dtype.kind not in ['U', 'S']:
                        if before[k].dtype != after[k].dtype or before[k].shape != after[k].shape or before[k].tobytes() != after[k].tobytes():
                            raise ValueError(f'Numerical NPZ field changed: {source}:{k}')
        elif a.assets_mode == 'link':
            destination.symlink_to(source.resolve())
        else:
            shutil.copyfile(source, destination)
            if source.read_bytes() != destination.read_bytes():
                raise ValueError(f'Copied file changed: {source}')
        moved_files[text] = str(destination)
        return str(destination)

    def compare_cache(old, new, field='cache'):
        if isinstance(old, torch.Tensor):
            if not isinstance(new, torch.Tensor) or old.dtype != new.dtype or old.shape != new.shape or not torch.equal(old, new):
                raise ValueError(f'Tensor changed: {field}')
        elif isinstance(old, np.ndarray):
            if old.dtype != new.dtype or old.shape != new.shape or old.tobytes() != new.tobytes():
                raise ValueError(f'Array changed: {field}')
        elif isinstance(old, dict):
            if old.keys() != new.keys():
                raise ValueError(f'Cache fields changed: {field}')
            for k in old:
                if k not in ['cache_key', 'identity_text', 'qwen_model_path']:
                    compare_cache(old[k], new[k], field + '.' + k)
        elif isinstance(old, (list, tuple)):
            if len(old) != len(new):
                raise ValueError(f'Cache list changed: {field}')
            for i, x in enumerate(old):
                compare_cache(x, new[i], field + f'[{i}]')
        elif isinstance(old, str) and Path(old).is_absolute():
            if relocate(old) != new:
                raise ValueError(f'Path mapping changed: {field}')
        elif old != new:
            raise ValueError(f'Cache value changed: {field}')

    with a.input_jsonl.open() as src, output_jsonl.open('w') as dst, (root / 'ID_MAP.tsv').open('w') as ids:
        writer = csv.writer(ids, delimiter='\t')
        writer.writerow(['order', 'sample_id', 'occurrence'])
        count = 0
        for count, line in enumerate(src, 1):
            old_row = json.loads(line)
            row = inference_record(old_row) if a.inference_only else copy.deepcopy(old_row)
            old_identity = cache_identity_from_row(old_row, scannet_frame_root=a.historical_scannet_frame_root,
                                                   max_visual_frames=a.max_visual_frames,
                                                   qwen_model_path=a.historical_model_path, max_pixels=a.max_pixels)
            if old_identity is None:
                raise ValueError(f'No original cache identity: {old_row["sample_id"]}')
            old_key, old_text = old_identity
            old_path = find_visual_cache_path(a.visual_cache_root, old_key)
            if old_path is None:
                raise FileNotFoundError(f'Historical visual cache missing: {old_key}')
            geometry = row['refs']['geometry']
            cut = geometry.get('cut3r_npz_path')
            if cut:
                materialize(cut, rewrite_npz=True)
            elif old_text.startswith('scannet_scene|'):
                if a.historical_scannet_frame_root is None or a.scannet_frame_root is None:
                    raise ValueError('Scene input requires both ScanNet frame roots')
                name = str(row['scene'].get('scene_name') or row['scene'].get('scene_id') or '').strip()
                scene_dir = a.historical_scannet_frame_root / name
                prepared_scene = a.scannet_frame_root / name
                if Path(relocate(str(scene_dir))) != prepared_scene:
                    raise ValueError(f'ScanNet root disagrees with path bindings: {name}')
                # 目录读取器按全场景有效帧取32帧。保留目录中的全部颜色、深度与姿态，
                # 才能保证迁移前后抽到相同的帧，而不是把选出的32帧误当成原始目录。
                color_paths = sorted((scene_dir / 'color').glob('*.jpg'), key=frame_path_sort_key)
                if not color_paths:
                    color_paths = sorted((scene_dir / 'color').glob('*.png'), key=frame_path_sort_key)
                for color_path in color_paths:
                    materialize(str(color_path))
                    for folder, suffix in [('depth', '.png'), ('pose', '.txt')]:
                        path = scene_dir / folder / (color_path.stem + suffix)
                        if path.is_file():
                            materialize(str(path))
                materialize(str(scene_dir / 'intrinsic' / 'intrinsic_depth.txt'))
                if not scene_dir_has_min_frames(prepared_scene, min_frames=a.max_visual_frames):
                    raise ValueError(f'Migrated ScanNet scene is incomplete: {prepared_scene}')
            else:
                for values in [row['inputs'].get('image_paths'), geometry.get('candidate_depth_paths'),
                               geometry.get('candidate_intrinsic_depth_paths'), row['refs']['pose'].get('candidate_pose_paths')]:
                    if not values:
                        raise ValueError(f'Incomplete explicit depth/pose input: {row["sample_id"]}')
                    for path in values:
                        materialize(path)
            aux = row['refs']['simple3_aux']
            if not a.inference_only and aux.get('aux_npz_path'):
                materialize(aux['aux_npz_path'])
            new_row = relocate(row)
            new_identity = cache_identity_from_row(new_row, scannet_frame_root=a.scannet_frame_root,
                                                   max_visual_frames=a.max_visual_frames,
                                                   qwen_model_path=a.model_path, max_pixels=a.max_pixels)
            if new_identity is None:
                raise ValueError(f'No migrated cache identity: {row["sample_id"]}')
            new_key, new_text = new_identity
            if old_key not in moved_caches:
                payload = torch.load(old_path, map_location='cpu', weights_only=False)
                for k, v in {'format_version': CACHE_FORMAT_VERSION, 'cache_key': old_key,
                             'identity_text': old_text, 'qwen_model_path': str(a.historical_model_path),
                             'max_pixels': a.max_pixels, 'max_visual_frames': a.max_visual_frames}.items():
                    if payload.get(k) != v:
                        raise ValueError(f'Historical cache {k} mismatch: {old_path}')
                # 先通过原身份检查，再重写路径和对应key；不删除或放宽加载器的检查。
                new_payload = {}
                for k, v in payload.items():
                    if k == 'spatial':
                        new_payload[k] = {name: relocate(data) if isinstance(data, (str, list)) else data for name, data in v.items()}
                    else:
                        new_payload[k] = v
                new_payload.update(cache_key=new_key, identity_text=new_text, qwen_model_path=str(a.model_path))
                target = visual_cache_path_for_key(root / 'visual_cache', new_key)
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(new_payload, target)
                readback = torch.load(target, map_location='cpu', weights_only=False)
                compare_cache(payload, readback)
                moved_caches[old_key] = {'original': str(old_path), 'prepared': str(target), 'new_key': new_key}
            elif moved_caches[old_key]['new_key'] != new_key:
                raise ValueError(f'One old cache mapped to different new keys: {old_key}')
            dst.write(json.dumps(new_row, ensure_ascii=False, separators=(',', ':')) + '\n')
            occurrences[row['sample_id']] += 1
            writer.writerow([count, row['sample_id'], occurrences[row['sample_id']]])
            if json.loads(json.dumps(new_row)) != relocate(row):
                raise ValueError(f'JSON field changed: {row["sample_id"]}')
    if count == 0:
        raise ValueError('No input records')
    receipt = {'status': 'PREPARED', 'rows': count, 'records': str(output_jsonl),
               'assets_mode': a.assets_mode, 'inference_only': a.inference_only,
               'files': moved_files, 'visual_caches': moved_caches,
               'max_pixels': a.max_pixels, 'max_visual_frames': a.max_visual_frames,
               'model_path': str(a.model_path), 'scannet_frame_root': str(a.scannet_frame_root) if a.scannet_frame_root else None}
    (root / 'PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'rows': count, 'files': len(moved_files), 'caches': len(moved_caches)}))


if __name__ == '__main__':
    main()
