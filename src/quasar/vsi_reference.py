"""Locate the station objects in public RGB frames, then compute an estimated reference from this run's CUT3R points; answers are not read."""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

VERSION = 'public-vsi-reference-v1'


class ReferenceInputBlocked(ValueError):
    """题意、可见物体或坐标证据不足；调用者应保留原因并阻塞相应整格。"""


def parse_station(text, qtype):
    """仅截取题首的站位约束，问的目标物体、路线答案和选项不送定位器。"""
    if qtype.startswith('object_rel_direction'):
        match = re.match(r'^If I am standing by (.+?) and facing (.+?),', text)
        mode = 'facing'
    elif qtype == 'route_planning':
        match = re.match(r'^You are a robot beginning (?:at|by) (.+?)(?: and)? facing (.+?)\. You want', text)
        mode = 'facing'
        if match is None:
            match = re.match(r'^You are a robot beginning at (.+?), with your back to (.+?)\. You want', text)
            mode = 'back_to'
    else:
        return None
    if match is None:
        raise ReferenceInputBlocked('Unrecognized station expression')
    return dict(mode=mode, station=match[1], orientation_object=match[2])


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Semantic pose must be a finite 4x4 matrix')
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Invalid homogeneous pose row')
    rot = pose[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rot), 1, atol=1e-4):
        raise ValueError('Semantic pose must have a proper rigid rotation')
    return pose


def object_points(anchor, geometry, selected_indices, *, minimum):
    """把定位框中央区域映射到同一张CUT3R裁剪图的网格；不猜缺失深度。"""
    if set(anchor) != {'frame', 'bbox'} or type(anchor['frame']) is not int or not 0 <= anchor['frame'] < 32:
        raise ReferenceInputBlocked('Invalid locator frame/schema')
    box = np.asarray(anchor['bbox'], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1000):
        raise ReferenceInputBlocked('Invalid normalized bbox')
    x1, y1, x2, y2 = box/1000
    if x1 >= x2 or y1 >= y2:
        raise ReferenceInputBlocked('Empty locator bbox')
    index = selected_indices[anchor['frame']]
    h, w = geometry['patch_grid_hw'][index]
    x, y = np.meshgrid((np.arange(w)+0.5)/w, (np.arange(h)+0.5)/h)
    mask = (x >= x1+(x2-x1)*0.25) & (x <= x2-(x2-x1)*0.25)
    mask &= (y >= y1+(y2-y1)*0.25) & (y <= y2-(y2-y1)*0.25)
    points = geometry['target_points'][index]
    valid = geometry['token_mask'][index] & np.isfinite(points).all(-1)
    indices = np.flatnonzero(mask.ravel() & valid)
    if len(indices) < minimum:
        raise ReferenceInputBlocked(f'Insufficient object patches: {len(indices)} < {minimum}')
    selected = points[indices].astype(float)
    evidence = dict(frame=anchor['frame'], geometry_index=int(index), bbox=anchor['bbox'],
        patch_indices=indices.tolist(), points=selected.tolist(), median=np.median(selected, axis=0).tolist())
    return selected, evidence


def compute_reference(station, located, geometry, selected_indices):
    """构造本次重建中的站位与朝向，并保留能独立重算的中间数值。"""
    expected = {'status', 'reason', 'station', 'orientation'}
    if set(located) != expected or located['status'] != 'unique':
        raise ReferenceInputBlocked('Locator did not identify unique anchors: '+str(located.get('reason', located)))
    points, start = object_points(located['station'], geometry, selected_indices, minimum=3)
    position = np.median(points, axis=0)
    cameras = geometry['camera_pose'][selected_indices].astype(float)
    ups = -cameras[:, :3, 1]
    mean_up = ups.mean(0)
    concentration = float(np.linalg.norm(mean_up))
    if not np.isfinite(concentration) or concentration < 0.8:
        raise ReferenceInputBlocked('Camera-up estimate is inconsistent')
    up = mean_up/concentration
    face_points, face = object_points(located['orientation'], geometry, selected_indices,
                                      minimum=6 if station['mode'] == 'back_to' else 3)
    evidence = dict(station=start, orientation=face, up=up.tolist(), up_concentration=concentration)
    if station['mode'] == 'back_to':
        centered = face_points-face_points.mean(0)
        _, values, axes = np.linalg.svd(centered, full_matrices=False)
        if values[1] <= 1e-5 or values[2]/values[1] > 0.2:
            raise ReferenceInputBlocked('Visible front face is not a supported plane')
        forward = axes[-1]
        to_camera = geometry['camera_pose'][face['geometry_index'], :3, 3]-face_points.mean(0)
        if abs(float(forward @ to_camera)) < 1e-5:
            raise ReferenceInputBlocked('Cannot orient front-face normal')
        if forward @ to_camera < 0:
            forward = -forward
        evidence.update(plane_singular_values=values.tolist(), outward_normal=forward.tolist())
    else:
        forward = np.median(face_points, axis=0)-position
    forward = forward-up*(forward @ up)
    if np.linalg.norm(forward) <= 1e-5:
        raise ReferenceInputBlocked('Station and heading do not define a horizontal direction')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = position
    validate_pose(pose)
    return pose.tolist(), evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl', type=Path, required=True, help='Only source_row/sample_id/video_id/question_text/question_type.')
    p.add_argument('--geometry-root', type=Path, required=True)
    p.add_argument('--qwen-model-path', type=Path, required=True)
    p.add_argument('--cut3r-repo', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(l) for l in a.input_jsonl.read_text().splitlines()]
    allowed = {'source_row', 'sample_id', 'video_id', 'question_text', 'question_type'}
    if any(set(r) != allowed for r in rows):
        raise ValueError('Locator input has unsupported or answer fields')
    a.output_root.mkdir(parents=True, exist_ok=True)
    from quasar.reviewer import write_jsonl
    write_jsonl(a.output_root/'INPUTS.jsonl', rows)
    from quasar.public_vsi_base import REVISION
    from quasar.reviewer_data import verify_backbone
    verify_backbone(a.qwen_model_path, REVISION)
    sys.path.insert(0, str(a.cut3r_repo))
    from src.dust3r.utils.image import load_images
    from PIL import Image
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = None
    references, outcomes = [], []
    for row in rows:
        station = parse_station(row['question_text'], row['question_type'])
        if station is None:
            continue
        folder = a.output_root/str(row['source_row'])
        folder.mkdir(exist_ok=True)
        geometry_path = a.geometry_root/'video_npz'/(row['video_id']+'.npz')
        # 只载入定位需要的原始点与相机；不载入训练对象或监督目标。
        with np.load(geometry_path, allow_pickle=False) as archive:
            names = ['video_id', 'frame_image_paths', 'frame_ids', 'target_points', 'token_mask', 'camera_pose', 'patch_grid_hw']
            geometry = {k: archive[k] for k in names}
        if str(geometry['video_id']) != row['video_id']:
            raise ValueError('Reference geometry belongs to another video')
        indices = torch.linspace(0, len(geometry['frame_ids'])-1, steps=32).round().long().numpy()
        paths = geometry['frame_image_paths'][indices].tolist()
        views = load_images(paths, size=512, verbose=False)
        crops = []
        for i, view in enumerate(views):
            pixels = ((view['img'][0].permute(1, 2, 0).numpy()*0.5+0.5)*255).round().clip(0, 255).astype('uint8')
            h, w = geometry['patch_grid_hw'][indices[i]]
            if pixels.shape[:2] != (int(h)*16, int(w)*16):
                raise ValueError('Locator crop does not match CUT3R patch-grid coordinates')
            path = folder/f'crop_{i:02d}.png'
            if not path.exists():
                Image.fromarray(pixels).save(path)
            crops.append(str(path))
        prompt = ('Locate ONLY the two reference objects in these 32 chronological images, numbered 0 to 31. '
            'Do not answer any spatial benchmark question. Station object: '+station['station']+'. '
            'Orientation object: '+station['orientation_object']+'. '
            + ('For orientation give a rectangle entirely on the visible FRONT FACE of the orientation object, '
               'excluding side panels and surrounding objects. ' if station['mode']=='back_to' else '')
            + 'Choose one clear frame for each object. If the wording does not identify a unique physical object '
            'or it is not visible, set status to ambiguous or missing and explain. Return ONLY JSON: '
            '{"status":"unique|ambiguous|missing","reason":"brief identity evidence",'
            '"station":{"frame":0,"bbox":[x1,y1,x2,y2]},"orientation":{"frame":0,"bbox":[x1,y1,x2,y2]}}. '
            'All box coordinates are integers normalized to 0..1000 in that image.')
        write_jsonl(folder/'REQUEST.jsonl', [dict(protocol=VERSION, revision=REVISION, station=station,
                                               prompt=prompt, crops=crops, geometry=str(geometry_path), indices=indices.tolist())])
        raw_path = folder/'LOCATOR.txt'
        if not raw_path.exists():
            if model is None:
                processor = AutoProcessor.from_pretrained(a.qwen_model_path, local_files_only=True)
                model = Qwen3VLForConditionalGeneration.from_pretrained(a.qwen_model_path,
                    local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
                model.eval().requires_grad_(False).to('cuda:0')
            content = []
            for i, path in enumerate(crops):
                content.extend([dict(type='text', text=f'Image {i}:'), dict(type='image', image=path, max_pixels=262144)])
            content.append(dict(type='text', text=prompt))
            messages = [dict(role='user', content=content)]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            if videos is not None:
                raise ValueError('Locator expected independent images')
            batch = processor(text=[text], images=images, return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                output = model.generate(**batch, do_sample=False, max_new_tokens=1024)
            raw_path.write_text(processor.decode(output[0, batch['input_ids'].shape[1]:], skip_special_tokens=True))
        # 模型文本是开放格式输入；解析失败只记录该样例，不更换提示或模型重抽答案。
        raw = raw_path.read_text().strip()
        if raw.startswith('```json') and raw.endswith('```'):
            raw = raw[7:-3].strip()
        try:
            located = json.loads(raw)
            pose, evidence = compute_reference(station, located, geometry, indices)
        except (json.JSONDecodeError, ReferenceInputBlocked) as error:
            result = dict(source_row=row['source_row'], sample_id=row['sample_id'], status='BLOCKED_INPUT', reason=str(error))
        else:
            reference = dict(source_row=row['source_row'], sample_id=row['sample_id'],
                geometry_gauge='cut3r_reconstruction:'+row['video_id'], source=VERSION+':estimated_rgb_surface_anchors', pose_4x4=pose)
            references.append(reference)
            result = dict(source_row=row['source_row'], sample_id=row['sample_id'], status='ESTIMATED',
                          station_expression=station, reference=reference, arithmetic=evidence)
        (folder/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
        outcomes.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in {'reference', 'arithmetic'}}), flush=True)
    write_jsonl(a.output_root/'references.jsonl', references)
    (a.output_root/'STATUS.json').write_text(json.dumps(dict(protocol=VERSION,
        status='BLOCKED_INPUT' if any(r['status']=='BLOCKED_INPUT' for r in outcomes) else 'ESTIMATED',
        independently_reviewed=False, cases=outcomes), indent=2)+'\n')


if __name__ == '__main__':
    main()
