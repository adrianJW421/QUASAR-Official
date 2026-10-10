"""Extract frames from the original video with the recorded 32-frame rule; model features are neither read nor created."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image

from quasar.export_observation_geometry import sanitize_filename


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-plan', type=Path, required=True,
                   help='JSONL: video_id, video_path, expected_frame_ids, image_long_edge.')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--num-frames', type=int, required=True)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    a = p.parse_args()
    if a.num_frames <= 0:
        raise ValueError('num-frames must be positive')
    plans = [json.loads(x) for x in a.video_plan.read_text().splitlines() if x.strip()]
    if not plans or len({x['video_id'] for x in plans}) != len(plans):
        raise ValueError('Video plan must be nonempty with one row per video')
    if a.output_root.exists():
        raise FileExistsError(a.output_root)
    a.output_root.mkdir(parents=True)
    frames = []
    receipts = []
    for plan in plans:
        path = Path(plan['video_path'])
        if not path.is_file():
            raise FileNotFoundError(path)
        probe = subprocess.run([a.ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,avg_frame_rate', '-of', 'json', str(path)],
            text=True, capture_output=True, check=True)
        stream = json.loads(probe.stdout)['streams'][0]
        count = int(stream['nb_frames'])
        if count < a.num_frames:
            raise ValueError(f'Video has only {count} real frames: {path}')
        # 沿原生产程序逐项round的均匀取帧规则；与已登记帧号不符就停下，不能换一批帧。
        indices = [0] if a.num_frames == 1 else [
            int(round(i * (count - 1) / (a.num_frames - 1))) for i in range(a.num_frames)]
        if plan.get('selection') == 'recorded_model_input':
            # VSI历史运行使用固定的选帧清单；不能换成均匀32帧。
            if count != plan['expected_total_frames']:
                raise ValueError(f'Original video version/frame count changed: {path}')
            indices = plan['expected_frame_ids']
            if len(indices) != a.num_frames or indices != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= count:
                raise ValueError('Recorded input must contain distinct ordered native frame numbers')
        if indices != plan['expected_frame_ids']:
            raise ValueError(f'Frame selection disagrees with the supplied member record: {plan["video_id"]}')
        edge = int(plan['image_long_edge'])
        if edge <= 0:
            raise ValueError('image_long_edge must be positive')
        video = str(plan['video_id'])
        folder = a.output_root / 'frames' / sanitize_filename(video)
        folder.mkdir(parents=True, exist_ok=False)
        select = 'select=' + '+'.join(f'eq(n\\,{i})' for i in indices)
        # JPEG编码沿历史命令使用默认线程设置；显式改线程会改变自动量化及最终像素。
        command = [a.ffmpeg, '-hide_banner', '-loglevel', 'error',
            '-i', str(path), '-vf', select, '-vsync', '0',
            str(folder / 'selected_%06d.jpg')]
        subprocess.run(command, check=True)
        extracted = sorted(folder.glob('selected_*.jpg'))
        if len(extracted) != len(indices):
            raise ValueError(f'Extracted {len(extracted)} frames, expected {len(indices)}')
        for order, (index, image_path) in enumerate(zip(indices, extracted)):
            target = folder / f'{sanitize_filename(video)}__frame_{index:06d}.jpg'
            # JPEG质量和长边缩放沿原帧导出程序，不让取帧入口静默改变模型所见的画面。
            with Image.open(image_path) as image:
                rgb = image.convert('RGB')
                width, height = rgb.size
                if max(width, height) > edge:
                    factor = edge / max(width, height)
                    rgb = rgb.resize((max(1, round(width * factor)), max(1, round(height * factor))),
                                     Image.Resampling.LANCZOS)
                rgb.save(target, quality=95)
            image_path.unlink()
            frames.append({'video_id': video, 'frame_candidate_id': f'{video}__frame_{index:06d}',
                'frame_id': index, 'sample_order': order, 'embedding_row_index': order,
                'frame_image_path': str(target.resolve())})
        receipts.append({'video_id': video, 'video_path': str(path), 'total_frames': count,
            'selected_frame_ids': indices, 'image_long_edge': edge, 'command': command})
    partial_manifest = a.output_root / 'video_frame_manifest.jsonl.partial'
    partial_manifest.write_text(
        ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in frames))
    (a.output_root / 'extraction.json').write_text(json.dumps(receipts, indent=2) + '\n')
    partial_manifest.replace(a.output_root / 'video_frame_manifest.jsonl')
    print(json.dumps({'videos': len(plans), 'frames': len(frames)}))


if __name__ == '__main__':
    main()
