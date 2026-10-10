"""从新提取的原视频画面生成纯视觉特征，供Base和Visual-only评测使用。"""
import argparse
import json
from pathlib import Path

import torch

from quasar.export_visual_cache import load_processor_and_qwen_for_cache, encode_qwen_images, cache_path_for_key, CACHE_FORMAT_VERSION
from quasar.train_native_qwen3vl2b_glow_patch_spatial import cache_identity_from_row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--qwen-model-path', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    a.qwen_cache_dir = None
    a.max_pixels = 50176
    a.max_visual_frames = 32
    rows = [json.loads(x) for x in a.input_jsonl.read_text().splitlines()]
    jobs = {}
    for row in rows:
        if set(row['answer']) - {'type', 'options', 'numeric_unit'}:
            raise ValueError('Gold answer in RGB inference input')
        paths = row['inputs']['image_paths']
        if len(paths) != 32 or any(not Path(x).is_file() for x in paths):
            raise ValueError('RGB input requires the complete ordered 32-frame set')
        if row['refs'] != {'geometry': {}, 'pose': {}}:
            raise ValueError('RGB export takes no geometry, references or training targets')
        key, identity = cache_identity_from_row(row, scannet_frame_root=None, max_visual_frames=32,
                                                qwen_model_path=a.qwen_model_path, max_pixels=50176)
        jobs[key] = (identity, paths)
    a.output_root.mkdir(parents=True, exist_ok=True)
    pending = [key for key in jobs if not cache_path_for_key(a.output_root, key).exists()]
    # 续跑已有完整特征时，不重新加载整个语言模型。
    if pending:
        processor, model, merge, meta = load_processor_and_qwen_for_cache(a)
        model.to(a.device)
    else:
        meta = {'all_features_reused': True}
    receipts = []
    for key, (identity, paths) in jobs.items():
        path = cache_path_for_key(a.output_root, key)
        reused = path.exists()
        if reused:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['identity_text'] != identity or payload['spatial']['image_paths'] != paths:
                raise ValueError('Existing RGB features belong to different input')
        else:
            encoded = encode_qwen_images(processor=processor, qwen=model, image_paths=paths,
                                         max_pixels=50176, device=torch.device(a.device))
            payload = dict(format_version=CACHE_FORMAT_VERSION, cache_key=key, identity_text=identity,
                           qwen_model_path=str(a.qwen_model_path), max_pixels=50176, max_visual_frames=32,
                           spatial_merge_size=merge, qwen=encoded, spatial={'image_paths': paths})
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            torch.save(payload, tmp)
            tmp.replace(path)
        # Base原入口读取独立的RGB字段，不能从这里取得几何或答案。
        base_path = path.with_name(path.stem + '.rgb.pt')
        if not base_path.exists():
            tmp = base_path.with_suffix('.tmp')
            torch.save(payload['qwen'], tmp)
            tmp.replace(base_path)
        receipts.append({'key': key, 'reused': reused, 'frames': len(paths), 'path': str(path)})
    (a.output_root/'RGB_COMPLETE.json').write_text(json.dumps({'jobs': receipts, 'model': meta}, indent=2)+'\n')


if __name__ == '__main__':
    main()
