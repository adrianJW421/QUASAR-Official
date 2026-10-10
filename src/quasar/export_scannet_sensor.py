"""Export original ScanNet scans with the checked historical image size, frame interval and text precision."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensor-reader', type=Path, required=True, help='Authorized upstream SensorData.py')
    parser.add_argument('--sens', type=Path, required=True)
    parser.add_argument('--scene-id', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.sens.stem != args.scene_id or args.sens.suffix != '.sens':
        raise ValueError('Scene identity does not match original sensor file')
    output = args.output_root / args.scene_id
    if output.exists():
        raise FileExistsError(output)
    spec = importlib.util.spec_from_file_location('upstream_sensor_reader', args.sensor_reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sensor = module.SensorData(str(args.sens))
    if sensor.depth_shift != 1000.0:
        raise ValueError('This historical route requires millimetre depth with depth_shift=1000')
    for folder in ['color', 'depth', 'pose', 'intrinsic']:
        (output / folder).mkdir(parents=True, exist_ok=False)
    ids = list(range(0, len(sensor.frames), 20))
    for i in ids:
        frame = sensor.frames[i]
        color = frame.decompress_color(sensor.color_compression_type)
        depth = np.frombuffer(frame.decompress_depth(sensor.depth_compression_type), dtype=np.uint16)
        depth = depth.reshape(sensor.depth_height, sensor.depth_width)
        # 历史原扫描导出将两种图像分别最近邻缩放到480×640；相机矩阵保持原值。
        # 不在这里重标定或补深度，否则模型会看到另一套几何输入。
        color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
        imageio.imwrite(output / 'color' / f'{i}.png', color)
        imageio.imwrite(output / 'depth' / f'{i}.png', depth)
        np.savetxt(output / 'pose' / f'{i}.txt', frame.camera_to_world, fmt='%f')
    for name in ['intrinsic_color', 'intrinsic_depth', 'extrinsic_color', 'extrinsic_depth']:
        np.savetxt(output / 'intrinsic' / (name + '.txt'), getattr(sensor, name), fmt='%f')
    (output / 'conversion.json').write_text(json.dumps({'scene_id': args.scene_id,
        'sensor_version': 4, 'native_frame_count': len(sensor.frames), 'exported_frame_ids': ids,
        'frame_skip': 20, 'image_hw': [480, 640], 'resize': 'INTER_NEAREST',
        'intrinsics': 'original values, six decimal places', 'depth_shift': 1000.0,
        'finite_pose_selection': 'performed by prepare-public-inputs'}, indent=2)+'\n')
    print(json.dumps({'scene_id': args.scene_id, 'frames': len(ids)}))


if __name__ == '__main__':
    main()
