"""创建坐标教学样例所需的假图与帧记录，只用于检查转换计算。"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    image = args.output_root.resolve()/'synthetic.jpg'
    Image.new('RGB', (8, 8)).save(image)
    frame = dict(video_id='teaching', frame_id=10, frame_candidate_id='teaching_10',
        sample_order=0, embedding_row_index=0, frame_image_path=str(image))
    (args.output_root/'frames.jsonl').write_text(json.dumps(frame)+'\n')
    np.savez(args.output_root/'binding.npz', video_id=np.asarray('teaching'),
        frame_candidate_ids=np.asarray(['teaching_10']), frame_image_paths=np.asarray([str(image)]))
    shutil.copyfile(Path(__file__).with_name('reference_definition.synthetic.json'), args.output_root/'definition.json')
    print('Synthetic teaching fixture only: ' + str(args.output_root))


if __name__ == '__main__':
    main()
