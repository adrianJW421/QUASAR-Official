#!/usr/bin/env python3
"""Export CUT3R geometry for a variable-length video_frame_manifest.

VSTI 是每个视频固定 32 帧；VSI official all-MCA 是每个 video 的候选帧全集，
不同 video 帧数不一样。现有 32f exporter 强制每个 video 同样长，所以这里单独
提供一个 variable-length 版本。它仍然保证同一个 video 里：

`embedding_row_index == cut3r_row_index == sample_order`

这样下游 Flow-as-Picker 只需要一份 candidate_cut3r_row_indices 就能同时索引
CLIP/DINO/CUT3R 三类数组。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


from quasar.cut3r_runtime import run_cut3r_sequence, load_cut3r_runtime


FORMAT_VERSION = "flow_as_picker_cut3r_variable_frame_bank_v1"
GEOMETRY_SOURCE = "cut3r_pseudo_pose_pointmap_decoder_tokens_variable_frames"


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()

def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

def sanitize_filename(value: object) -> str:
    text = normalize_text(value)
    safe_chars = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text]
    safe = "".join(safe_chars).strip("_")
    if not safe:
        raise ValueError(f"Cannot build filename from empty value: {value}")
    return safe

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export variable-length CUT3R geometry bank from frame manifest.")
    parser.add_argument("--video-frame-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cut3r-repo-root", type=Path, required=True)
    parser.add_argument("--cut3r-model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--video-id-list", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no} must be JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def load_frame_rows_by_video(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_candidate_ids: set[str] = set()
    for row in read_jsonl(path):
        candidate_id = str(row["frame_candidate_id"])
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate frame_candidate_id={candidate_id}")
        seen_candidate_ids.add(candidate_id)
        image_path = Path(str(row["frame_image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"missing frame image for candidate_id={candidate_id}: {image_path}")
        by_video[str(row["video_id"])].append(dict(row))

    for video_id, rows in by_video.items():
        rows.sort(key=lambda item: (int(item["sample_order"]), int(item["frame_id"]), str(item["frame_candidate_id"])))
        expected_orders = list(range(len(rows)))
        actual_orders = [int(row["sample_order"]) for row in rows]
        if actual_orders != expected_orders:
            raise ValueError(f"video_id={video_id} sample_order must be 0..N-1, got first={actual_orders[:8]}")
        for row_index, row in enumerate(rows):
            embedding_row_index = int(row.get("embedding_row_index", row_index))
            if embedding_row_index != row_index:
                raise ValueError(
                    f"video_id={video_id} candidate={row['frame_candidate_id']} "
                    f"embedding_row_index must equal sample order: {embedding_row_index} vs {row_index}"
                )
    return dict(by_video)


def select_video_ids(
    *,
    all_video_ids: Sequence[str],
    max_videos: int,
    num_shards: int,
    shard_index: int,
    video_id_list: Path | None,
) -> list[str]:
    if max_videos < 0:
        raise ValueError(f"max_videos must be >=0, got={max_videos}")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got={num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0,num_shards), got={shard_index}/{num_shards}")
    selected = list(sorted(all_video_ids))
    if video_id_list is not None:
        if not video_id_list.is_file():
            raise FileNotFoundError(f"video id list not found: {video_id_list}")
        requested = [line.strip() for line in video_id_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not requested:
            raise ValueError(f"video id list is empty: {video_id_list}")
        missing = sorted(set(requested) - set(selected))
        if missing:
            raise KeyError(f"video id list contains ids missing from frame manifest: {missing[:10]}")
        selected = requested
    if max_videos:
        selected = selected[: int(max_videos)]
    if num_shards > 1:
        selected = [video_id for index, video_id in enumerate(selected) if index % num_shards == shard_index]
    if not selected:
        raise ValueError("no videos selected after max_videos/sharding/video_id_list")
    return selected


def validate_existing_npz(*, npz_path: Path, video_id: str, frame_rows: Sequence[Mapping[str, Any]]) -> None:
    if not npz_path.is_file():
        raise FileNotFoundError(f"existing CUT3R npz missing: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        required = (
            "format_version",
            "geometry_source",
            "video_id",
            "frame_candidate_ids",
            "context_tokens",
            "target_points",
            "token_confidence",
            "token_mask",
            "camera_pose",
            "patch_grid_hw",
        )
        for key in required:
            if key not in payload.files:
                raise KeyError(f"existing CUT3R npz missing key={key}: {npz_path}")
        if str(payload["format_version"].tolist()) != FORMAT_VERSION:
            raise ValueError(f"existing CUT3R npz format mismatch: {npz_path}")
        if str(payload["geometry_source"].tolist()) != GEOMETRY_SOURCE:
            raise ValueError(f"existing CUT3R npz geometry_source mismatch: {npz_path}")
        if str(payload["video_id"].tolist()) != video_id:
            raise ValueError(f"existing CUT3R npz video_id mismatch: {npz_path}")
        expected_ids = [str(row["frame_candidate_id"]) for row in frame_rows]
        actual_ids = [str(value) for value in payload["frame_candidate_ids"].tolist()]
        if actual_ids != expected_ids:
            raise ValueError(f"existing CUT3R npz frame ids mismatch for video_id={video_id}")
        frame_count = len(frame_rows)
        if payload["camera_pose"].shape != (frame_count, 4, 4):
            raise ValueError(f"existing camera_pose shape mismatch: {payload['camera_pose'].shape}")
        if payload["target_points"].ndim != 3 or payload["target_points"].shape[0] != frame_count:
            raise ValueError(f"existing target_points shape mismatch: {payload['target_points'].shape}")


def save_video_npz(
    *,
    output_npz: Path,
    video_id: str,
    frame_rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    # 完整写好才成为可复用的结果；中断不会留下同名的半份几何文件。
    partial = output_npz.with_suffix('.partial.npz')
    np.savez_compressed(
        partial,
        format_version=np.asarray(FORMAT_VERSION),
        geometry_source=np.asarray(GEOMETRY_SOURCE),
        video_id=np.asarray(video_id),
        frame_candidate_ids=np.asarray([str(row["frame_candidate_id"]) for row in frame_rows]),
        frame_ids=np.asarray([int(row["frame_id"]) for row in frame_rows], dtype=np.int64),
        sample_orders=np.asarray([int(row["sample_order"]) for row in frame_rows], dtype=np.int64),
        frame_image_paths=np.asarray([str(row["frame_image_path"]) for row in frame_rows]),
        context_tokens=payload["context_tokens"],
        target_points=payload["target_points"],
        token_confidence=payload["token_confidence"],
        token_mask=payload["token_mask"],
        camera_pose=payload["camera_pose"],
        patch_grid_hw=payload["patch_grid_hw"],
        is_query_frame=np.asarray([index == len(frame_rows) - 1 for index in range(len(frame_rows))], dtype=bool),
    )
    partial.replace(output_npz)


def manifest_rows_for_video(*, video_id: str, frame_rows: Sequence[Mapping[str, Any]], output_npz: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_candidate_id = str(frame_rows[-1]["frame_candidate_id"])
    for row_index, row in enumerate(frame_rows):
        rows.append(
            {
                "format_version": FORMAT_VERSION,
                "geometry_source": GEOMETRY_SOURCE,
                "video_id": video_id,
                "frame_candidate_id": str(row["frame_candidate_id"]),
                "frame_id": int(row["frame_id"]),
                "sample_order": int(row["sample_order"]),
                "frame_image_path": str(row["frame_image_path"]),
                "cut3r_npz_path": str(output_npz),
                "cut3r_row_index": int(row_index),
                "embedding_row_index": int(row.get("embedding_row_index", row_index)),
                "is_query_frame": row_index == len(frame_rows) - 1,
                "query_frame_candidate_id": query_candidate_id,
            }
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    if not args.cut3r_repo_root.is_dir():
        raise FileNotFoundError(f"CUT3R repo root not found: {args.cut3r_repo_root}")
    if not args.cut3r_model_path.is_file():
        raise FileNotFoundError(f"CUT3R model checkpoint not found: {args.cut3r_model_path}")
    if args.image_size <= 0:
        raise ValueError(f"image_size must be positive, got={args.image_size}")
    if args.progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got={args.progress_every}")

    rows_by_video = load_frame_rows_by_video(args.video_frame_manifest)
    selected_video_ids = select_video_ids(
        all_video_ids=sorted(rows_by_video),
        max_videos=int(args.max_videos),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        video_id_list=args.video_id_list,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    video_npz_root = args.output_root / "video_npz"

    model, prepare_input, pose_encoding_to_camera, device = load_cut3r_runtime(
        cut3r_repo_root=args.cut3r_repo_root,
        model_path=args.cut3r_model_path,
        device_text=str(args.device),
    )

    manifest_rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    frame_counts: list[int] = []
    for video_number, video_id in enumerate(selected_video_ids, start=1):
        frame_rows = rows_by_video[video_id]
        frame_counts.append(len(frame_rows))
        output_npz = video_npz_root / f"{sanitize_filename(video_id)}.npz"
        if args.skip_existing and output_npz.is_file():
            validate_existing_npz(npz_path=output_npz, video_id=video_id, frame_rows=frame_rows)
            skipped += 1
        else:
            payload = run_cut3r_sequence(
                model=model,
                prepare_input=prepare_input,
                pose_encoding_to_camera=pose_encoding_to_camera,
                device=device,
                frame_rows=frame_rows,
                image_size=int(args.image_size),
            )
            save_video_npz(output_npz=output_npz, video_id=video_id, frame_rows=frame_rows, payload=payload)
            processed += 1
        manifest_rows.extend(manifest_rows_for_video(video_id=video_id, frame_rows=frame_rows, output_npz=output_npz))
        if video_number % int(args.progress_every) == 0 or video_number == len(selected_video_ids):
            print(
                f"[cut3r] shard={args.shard_index}/{args.num_shards} videos={video_number}/{len(selected_video_ids)} "
                f"processed={processed} skipped={skipped} last_video={video_id} frames={len(frame_rows)}",
                flush=True,
            )

    suffix = "" if int(args.num_shards) == 1 else f".shard_{int(args.shard_index):02d}_of_{int(args.num_shards):02d}"
    manifest_path = args.output_root / f"cut3r_frame_manifest{suffix}.jsonl"
    summary_path = args.output_root / f"summary{suffix}.json"
    write_jsonl(manifest_path, manifest_rows)
    summary = {
        "format_version": FORMAT_VERSION,
        "geometry_source": GEOMETRY_SOURCE,
        "input_video_frame_manifest": str(args.video_frame_manifest),
        "output_root": str(args.output_root),
        "manifest_jsonl": str(manifest_path),
        "cut3r_repo_root": str(args.cut3r_repo_root),
        "cut3r_model_path": str(args.cut3r_model_path),
        "device": str(args.device),
        "image_size": int(args.image_size),
        "num_input_videos": int(len(rows_by_video)),
        "num_selected_videos": int(len(selected_video_ids)),
        "num_manifest_rows": int(len(manifest_rows)),
        "num_processed_videos": int(processed),
        "num_skipped_videos": int(skipped),
        "frame_count_per_video": {
            "min": int(min(frame_counts)),
            "mean": float(sum(frame_counts) / len(frame_counts)),
            "max": int(max(frame_counts)),
        },
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
