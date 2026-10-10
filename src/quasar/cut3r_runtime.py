"""从RGB序列导出观测几何；数值计算沿既有导出器。"""
from __future__ import annotations
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

def infer_patch_grid(*, token_count: int, image_height: int, image_width: int) -> tuple[int, int]:
    if token_count <= 0:
        raise ValueError(f"token_count must be positive, got {token_count}")
    candidates: list[tuple[float, int, int]] = []
    target_ratio = float(image_width) / float(image_height)
    for grid_h in range(1, token_count + 1):
        if token_count % grid_h != 0:
            continue
        grid_w = token_count // grid_h
        if image_height % grid_h != 0 or image_width % grid_w != 0:
            continue
        ratio_error = abs((float(grid_w) / float(grid_h)) - target_ratio)
        candidates.append((ratio_error, grid_h, grid_w))
    if not candidates:
        raise ValueError(
            f"cannot infer patch grid for token_count={token_count}, image_shape={(image_height, image_width)}"
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]

def patch_points_from_dense_pointmap(pointmap_world, confidence_map, *, token_count: int):
    import numpy as np

    points = np.asarray(pointmap_world, dtype=np.float32)
    conf = np.asarray(confidence_map, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"pointmap_world must be [H,W,3], got {points.shape}")
    if conf.shape != points.shape[:2]:
        raise ValueError(f"confidence_map shape mismatch: {conf.shape} vs {points.shape[:2]}")
    height, width = points.shape[:2]
    grid_h, grid_w = infer_patch_grid(token_count=token_count, image_height=height, image_width=width)
    patch_h = height // grid_h
    patch_w = width // grid_w
    token_points: list[np.ndarray] = []
    token_conf: list[float] = []
    token_mask: list[bool] = []
    for row in range(grid_h):
        for col in range(grid_w):
            patch_points = points[row * patch_h : (row + 1) * patch_h, col * patch_w : (col + 1) * patch_w]
            patch_conf = conf[row * patch_h : (row + 1) * patch_h, col * patch_w : (col + 1) * patch_w]
            flat_points = patch_points.reshape(-1, 3)
            flat_conf = patch_conf.reshape(-1)
            valid = np.isfinite(flat_points).all(axis=1) & np.isfinite(flat_conf)
            if bool(valid.any()):
                token_points.append(flat_points[valid].mean(axis=0).astype(np.float32))
                token_conf.append(float(flat_conf[valid].mean()))
                token_mask.append(True)
            else:
                token_points.append(np.zeros((3,), dtype=np.float32))
                token_conf.append(0.0)
                token_mask.append(False)
    return (
        np.stack(token_points, axis=0).astype(np.float32),
        np.asarray(token_conf, dtype=np.float32),
        np.asarray(token_mask, dtype=bool),
        (grid_h, grid_w),
    )

def transform_points_to_world(points_self, camera_pose):
    import torch

    rotation = camera_pose[:, :3, :3]
    translation = camera_pose[:, :3, 3]
    return torch.einsum("bhwc,bdc->bhwd", points_self, rotation) + translation[:, None, None, :]

def load_cut3r_runtime(*, cut3r_repo_root: Path, model_path: Path, device_text: str):
    import numpy as np
    import torch
    from copy import deepcopy

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if str(cut3r_repo_root) not in sys.path:
        sys.path.insert(0, str(cut3r_repo_root))

    # 代码根由调用者指定；权重可位于其他挂载路径，不能用权重目录决定代码来源。
    dust3r_source = str(cut3r_repo_root / "src")
    if dust3r_source not in sys.path:
        sys.path.insert(0, dust3r_source)
    croco_source = str(cut3r_repo_root / "src" / "croco")
    if croco_source not in sys.path:
        sys.path.insert(0, croco_source)
    # CUT3R的相机token使用负位置；原CUDA实现能计算它，查表版会越界，故这里要求真实组件已安装。
    from models.curope import cuRoPE2D
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.image import load_images

    def prepare_input(img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True):
        """准备 CUT3R image-only 输入。

        原版 demo.prepare_input 顶层会 import imageio，只是为了视频输出；
        我们导出训练/eval 几何时只读图片，所以直接复用 CUT3R 的 load_images，
        避免把无关可视化依赖绑进实验环境。
        """

        images = load_images(img_paths, size=size)
        views = []
        if raymaps is not None or raymap_mask is not None:
            raise ValueError("QUASAR CUT3R geometry export only supports image-only inputs")
        if len(img_mask) != len(images):
            raise ValueError(f"img_mask length mismatch: {len(img_mask)} vs images={len(images)}")
        for index, image in enumerate(images):
            view = {
                "img": image["img"],
                "ray_map": torch.full(
                    (
                        image["img"].shape[0],
                        6,
                        image["img"].shape[-2],
                        image["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(image["true_shape"]),
                "idx": index,
                "instance": str(index),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(bool(img_mask[index])).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(bool(update)).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)

        if revisit > 1:
            new_views = []
            for round_index in range(revisit):
                for view_index, view in enumerate(views):
                    new_view = deepcopy(view)
                    new_view["idx"] = round_index * len(views) + view_index
                    new_view["instance"] = str(round_index * len(views) + view_index)
                    if round_index > 0 and not update:
                        new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    device = torch.device(device_text)
    model = ARCroco3DStereo.from_pretrained(str(model_path)).to(device).eval()
    return model, prepare_input, pose_encoding_to_camera, device

def run_cut3r_sequence(
    *,
    model,
    prepare_input,
    pose_encoding_to_camera,
    device,
    frame_rows: Sequence[Mapping[str, Any]],
    image_size: int,
) -> dict[str, Any]:
    """运行 CUT3R，并显式取出 decoder patch tokens、pointmap 和 camera pose。"""

    import numpy as np
    import torch

    image_paths = [str(row["frame_image_path"]) for row in frame_rows]
    views = prepare_input(
        img_paths=image_paths,
        img_mask=[True] * len(image_paths),
        size=image_size,
        revisit=1,
        update=True,
    )
    for view in views:
        for key, value in list(view.items()):
            if key in {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}:
                continue
            if isinstance(value, torch.Tensor):
                view[key] = value.to(device, non_blocking=True)

    context_tokens = []
    target_points = []
    token_confidences = []
    token_masks = []
    camera_poses = []
    patch_grids = []

    with torch.no_grad():
        shape, feat_ls, pos = model._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = model._init_state(feat[0], pos[0])
        mem = model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        for index in range(len(views)):
            feat_i = feat[index]
            pos_i = pos[index]
            global_img_feat_i = model._get_img_level_feat(feat_i)
            if model.pose_head_flag:
                if index == 0:
                    pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = model.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0],
                    1,
                    2,
                    device=feat_i.device,
                    dtype=pos_i.dtype,
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec = model._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[index]["img_mask"],
                reset_mask=views[index]["reset"],
                update=views[index].get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = model.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
            head_input = [
                dec[0].float(),
                dec[model.dec_depth * 2 // 4][:, 1:].float(),
                dec[model.dec_depth * 3 // 4][:, 1:].float(),
                dec[model.dec_depth].float(),
            ]
            res = model._downstream_head(head_input, shape[index], pos=pos_i)
            cam2world = pose_encoding_to_camera(res["camera_pose"].clone()).float()
            pointmap_self = res["pts3d_in_self_view"].float()
            conf_self = res["conf_self"].float()
            pointmap_world = transform_points_to_world(pointmap_self, cam2world)
            tokens = dec[model.dec_depth][:, 1:].float()

            if tokens.shape[0] != 1:
                raise ValueError(f"CUT3R export expects batch size 1, got {tokens.shape[0]}")
            dense_points_np = pointmap_world[0].detach().cpu().numpy().astype(np.float32)
            dense_conf_np = conf_self[0].detach().cpu().numpy().astype(np.float32)
            token_points, token_conf, token_mask, patch_grid = patch_points_from_dense_pointmap(
                dense_points_np,
                dense_conf_np,
                token_count=int(tokens.shape[1]),
            )
            context_tokens.append(tokens[0].detach().cpu().numpy().astype(np.float32))
            target_points.append(token_points)
            token_confidences.append(token_conf)
            token_masks.append(token_mask)
            camera_poses.append(cam2world[0].detach().cpu().numpy().astype(np.float32))
            patch_grids.append(patch_grid)

            img_mask = views[index]["img_mask"]
            update = views[index].get("update", None)
            update_mask = img_mask & update if update is not None else img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
            mem = new_mem * update_mask + mem * (1 - update_mask)
            reset_mask = views[index]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)

    token_shapes = {tuple(item.shape) for item in context_tokens}
    point_shapes = {tuple(item.shape) for item in target_points}
    if len(token_shapes) != 1:
        raise ValueError(f"CUT3R context token shapes differ in one video: {sorted(token_shapes)}")
    if len(point_shapes) != 1:
        raise ValueError(f"CUT3R target point shapes differ in one video: {sorted(point_shapes)}")
    return {
        "context_tokens": np.stack(context_tokens, axis=0).astype(np.float32),
        "target_points": np.stack(target_points, axis=0).astype(np.float32),
        "token_confidence": np.stack(token_confidences, axis=0).astype(np.float32),
        "token_mask": np.stack(token_masks, axis=0).astype(bool),
        "camera_pose": np.stack(camera_poses, axis=0).astype(np.float32),
        "patch_grid_hw": np.asarray(patch_grids, dtype=np.int32),
    }
