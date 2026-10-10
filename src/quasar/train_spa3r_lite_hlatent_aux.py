#!/usr/bin/env python3
"""Train an H-latent Spa3R-lite auxiliary-only model.

这个脚本先不碰 Flow 答案模型，只验证一件事：

    H-latent 能不能根据 5-view window 里的 context views，
    预测 held-out target views 的 frozen spatial / semantic features。

输入来自 `build_spa3r_lite_5view_aux_dataset.py` 生成的 JSONL：
- context views: positions [-2, 0, +2]
- target views: positions [-1, +1]
- frozen targets: DINO token features、CUT3R context tokens、CUT3R target points

为了让闭环先可靠跑起来，模型预测：
- target DINO token field: 默认 256 个 DINO tokens；
- target CUT3R feature field: 默认从 768 个 CUT3R tokens 中均匀采样 192 个；
- target CUT3R point field: 同一批 CUT3R token 对应的 3D points。

同时报告一个无训练 baseline：直接用 context views 的平均 feature 去猜 target。
注意：baseline 只用于日志对照，不参与预测公式。真正的 auxiliary 预测必须是
`H latent + target pose -> target feature/point`，不能写成 `context mean + H residual`，
否则 H latent 会退化成“平均特征的修正项”，和正式 pipeline 里“环境偏置 latent”的角色不一致。
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


FORMAT_VERSION = "glow_spa3r_lite_hlatent_aux_train_v1"
EXPECTED_DATASET_FORMAT = "glow_spa3r_lite_anchor_5view_aux_dataset_v1"
CONTEXT_POSITIONS = (-2, 0, 2)
TARGET_POSITIONS = (-1, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train H-latent Spa3R-lite auxiliary-only model.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-h-latents", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--cut3r-token-subsample", type=int, default=192)
    parser.add_argument("--dino-token-subsample", type=int, default=256)
    parser.add_argument("--feature-cache-items", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dino-loss-weight", type=float, default=1.0)
    parser.add_argument("--cut3r-loss-weight", type=float, default=1.0)
    parser.add_argument("--point-loss-weight", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--dataloader-audit-only", action="store_true")
    return parser


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no} must be JSON object")
            yield line_no, row


def read_rows(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in iter_jsonl(path):
        rows.append(row)
        if max_rows and len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_even_indices(total: int, keep: int) -> np.ndarray:
    if keep <= 0:
        raise ValueError(f"keep must be positive, got {keep}")
    if keep > total:
        raise ValueError(f"cannot keep {keep} tokens from total={total}")
    if keep == total:
        return np.arange(total, dtype=np.int64)
    # 均匀取 token，避免只看左上角局部 patch。
    return np.linspace(0, total - 1, num=keep, dtype=np.int64)


def make_patch_xy(*, height: int, width: int, indices: Sequence[int]) -> torch.Tensor:
    xy = []
    for index in indices:
        y = int(index) // width
        x = int(index) % width
        x_norm = (float(x) / float(max(width - 1, 1))) * 2.0 - 1.0
        y_norm = (float(y) / float(max(height - 1, 1))) * 2.0 - 1.0
        xy.append([x_norm, y_norm])
    return torch.tensor(xy, dtype=torch.float32)


class NpzFeatureCache:
    """小型 LRU cache，避免同一个视频的 npz 在一个 epoch 里被反复打开。

    NpzFile 会持有文件句柄，所以被 LRU 挤掉时要显式 close。
    """

    def __init__(self, max_items: int) -> None:
        if max_items <= 0:
            raise ValueError(f"feature cache max_items must be positive, got {max_items}")
        self.max_items = max_items
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: str) -> Any:
        if path in self._cache:
            item = self._cache.pop(path)
            self._cache[path] = item
            return item
        if not Path(path).is_file():
            raise FileNotFoundError(f"feature npz not found: {path}")
        item = np.load(path, allow_pickle=False)
        self._cache[path] = item
        while len(self._cache) > self.max_items:
            _, old_item = self._cache.popitem(last=False)
            old_item.close()
        return item

    def close(self) -> None:
        while self._cache:
            _, item = self._cache.popitem(last=False)
            item.close()


class Spa3RLite5ViewAuxDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        dino_indices: np.ndarray,
        cut3r_indices: np.ndarray,
        feature_cache_items: int,
    ) -> None:
        self.rows = list(rows)
        self.dino_indices = dino_indices
        self.cut3r_indices = cut3r_indices
        self.cache = NpzFeatureCache(feature_cache_items)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if str(row["format_version"]) != EXPECTED_DATASET_FORMAT:
            raise ValueError(f"row_id={row.get('row_id')} unexpected format_version={row.get('format_version')}")

        feature_bank = row["feature_bank"]
        cut3r = self.cache.get(str(feature_bank["cut3r_npz_path"]))
        dino = self.cache.get(str(feature_bank["dino_token_path"]))

        views = list(row["views"])
        if len(views) != 5:
            raise ValueError(f"row_id={row['row_id']} expected 5 views, got {len(views)}")

        context_views = [view for view in views if str(view["role"]) == "context"]
        target_views = [view for view in views if str(view["role"]) == "target"]
        if [int(view["position"]) for view in context_views] != list(CONTEXT_POSITIONS):
            raise ValueError(f"row_id={row['row_id']} context positions mismatch")
        if [int(view["position"]) for view in target_views] != list(TARGET_POSITIONS):
            raise ValueError(f"row_id={row['row_id']} target positions mismatch")

        context_rows = np.asarray([int(view["cut3r_row_index"]) for view in context_views], dtype=np.int64)
        target_rows = np.asarray([int(view["cut3r_row_index"]) for view in target_views], dtype=np.int64)
        context_pose = np.asarray([view["camera_pose"] for view in context_views], dtype=np.float32)
        target_pose = np.asarray([view["camera_pose"] for view in target_views], dtype=np.float32)

        context_dino = np.asarray(dino["dino_token_features"][context_rows][:, self.dino_indices, :], dtype=np.float32)
        target_dino = np.asarray(dino["dino_token_features"][target_rows][:, self.dino_indices, :], dtype=np.float32)
        context_cut3r = np.asarray(cut3r["context_tokens"][context_rows][:, self.cut3r_indices, :], dtype=np.float32)
        target_cut3r = np.asarray(cut3r["context_tokens"][target_rows][:, self.cut3r_indices, :], dtype=np.float32)
        context_points = np.asarray(cut3r["target_points"][context_rows][:, self.cut3r_indices, :], dtype=np.float32)
        target_points = np.asarray(cut3r["target_points"][target_rows][:, self.cut3r_indices, :], dtype=np.float32)

        return {
            "row_id": str(row["row_id"]),
            "video_id": str(row["video_id"]),
            "context_positions": torch.tensor(CONTEXT_POSITIONS, dtype=torch.long),
            "target_positions": torch.tensor(TARGET_POSITIONS, dtype=torch.long),
            "context_dino": torch.from_numpy(context_dino),
            "target_dino": torch.from_numpy(target_dino),
            "context_cut3r": torch.from_numpy(context_cut3r),
            "target_cut3r": torch.from_numpy(target_cut3r),
            "context_points": torch.from_numpy(context_points),
            "target_points": torch.from_numpy(target_points),
            "context_pose": torch.from_numpy(context_pose),
            "target_pose": torch.from_numpy(target_pose),
        }


class HLatentSpa3RLiteAuxModel(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_h_latents: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        dino_patch_xy: torch.Tensor,
        cut3r_patch_xy: torch.Tensor,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_h_latents = num_h_latents

        self.dino_proj = nn.Linear(768, hidden_dim)
        self.cut3r_proj = nn.Linear(768, hidden_dim)
        self.point_xyz_proj = nn.Linear(3, hidden_dim)
        self.pose_proj = nn.Linear(16, hidden_dim)
        self.position_embedding = nn.Embedding(5, hidden_dim)
        self.token_type_embedding = nn.Embedding(8, hidden_dim)
        self.h_queries = nn.Parameter(torch.randn(num_h_latents, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.target_pose_proj = nn.Linear(16, hidden_dim)
        self.patch_xy_proj = nn.Linear(2, hidden_dim)
        self.decoder_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dino_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 768),
        )
        self.cut3r_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 768),
        )
        self.point_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        self.register_buffer("dino_patch_xy", dino_patch_xy, persistent=False)
        self.register_buffer("cut3r_patch_xy", cut3r_patch_xy, persistent=False)

        # 这里不要做 residual zero init。
        # 这版 auxiliary 的目标是：H latent 本身携带局部环境，再配合 target pose 直接预测该视角特征。
        # 如果再写成 context mean + residual，H 就不再是环境偏置，而只是“平均图像特征的补丁”。

    @staticmethod
    def _relative_pose_flat(pose: torch.Tensor, anchor_pose: torch.Tensor) -> torch.Tensor:
        # pose: [B,V,4,4], anchor_pose: [B,4,4]
        inv_anchor = torch.linalg.inv(anchor_pose)
        rel = torch.matmul(inv_anchor[:, None, :, :], pose)
        return rel.reshape(rel.shape[0], rel.shape[1], 16)

    def encode_h_latent(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        context_dino = batch["context_dino"]
        context_cut3r = batch["context_cut3r"]
        context_points = batch["context_points"]
        context_pose = batch["context_pose"]
        context_positions = batch["context_positions"]
        batch_size = context_dino.shape[0]
        num_context = context_dino.shape[1]
        num_dino_patches = context_dino.shape[2]
        num_cut3r_patches = context_cut3r.shape[2]

        # context 的第 2 个 view 是 anchor position 0。所有 pose 都先转到 anchor 坐标系，
        # 这样 H-latent 学的是“同一个局部空间”，而不是五张互不相干的图。
        pose_flat = self._relative_pose_flat(context_pose, context_pose[:, 1])
        pos_index = (context_positions + 2).clamp(0, 4)
        pos_emb = self.position_embedding(pos_index)
        view_emb = self.pose_proj(pose_flat) + pos_emb

        dino_xy = self.patch_xy_proj(self.dino_patch_xy).view(1, 1, num_dino_patches, self.hidden_dim)
        cut3r_xy = self.patch_xy_proj(self.cut3r_patch_xy).view(1, 1, num_cut3r_patches, self.hidden_dim)

        # 不再只喂每帧 mean feature。这里保留 token-level 空间布局，
        # 否则模型只能猜“平均画面”，很难证明 H-latent 真的学到了环境结构。
        dino_tokens = (
            self.dino_proj(context_dino)
            + dino_xy
            + view_emb.view(batch_size, num_context, 1, self.hidden_dim)
            + self.token_type_embedding.weight[0].view(1, 1, 1, self.hidden_dim)
        ).reshape(batch_size, num_context * num_dino_patches, self.hidden_dim)
        cut3r_tokens = (
            self.cut3r_proj(context_cut3r)
            + self.point_xyz_proj(context_points)
            + cut3r_xy
            + view_emb.view(batch_size, num_context, 1, self.hidden_dim)
            + self.token_type_embedding.weight[1].view(1, 1, 1, self.hidden_dim)
        ).reshape(batch_size, num_context * num_cut3r_patches, self.hidden_dim)

        context_tokens = torch.cat([dino_tokens, cut3r_tokens], dim=1)
        h_queries = self.h_queries.unsqueeze(0).expand(batch_size, -1, -1)
        sequence = torch.cat([h_queries, context_tokens], dim=1)
        encoded = self.encoder(sequence)
        return encoded[:, : self.num_h_latents, :]

    def _decode_field(
        self,
        *,
        h_latent: torch.Tensor,
        target_pose: torch.Tensor,
        anchor_pose: torch.Tensor,
        target_positions: torch.Tensor,
        patch_xy: torch.Tensor,
        decoder: nn.Module,
        token_type_id: int,
    ) -> torch.Tensor:
        batch_size = h_latent.shape[0]
        num_targets = target_pose.shape[1]
        num_patches = patch_xy.shape[0]

        pose_flat = self._relative_pose_flat(target_pose, anchor_pose)
        target_pos_index = (target_positions + 2).clamp(0, 4)
        target_emb = self.target_pose_proj(pose_flat) + self.position_embedding(target_pos_index)
        patch_emb = self.patch_xy_proj(patch_xy).view(1, 1, num_patches, self.hidden_dim)

        query = (
            target_emb.view(batch_size, num_targets, 1, self.hidden_dim)
            + patch_emb
            + self.token_type_embedding.weight[token_type_id].view(1, 1, 1, self.hidden_dim)
        ).reshape(batch_size, num_targets * num_patches, self.hidden_dim)
        attended, _weights = self.decoder_attn(query=query, key=h_latent, value=h_latent, need_weights=False)
        combined = query + attended
        combined = combined + self.decoder_ffn(self.decoder_norm(combined))
        return decoder(combined).view(batch_size, num_targets, num_patches, -1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h_latent = self.encode_h_latent(batch)
        target_pose = batch["target_pose"]
        target_positions = batch["target_positions"]
        anchor_pose = batch["context_pose"][:, 1]

        # 辅助任务只把 target pose 当作查询条件，让 decoder 从 H latent 里读出
        # 目标视角应该是什么样的 feature/point。这里不再加 context mean baseline。
        dino_pred = self._decode_field(
            h_latent=h_latent,
            target_pose=target_pose,
            anchor_pose=anchor_pose,
            target_positions=target_positions,
            patch_xy=self.dino_patch_xy,
            decoder=self.dino_decoder,
            token_type_id=3,
        )
        cut3r_pred = self._decode_field(
            h_latent=h_latent,
            target_pose=target_pose,
            anchor_pose=anchor_pose,
            target_positions=target_positions,
            patch_xy=self.cut3r_patch_xy,
            decoder=self.cut3r_decoder,
            token_type_id=4,
        )
        point_pred_norm = self._decode_field(
            h_latent=h_latent,
            target_pose=target_pose,
            anchor_pose=anchor_pose,
            target_positions=target_positions,
            patch_xy=self.cut3r_patch_xy,
            decoder=self.point_decoder,
            token_type_id=5,
        )
        return {
            "h_latent": h_latent,
            "target_dino_pred": dino_pred,
            "target_cut3r_pred": cut3r_pred,
            "target_points_pred_norm": point_pred_norm,
        }


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output


def normalize_points(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_points = batch["context_points"]
    target_points = batch["target_points"]
    center = context_points.mean(dim=(1, 2), keepdim=True)
    scale = context_points.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-3)
    target_norm = (target_points - center) / scale
    context_norm = (context_points - center) / scale
    return context_norm, target_norm, scale


def cosine_feature_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def compute_losses(
    *,
    batch: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    dino_loss_weight: float,
    cut3r_loss_weight: float,
    point_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_dino = batch["target_dino"]
    target_cut3r = batch["target_cut3r"]
    _context_points_norm, target_points_norm, _scale = normalize_points(batch)

    dino_cos = cosine_feature_loss(outputs["target_dino_pred"], target_dino)
    cut3r_cos = cosine_feature_loss(outputs["target_cut3r_pred"], target_cut3r)
    point_l1 = F.smooth_l1_loss(outputs["target_points_pred_norm"], target_points_norm)
    total = dino_loss_weight * dino_cos + cut3r_loss_weight * cut3r_cos + point_loss_weight * point_l1
    metrics = {
        "loss": float(total.detach().cpu()),
        "dino_cos_loss": float(dino_cos.detach().cpu()),
        "cut3r_cos_loss": float(cut3r_cos.detach().cpu()),
        "point_smooth_l1": float(point_l1.detach().cpu()),
    }
    baseline_metrics = compute_baseline_metrics(batch)
    metrics.update(baseline_metrics)
    metrics["dino_cos_improvement"] = metrics["baseline_dino_cos_loss"] - metrics["dino_cos_loss"]
    metrics["cut3r_cos_improvement"] = metrics["baseline_cut3r_cos_loss"] - metrics["cut3r_cos_loss"]
    metrics["point_l1_improvement"] = metrics["baseline_point_smooth_l1"] - metrics["point_smooth_l1"]
    return total, metrics


def compute_baseline_metrics(batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
    context_dino = batch["context_dino"]
    target_dino = batch["target_dino"]
    context_cut3r = batch["context_cut3r"]
    target_cut3r = batch["target_cut3r"]
    context_points_norm, target_points_norm, _scale = normalize_points(batch)
    baseline_dino = context_dino.mean(dim=1, keepdim=True).expand_as(target_dino)
    baseline_cut3r = context_cut3r.mean(dim=1, keepdim=True).expand_as(target_cut3r)
    baseline_points = context_points_norm.mean(dim=1, keepdim=True).expand_as(target_points_norm)
    return {
        "baseline_dino_cos_loss": float(cosine_feature_loss(baseline_dino, target_dino).detach().cpu()),
        "baseline_cut3r_cos_loss": float(cosine_feature_loss(baseline_cut3r, target_cut3r).detach().cpu()),
        "baseline_point_smooth_l1": float(F.smooth_l1_loss(baseline_points, target_points_norm).detach().cpu()),
    }


def aggregate_metrics(metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("cannot aggregate empty metrics")
    keys = sorted(metrics[0].keys())
    output = {}
    for key in keys:
        values = [float(item[key]) for item in metrics]
        output[key] = float(sum(values) / len(values))
    return output


@torch.no_grad()
def evaluate(
    *,
    model: HLatentSpa3RLiteAuxModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    metrics: list[dict[str, float]] = []
    start = time.time()
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        _loss, batch_metrics = compute_losses(
            batch=batch,
            outputs=outputs,
            dino_loss_weight=float(args.dino_loss_weight),
            cut3r_loss_weight=float(args.cut3r_loss_weight),
            point_loss_weight=float(args.point_loss_weight),
        )
        metrics.append(batch_metrics)
        if args.eval_max_batches and batch_index >= int(args.eval_max_batches):
            break
    return {
        "batch_count": len(metrics),
        "metrics": aggregate_metrics(metrics),
        "elapsed_sec": time.time() - start,
    }


def save_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, args: argparse.Namespace, step: int, epoch: int, val_summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "step": step,
            "epoch": epoch,
            "val_summary": dict(val_summary),
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_rows(args.train_jsonl, max_rows=int(args.max_train_rows))
    val_rows = read_rows(args.val_jsonl, max_rows=int(args.max_val_rows))
    dino_indices = make_even_indices(256, int(args.dino_token_subsample))
    cut3r_indices = make_even_indices(768, int(args.cut3r_token_subsample))

    train_dataset = Spa3RLite5ViewAuxDataset(
        rows=train_rows,
        dino_indices=dino_indices,
        cut3r_indices=cut3r_indices,
        feature_cache_items=int(args.feature_cache_items),
    )
    val_dataset = Spa3RLite5ViewAuxDataset(
        rows=val_rows,
        dino_indices=dino_indices,
        cut3r_indices=cut3r_indices,
        feature_cache_items=int(args.feature_cache_items),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    # argparse 里有 Path 对象。JSON 日志必须写成普通字符串，否则 audit 阶段一开始就会失败。
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config = {
        "format_version": FORMAT_VERSION,
        "args": serializable_args,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "dino_indices_count": len(dino_indices),
        "cut3r_indices_count": len(cut3r_indices),
    }
    write_json(output_dir / "run_config.json", run_config)
    print(json.dumps({"event": "run_config", **run_config}, ensure_ascii=False), flush=True)

    audit_batch = next(iter(train_loader))
    audit_shapes = {
        key: list(value.shape)
        for key, value in audit_batch.items()
        if torch.is_tensor(value)
    }
    write_json(output_dir / "dataloader_audit.json", {"status": "passed", "tensor_shapes": audit_shapes})
    print(json.dumps({"event": "dataloader_audit", "status": "passed", "tensor_shapes": audit_shapes}, ensure_ascii=False), flush=True)
    if args.dataloader_audit_only:
        summary = {"status": "dataloader_audit_only_completed", "run_config": run_config, "tensor_shapes": audit_shapes}
        write_json(output_dir / "summary.json", summary)
        return summary

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    dino_patch_xy = make_patch_xy(height=16, width=16, indices=dino_indices)
    cut3r_patch_xy = make_patch_xy(height=24, width=32, indices=cut3r_indices)
    model = HLatentSpa3RLiteAuxModel(
        hidden_dim=int(args.hidden_dim),
        num_h_latents=int(args.num_h_latents),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        dino_patch_xy=dino_patch_xy,
        cut3r_patch_xy=cut3r_patch_xy,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val_loss = float("inf")
    best_val_summary: dict[str, Any] | None = None
    step = 0
    train_log_path = output_dir / "train_log.jsonl"
    with train_log_path.open("w", encoding="utf-8") as train_log:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            epoch_metrics: list[dict[str, float]] = []
            epoch_start = time.time()
            for batch in train_loader:
                step += 1
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                loss, batch_metrics = compute_losses(
                    batch=batch,
                    outputs=outputs,
                    dino_loss_weight=float(args.dino_loss_weight),
                    cut3r_loss_weight=float(args.cut3r_loss_weight),
                    point_loss_weight=float(args.point_loss_weight),
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
                batch_metrics["grad_norm"] = float(grad_norm.detach().cpu())
                epoch_metrics.append(batch_metrics)

                if int(args.log_every) and step % int(args.log_every) == 0:
                    recent = aggregate_metrics(epoch_metrics[-int(args.log_every) :])
                    event = {"event": "train_step", "step": step, "epoch": epoch, "metrics": recent}
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    train_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    train_log.flush()

                if int(args.eval_every_steps) and step % int(args.eval_every_steps) == 0:
                    val_summary = evaluate(model=model, loader=val_loader, device=device, args=args)
                    event = {"event": "eval_step", "step": step, "epoch": epoch, "val": val_summary}
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    train_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    train_log.flush()
                    val_loss = float(val_summary["metrics"]["loss"])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_summary = dict(val_summary)
                        if args.save_best:
                            save_checkpoint(
                                output_dir / "best_checkpoint.pt",
                                model=model,
                                optimizer=optimizer,
                                args=args,
                                step=step,
                                epoch=epoch,
                                val_summary=val_summary,
                            )

            epoch_summary = {
                "event": "epoch_completed",
                "step": step,
                "epoch": epoch,
                "train": {
                    "batch_count": len(epoch_metrics),
                    "metrics": aggregate_metrics(epoch_metrics),
                    "elapsed_sec": time.time() - epoch_start,
                },
            }
            val_summary = evaluate(model=model, loader=val_loader, device=device, args=args)
            epoch_summary["val"] = val_summary
            print(json.dumps(epoch_summary, ensure_ascii=False), flush=True)
            train_log.write(json.dumps(epoch_summary, ensure_ascii=False) + "\n")
            train_log.flush()
            val_loss = float(val_summary["metrics"]["loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_summary = dict(val_summary)
                if args.save_best:
                    save_checkpoint(
                        output_dir / "best_checkpoint.pt",
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        step=step,
                        epoch=epoch,
                        val_summary=val_summary,
                    )

    final_val = evaluate(model=model, loader=val_loader, device=device, args=args)
    summary = {
        "status": "completed",
        "format_version": FORMAT_VERSION,
        "run_config": run_config,
        "final_step": step,
        "final_val": final_val,
        "best_val_loss": best_val_loss,
        "best_val": best_val_summary,
        "proof": {
            "beats_mean_context_dino_baseline": final_val["metrics"]["dino_cos_loss"] < final_val["metrics"]["baseline_dino_cos_loss"],
            "beats_mean_context_cut3r_baseline": final_val["metrics"]["cut3r_cos_loss"] < final_val["metrics"]["baseline_cut3r_cos_loss"],
            "beats_mean_context_point_baseline": final_val["metrics"]["point_smooth_l1"] < final_val["metrics"]["baseline_point_smooth_l1"],
            "dino_cos_improvement": final_val["metrics"]["dino_cos_improvement"],
            "cut3r_cos_improvement": final_val["metrics"]["cut3r_cos_improvement"],
            "point_l1_improvement": final_val["metrics"]["point_l1_improvement"],
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = train(args)
    print(json.dumps({"event": "completed", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
