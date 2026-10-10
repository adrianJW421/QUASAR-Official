from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_INVALID_REASON_CLASS_COUNT = 4


@dataclass(frozen=True, slots=True)
class WorkspaceGeometryAuxBatch:
    """MLLM workspace 几何辅助损失需要的一小批数据。

    每条样本有一串 source object 的几何 token，也有同一串点在 source camera 下的目标坐标。
    batch 里 token 数不一定一样，所以这里用 mask 标出哪些位置是真的点，哪些只是 padding。
    """

    context_tokens: torch.Tensor
    target_points: torch.Tensor
    point_mask: torch.Tensor
    pose_features: torch.Tensor
    compass16: torch.Tensor
    range_bin: torch.Tensor
    valid_evidence: torch.Tensor
    answer_loss_mask: torch.Tensor
    point_loss_mask: torch.Tensor
    center_loss_mask: torch.Tensor
    compass_loss_mask: torch.Tensor
    range_loss_mask: torch.Tensor
    invalid_reason: torch.Tensor
    answer_option: torch.Tensor
    spatial_option_compass16: torch.Tensor
    spatial_option_range_bin: torch.Tensor
    typed_bucket: torch.Tensor
    typed_numeric_scalar: torch.Tensor
    typed_numeric_scalar_raw: torch.Tensor
    typed_numeric_scalar_mask: torch.Tensor
    typed_numeric_kind: torch.Tensor
    typed_numeric_kind_mask: torch.Tensor
    typed_choice_label: torch.Tensor
    typed_choice_mask: torch.Tensor
    typed_choice_loss_mask: torch.Tensor
    typed_spatial_label: torch.Tensor
    typed_spatial_mask: torch.Tensor
    typed_answer_type: torch.Tensor
    typed_answer_type_mask: torch.Tensor
    typed_answer_embedding_mask: torch.Tensor
    source_row_ids: tuple[str, ...]
    typed_answer_texts: tuple[str, ...]
    compass16_class_weights: torch.Tensor
    range_bin_class_weights: torch.Tensor
    invalid_reason_class_weights: torch.Tensor
    answer_option_class_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class WorkspaceGeometryAuxTable:
    """按 Stage1 `idx` 查几何辅助监督的表。

    Stage1 trainer 每个 batch 只知道样本 idx。
    这张表负责把 idx 映射到：
    - source object 的 point/geometry tokens；
    - 这些点在 source camera 下的原始位置；
    - source camera 下的方向和粗距离标签。
    source->query 的变化不写进这里的监督标签，只通过 pose_features 交给 updater。
    """

    row_ids: tuple[str, ...]
    context_tokens: np.ndarray
    target_points: np.ndarray
    token_offsets: np.ndarray
    pose_features: np.ndarray
    compass16: np.ndarray
    range_bin: np.ndarray
    valid_evidence: np.ndarray
    invalid_reason: np.ndarray
    answer_option: np.ndarray
    spatial_option_compass16: np.ndarray
    spatial_option_range_bin: np.ndarray
    source_row_ids: tuple[str, ...]
    box_modes: tuple[str, ...] | None = None
    counterfactual_modes: tuple[str, ...] | None = None
    answer_loss_mask: np.ndarray | None = None
    point_loss_mask: np.ndarray | None = None
    center_loss_mask: np.ndarray | None = None
    compass_loss_mask: np.ndarray | None = None
    range_loss_mask: np.ndarray | None = None
    answer_option_class_count_override: int | None = None
    compass16_class_weights: np.ndarray | None = None
    range_bin_class_weights: np.ndarray | None = None
    invalid_reason_class_weights: np.ndarray | None = None
    answer_option_class_weights: np.ndarray | None = None
    typed_bucket: np.ndarray | None = None
    typed_numeric_scalar: np.ndarray | None = None
    typed_numeric_scalar_raw: np.ndarray | None = None
    typed_numeric_scalar_mask: np.ndarray | None = None
    typed_numeric_kind: np.ndarray | None = None
    typed_numeric_kind_mask: np.ndarray | None = None
    typed_choice_label: np.ndarray | None = None
    typed_choice_mask: np.ndarray | None = None
    typed_choice_loss_mask: np.ndarray | None = None
    typed_spatial_label: np.ndarray | None = None
    typed_spatial_mask: np.ndarray | None = None
    typed_answer_type: np.ndarray | None = None
    typed_answer_type_mask: np.ndarray | None = None
    typed_answer_embedding_mask: np.ndarray | None = None
    typed_answer_texts: tuple[str, ...] | None = None
    typed_numeric_kind_class_count_override: int | None = None
    typed_spatial_class_count_override: int | None = None
    typed_answer_type_class_count_override: int | None = None

    @property
    def invalid_reason_class_count(self) -> int:
        """返回当前表实际需要的 invalid reason 类数。

        旧实验只有 valid/wrong_object/shifted/wrong_pose 四类；
        新实验额外加入 latent_mismatch，所以需要五类。
        类数从数据表里推断，避免新旧实验互相弄坏。
        """

        max_label = int(np.asarray(self.invalid_reason, dtype=np.int64).max())
        return max(DEFAULT_INVALID_REASON_CLASS_COUNT, max_label + 1)

    @property
    def answer_option_class_count(self) -> int:
        """返回当前表实际需要的选项类数。A-D 是 4，A-E 是 5，A-G 是 7，A-H 是 8。

        当前 correct-evidence teacher 只做 A-D 空间选择题。
        这时数据里没有 invalid 行，不能因为 invalid_reason head 默认保留 4 类，
        就把 answer classifier 也硬扩成 A-G。否则训练时会要求 4 个选项文本去匹配 7 个类别。
        """

        if self.answer_option_class_count_override is not None:
            return int(self.answer_option_class_count_override)
        max_label = int(np.asarray(self.answer_option, dtype=np.int64).max())
        valid_evidence = np.asarray(self.valid_evidence, dtype=np.int64)
        invalid_reason = np.asarray(self.invalid_reason, dtype=np.int64)
        if bool((valid_evidence == 1).all()) and bool((invalid_reason == 0).all()):
            return max(4, max_label + 1)
        return max(4 + self.invalid_reason_class_count - 1, max_label + 1)

    @property
    def has_typed_latent_labels(self) -> bool:
        return self.typed_bucket is not None

    @property
    def typed_numeric_kind_class_count(self) -> int:
        if self.typed_numeric_kind_class_count_override is not None:
            return int(self.typed_numeric_kind_class_count_override)
        if self.typed_numeric_kind is None:
            return 1
        return max(1, int(np.asarray(self.typed_numeric_kind, dtype=np.int64).max()) + 1)

    @property
    def typed_spatial_class_count(self) -> int:
        if self.typed_spatial_class_count_override is not None:
            return int(self.typed_spatial_class_count_override)
        if self.typed_spatial_label is None:
            return 1
        return max(1, int(np.asarray(self.typed_spatial_label, dtype=np.int64).max()) + 1)

    @property
    def typed_answer_type_class_count(self) -> int:
        if self.typed_answer_type_class_count_override is not None:
            return int(self.typed_answer_type_class_count_override)
        if self.typed_answer_type is None:
            return 1
        return max(1, int(np.asarray(self.typed_answer_type, dtype=np.int64).max()) + 1)

    @classmethod
    def load(cls, path: str | Path) -> "WorkspaceGeometryAuxTable":
        table_path = Path(path)
        if not table_path.is_file():
            raise FileNotFoundError(f"workspace geometry aux npz does not exist: {table_path}")
        payload = np.load(table_path, allow_pickle=False)
        required_keys = (
            "row_ids",
            "source_row_ids",
            "context_tokens",
            "target_points",
            "token_offsets",
            "pose_features",
            "compass16",
            "range_bin",
            "valid_evidence",
            "invalid_reason",
            "answer_option",
            "spatial_option_compass16",
            "spatial_option_range_bin",
        )
        for key in required_keys:
            if key not in payload.files:
                raise KeyError(f"workspace geometry aux npz missing key={key}: {table_path}")

        row_ids_array = payload["row_ids"]
        source_row_ids_array = payload["source_row_ids"]
        context_tokens = np.asarray(payload["context_tokens"], dtype=np.float32)
        target_points = np.asarray(payload["target_points"], dtype=np.float32)
        token_offsets = np.asarray(payload["token_offsets"], dtype=np.int64)
        pose_features = np.asarray(payload["pose_features"], dtype=np.float32)
        compass16 = np.asarray(payload["compass16"], dtype=np.int64)
        range_bin = np.asarray(payload["range_bin"], dtype=np.int64)
        valid_evidence = np.asarray(payload["valid_evidence"], dtype=np.int64)
        if "answer_loss_mask" in payload.files:
            answer_loss_mask = np.asarray(payload["answer_loss_mask"], dtype=np.int64)
        else:
            # 旧 npz 没有这个字段。旧实验默认每一行答案都参与监督，
            # 所以这里补全 1，不改变历史实验口径。
            answer_loss_mask = np.ones_like(valid_evidence, dtype=np.int64)
        component_masks: dict[str, np.ndarray] = {}
        for mask_name in ("point_loss_mask", "center_loss_mask", "compass_loss_mask", "range_loss_mask"):
            if mask_name in payload.files:
                component_masks[mask_name] = np.asarray(payload[mask_name], dtype=np.int64)
            else:
                # 旧 aux npz 没有分项 mask。默认全部参与，保证历史实验严格兼容。
                component_masks[mask_name] = np.ones_like(valid_evidence, dtype=np.int64)
        invalid_reason = np.asarray(payload["invalid_reason"], dtype=np.int64)
        answer_option = np.asarray(payload["answer_option"], dtype=np.int64)
        spatial_option_compass16 = np.asarray(payload["spatial_option_compass16"], dtype=np.int64)
        spatial_option_range_bin = np.asarray(payload["spatial_option_range_bin"], dtype=np.int64)
        box_modes: tuple[str, ...] | None = None
        counterfactual_modes: tuple[str, ...] | None = None
        if "box_modes" in payload.files:
            box_modes_array = payload["box_modes"]
            if box_modes_array.ndim != 1:
                raise ValueError(f"box_modes must be rank-1, got {box_modes_array.shape}")
            box_modes = tuple(str(value) for value in box_modes_array.tolist())
        if "counterfactual_modes" in payload.files:
            counterfactual_modes_array = payload["counterfactual_modes"]
            if counterfactual_modes_array.ndim != 1:
                raise ValueError(f"counterfactual_modes must be rank-1, got {counterfactual_modes_array.shape}")
            counterfactual_modes = tuple(str(value) for value in counterfactual_modes_array.tolist())
        answer_option_class_count_override: int | None = None
        if "answer_option_class_count" in payload.files:
            answer_option_class_count_array = np.asarray(payload["answer_option_class_count"], dtype=np.int64)
            if answer_option_class_count_array.shape not in {(), (1,)}:
                raise ValueError(
                    "answer_option_class_count must be a scalar or length-1 array, "
                    f"got {answer_option_class_count_array.shape}"
                )
            answer_option_class_count_override = int(answer_option_class_count_array.reshape(-1)[0])
            if answer_option_class_count_override < 4:
                raise ValueError("answer_option_class_count must be >= 4")
        typed_required_keys = (
            "typed_bucket",
            "typed_numeric_scalar",
            "typed_numeric_scalar_raw",
            "typed_numeric_scalar_mask",
            "typed_numeric_kind",
            "typed_numeric_kind_mask",
            "typed_choice_label",
            "typed_choice_mask",
            "typed_choice_loss_mask",
            "typed_spatial_label",
            "typed_spatial_mask",
            "typed_answer_type",
            "typed_answer_type_mask",
            "typed_answer_embedding_mask",
            "typed_answer_texts",
            "typed_numeric_kind_class_count",
            "typed_spatial_class_count",
            "typed_answer_type_class_count",
        )
        typed_present = [key for key in typed_required_keys if key in payload.files]
        typed_values: dict[str, np.ndarray] = {}
        typed_answer_texts: tuple[str, ...] | None = None
        typed_numeric_kind_class_count_override: int | None = None
        typed_spatial_class_count_override: int | None = None
        typed_answer_type_class_count_override: int | None = None
        if typed_present:
            missing_typed = sorted(set(typed_required_keys) - set(typed_present))
            if missing_typed:
                raise KeyError(f"typed latent aux labels are incomplete; missing={missing_typed}")
            for key in typed_required_keys:
                if key == "typed_answer_texts":
                    typed_answer_texts = tuple(str(value) for value in payload[key].tolist())
                elif key.endswith("_class_count"):
                    scalar = np.asarray(payload[key], dtype=np.int64).reshape(-1)
                    if int(scalar.shape[0]) != 1:
                        raise ValueError(f"{key} must be scalar or length-1 array")
                    value = int(scalar[0])
                    if value <= 0:
                        raise ValueError(f"{key} must be positive, got {value}")
                    if key == "typed_numeric_kind_class_count":
                        typed_numeric_kind_class_count_override = value
                    elif key == "typed_spatial_class_count":
                        typed_spatial_class_count_override = value
                    else:
                        typed_answer_type_class_count_override = value
                elif key in {"typed_numeric_scalar", "typed_numeric_scalar_raw"}:
                    typed_values[key] = np.asarray(payload[key], dtype=np.float32)
                elif key == "typed_choice_mask":
                    typed_values[key] = np.asarray(payload[key], dtype=np.bool_)
                else:
                    typed_values[key] = np.asarray(payload[key], dtype=np.int64)

        if row_ids_array.ndim != 1:
            raise ValueError(f"row_ids must be rank-1, got {row_ids_array.shape}")
        if source_row_ids_array.ndim != 1:
            raise ValueError(f"source_row_ids must be rank-1, got {source_row_ids_array.shape}")
        if int(source_row_ids_array.shape[0]) != int(row_ids_array.shape[0]):
            raise ValueError("source_row_ids row count must match row_ids")
        if context_tokens.ndim != 2:
            raise ValueError(f"context_tokens must be [M,D], got {context_tokens.shape}")
        if target_points.ndim != 2 or int(target_points.shape[1]) != 3:
            raise ValueError(f"target_points must be [M,3], got {target_points.shape}")
        if int(context_tokens.shape[0]) != int(target_points.shape[0]):
            raise ValueError("context_tokens and target_points must have the same token count")
        if token_offsets.ndim != 1:
            raise ValueError(f"token_offsets must be rank-1, got {token_offsets.shape}")
        if int(token_offsets.shape[0]) != int(row_ids_array.shape[0]) + 1:
            raise ValueError("token_offsets length must equal row count + 1")
        if int(token_offsets[0]) != 0 or int(token_offsets[-1]) != int(context_tokens.shape[0]):
            raise ValueError("token_offsets must start at 0 and end at total token count")
        if np.any(token_offsets[1:] <= token_offsets[:-1]):
            raise ValueError("each geometry aux row must contain at least one token")
        row_count = int(row_ids_array.shape[0])
        if int(pose_features.shape[0]) != row_count:
            raise ValueError("pose_features row count must match row_ids")
        if int(compass16.shape[0]) != row_count or int(range_bin.shape[0]) != row_count:
            raise ValueError("relation label row count must match row_ids")
        if int(valid_evidence.shape[0]) != row_count:
            raise ValueError("valid_evidence row count must match row_ids")
        if int(answer_loss_mask.shape[0]) != row_count:
            raise ValueError("answer_loss_mask row count must match row_ids")
        if not np.isin(answer_loss_mask, np.asarray([0, 1], dtype=np.int64)).all():
            raise ValueError("answer_loss_mask must contain only 0/1 labels")
        for mask_name, mask_value in component_masks.items():
            if int(mask_value.shape[0]) != row_count:
                raise ValueError(f"{mask_name} row count must match row_ids")
            if not np.isin(mask_value, np.asarray([0, 1], dtype=np.int64)).all():
                raise ValueError(f"{mask_name} must contain only 0/1 labels")
        if box_modes is not None and len(box_modes) != row_count:
            raise ValueError("box_modes row count must match row_ids")
        if counterfactual_modes is not None and len(counterfactual_modes) != row_count:
            raise ValueError("counterfactual_modes row count must match row_ids")
        if int(invalid_reason.shape[0]) != row_count:
            raise ValueError("invalid_reason row count must match row_ids")
        if int(answer_option.shape[0]) != row_count:
            raise ValueError("answer_option row count must match row_ids")
        if spatial_option_compass16.shape != (row_count, 4):
            raise ValueError(f"spatial_option_compass16 must be [N,4], got {spatial_option_compass16.shape}")
        if spatial_option_range_bin.shape != (row_count, 4):
            raise ValueError(f"spatial_option_range_bin must be [N,4], got {spatial_option_range_bin.shape}")
        if typed_present:
            for key, value in typed_values.items():
                if key == "typed_choice_mask":
                    if value.shape != (row_count, 4):
                        raise ValueError(f"{key} must be [N,4], got {value.shape}")
                elif value.shape != (row_count,):
                    raise ValueError(f"{key} must be rank-1 with row count {row_count}, got {value.shape}")
            if typed_answer_texts is None or len(typed_answer_texts) != row_count:
                raise ValueError("typed_answer_texts row count must match row_ids")
            for mask_key in (
                "typed_numeric_scalar_mask",
                "typed_numeric_kind_mask",
                "typed_choice_loss_mask",
                "typed_spatial_mask",
                "typed_answer_type_mask",
                "typed_answer_embedding_mask",
            ):
                if not np.isin(typed_values[mask_key], np.asarray([0, 1], dtype=np.int64)).all():
                    raise ValueError(f"{mask_key} must contain only 0/1 labels")
            choice_mask = typed_values["typed_choice_mask"]
            if not bool(choice_mask.any(axis=1).all()):
                raise ValueError("typed_choice_mask must leave at least one legal option per row")
            choice_labels = typed_values["typed_choice_label"]
            choice_loss_mask = typed_values["typed_choice_loss_mask"]
            for index in np.nonzero(choice_loss_mask == 1)[0]:
                label = int(choice_labels[index])
                if label < 0 or label >= 4 or not bool(choice_mask[index, label]):
                    raise ValueError(f"typed choice label must be legal under option mask at row={index}")
            if not np.isfinite(typed_values["typed_numeric_scalar"]).all():
                raise ValueError("typed_numeric_scalar contains NaN or Inf")
            if not np.isfinite(typed_values["typed_numeric_scalar_raw"]).all():
                raise ValueError("typed_numeric_scalar_raw contains NaN or Inf")
        if not np.isin(valid_evidence, np.asarray([0, 1], dtype=np.int64)).all():
            raise ValueError("valid_evidence must contain only 0/1 labels")
        invalid_reason_class_count = max(DEFAULT_INVALID_REASON_CLASS_COUNT, int(invalid_reason.max()) + 1)
        if answer_option_class_count_override is not None:
            answer_option_class_count = int(answer_option_class_count_override)
        elif bool((valid_evidence == 1).all()) and bool((invalid_reason == 0).all()):
            answer_option_class_count = max(4, int(answer_option.max()) + 1)
        else:
            answer_option_class_count = max(4 + invalid_reason_class_count - 1, int(answer_option.max()) + 1)
        if int(invalid_reason.min()) < 0 or int(invalid_reason.max()) >= invalid_reason_class_count:
            raise ValueError(
                f"invalid_reason labels must be in [0, {invalid_reason_class_count - 1}]"
            )
        if int(answer_option.min()) < 0 or int(answer_option.max()) >= answer_option_class_count:
            raise ValueError(f"answer_option labels must be in [0, {answer_option_class_count - 1}]")
        if int(spatial_option_compass16.min()) < 0 or int(spatial_option_compass16.max()) >= 16:
            raise ValueError("spatial_option_compass16 labels must be in [0, 15]")
        if int(spatial_option_range_bin.min()) < 0 or int(spatial_option_range_bin.max()) >= 5:
            raise ValueError("spatial_option_range_bin labels must be in [0, 4]")
        if not np.array_equal((invalid_reason == 0).astype(np.int64), valid_evidence):
            raise ValueError("invalid_reason=0 must exactly match valid_evidence=1")
        if not np.isfinite(context_tokens).all():
            raise ValueError("context_tokens contains NaN or Inf")
        if not np.isfinite(target_points).all():
            raise ValueError("target_points contains NaN or Inf")
        if not np.isfinite(pose_features).all():
            raise ValueError("pose_features contains NaN or Inf")

        row_ids = tuple(str(value) for value in row_ids_array.tolist())
        source_row_ids = tuple(str(value) for value in source_row_ids_array.tolist())
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("workspace geometry aux row_ids must be unique")
        return cls(
            row_ids=row_ids,
            context_tokens=context_tokens,
            target_points=target_points,
            token_offsets=token_offsets,
            pose_features=pose_features,
            compass16=compass16,
            range_bin=range_bin,
            valid_evidence=valid_evidence,
            answer_loss_mask=answer_loss_mask,
            point_loss_mask=component_masks["point_loss_mask"],
            center_loss_mask=component_masks["center_loss_mask"],
            compass_loss_mask=component_masks["compass_loss_mask"],
            range_loss_mask=component_masks["range_loss_mask"],
            invalid_reason=invalid_reason,
            answer_option=answer_option,
            spatial_option_compass16=spatial_option_compass16,
            spatial_option_range_bin=spatial_option_range_bin,
            source_row_ids=source_row_ids,
            # 这些字段不参与普通几何 loss；它们只给反事实负样本选择器用。
            # 旧 npz 可能没有这些字段，真正请求 counterfactual negative 时会再显式报错。
            box_modes=box_modes,
            counterfactual_modes=counterfactual_modes,
            answer_option_class_count_override=answer_option_class_count_override,
            compass16_class_weights=balanced_class_weights_for_valid_rows_or_ones(
                labels=compass16,
                valid_evidence=valid_evidence * component_masks["compass_loss_mask"],
                class_count=16,
            ),
            range_bin_class_weights=balanced_class_weights_for_valid_rows_or_ones(
                labels=range_bin,
                valid_evidence=valid_evidence * component_masks["range_loss_mask"],
                class_count=5,
            ),
            invalid_reason_class_weights=balanced_class_weights_for_labels(
                labels=invalid_reason,
                class_count=invalid_reason_class_count,
            ),
            answer_option_class_weights=balanced_class_weights_for_labels(
                labels=answer_option[answer_loss_mask == 1] if bool((answer_loss_mask == 1).any()) else answer_option,
                class_count=answer_option_class_count,
            ),
            typed_bucket=typed_values.get("typed_bucket"),
            typed_numeric_scalar=typed_values.get("typed_numeric_scalar"),
            typed_numeric_scalar_raw=typed_values.get("typed_numeric_scalar_raw"),
            typed_numeric_scalar_mask=typed_values.get("typed_numeric_scalar_mask"),
            typed_numeric_kind=typed_values.get("typed_numeric_kind"),
            typed_numeric_kind_mask=typed_values.get("typed_numeric_kind_mask"),
            typed_choice_label=typed_values.get("typed_choice_label"),
            typed_choice_mask=typed_values.get("typed_choice_mask"),
            typed_choice_loss_mask=typed_values.get("typed_choice_loss_mask"),
            typed_spatial_label=typed_values.get("typed_spatial_label"),
            typed_spatial_mask=typed_values.get("typed_spatial_mask"),
            typed_answer_type=typed_values.get("typed_answer_type"),
            typed_answer_type_mask=typed_values.get("typed_answer_type_mask"),
            typed_answer_embedding_mask=typed_values.get("typed_answer_embedding_mask"),
            typed_answer_texts=typed_answer_texts,
            typed_numeric_kind_class_count_override=typed_numeric_kind_class_count_override,
            typed_spatial_class_count_override=typed_spatial_class_count_override,
            typed_answer_type_class_count_override=typed_answer_type_class_count_override,
        )

    @property
    def context_dim(self) -> int:
        return int(self.context_tokens.shape[1])

    @property
    def pose_dim(self) -> int:
        return int(self.pose_features.shape[1])

    def batch_for_indices(
        self,
        idx_values: list[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> WorkspaceGeometryAuxBatch:
        index_by_row_id = {row_id: index for index, row_id in enumerate(self.row_ids)}
        table_indices: list[int] = []
        for idx_value in idx_values:
            row_id = str(int(idx_value))
            if row_id not in index_by_row_id:
                raise KeyError(f"workspace geometry aux table missing idx={row_id}")
            table_indices.append(index_by_row_id[row_id])

        token_lengths = [
            int(self.token_offsets[index + 1] - self.token_offsets[index])
            for index in table_indices
        ]
        max_length = max(token_lengths)
        batch_size = len(table_indices)
        context = np.zeros((batch_size, max_length, self.context_dim), dtype=np.float32)
        target = np.zeros((batch_size, max_length, 3), dtype=np.float32)
        mask = np.zeros((batch_size, max_length), dtype=np.bool_)
        for batch_index, table_index in enumerate(table_indices):
            start = int(self.token_offsets[table_index])
            end = int(self.token_offsets[table_index + 1])
            length = end - start
            context[batch_index, :length] = self.context_tokens[start:end]
            target[batch_index, :length] = self.target_points[start:end]
            mask[batch_index, :length] = True
        zero_int = np.zeros((batch_size,), dtype=np.int64)
        zero_float = np.zeros((batch_size,), dtype=np.float32)
        default_choice_mask = np.ones((batch_size, 4), dtype=np.bool_)

        return WorkspaceGeometryAuxBatch(
            context_tokens=torch.tensor(context, device=device, dtype=dtype),
            target_points=torch.tensor(target, device=device, dtype=dtype),
            point_mask=torch.tensor(mask, device=device, dtype=torch.bool),
            pose_features=torch.tensor(self.pose_features[table_indices], device=device, dtype=dtype),
            compass16=torch.tensor(self.compass16[table_indices], device=device, dtype=torch.long),
            range_bin=torch.tensor(self.range_bin[table_indices], device=device, dtype=torch.long),
            valid_evidence=torch.tensor(self.valid_evidence[table_indices], device=device, dtype=torch.long),
            answer_loss_mask=torch.tensor(
                self.answer_loss_mask[table_indices]
                if self.answer_loss_mask is not None
                else np.ones((len(table_indices),), dtype=np.int64),
                device=device,
                dtype=torch.long,
            ),
            point_loss_mask=torch.tensor(
                self.point_loss_mask[table_indices]
                if self.point_loss_mask is not None
                else np.ones((len(table_indices),), dtype=np.int64),
                device=device,
                dtype=torch.long,
            ),
            center_loss_mask=torch.tensor(
                self.center_loss_mask[table_indices]
                if self.center_loss_mask is not None
                else np.ones((len(table_indices),), dtype=np.int64),
                device=device,
                dtype=torch.long,
            ),
            compass_loss_mask=torch.tensor(
                self.compass_loss_mask[table_indices]
                if self.compass_loss_mask is not None
                else np.ones((len(table_indices),), dtype=np.int64),
                device=device,
                dtype=torch.long,
            ),
            range_loss_mask=torch.tensor(
                self.range_loss_mask[table_indices]
                if self.range_loss_mask is not None
                else np.ones((len(table_indices),), dtype=np.int64),
                device=device,
                dtype=torch.long,
            ),
            invalid_reason=torch.tensor(self.invalid_reason[table_indices], device=device, dtype=torch.long),
            answer_option=torch.tensor(self.answer_option[table_indices], device=device, dtype=torch.long),
            spatial_option_compass16=torch.tensor(
                self.spatial_option_compass16[table_indices],
                device=device,
                dtype=torch.long,
            ),
            spatial_option_range_bin=torch.tensor(
                self.spatial_option_range_bin[table_indices],
                device=device,
                dtype=torch.long,
            ),
            typed_bucket=torch.tensor(
                self.typed_bucket[table_indices] if self.typed_bucket is not None else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_numeric_scalar=torch.tensor(
                self.typed_numeric_scalar[table_indices] if self.typed_numeric_scalar is not None else zero_float,
                device=device,
                dtype=dtype,
            ),
            typed_numeric_scalar_raw=torch.tensor(
                self.typed_numeric_scalar_raw[table_indices]
                if self.typed_numeric_scalar_raw is not None
                else zero_float,
                device=device,
                dtype=dtype,
            ),
            typed_numeric_scalar_mask=torch.tensor(
                self.typed_numeric_scalar_mask[table_indices]
                if self.typed_numeric_scalar_mask is not None
                else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_numeric_kind=torch.tensor(
                self.typed_numeric_kind[table_indices] if self.typed_numeric_kind is not None else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_numeric_kind_mask=torch.tensor(
                self.typed_numeric_kind_mask[table_indices]
                if self.typed_numeric_kind_mask is not None
                else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_choice_label=torch.tensor(
                self.typed_choice_label[table_indices] if self.typed_choice_label is not None else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_choice_mask=torch.tensor(
                self.typed_choice_mask[table_indices]
                if self.typed_choice_mask is not None
                else default_choice_mask,
                device=device,
                dtype=torch.bool,
            ),
            typed_choice_loss_mask=torch.tensor(
                self.typed_choice_loss_mask[table_indices]
                if self.typed_choice_loss_mask is not None
                else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_spatial_label=torch.tensor(
                self.typed_spatial_label[table_indices] if self.typed_spatial_label is not None else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_spatial_mask=torch.tensor(
                self.typed_spatial_mask[table_indices]
                if self.typed_spatial_mask is not None
                else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_answer_type=torch.tensor(
                self.typed_answer_type[table_indices] if self.typed_answer_type is not None else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_answer_type_mask=torch.tensor(
                self.typed_answer_type_mask[table_indices]
                if self.typed_answer_type_mask is not None
                else zero_int,
                device=device,
                dtype=torch.long,
            ),
            typed_answer_embedding_mask=torch.tensor(
                self.typed_answer_embedding_mask[table_indices]
                if self.typed_answer_embedding_mask is not None
                else zero_int,
                device=device,
                dtype=torch.long,
            ),
            source_row_ids=tuple(self.source_row_ids[index] for index in table_indices),
            typed_answer_texts=(
                tuple(self.typed_answer_texts[index] for index in table_indices)
                if self.typed_answer_texts is not None
                else tuple("" for _ in table_indices)
            ),
            compass16_class_weights=torch.tensor(
                self._compass16_class_weights(),
                device=device,
                dtype=torch.float32,
            ),
            range_bin_class_weights=torch.tensor(
                self._range_bin_class_weights(),
                device=device,
                dtype=torch.float32,
            ),
            invalid_reason_class_weights=torch.tensor(
                self._invalid_reason_class_weights(),
                device=device,
                dtype=torch.float32,
            ),
            answer_option_class_weights=torch.tensor(
                self._answer_option_class_weights(),
                device=device,
                dtype=torch.float32,
            ),
        )

    def _compass16_class_weights(self) -> np.ndarray:
        if self.compass16_class_weights is not None:
            return np.asarray(self.compass16_class_weights, dtype=np.float32)
        return balanced_class_weights_for_valid_rows_or_ones(
            labels=self.compass16,
            valid_evidence=(
                self.valid_evidence
                if self.compass_loss_mask is None
                else np.asarray(self.valid_evidence, dtype=np.int64) * np.asarray(self.compass_loss_mask, dtype=np.int64)
            ),
            class_count=16,
        )

    def _range_bin_class_weights(self) -> np.ndarray:
        if self.range_bin_class_weights is not None:
            return np.asarray(self.range_bin_class_weights, dtype=np.float32)
        return balanced_class_weights_for_valid_rows_or_ones(
            labels=self.range_bin,
            valid_evidence=(
                self.valid_evidence
                if self.range_loss_mask is None
                else np.asarray(self.valid_evidence, dtype=np.int64) * np.asarray(self.range_loss_mask, dtype=np.int64)
            ),
            class_count=5,
        )

    def _invalid_reason_class_weights(self) -> np.ndarray:
        if self.invalid_reason_class_weights is not None:
            return np.asarray(self.invalid_reason_class_weights, dtype=np.float32)
        return balanced_class_weights_for_labels(
            labels=self.invalid_reason,
            class_count=self.invalid_reason_class_count,
        )

    def _answer_option_class_weights(self) -> np.ndarray:
        if self.answer_option_class_weights is not None:
            return np.asarray(self.answer_option_class_weights, dtype=np.float32)
        if self.answer_loss_mask is not None and bool((self.answer_loss_mask == 1).any()):
            return balanced_class_weights_for_labels(
                labels=self.answer_option[self.answer_loss_mask == 1],
                class_count=self.answer_option_class_count,
            )
        return balanced_class_weights_for_labels(
            labels=self.answer_option,
            class_count=self.answer_option_class_count,
        )


def balanced_class_weights_for_labels(*, labels: np.ndarray, class_count: int) -> np.ndarray:
    """按整张表的标签分布生成 class-balanced CE 权重。

    invalid reason 和 A-G option 不是只在 valid 行上训练。
    它们要覆盖 correct/wrong/shifted/wrong-pose 全部样本，所以这里直接用全体标签分布。
    """

    if int(class_count) <= 0:
        raise ValueError("class_count must be positive")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be rank-1, got {labels.shape}")
    if int(labels.min()) < 0 or int(labels.max()) >= int(class_count):
        raise ValueError("label value out of class range when building class weights")
    counts = np.bincount(labels, minlength=int(class_count)).astype(np.float32)
    present = counts > 0
    weights = np.zeros((int(class_count),), dtype=np.float32)
    weights[present] = counts[present].sum() / (float(present.sum()) * counts[present])
    return weights


def balanced_class_weights_for_valid_rows(
    *,
    labels: np.ndarray,
    valid_evidence: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """按 valid evidence 的全局标签分布生成 class-balanced CE 权重。

    组 batch 里每个 source 只有 1 个 correct 行。
    如果 direction/range 标签本来就偏，普通 CE 会鼓励模型永远猜最多的类。
    这里用全局正样本分布给少数类更大权重，避免 relation head 学成多数类常数。
    """

    if int(class_count) <= 0:
        raise ValueError("class_count must be positive")
    labels = np.asarray(labels, dtype=np.int64)
    valid_evidence = np.asarray(valid_evidence, dtype=np.int64)
    if labels.shape != valid_evidence.shape:
        raise ValueError("labels and valid_evidence must have the same shape")
    positive_labels = labels[valid_evidence == 1]
    if positive_labels.size == 0:
        raise ValueError("cannot build class weights without valid evidence rows")
    if int(positive_labels.min()) < 0 or int(positive_labels.max()) >= int(class_count):
        raise ValueError("label value out of class range when building class weights")
    counts = np.bincount(positive_labels, minlength=int(class_count)).astype(np.float32)
    present = counts > 0
    weights = np.zeros((int(class_count),), dtype=np.float32)
    weights[present] = counts[present].sum() / (float(present.sum()) * counts[present])
    return weights


def balanced_class_weights_for_valid_rows_or_ones(
    *,
    labels: np.ndarray,
    valid_evidence: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """和 balanced_class_weights_for_valid_rows 相同，但允许该分项没有启用行。

    VLM-3R 的分项 mask 可能把 compass 全关掉。这不是数据错误，只表示这个 head 本轮不训练；
    这时返回全 1 权重即可，真正的 loss 会因为 row mask 为空而变成 0。
    """

    valid_evidence_array = np.asarray(valid_evidence, dtype=np.int64)
    if int(valid_evidence_array.sum()) == 0:
        return np.ones((int(class_count),), dtype=np.float32)
    return balanced_class_weights_for_valid_rows(
        labels=labels,
        valid_evidence=valid_evidence_array,
        class_count=class_count,
    )


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    """发现 NaN/Inf 就立刻报错，不继续训练假结果。

    这个实验最怕“训练看似跑完，但其实某个中间张量早就炸了”。
    所以 auxiliary head 的关键输入/输出都做硬检查；一旦非有限，直接停。
    """

    if not bool(torch.isfinite(tensor.detach()).all().item()):
        finite = torch.isfinite(tensor.detach())
        finite_values = tensor.detach()[finite]
        if int(finite_values.numel()) == 0:
            raise ValueError(f"{name} contains no finite values")
        raise ValueError(
            f"{name} contains NaN or Inf: "
            f"finite_min={float(finite_values.float().min().item())}, "
            f"finite_max={float(finite_values.float().max().item())}"
        )


class WorkspaceSigRegProjectionHead(nn.Module):
    """给 workspace updated-summary latent 做 SIGReg 前的投影头。

    不直接把 Qwen hidden state 强行压成高斯。先过一个小投影头，再在投影空间里
    检查表示有没有塌成一团。这样 SIGReg 更像“体检”，不是直接改写语言模型脑子。
    """

    def __init__(self, hidden_size: int, projection_dim: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.projection_dim = int(projection_dim)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.proj = nn.Linear(self.hidden_size, self.projection_dim, bias=False)
        self.norm = nn.BatchNorm1d(self.projection_dim, affine=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"SIGReg hidden_states must be [B,L,D], got {tuple(hidden_states.shape)}")
        if int(hidden_states.shape[-1]) != self.hidden_size:
            raise ValueError(
                "SIGReg hidden size mismatch: "
                f"got={int(hidden_states.shape[-1])}, expected={self.hidden_size}"
            )
        flat = hidden_states.float().reshape(-1, self.hidden_size)
        if int(flat.shape[0]) < 2:
            raise ValueError("SIGReg projection needs at least two latent tokens in the batch")
        projected = self.proj(flat)
        projected = self.norm(projected)
        projected = projected.reshape(hidden_states.shape[0], hidden_states.shape[1], self.projection_dim)
        _require_finite("sigreg_projected_hidden_states", projected)
        return projected


def compute_workspace_sigreg_loss(
    embeddings: torch.Tensor,
    *,
    num_projections: int,
    num_integration_knots: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 Sketched-Isotropic-Gaussian Regularizer。

    把一批 latent 沿很多个随机方向压成一维；如果每个一维投影都像标准高斯，
    那整批 latent 就不太可能全部挤在同一个点上。
    """

    if embeddings.ndim != 3:
        raise ValueError(f"SIGReg embeddings must be [B,L,D], got {tuple(embeddings.shape)}")
    if int(num_projections) <= 0:
        raise ValueError("num_projections must be positive")
    if int(num_integration_knots) < 2:
        raise ValueError("num_integration_knots must be >= 2")

    flat = embeddings.float().reshape(-1, embeddings.shape[-1])
    sample_count = int(flat.shape[0])
    dim = int(flat.shape[1])
    if sample_count < 2:
        raise ValueError("SIGReg needs at least two projected embeddings")
    _require_finite("sigreg_embeddings", flat)

    generator = torch.Generator(device=flat.device)
    generator.manual_seed(int(seed))
    directions = torch.randn(
        int(num_projections),
        dim,
        generator=generator,
        device=flat.device,
        dtype=flat.dtype,
    )
    directions = F.normalize(directions, dim=-1)
    projections = flat @ directions.t()

    knots = torch.linspace(
        0.2,
        4.0,
        steps=int(num_integration_knots),
        device=flat.device,
        dtype=flat.dtype,
    )
    phases = projections.unsqueeze(-1) * knots.view(1, 1, -1)
    empirical_real = torch.cos(phases).mean(dim=0)
    empirical_imag = torch.sin(phases).mean(dim=0)
    gaussian_real = torch.exp(-0.5 * knots.pow(2)).view(1, -1)
    gaussian_imag = torch.zeros_like(gaussian_real)
    weight = torch.exp(-0.5 * knots.pow(2)).view(1, -1)
    squared_error = (empirical_real - gaussian_real).pow(2) + (empirical_imag - gaussian_imag).pow(2)
    sigreg_by_projection = torch.trapezoid(weight * squared_error, knots, dim=-1)
    loss = sigreg_by_projection.mean()
    _require_finite("sigreg_loss", loss)

    centered = flat - flat.mean(dim=0, keepdim=True)
    feature_std = flat.std(dim=0, unbiased=False)
    unit = F.normalize(centered, dim=-1)
    cosine_matrix = unit @ unit.t()
    off_diagonal_mask = ~torch.eye(sample_count, device=flat.device, dtype=torch.bool)
    off_diagonal_cosine = cosine_matrix[off_diagonal_mask].mean()
    metrics = {
        "workspace_sigreg_loss": loss.detach(),
        "workspace_sigreg_projection_mean_abs": flat.mean(dim=0).abs().mean().detach(),
        "workspace_sigreg_projection_std_mean": feature_std.mean().detach(),
        "workspace_sigreg_projection_std_min": feature_std.min().detach(),
        "workspace_sigreg_offdiag_cosine": off_diagonal_cosine.detach(),
    }
    return loss, metrics


class CrossAttentionFfnBlock(nn.Module):
    """一个小的 cross-attention + FFN 块。

    它让一组 latent slot 去读取另一组 token。
    这里不是为了做花哨结构，只是把“某一组 latent 主动读另一组 token”这件事写清楚。
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.cross_attn(
            query,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        query = self.norm1(query + attended)
        return self.norm2(query + self.ffn(query))


class WorkspaceGeometryAuxHead(nn.Module):
    """用 summary latent 预测 source-pose geometry，并把结果写回 updated summary。

    这版严格对应以下设计结构：
    old summary -> work latent -> updated summary -> Response。
    - old summary 先把 source 几何上下文压成一个旧摘要，并接受 source-pose 几何监督；
    - work latent 负责读 old summary、source context 和 source->query pose，做空间转换草稿；
    - updated summary 是唯一被 aux 写回的地方；
    - 后续 answer forward 继续走 MLLM 自回归路径，而不是 aux 旁路直接选答案。
    """

    def __init__(
        self,
        *,
        llm_hidden_size: int,
        context_dim: int,
        pose_dim: int,
        summary_latent_size: int,
        work_latent_size: int,
        updated_summary_latent_size: int,
        aux_hidden_size: int,
        num_heads: int,
        dropout: float,
        range_bin_count: int,
        invalid_reason_count: int = DEFAULT_INVALID_REASON_CLASS_COUNT,
        updated_summary_delta_scale: float = 0.1,
        use_context_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.summary_latent_size = int(summary_latent_size)
        self.work_latent_size = int(work_latent_size)
        self.updated_summary_latent_size = int(updated_summary_latent_size)
        self.invalid_reason_count = int(invalid_reason_count)
        self.use_context_tokens = bool(use_context_tokens)
        if self.invalid_reason_count < DEFAULT_INVALID_REASON_CLASS_COUNT:
            raise ValueError(
                f"invalid_reason_count must be >= {DEFAULT_INVALID_REASON_CLASS_COUNT}, "
                f"got {self.invalid_reason_count}"
            )
        if float(updated_summary_delta_scale) <= 0.0:
            raise ValueError("updated_summary_delta_scale must be positive")
        self.updated_summary_delta_scale = float(updated_summary_delta_scale)
        self.latent_input_norm = nn.LayerNorm(llm_hidden_size)
        self.latent_proj = nn.Linear(llm_hidden_size, aux_hidden_size)
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, aux_hidden_size),
            nn.GELU(),
            nn.LayerNorm(aux_hidden_size),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, aux_hidden_size),
            nn.GELU(),
            nn.Linear(aux_hidden_size, aux_hidden_size),
        )
        self.summary_reader = CrossAttentionFfnBlock(aux_hidden_size, num_heads, dropout)
        self.work_reader = CrossAttentionFfnBlock(aux_hidden_size, num_heads, dropout)
        self.updated_summary_writer = CrossAttentionFfnBlock(aux_hidden_size, num_heads, dropout)
        self.latent_out_proj = nn.Sequential(
            nn.LayerNorm(aux_hidden_size),
            nn.Linear(aux_hidden_size, llm_hidden_size),
        )
        if not self.use_context_tokens:
            # strict VLM-3R top-k 路径里，summary 已经是 picker 从证据帧读出的摘要。
            # work / updated_summary 不能再从 Qwen 原生 hidden 继承信息，否则就是旁路。
            # 所以这里用可训练槽位当“空白稿纸”，只允许它们读 selected summary 和 pose。
            self.strict_work_slot_embedding = nn.Embedding(self.work_latent_size, aux_hidden_size)
            self.strict_updated_summary_slot_embedding = nn.Embedding(
                self.updated_summary_latent_size,
                aux_hidden_size,
            )
            self.strict_updated_summary_output_base = nn.Embedding(
                self.updated_summary_latent_size,
                llm_hidden_size,
            )
        self.point_head = nn.Sequential(
            nn.LayerNorm(aux_hidden_size * 2),
            nn.Linear(aux_hidden_size * 2, aux_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden_size, 3),
        )
        # compass/range 不能只读 work latent。
        # 前面 point_head 已经预测了“目标点在 source pose 下的位置”，
        # 离散方向/距离应该贴着这个几何结果走，否则 relation head 很容易退化成标签先验。
        relation_hidden_size = aux_hidden_size + 3
        self.compass_head = nn.Sequential(nn.LayerNorm(relation_hidden_size), nn.Linear(relation_hidden_size, 16))
        self.range_head = nn.Sequential(nn.LayerNorm(relation_hidden_size), nn.Linear(relation_hidden_size, range_bin_count))
        self.valid_head = nn.Sequential(nn.LayerNorm(aux_hidden_size), nn.Linear(aux_hidden_size, 2))
        self.invalid_reason_head = nn.Sequential(
            nn.LayerNorm(aux_hidden_size),
            nn.Linear(aux_hidden_size, self.invalid_reason_count),
        )

    def forward(self, latent_hidden_states: torch.Tensor, batch: WorkspaceGeometryAuxBatch) -> dict[str, torch.Tensor]:
        if latent_hidden_states.ndim != 3:
            raise ValueError(f"latent_hidden_states must be [B,L,D], got {latent_hidden_states.shape}")
        expected_latents = (
            self.summary_latent_size + self.work_latent_size + self.updated_summary_latent_size
        )
        if int(latent_hidden_states.shape[1]) != expected_latents:
            raise ValueError(
                "workspace latent length mismatch: "
                f"expected={expected_latents}, got={int(latent_hidden_states.shape[1])}"
            )

        # geometry aux 是一个小头，但它同时要吃 pointmap、pose、Qwen hidden，
        # 再把结果写回大模型 latent。这里按 head 参数 dtype 统一计算。
        # 主训练会把这个 head 固定放在 float32；不要用 bf16 训练这个小头，
        # 否则 AdamW 更新后很容易把 head 权重打成 NaN。
        target_dtype = self.latent_proj.weight.dtype
        target_device = self.latent_proj.weight.device
        latent_hidden_states = latent_hidden_states.to(device=target_device, dtype=target_dtype)
        context_tokens = batch.context_tokens.to(device=target_device, dtype=target_dtype)
        pose_features = batch.pose_features.to(device=target_device, dtype=target_dtype)
        point_mask = batch.point_mask.to(device=target_device)
        _require_finite("latent_hidden_states", latent_hidden_states)
        _require_finite("context_tokens", context_tokens)
        _require_finite("pose_features", pose_features)

        # Qwen 的 workspace hidden state 数值尺度很大，真实实验里 max 可能到几百。
        # 如果直接 Linear 投影，它会压过 pointmap context 和 pose，aux 分支就学成“看 latent 噪声”。
        # 这里先做 LayerNorm，让 summary/work latent 只提供“槽位内容”，不要靠原始幅值主导几何分支。
        normalized_latents = self.latent_input_norm(latent_hidden_states)
        summary_end = self.summary_latent_size
        work_end = summary_end + self.work_latent_size
        summary_raw = normalized_latents[:, :summary_end, :]
        work_raw = normalized_latents[:, summary_end:work_end, :]
        updated_summary_raw = normalized_latents[:, work_end:, :]
        pose_hidden = self.pose_proj(pose_features).unsqueeze(1)

        # summary 是“原始观察记忆”，它不应该提前混入 source->query pose。
        # pose 只在后面的 work/update 阶段使用，否则 summary head 会被训练成 query 条件表征，
        # 与“先看见，再搬运”的 pose-conditioned flow 结构相反。
        summary = self.latent_proj(summary_raw)
        if self.use_context_tokens:
            context_hidden = self.context_proj(context_tokens)
            context_padding_mask = ~point_mask
            work = self.latent_proj(work_raw) + pose_hidden
            updated_summary_query = self.latent_proj(updated_summary_raw) + pose_hidden

            # old summary 只读 source object 的几何上下文，形成“这条历史证据是什么”的摘要。
            # 真正执行 source->query 空间转换的是 work latent；后面写回 updated summary 时，
            # 只允许通过 summary/work 这两块传递信息，避免 answer 分支绕开 workspace。
            summary = self.summary_reader(summary, context_hidden, memory_padding_mask=context_padding_mask)

            work_memory = torch.cat([summary, context_hidden], dim=1)
            summary_padding = torch.zeros(
                (context_tokens.shape[0], self.summary_latent_size),
                device=target_device,
                dtype=torch.bool,
            )
            work_memory_padding = torch.cat([summary_padding, context_padding_mask], dim=1)
            work = self.work_reader(work, work_memory, memory_padding_mask=work_memory_padding)

            update_memory = torch.cat([summary, work], dim=1)
            update_memory_padding = torch.zeros(
                (context_tokens.shape[0], self.summary_latent_size + self.work_latent_size),
                device=target_device,
                dtype=torch.bool,
            )
            updated_summary = self.updated_summary_writer(
                updated_summary_query,
                update_memory,
                memory_padding_mask=update_memory_padding,
            )
        else:
            # strict no-context 模式是为了验证“答案信息必须来自 picker 选出的 top-k 证据帧”。
            # 这里如果几何表塞进了任何非零 dense token，直接报错；不静默忽略。
            if context_tokens.ndim != 3:
                raise ValueError(f"strict no-context context_tokens must be [B,T,D], got {context_tokens.shape}")
            if int(context_tokens.shape[1]) != 1:
                raise ValueError(
                    "strict no-context geometry table must contain exactly one zero sentinel token per row, "
                    f"got token_count={int(context_tokens.shape[1])}"
                )
            if not bool(point_mask.all().item()):
                raise ValueError("strict no-context geometry table sentinel token must be marked valid")
            if int(torch.count_nonzero(context_tokens).item()) != 0:
                raise ValueError("strict no-context geometry table contains non-zero dense context tokens")

            summary_padding = torch.zeros(
                (context_tokens.shape[0], self.summary_latent_size),
                device=target_device,
                dtype=torch.bool,
            )
            work_slot_ids = torch.arange(
                self.work_latent_size,
                device=target_device,
                dtype=torch.long,
            ).view(1, self.work_latent_size)
            work = self.strict_work_slot_embedding(work_slot_ids).expand(context_tokens.shape[0], -1, -1)
            work = work + pose_hidden
            work = self.work_reader(work, summary, memory_padding_mask=summary_padding)

            updated_slot_ids = torch.arange(
                self.updated_summary_latent_size,
                device=target_device,
                dtype=torch.long,
            ).view(1, self.updated_summary_latent_size)
            updated_summary_query = self.strict_updated_summary_slot_embedding(updated_slot_ids).expand(
                context_tokens.shape[0],
                -1,
                -1,
            )
            updated_summary_query = updated_summary_query + pose_hidden
            update_memory = torch.cat([summary, work], dim=1)
            update_memory_padding = torch.zeros(
                (context_tokens.shape[0], self.summary_latent_size + self.work_latent_size),
                device=target_device,
                dtype=torch.bool,
            )
            updated_summary = self.updated_summary_writer(
                updated_summary_query,
                update_memory,
                memory_padding_mask=update_memory_padding,
            )
            context_hidden = torch.zeros(
                (context_tokens.shape[0], context_tokens.shape[1], summary.shape[-1]),
                device=target_device,
                dtype=target_dtype,
            )

        summary_pooled = summary.mean(dim=1)
        summary_per_point = summary_pooled.unsqueeze(1).expand(-1, context_hidden.shape[1], -1)
        point_input = torch.cat([context_hidden, summary_per_point], dim=-1)
        # aux 只能“小幅改写” updated summary，不能把 Qwen hidden 整段推飞。
        # 这里用 tanh + scale 把写回 residual 变成有界更新：
        # - 有梯度，可以训练；
        # - 不会因为一两个 batch 的 answer CE 反传过大，下一步直接 NaN；
        # - 结构上仍然是 work latent 更新 summary，再回到自回归回答路径。
        raw_updated_summary_delta = self.latent_out_proj(updated_summary)
        updated_summary_delta = torch.tanh(raw_updated_summary_delta) * self.updated_summary_delta_scale

        predicted_points = self.point_head(point_input)
        _require_finite("predicted_points", predicted_points)
        _require_finite("updated_summary_delta", updated_summary_delta)
        point_mask_float = point_mask.unsqueeze(-1).to(dtype=predicted_points.dtype)
        point_count = point_mask_float.sum(dim=1).clamp_min(1.0)
        predicted_center = (predicted_points * point_mask_float).sum(dim=1) / point_count
        relation_input = torch.cat([summary_pooled, predicted_center], dim=-1)

        # auxiliary head 不能再把所有 latent 一起改掉。
        # 设计上 work latent 只读几何，再把结果写回 updated summary。
        # 因此 old summary/work 保持 MLLM 原生 hidden state，只有 updated summary 段加 residual。
        if self.use_context_tokens:
            updated_summary_hidden_states = latent_hidden_states[:, work_end:, :] + updated_summary_delta.to(
                device=latent_hidden_states.device,
                dtype=latent_hidden_states.dtype,
            )
        else:
            updated_slot_ids = torch.arange(
                self.updated_summary_latent_size,
                device=target_device,
                dtype=torch.long,
            ).view(1, self.updated_summary_latent_size)
            updated_summary_hidden_states = self.strict_updated_summary_output_base(updated_slot_ids).expand(
                latent_hidden_states.shape[0],
                -1,
                -1,
            )
            updated_summary_hidden_states = updated_summary_hidden_states + updated_summary_delta.to(
                device=target_device,
                dtype=target_dtype,
            )
            updated_summary_hidden_states = updated_summary_hidden_states.to(
                device=latent_hidden_states.device,
                dtype=latent_hidden_states.dtype,
            )
        updated_latent_hidden_states = torch.cat(
            [latent_hidden_states[:, :work_end, :], updated_summary_hidden_states],
            dim=1,
        )
        compass_logits = self.compass_head(relation_input)
        range_logits = self.range_head(relation_input)
        valid_logits = self.valid_head(summary_pooled)
        invalid_reason_logits = self.invalid_reason_head(summary_pooled)
        _require_finite("compass_logits", compass_logits)
        _require_finite("range_logits", range_logits)
        _require_finite("valid_logits", valid_logits)
        _require_finite("invalid_reason_logits", invalid_reason_logits)
        return {
            "points": predicted_points,
            "compass_logits": compass_logits,
            "range_logits": range_logits,
            "valid_logits": valid_logits,
            "invalid_reason_logits": invalid_reason_logits,
            "updated_latent_hidden_states": updated_latent_hidden_states,
            "updated_summary_hidden_states": updated_summary_hidden_states,
            "updated_summary_delta": updated_summary_delta,
            "raw_updated_summary_delta": raw_updated_summary_delta,
        }


class WorkspaceAnswerClassifierHead(nn.Module):
    """只读 updated memory 的答案分类头。

    geometry aux 的职责是“把旧空间记忆按 query pose 转成 updated memory”。
    它不是最终答案老师。这个 head 才是答案老师：
    - 输入：updated 后的 workspace latent、query pose、选项的方向/距离/无效原因描述；
    - 不输入：RGB 图像 token、整段视觉特征；
    - 输出：A-H 每个选项的 logit。

    这样后续自回归头对齐的是“updated memory 能推出哪个答案”，
    而不是拿几何 head 的 point/valid 子任务硬拼成 teacher。
    """

    def __init__(
        self,
        *,
        llm_hidden_size: int,
        pose_dim: int,
        summary_latent_size: int,
        work_latent_size: int,
        updated_summary_latent_size: int,
        classifier_hidden_size: int,
        dropout: float,
        range_bin_count: int,
        invalid_reason_count: int,
        answer_option_class_count: int,
        latent_scope: str = "updated_summary",
    ) -> None:
        super().__init__()
        self.summary_latent_size = int(summary_latent_size)
        self.work_latent_size = int(work_latent_size)
        self.updated_summary_latent_size = int(updated_summary_latent_size)
        self.invalid_reason_count = int(invalid_reason_count)
        self.answer_option_class_count = int(answer_option_class_count)
        self.latent_scope = str(latent_scope)
        binary_invalid_option_count = 5
        expected_option_count = 4 + self.invalid_reason_count - 1
        if self.answer_option_class_count not in {4, binary_invalid_option_count, expected_option_count}:
            raise ValueError(
                "answer_option_class_count must be 4 for spatial-only, 5 for binary invalid, "
                "or 4 + invalid_reason_count - 1 for reason-specific counterfactual tasks, "
                f"got answer_option_class_count={self.answer_option_class_count}, "
                f"invalid_reason_count={self.invalid_reason_count}"
            )
        if self.latent_scope not in {"updated_summary", "all_workspace"}:
            raise ValueError(f"unsupported latent_scope={self.latent_scope!r}")

        self.latent_norm = nn.LayerNorm(llm_hidden_size)
        self.latent_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, classifier_hidden_size),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_size),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, classifier_hidden_size),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_size),
        )
        self.option_text_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, classifier_hidden_size),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_size),
        )
        self.option_type_embedding = nn.Embedding(2, classifier_hidden_size)
        self.option_norm = nn.LayerNorm(classifier_hidden_size)
        self.option_scorer = nn.Sequential(
            nn.LayerNorm(classifier_hidden_size * 4),
            nn.Linear(classifier_hidden_size * 4, classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, 1),
        )

    def _select_latents(self, latent_hidden_states: torch.Tensor) -> torch.Tensor:
        """按配置取 answer classifier 可以看的 workspace latent。"""

        expected_latents = (
            self.summary_latent_size + self.work_latent_size + self.updated_summary_latent_size
        )
        if latent_hidden_states.ndim != 3:
            raise ValueError(f"latent_hidden_states must be [B,L,D], got {latent_hidden_states.shape}")
        if int(latent_hidden_states.shape[1]) != expected_latents:
            raise ValueError(
                "workspace latent length mismatch for answer classifier: "
                f"expected={expected_latents}, got={int(latent_hidden_states.shape[1])}"
            )
        if self.latent_scope == "all_workspace":
            return latent_hidden_states
        start = self.summary_latent_size + self.work_latent_size
        return latent_hidden_states[:, start:, :]

    def forward(
        self,
        latent_hidden_states: torch.Tensor,
        batch: WorkspaceGeometryAuxBatch,
        option_text_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_dtype = self.option_type_embedding.weight.dtype
        target_device = self.option_type_embedding.weight.device
        selected_latents = self._select_latents(latent_hidden_states).to(
            device=target_device,
            dtype=target_dtype,
        )
        option_text_embeddings = option_text_embeddings.to(device=target_device, dtype=target_dtype)
        pose_features = batch.pose_features.to(device=target_device, dtype=target_dtype)
        _require_finite("answer_classifier_latents", selected_latents)
        _require_finite("answer_classifier_option_text_embeddings", option_text_embeddings)
        _require_finite("answer_classifier_pose_features", pose_features)
        if option_text_embeddings.ndim != 3:
            raise ValueError(
                f"option_text_embeddings must be [B,C,D], got {option_text_embeddings.shape}"
            )
        if option_text_embeddings.shape[:2] != (selected_latents.shape[0], self.answer_option_class_count):
            raise ValueError(
                "option_text_embeddings shape mismatch: "
                f"expected=({selected_latents.shape[0]}, {self.answer_option_class_count}, D), "
                f"got={tuple(option_text_embeddings.shape)}"
            )
        option_text_hidden = self.option_text_proj(option_text_embeddings)

        latent_tokens = self.latent_proj(self.latent_norm(selected_latents))
        pose_hidden = self.pose_proj(pose_features)
        memory = latent_tokens.mean(dim=1) + pose_hidden

        # 选项语义来自真实 option text embedding，不再靠 compass/range id 手工喂答案。
        # 这里唯一保留的是“空间选项/无效证据选项”的粗类型 embedding；
        # 这不是答案标签，而是让 classifier 知道 A-D 和 E-H 的选项性质不同。
        spatial_type = torch.zeros((1,), device=target_device, dtype=torch.long)
        invalid_type = torch.ones((1,), device=target_device, dtype=torch.long)
        spatial_options = (
            option_text_hidden[:, :4, :]
            + self.option_type_embedding(spatial_type).view(1, 1, -1)
        )
        invalid_options = (
            option_text_hidden[:, 4:, :]
            + self.option_type_embedding(invalid_type).view(1, 1, -1)
        )
        option_hidden = self.option_norm(torch.cat([spatial_options, invalid_options], dim=1))
        if int(option_hidden.shape[1]) != self.answer_option_class_count:
            raise ValueError(
                "answer classifier option count mismatch: "
                f"built={int(option_hidden.shape[1])}, expected={self.answer_option_class_count}"
            )

        memory_per_option = memory.unsqueeze(1).expand(-1, option_hidden.shape[1], -1)
        scorer_input = torch.cat(
            [
                memory_per_option,
                option_hidden,
                memory_per_option * option_hidden,
                torch.abs(memory_per_option - option_hidden),
            ],
            dim=-1,
        )
        answer_option_logits = self.option_scorer(scorer_input).squeeze(-1)
        _require_finite("answer_option_logits", answer_option_logits)
        return {"answer_option_logits": answer_option_logits}

class WorkspaceTypedLatentAuxHead(nn.Module):
    """只读 typed updated_summary slot 的辅助监督头。

    20 个 updated_summary slot 被固定分成 5 组，每组 4 个。
    这个 head 不看整段 workspace，也不看 RGB/候选特征，只检查对应那 4 个 slot
    里是不是已经刻进了数量、选择、空间关系和答案类型这些“可计算信息”。
    """

    GROUP_SLICES = {
        "general": (0, 4),
        "numeric": (4, 8),
        "choice": (8, 12),
        "spatial": (12, 16),
        "free_text": (16, 20),
    }

    def __init__(
        self,
        *,
        llm_hidden_size: int,
        updated_summary_latent_size: int,
        aux_hidden_size: int,
        numeric_kind_class_count: int,
        spatial_class_count: int,
        answer_type_class_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(updated_summary_latent_size) != 20:
            raise ValueError(
                "typed latent aux requires workspace_updated_summary_latent_size=20, "
                f"got {updated_summary_latent_size}"
            )
        self.llm_hidden_size = int(llm_hidden_size)
        self.updated_summary_latent_size = int(updated_summary_latent_size)
        self.numeric_kind_class_count = int(numeric_kind_class_count)
        self.spatial_class_count = int(spatial_class_count)
        self.answer_type_class_count = int(answer_type_class_count)
        if self.numeric_kind_class_count <= 0 or self.spatial_class_count <= 0 or self.answer_type_class_count <= 0:
            raise ValueError("typed latent aux class counts must be positive")
        group_dim = int(llm_hidden_size) * 4

        def build_mlp(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(group_dim),
                nn.Linear(group_dim, aux_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden_size, output_dim),
            )

        self.numeric_scalar_head = build_mlp(1)
        self.numeric_kind_head = build_mlp(self.numeric_kind_class_count)
        self.choice_head = build_mlp(4)
        self.spatial_head = build_mlp(self.spatial_class_count)
        self.answer_type_head = build_mlp(self.answer_type_class_count)
        self.answer_embedding_head = build_mlp(self.llm_hidden_size)

    def _group(self, updated_summary_hidden_states: torch.Tensor, name: str) -> torch.Tensor:
        start, end = self.GROUP_SLICES[name]
        group = updated_summary_hidden_states[:, start:end, :]
        if int(group.shape[1]) != 4:
            raise ValueError(f"typed latent group {name} must contain 4 slots, got {tuple(group.shape)}")
        return group.reshape(group.shape[0], -1)

    def forward(self, updated_summary_hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        if updated_summary_hidden_states.ndim != 3:
            raise ValueError(
                "updated_summary_hidden_states must be [B,20,D], "
                f"got {tuple(updated_summary_hidden_states.shape)}"
            )
        if int(updated_summary_hidden_states.shape[1]) != self.updated_summary_latent_size:
            raise ValueError(
                "typed latent aux updated summary length mismatch: "
                f"expected={self.updated_summary_latent_size}, got={int(updated_summary_hidden_states.shape[1])}"
            )
        numeric_hidden = self._group(updated_summary_hidden_states, "numeric")
        choice_hidden = self._group(updated_summary_hidden_states, "choice")
        spatial_hidden = self._group(updated_summary_hidden_states, "spatial")
        free_text_hidden = self._group(updated_summary_hidden_states, "free_text")
        output = {
            "typed_numeric_scalar": self.numeric_scalar_head(numeric_hidden).squeeze(-1),
            "typed_numeric_kind_logits": self.numeric_kind_head(numeric_hidden),
            "typed_choice_logits": self.choice_head(choice_hidden),
            "typed_spatial_logits": self.spatial_head(spatial_hidden),
            "typed_answer_type_logits": self.answer_type_head(free_text_hidden),
            "typed_answer_embedding": self.answer_embedding_head(free_text_hidden),
        }
        for key, value in output.items():
            _require_finite(key, value)
        return output


def _masked_mean_loss(loss_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=loss_values.device, dtype=loss_values.dtype)
    return (loss_values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_workspace_typed_latent_aux_loss_from_output(
    *,
    output: dict[str, torch.Tensor],
    batch: WorkspaceGeometryAuxBatch,
    answer_embedding_targets: torch.Tensor,
    numeric_scalar_weight: float,
    numeric_kind_weight: float,
    choice_weight: float,
    spatial_weight: float,
    answer_type_weight: float,
    answer_embedding_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 typed latent auxiliary loss。

    这不是替代最终答案 CE，而是给 updated_summary 的四组 typed slot 各自上尺子：
    numeric 组要会说数值，choice 组要会选 A-D，spatial 组要会说空间关系，
    free-text/type 组要会说答案类型，并且靠近冻结 Qwen answer embedding。
    """

    numeric_scalar = output["typed_numeric_scalar"].float()
    numeric_kind_logits = output["typed_numeric_kind_logits"].float()
    choice_logits = output["typed_choice_logits"].float()
    spatial_logits = output["typed_spatial_logits"].float()
    answer_type_logits = output["typed_answer_type_logits"].float()
    predicted_answer_embedding = output["typed_answer_embedding"].float()

    numeric_scalar_mask = batch.typed_numeric_scalar_mask.to(device=numeric_scalar.device, dtype=torch.float32)
    numeric_kind_mask = batch.typed_numeric_kind_mask.to(device=numeric_kind_logits.device, dtype=torch.float32)
    choice_loss_mask = batch.typed_choice_loss_mask.to(device=choice_logits.device, dtype=torch.float32)
    spatial_mask = batch.typed_spatial_mask.to(device=spatial_logits.device, dtype=torch.float32)
    answer_type_mask = batch.typed_answer_type_mask.to(device=answer_type_logits.device, dtype=torch.float32)
    answer_embedding_mask = batch.typed_answer_embedding_mask.to(
        device=predicted_answer_embedding.device,
        dtype=torch.float32,
    )

    scalar_target = batch.typed_numeric_scalar.to(device=numeric_scalar.device, dtype=torch.float32)
    scalar_loss_values = F.smooth_l1_loss(numeric_scalar, scalar_target, reduction="none")
    scalar_loss = _masked_mean_loss(scalar_loss_values, numeric_scalar_mask)

    numeric_kind_labels = batch.typed_numeric_kind.to(device=numeric_kind_logits.device, dtype=torch.long)
    numeric_kind_loss = _masked_mean_loss(
        F.cross_entropy(numeric_kind_logits, numeric_kind_labels, reduction="none"),
        numeric_kind_mask,
    )

    choice_labels = batch.typed_choice_label.to(device=choice_logits.device, dtype=torch.long)
    choice_mask = batch.typed_choice_mask.to(device=choice_logits.device, dtype=torch.bool)
    if choice_logits.shape != choice_mask.shape:
        raise ValueError(f"typed choice mask shape mismatch: logits={choice_logits.shape}, mask={choice_mask.shape}")
    masked_choice_logits = choice_logits.masked_fill(~choice_mask, torch.finfo(choice_logits.dtype).min)
    choice_loss = _masked_mean_loss(
        F.cross_entropy(masked_choice_logits, choice_labels, reduction="none"),
        choice_loss_mask,
    )

    spatial_labels = batch.typed_spatial_label.to(device=spatial_logits.device, dtype=torch.long)
    spatial_loss = _masked_mean_loss(
        F.cross_entropy(spatial_logits, spatial_labels, reduction="none"),
        spatial_mask,
    )

    answer_type_labels = batch.typed_answer_type.to(device=answer_type_logits.device, dtype=torch.long)
    answer_type_loss = _masked_mean_loss(
        F.cross_entropy(answer_type_logits, answer_type_labels, reduction="none"),
        answer_type_mask,
    )

    answer_embedding_targets = answer_embedding_targets.to(
        device=predicted_answer_embedding.device,
        dtype=predicted_answer_embedding.dtype,
    )
    if answer_embedding_targets.shape != predicted_answer_embedding.shape:
        raise ValueError(
            "answer embedding target shape mismatch: "
            f"target={tuple(answer_embedding_targets.shape)}, pred={tuple(predicted_answer_embedding.shape)}"
        )
    cosine = F.cosine_similarity(predicted_answer_embedding, answer_embedding_targets, dim=-1)
    answer_embedding_loss = _masked_mean_loss(1.0 - cosine, answer_embedding_mask)

    total = (
        float(numeric_scalar_weight) * scalar_loss
        + float(numeric_kind_weight) * numeric_kind_loss
        + float(choice_weight) * choice_loss
        + float(spatial_weight) * spatial_loss
        + float(answer_type_weight) * answer_type_loss
        + float(answer_embedding_weight) * answer_embedding_loss
    )

    zero = total * 0.0
    numeric_kind_acc = (
        (numeric_kind_logits.argmax(dim=-1) == numeric_kind_labels).to(torch.float32)[numeric_kind_mask > 0].mean()
        if bool((numeric_kind_mask > 0).any().item())
        else zero
    )
    choice_acc = (
        (masked_choice_logits.argmax(dim=-1) == choice_labels).to(torch.float32)[choice_loss_mask > 0].mean()
        if bool((choice_loss_mask > 0).any().item())
        else zero
    )
    spatial_acc = (
        (spatial_logits.argmax(dim=-1) == spatial_labels).to(torch.float32)[spatial_mask > 0].mean()
        if bool((spatial_mask > 0).any().item())
        else zero
    )
    answer_type_acc = (
        (answer_type_logits.argmax(dim=-1) == answer_type_labels).to(torch.float32)[answer_type_mask > 0].mean()
        if bool((answer_type_mask > 0).any().item())
        else zero
    )
    numeric_scalar_mae = _masked_mean_loss(torch.abs(numeric_scalar - scalar_target), numeric_scalar_mask)
    metrics = {
        "workspace_typed_numeric_scalar_loss": scalar_loss.detach(),
        "workspace_typed_numeric_kind_loss": numeric_kind_loss.detach(),
        "workspace_typed_choice_loss": choice_loss.detach(),
        "workspace_typed_spatial_loss": spatial_loss.detach(),
        "workspace_typed_answer_type_loss": answer_type_loss.detach(),
        "workspace_typed_answer_embedding_loss": answer_embedding_loss.detach(),
        "workspace_typed_numeric_kind_acc": numeric_kind_acc.detach(),
        "workspace_typed_numeric_scalar_mae": numeric_scalar_mae.detach(),
        "workspace_typed_choice_acc": choice_acc.detach(),
        "workspace_typed_spatial_acc": spatial_acc.detach(),
        "workspace_typed_answer_type_acc": answer_type_acc.detach(),
        "workspace_typed_answer_embedding_cosine": _masked_mean_loss(cosine, answer_embedding_mask).detach(),
        "workspace_typed_numeric_rows": numeric_scalar_mask.sum().detach(),
        "workspace_typed_choice_rows": choice_loss_mask.sum().detach(),
        "workspace_typed_spatial_rows": spatial_mask.sum().detach(),
        "workspace_typed_free_text_rows": answer_embedding_mask.sum().detach(),
    }
    return total, metrics


class WorkspaceLatentFlowHead(nn.Module):
    """学习 source workspace latent 到 query-oriented latent 的条件速度场。

    geometry aux teacher 会给出一个 answer 前的目标 updated summary。
    这个 head 不直接复制目标 hidden，而是学习“从 source hidden 走向 target hidden”
    的速度。训练时用 flow matching；回答时用一步更新得到新的 updated summary。
    """

    def __init__(
        self,
        *,
        llm_hidden_size: int,
        context_dim: int,
        pose_dim: int,
        summary_latent_size: int,
        work_latent_size: int,
        updated_summary_latent_size: int,
        flow_hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.llm_hidden_size = int(llm_hidden_size)
        self.summary_latent_size = int(summary_latent_size)
        self.work_latent_size = int(work_latent_size)
        self.updated_summary_latent_size = int(updated_summary_latent_size)
        self.flow_hidden_size = int(flow_hidden_size)
        self.latent_norm = nn.LayerNorm(llm_hidden_size)
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, flow_hidden_size),
            nn.GELU(),
            nn.LayerNorm(flow_hidden_size),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, flow_hidden_size),
            nn.GELU(),
            nn.LayerNorm(flow_hidden_size),
        )
        self.workspace_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, flow_hidden_size),
            nn.GELU(),
            nn.LayerNorm(flow_hidden_size),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, flow_hidden_size),
            nn.GELU(),
            nn.LayerNorm(flow_hidden_size),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, flow_hidden_size),
            nn.GELU(),
            nn.LayerNorm(flow_hidden_size),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(flow_hidden_size * 5),
            nn.Linear(flow_hidden_size * 5, flow_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flow_hidden_size, llm_hidden_size),
        )

    def _condition_hidden(
        self,
        *,
        workspace_latents: torch.Tensor,
        batch: WorkspaceGeometryAuxBatch,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """把 workspace/context/pose 压成每个样本一个条件向量。"""

        workspace_latents = workspace_latents.to(device=target_device, dtype=target_dtype)
        context_tokens = batch.context_tokens.to(device=target_device, dtype=target_dtype)
        pose_features = batch.pose_features.to(device=target_device, dtype=target_dtype)
        point_mask = batch.point_mask.to(device=target_device)
        _require_finite("flow_workspace_latents", workspace_latents)
        _require_finite("flow_context_tokens", context_tokens)
        _require_finite("flow_pose_features", pose_features)

        normalized_latents = self.latent_norm(workspace_latents)
        workspace_hidden = self.workspace_proj(normalized_latents).mean(dim=1)
        context_hidden = self.context_proj(context_tokens)
        point_mask_float = point_mask.unsqueeze(-1).to(dtype=context_hidden.dtype)
        point_count = point_mask_float.sum(dim=1).clamp_min(1.0)
        context_pooled = (context_hidden * point_mask_float).sum(dim=1) / point_count
        pose_hidden = self.pose_proj(pose_features)
        return workspace_hidden + context_pooled + pose_hidden

    def velocity(
        self,
        *,
        z_t: torch.Tensor,
        t: torch.Tensor,
        workspace_latents: torch.Tensor,
        batch: WorkspaceGeometryAuxBatch,
    ) -> torch.Tensor:
        """预测当前 latent 位置的速度场。"""

        if z_t.ndim != 3:
            raise ValueError(f"z_t must be [B,U,D], got {tuple(z_t.shape)}")
        if t.ndim != 2 or int(t.shape[1]) != 1:
            raise ValueError(f"t must be [B,1], got {tuple(t.shape)}")
        if int(z_t.shape[1]) != self.updated_summary_latent_size:
            raise ValueError(
                "z_t updated summary length mismatch: "
                f"expected={self.updated_summary_latent_size}, got={int(z_t.shape[1])}"
            )
        target_dtype = self.velocity_head[-1].weight.dtype
        target_device = self.velocity_head[-1].weight.device
        z_t = z_t.to(device=target_device, dtype=target_dtype)
        t = t.to(device=target_device, dtype=target_dtype)
        condition = self._condition_hidden(
            workspace_latents=workspace_latents,
            batch=batch,
            target_device=target_device,
            target_dtype=target_dtype,
        )
        token_hidden = self.token_proj(self.latent_norm(z_t))
        condition_per_token = condition.unsqueeze(1).expand(-1, z_t.shape[1], -1)
        time_hidden = self.time_proj(t).unsqueeze(1).expand(-1, z_t.shape[1], -1)
        # token_hidden 是当前位置；condition 是问题/evidence/pose 的条件；
        # time_hidden 告诉 head 当前在 source->query 轨迹上的哪个时刻。
        model_input = torch.cat(
            [
                token_hidden,
                condition_per_token,
                time_hidden,
                token_hidden * condition_per_token,
                torch.abs(token_hidden - condition_per_token),
            ],
            dim=-1,
        )
        velocity = self.velocity_head(model_input)
        _require_finite("workspace_latent_flow_velocity", velocity)
        return velocity

    def forward(
        self,
        *,
        source_summary_hidden_states: torch.Tensor,
        teacher_updated_summary_hidden_states: torch.Tensor,
        workspace_latents: torch.Tensor,
        batch: WorkspaceGeometryAuxBatch,
        integration_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        """计算 flow matching loss，并给 answer 路径产出多步积分后的 latent。"""

        if source_summary_hidden_states.shape != teacher_updated_summary_hidden_states.shape:
            raise ValueError(
                "flow source/teacher target shape mismatch: "
                f"source={tuple(source_summary_hidden_states.shape)}, "
                f"teacher={tuple(teacher_updated_summary_hidden_states.shape)}"
            )
        integration_steps = int(integration_steps)
        if integration_steps <= 0:
            raise ValueError(f"integration_steps must be positive, got {integration_steps}")
        target_dtype = self.velocity_head[-1].weight.dtype
        target_device = self.velocity_head[-1].weight.device
        z_source = source_summary_hidden_states.to(device=target_device, dtype=target_dtype)
        z_target = teacher_updated_summary_hidden_states.to(device=target_device, dtype=target_dtype)
        batch_size = int(z_source.shape[0])
        if batch_size <= 0:
            raise ValueError("flow batch_size must be positive")

        # FM 监督只训练速度场，不允许模型通过移动 teacher target 或 source target
        # 来把 MSE 变小；answer loss 仍然会通过 one-step latent 回传到可训练模块。
        z_source_target = z_source.detach()
        z_target_target = z_target.detach()
        t = torch.rand((batch_size, 1), device=target_device, dtype=target_dtype)
        z_t = (1.0 - t.unsqueeze(-1)) * z_source_target + t.unsqueeze(-1) * z_target_target
        target_velocity = z_target_target - z_source_target
        pred_velocity = self.velocity(
            z_t=z_t,
            t=t,
            workspace_latents=workspace_latents.detach(),
            batch=batch,
        )
        flow_matching_loss = F.mse_loss(pred_velocity.float(), target_velocity.float()).to(dtype=target_dtype)

        # 训练 flow loss 时随机抽 t；真正给 answer 用时，从 source memory
        # 沿同一个速度场走 K 小步。K=1 时退化为早期试验版本的一步更新。
        z_integrated = z_source
        dt = 1.0 / float(integration_steps)
        for step_index in range(integration_steps):
            step_t_value = float(step_index) / float(integration_steps)
            step_t = torch.full((batch_size, 1), step_t_value, device=target_device, dtype=target_dtype)
            step_velocity = self.velocity(
                z_t=z_integrated,
                t=step_t,
                workspace_latents=workspace_latents,
                batch=batch,
            )
            z_integrated = z_integrated + dt * step_velocity.to(dtype=z_integrated.dtype)

        target_l2 = (z_target_target.float() - z_source_target.float()).pow(2).mean().sqrt()
        pred_target_mse = F.mse_loss(z_integrated.float(), z_target_target.float())
        pred_target_cosine = F.cosine_similarity(
            z_integrated.float().flatten(start_dim=1),
            z_target_target.float().flatten(start_dim=1),
            dim=-1,
        ).mean()
        return {
            "updated_summary_hidden_states": z_integrated.to(
                device=teacher_updated_summary_hidden_states.device,
                dtype=teacher_updated_summary_hidden_states.dtype,
            ),
            "velocity": step_velocity,
            "flow_matching_loss": flow_matching_loss,
            "target_l2": target_l2.to(dtype=target_dtype),
            "pred_target_mse": pred_target_mse.to(dtype=target_dtype),
            "pred_target_cosine": pred_target_cosine.to(dtype=target_dtype),
            "integration_steps": torch.tensor(float(integration_steps), device=target_device, dtype=target_dtype),
        }


def compute_semantic_option_logits(
    *,
    output: dict[str, torch.Tensor],
    batch: WorkspaceGeometryAuxBatch,
) -> torch.Tensor:
    """把几何语义头转换成 A-G/A-H 多选题分数。

    A-D 的字母每行都随机打乱，aux head 不能直接猜“选 A 还是 B”。
    正确做法是：
    - A-D 分数 = valid 分数 + 该选项方向分数 + 该选项距离分数；
    - E-G/H 分数 = invalid 分数 + 对应无效原因分数。
    这样语义选项 loss 会反向监督 valid、方向、距离和无效原因，而不是训练一个容易背偏置的黑盒分类器。
    """

    compass_logits = output["compass_logits"].float()
    range_logits = output["range_logits"].float()
    valid_logits = output["valid_logits"].float()
    invalid_reason_logits = output["invalid_reason_logits"].float()
    spatial_option_compass16 = batch.spatial_option_compass16.to(device=compass_logits.device)
    spatial_option_range_bin = batch.spatial_option_range_bin.to(device=range_logits.device)
    if spatial_option_compass16.shape != (compass_logits.shape[0], 4):
        raise ValueError(f"spatial_option_compass16 must be [B,4], got {spatial_option_compass16.shape}")
    if spatial_option_range_bin.shape != (range_logits.shape[0], 4):
        raise ValueError(f"spatial_option_range_bin must be [B,4], got {spatial_option_range_bin.shape}")

    compass_scores = torch.gather(compass_logits, dim=1, index=spatial_option_compass16)
    range_scores = torch.gather(range_logits, dim=1, index=spatial_option_range_bin)
    spatial_scores = valid_logits[:, 1:2] + compass_scores + range_scores
    # invalid_reason 的 0 类是 valid/correct，不对应多选题里的无效选项。
    # 1..N-1 分别映射到 E/F/G/H；旧 A-G 数据 N=4，新 A-H 数据 N=5。
    invalid_scores = valid_logits[:, 0:1] + invalid_reason_logits[:, 1:]
    option_logits = torch.cat([spatial_scores, invalid_scores], dim=1)
    _require_finite("semantic_option_logits", option_logits)
    return option_logits


def compute_workspace_answer_classifier_loss_from_output(
    *,
    output: dict[str, torch.Tensor],
    batch: WorkspaceGeometryAuxBatch,
    class_balance: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """训练 updated-memory answer classifier。

    这个 loss 直接监督 classifier 选 A-H。
    它和 geometry aux 的区别是：
    - geometry aux 学“点云/方向/距离/valid”；
    - answer classifier 学“updated memory + 当前选项 -> 最终哪个字母”。
    后续 AR 头只应该向这个分类头学习答案分布。
    """

    logits = output["answer_option_logits"].float()
    labels = batch.answer_option.to(device=logits.device, dtype=torch.long)
    if logits.ndim != 2:
        raise ValueError(f"answer_option_logits must be [B,C], got {logits.shape}")
    if labels.ndim != 1 or int(labels.shape[0]) != int(logits.shape[0]):
        raise ValueError("answer_option labels must be [B] and aligned with answer classifier logits")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= int(logits.shape[1]):
        raise ValueError(
            "answer_option labels out of classifier range: "
            f"min={int(labels.min().item())}, max={int(labels.max().item())}, class_count={int(logits.shape[1])}"
        )

    per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
    if class_balance:
        sample_weight = batch.answer_option_class_weights.to(device=logits.device, dtype=torch.float32)[labels]
    else:
        sample_weight = torch.ones_like(per_sample_loss, dtype=torch.float32)
    answer_loss_mask = batch.answer_loss_mask.to(device=logits.device, dtype=torch.float32)
    if answer_loss_mask.ndim != 1 or int(answer_loss_mask.shape[0]) != int(logits.shape[0]):
        raise ValueError("answer_loss_mask must be [B] and aligned with answer classifier logits")
    sample_weight = sample_weight * answer_loss_mask
    loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

    pred = logits.argmax(dim=-1)
    supervised_rows = answer_loss_mask > 0
    valid_rows = batch.valid_evidence.to(device=logits.device, dtype=torch.long) == 1
    invalid_rows = ~valid_rows
    zero = logits.sum() * 0.0
    acc = (
        (pred[supervised_rows] == labels[supervised_rows]).to(dtype=torch.float32).mean()
        if bool(supervised_rows.any().item())
        else zero
    )
    valid_supervised_rows = valid_rows & supervised_rows
    invalid_supervised_rows = invalid_rows & supervised_rows
    valid_acc = (
        (pred[valid_supervised_rows] == labels[valid_supervised_rows]).to(dtype=torch.float32).mean()
        if bool(valid_supervised_rows.any().item())
        else zero
    )
    invalid_acc = (
        (pred[invalid_supervised_rows] == labels[invalid_supervised_rows]).to(dtype=torch.float32).mean()
        if bool(invalid_supervised_rows.any().item())
        else zero
    )
    spatial_pred = pred < 4
    evidence_validity_pred = spatial_pred.to(dtype=torch.long)
    evidence_validity_acc = (
        evidence_validity_pred == batch.valid_evidence.to(device=logits.device, dtype=torch.long)
    ).to(dtype=torch.float32).mean()
    entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    metrics = {
        "workspace_answer_classifier_acc": acc.detach(),
        "workspace_answer_classifier_valid_acc": valid_acc.detach(),
        "workspace_answer_classifier_invalid_acc": invalid_acc.detach(),
        "workspace_answer_classifier_evidence_validity_acc": evidence_validity_acc.detach(),
        "workspace_answer_classifier_entropy": entropy.detach(),
    }
    return loss, metrics


def compute_workspace_geometry_aux_loss_from_output(
    *,
    output: dict[str, torch.Tensor],
    batch: WorkspaceGeometryAuxBatch,
    point_loss_weight: float,
    center_loss_weight: float,
    compass_loss_weight: float,
    range_loss_weight: float,
    invalid_reason_loss_weight: float = 0.0,
    semantic_option_loss_weight: float = 0.0,
    relation_class_balance: bool = False,
    valid_loss_weight: float,
    valid_positive_weight: float = 1.0,
    valid_rank_loss_weight: float = 0.0,
    valid_rank_margin: float = 0.5,
    require_valid_rank_pairs: bool = False,
    negative_center_loss_weight: float,
    negative_center_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 workspace 几何辅助损失。

    这个 loss 不替代最终答案 CE。
    它只是在训练时逼 summary latent 真的记住 source 视角下的几何证据。
    source->query 的变化由后面的 work/update latent 做，不在这里把 summary 直接监督成 query pose。
    """

    predicted_points = output["points"]
    target_points = batch.target_points.to(device=predicted_points.device, dtype=predicted_points.dtype)
    mask = batch.point_mask.to(device=predicted_points.device).unsqueeze(-1).to(dtype=predicted_points.dtype)
    point_count = mask.sum(dim=1).clamp_min(1.0)
    predicted_center = (predicted_points * mask).sum(dim=1) / point_count
    target_center = (target_points * mask).sum(dim=1) / point_count

    valid_labels = batch.valid_evidence.to(device=predicted_points.device, dtype=torch.long)
    positive_rows = valid_labels == 1
    negative_rows = valid_labels == 0
    point_rows = positive_rows & (batch.point_loss_mask.to(device=predicted_points.device, dtype=torch.long) == 1)
    center_rows = positive_rows & (batch.center_loss_mask.to(device=predicted_points.device, dtype=torch.long) == 1)
    compass_rows = positive_rows & (batch.compass_loss_mask.to(device=predicted_points.device, dtype=torch.long) == 1)
    range_rows = positive_rows & (batch.range_loss_mask.to(device=predicted_points.device, dtype=torch.long) == 1)
    positive_count = positive_rows.sum()
    negative_count = negative_rows.sum()
    zero = predicted_points.sum() * 0.0

    # 这里不能直接用 `F.cross_entropy(..., weight=...)`。
    # 正式训练使用 batch_size=1 + gradient accumulation；PyTorch 的默认 mean reduction
    # 会把单样本自己的 class weight 又除回去，导致 `valid_positive_weight=3`
    # 在每个 micro batch 里几乎等于没开。结果就是 1 个 correct 对 3 个 invalid，
    # valid head 很容易塌成“全部判 invalid”。
    #
    # 所以这里显式取 per-sample loss，再手动乘正样本权重；这样即使 batch_size=1，
    # correct evidence 的 valid 监督也真的会变强。
    valid_per_sample_loss = F.cross_entropy(
        output["valid_logits"].float(),
        valid_labels,
        reduction="none",
    )
    valid_sample_weight = torch.where(
        valid_labels == 1,
        torch.full_like(valid_per_sample_loss, float(valid_positive_weight)),
        torch.ones_like(valid_per_sample_loss),
    )
    valid_loss = (valid_per_sample_loss * valid_sample_weight).mean()

    invalid_reason_labels = batch.invalid_reason.to(device=predicted_points.device, dtype=torch.long)
    invalid_reason_per_sample_loss = F.cross_entropy(
        output["invalid_reason_logits"].float(),
        invalid_reason_labels,
        reduction="none",
    )
    invalid_reason_sample_weight = batch.invalid_reason_class_weights.to(
        device=predicted_points.device,
        dtype=torch.float32,
    )[invalid_reason_labels]
    invalid_reason_loss = (invalid_reason_per_sample_loss * invalid_reason_sample_weight).mean()

    semantic_option_logits = compute_semantic_option_logits(output=output, batch=batch)
    answer_option_labels = batch.answer_option.to(device=predicted_points.device, dtype=torch.long)
    semantic_option_per_sample_loss = F.cross_entropy(
        semantic_option_logits,
        answer_option_labels,
        reduction="none",
    )
    semantic_option_sample_weight = batch.answer_option_class_weights.to(
        device=predicted_points.device,
        dtype=torch.float32,
    )[answer_option_labels]
    semantic_option_loss = (semantic_option_per_sample_loss * semantic_option_sample_weight).mean()

    # 单独看每条样本做 valid 二分类，模型很容易学成“全判无效”或“全判有效”。
    # 反事实实验需要同一个 source object 的 correct/wrong 放在同一批里比较：
    # correct 的 valid 分数必须比 wrong 高至少一个 margin。
    valid_rank_losses: list[torch.Tensor] = []
    valid_score = output["valid_logits"].float()[:, 1] - output["valid_logits"].float()[:, 0]
    for group_id in sorted(set(batch.source_row_ids)):
        group_indices = [
            index for index, source_row_id in enumerate(batch.source_row_ids)
            if source_row_id == group_id
        ]
        if len(group_indices) < 2:
            continue
        group_index_tensor = torch.tensor(group_indices, device=valid_score.device, dtype=torch.long)
        group_labels = valid_labels[group_index_tensor]
        positive_group_indices = group_index_tensor[group_labels == 1]
        negative_group_indices = group_index_tensor[group_labels == 0]
        if int(positive_group_indices.numel()) == 0 or int(negative_group_indices.numel()) == 0:
            continue
        positive_scores = valid_score[positive_group_indices].view(-1, 1)
        negative_scores = valid_score[negative_group_indices].view(1, -1)
        valid_rank_losses.append(F.relu(float(valid_rank_margin) - (positive_scores - negative_scores)).mean())
    if valid_rank_losses:
        valid_rank_loss = torch.stack(valid_rank_losses).mean()
    else:
        valid_rank_loss = zero
        if require_valid_rank_pairs and float(valid_rank_loss_weight) > 0.0:
            raise ValueError(
                "valid rank loss is enabled, but this batch has no source group containing both "
                "valid and invalid evidence. Enable grouped counterfactual batches."
            )

    if bool(point_rows.any().item()):
        positive_mask = mask[point_rows]
        positive_valid_count = positive_mask.sum().clamp_min(1.0)
        point_loss = F.smooth_l1_loss(
            predicted_points[point_rows] * positive_mask,
            target_points[point_rows] * positive_mask,
            reduction="sum",
        ) / (positive_valid_count * 3.0)
    else:
        point_loss = zero

    if bool(center_rows.any().item()):
        center_loss = F.smooth_l1_loss(predicted_center[center_rows], target_center[center_rows])
    else:
        center_loss = zero

    # VLM-3R 里很多题是数值距离/相机运动；此时 full-frame pointmap centroid 的 compass/range
    # 不一定等于题目答案。分项 mask 允许保留通用 pointmap metric supervision，同时关掉噪声关系标签。
    if bool(compass_rows.any().item()):
        compass_per_sample_loss = F.cross_entropy(
            output["compass_logits"][compass_rows].float(),
            batch.compass16.to(device=predicted_points.device)[compass_rows],
            reduction="none",
        )
        if relation_class_balance:
            compass_class_weight = batch.compass16_class_weights.to(device=predicted_points.device, dtype=torch.float32)
            compass_sample_weight = compass_class_weight[
                batch.compass16.to(device=predicted_points.device)[compass_rows]
            ]
            compass_loss = (compass_per_sample_loss * compass_sample_weight).mean()
        else:
            compass_loss = compass_per_sample_loss.mean()
    else:
        compass_loss = zero

    if bool(range_rows.any().item()):
        range_per_sample_loss = F.cross_entropy(
            output["range_logits"][range_rows].float(),
            batch.range_bin.to(device=predicted_points.device)[range_rows],
            reduction="none",
        )
        if relation_class_balance:
            range_class_weight = batch.range_bin_class_weights.to(device=predicted_points.device, dtype=torch.float32)
            range_sample_weight = range_class_weight[
                batch.range_bin.to(device=predicted_points.device)[range_rows]
            ]
            range_loss = (range_per_sample_loss * range_sample_weight).mean()
        else:
            range_loss = range_per_sample_loss.mean()
    else:
        range_loss = zero

    if int(negative_count.item()) > 0:
        # 负样本不是“没有监督”。它的监督是：你不该把错框/错物体也投到正确目标上。
        # 这里用中心距离 margin，把 wrong evidence 预测到正确物体附近的行为显式惩罚掉。
        negative_distance = torch.linalg.norm(
            predicted_center[negative_rows] - target_center[negative_rows],
            dim=-1,
        )
        negative_center_loss = F.relu(float(negative_center_margin) - negative_distance).pow(2).mean()
    else:
        negative_center_loss = zero

    total_loss = (
        float(point_loss_weight) * point_loss
        + float(center_loss_weight) * center_loss
        + float(compass_loss_weight) * compass_loss
        + float(range_loss_weight) * range_loss
        + float(valid_loss_weight) * valid_loss
        + float(invalid_reason_loss_weight) * invalid_reason_loss
        + float(semantic_option_loss_weight) * semantic_option_loss
        + float(valid_rank_loss_weight) * valid_rank_loss
        + float(negative_center_loss_weight) * negative_center_loss
    )
    valid_pred = output["valid_logits"].argmax(dim=-1)
    valid_acc = (valid_pred == valid_labels).to(dtype=torch.float32).mean()
    invalid_reason_pred = output["invalid_reason_logits"].argmax(dim=-1)
    invalid_reason_acc = (invalid_reason_pred == invalid_reason_labels).to(dtype=torch.float32).mean()
    semantic_option_pred = semantic_option_logits.argmax(dim=-1)
    semantic_option_acc = (semantic_option_pred == answer_option_labels).to(dtype=torch.float32).mean()
    metrics = {
        "workspace_geometry_point_loss": point_loss.detach(),
        "workspace_geometry_center_loss": center_loss.detach(),
        "workspace_geometry_compass_loss": compass_loss.detach(),
        "workspace_geometry_range_loss": range_loss.detach(),
        "workspace_geometry_valid_loss": valid_loss.detach(),
        "workspace_geometry_invalid_reason_loss": invalid_reason_loss.detach(),
        "workspace_geometry_semantic_option_loss": semantic_option_loss.detach(),
        "workspace_geometry_valid_rank_loss": valid_rank_loss.detach(),
        "workspace_geometry_valid_rank_pair_groups": torch.tensor(
            float(len(valid_rank_losses)),
            device=predicted_points.device,
            dtype=torch.float32,
        ),
        "workspace_geometry_negative_center_loss": negative_center_loss.detach(),
        "workspace_geometry_valid_points": batch.point_mask.sum().detach().to(dtype=torch.float32),
        "workspace_geometry_positive_rows": positive_count.detach().to(dtype=torch.float32),
        "workspace_geometry_negative_rows": negative_count.detach().to(dtype=torch.float32),
        "workspace_geometry_point_rows": point_rows.sum().detach().to(dtype=torch.float32),
        "workspace_geometry_center_rows": center_rows.sum().detach().to(dtype=torch.float32),
        "workspace_geometry_compass_rows": compass_rows.sum().detach().to(dtype=torch.float32),
        "workspace_geometry_range_rows": range_rows.sum().detach().to(dtype=torch.float32),
        "workspace_geometry_valid_acc": valid_acc.detach(),
        "workspace_geometry_invalid_reason_acc": invalid_reason_acc.detach(),
        "workspace_geometry_semantic_option_acc": semantic_option_acc.detach(),
    }
    if "updated_summary_delta" in output:
        metrics["workspace_geometry_latent_delta_norm"] = (
            output["updated_summary_delta"].detach().float().norm(dim=-1).mean()
        )
    if "raw_updated_summary_delta" in output:
        metrics["workspace_geometry_raw_latent_delta_norm"] = (
            output["raw_updated_summary_delta"].detach().float().norm(dim=-1).mean()
        )
    return total_loss, metrics


def compute_workspace_geometry_aux_loss(
    *,
    head: WorkspaceGeometryAuxHead,
    latent_hidden_states: torch.Tensor,
    batch: WorkspaceGeometryAuxBatch,
    point_loss_weight: float,
    center_loss_weight: float,
    compass_loss_weight: float,
    range_loss_weight: float,
    invalid_reason_loss_weight: float = 0.0,
    semantic_option_loss_weight: float = 0.0,
    relation_class_balance: bool = False,
    valid_loss_weight: float,
    valid_positive_weight: float = 1.0,
    valid_rank_loss_weight: float = 0.0,
    valid_rank_margin: float = 0.5,
    require_valid_rank_pairs: bool = False,
    negative_center_loss_weight: float,
    negative_center_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = head(latent_hidden_states, batch)
    return compute_workspace_geometry_aux_loss_from_output(
        output=output,
        batch=batch,
        point_loss_weight=point_loss_weight,
        center_loss_weight=center_loss_weight,
        compass_loss_weight=compass_loss_weight,
        range_loss_weight=range_loss_weight,
        invalid_reason_loss_weight=invalid_reason_loss_weight,
        semantic_option_loss_weight=semantic_option_loss_weight,
        relation_class_balance=relation_class_balance,
        valid_loss_weight=valid_loss_weight,
        valid_positive_weight=valid_positive_weight,
        valid_rank_loss_weight=valid_rank_loss_weight,
        valid_rank_margin=valid_rank_margin,
        require_valid_rank_pairs=require_valid_rank_pairs,
        negative_center_loss_weight=negative_center_loss_weight,
        negative_center_margin=negative_center_margin,
    )
