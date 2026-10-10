"""Copy the independently loadable model parts, keeping parameter values and removing training-continuation state."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import torch


MODEL_COMPONENTS = {'spatial_flow_state_dict', 'decoder_aux_state_dict',
                    'qwen_lora_state_dict', 'internvl_lora_state_dict'}
METADATA = {'format_version', 'base_metric_format_version', 'mode', 'step', 'epoch', 'args'}
TRAINING_STATE = {'optimizer_state_dict', 'rng_state', 'best_state', 'eval_state', 'batch_i'}


def relocate_metadata(value, replacements, forbidden):
    """路径只能按审过的对照表替换；未知路径或身份信息直接报错。"""
    if isinstance(value, dict):
        return {relocate_metadata(k, replacements, forbidden):
                relocate_metadata(v, replacements, forbidden) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(relocate_metadata(x, replacements, forbidden) for x in value)
    if isinstance(value, str):
        result = replacements.get(value, value)
        if Path(result).is_absolute() or any(token and token in result for token in forbidden):
            raise ValueError('Unresolved private metadata: ' + value)
        return result
    if value is None or type(value) in (bool, int, float):
        return value
    raise TypeError('Unsupported checkpoint metadata type: ' + type(value).__name__)


def export_checkpoint(source, destination, *, replacements, required_components, forbidden):
    """只写新文件；逐张量读回检查，避免漏掉暂时关闭损失的已保存读出参数。"""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    original = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    unknown = set(original) - MODEL_COMPONENTS - METADATA - TRAINING_STATE
    if unknown:
        raise ValueError(f'Unreviewed checkpoint fields: {sorted(unknown)}')
    present = set(original) & MODEL_COMPONENTS
    if present != set(required_components):
        raise KeyError(f'Checkpoint components mismatch: expected={required_components}, got={sorted(present)}')
    for key in ['format_version', 'step', 'epoch', 'args']:
        if key not in original:
            raise KeyError('Missing checkpoint metadata: ' + key)
    public = {k: relocate_metadata(v, replacements, forbidden)
              for k, v in original.items() if k in METADATA}
    for group in sorted(present):
        if not original[group]:
            raise ValueError('Empty model component: ' + group)
        public[group] = OrderedDict()
        for name, value in original[group].items():
            relocate_metadata(name, {}, forbidden)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'Non-tensor model parameter: {group}.{name}')
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite model parameter: {group}.{name}')
            # 独立存储，不让原优化器共享的底层空间混进模型文件；精度和数值保持不变。
            public[group][name] = value.detach().clone()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public, destination)
    restored = torch.load(destination, map_location='cpu', weights_only=True, mmap=True)
    if set(restored) != set(public):
        raise KeyError('Serialized checkpoint changed top-level keys')
    records = []
    for group in sorted(present):
        if set(restored[group]) != set(original[group]):
            raise KeyError('Serialized checkpoint changed tensor names: ' + group)
        for name, expected in original[group].items():
            actual = restored[group][name]
            if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError(f'Checkpoint tensor changed: {group}.{name}')
            records.append({'component': group, 'name': name, 'shape': list(actual.shape),
                            'dtype': str(actual.dtype), 'exact_equal': True,
                            'bytes': actual.numel() * actual.element_size()})
    for key in METADATA & set(public):
        if restored[key] != public[key]:
            raise ValueError('Serialized checkpoint changed metadata: ' + key)
    return {'passed': True, 'tensor_count': len(records), 'tensors': records,
            'checkpoint_bytes': destination.stat().st_size,
            'removed_fields': sorted(set(original) & TRAINING_STATE),
            'components': sorted(present), 'step': restored['step'], 'dtype_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-spec', type=Path, required=True,
                        help='Reviewed path replacements, required_components and forbidden identity fragments; do not publish.')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.private_spec.read_text())
    result = export_checkpoint(args.source, args.output, **spec)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
