from pathlib import Path

import pytest
import torch

from quasar.export_checkpoint import export_checkpoint


def source(tmp_path):
    path = tmp_path / 'original.pt'
    torch.save({'format_version': 'test', 'step': 3, 'epoch': 0,
                'args': {'qwen_model_path': '/private/alice/base', 'spatial_query_count': 4},
                'qwen_lora_state_dict': {'layer': torch.arange(8, dtype=torch.bfloat16)},
                'decoder_aux_state_dict': {'sigreg.weight': torch.arange(4, dtype=torch.float32)},
                'optimizer_state_dict': {'private': '/private/alice/optimizer'},
                'rng_state': torch.ones(2, dtype=torch.uint8)}, path)
    return path


def test_export_retains_sigreg_dtype_and_values(tmp_path):
    original = source(tmp_path)
    result = export_checkpoint(original, tmp_path/'public.pt',
        replacements={'/private/alice/base': './upstream_models/base'},
        required_components=['qwen_lora_state_dict', 'decoder_aux_state_dict'], forbidden=['alice'])
    public = torch.load(tmp_path/'public.pt', weights_only=True)
    assert result['tensor_count'] == 2
    assert public['args']['spatial_query_count'] == 4
    assert 'optimizer_state_dict' not in public and 'rng_state' not in public
    assert public['qwen_lora_state_dict']['layer'].dtype == torch.bfloat16
    assert public['decoder_aux_state_dict']['sigreg.weight'].tolist() == [0., 1., 2., 3.]
    assert torch.load(original, weights_only=True)['args']['qwen_model_path'] == '/private/alice/base'


def test_export_rejects_missing_component(tmp_path):
    with pytest.raises(KeyError, match='components mismatch'):
        export_checkpoint(source(tmp_path), tmp_path/'public.pt', replacements={},
                          required_components=['qwen_lora_state_dict'], forbidden=[])
    assert not (tmp_path/'public.pt').exists()


@pytest.mark.parametrize('replacement', [None, './alice/base'])
def test_export_rejects_unresolved_identity(tmp_path, replacement):
    replacements = {} if replacement is None else {'/private/alice/base': replacement}
    with pytest.raises(ValueError, match='Unresolved private metadata'):
        export_checkpoint(source(tmp_path), tmp_path/'public.pt', replacements=replacements,
                          required_components=['qwen_lora_state_dict', 'decoder_aux_state_dict'], forbidden=['alice'])
    assert not (tmp_path/'public.pt').exists()
