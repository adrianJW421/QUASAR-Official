"""确认公开命令保留原配方，只在明确要求时加入数值参照对照。"""
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mode', [None, 'current', 'hard_off'])
def test_recorded_arguments_and_explicit_modes(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    role = 'decoderaux_grpo_alltask_final_vsti'
    entry = next(x for x in json.loads((root / 'configs/commands.json').read_text())['commands'] if x['role'] == role)
    keys = set(re.findall(r'\$\{([^}]+)\}', ' '.join(entry['argv'])))
    parameters = {key: str(tmp_path / key) for key in keys}
    weight = tmp_path / 'glow_patch_spatial.pt'
    weight.write_bytes(b'command-binding-fixture')
    parameters['DECODERAUX_GRPO_ALLTASK_FINAL_2048'] = str(weight)
    config = tmp_path / 'paths.json'
    config.write_text(json.dumps(parameters))
    command = [sys.executable, '-I', '-B', str(root / 'scripts/run_recorded.py'),
               '--role', role, '--parameters', str(config), '--inference-only']
    if mode is not None:
        command += ['--numeric-pose-mode', mode]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    actual = shlex.split(result.stdout.strip())
    original = [re.sub(r'\$\{([^}]+)\}', lambda match: parameters[match[1]], value) for value in entry['argv']]
    expected = [sys.executable, '-m', entry['module'], *original, '--inference-only']
    assert actual[:len(expected)] == expected
    if mode is None:
        assert actual == expected
    else:
        assert actual[len(expected):-1] == ['--gate1-arm', mode, '--gate1-reference-family', 'all',
                                           '--gate1-expected-checkpoint-sha256']
        assert len(actual[-1]) == 64
        # 同一路径不可重复开新对照，避免把两次作答混进一个输出目录。
        Path(parameters['OUTPUT_DIR_44']).mkdir()
        repeated = subprocess.run(command, text=True, capture_output=True)
        assert repeated.returncode != 0
        assert 'Numeric-pose comparison needs a new output directory' in repeated.stderr
