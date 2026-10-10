import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('package_source', ROOT / 'scripts' / 'package_source.py')
package_source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_source)


def test_manifest_matches_public_file_list():
    names = (ROOT / 'PUBLIC_FILES.txt').read_text().splitlines()
    package_source.check_manifest(ROOT, names)
    assert 'docs/figures/teaser.png' in names and 'docs/figures/framework.png' in names


def test_manifest_missing_or_extra_member_is_rejected(tmp_path):
    names = ['MANIFEST.in', 'README.md', 'docs/figures/teaser.png']
    (tmp_path / 'MANIFEST.in').write_text('include README.md\n')
    with pytest.raises(ValueError, match='docs/figures/teaser.png'):
        package_source.check_manifest(tmp_path, names)
    (tmp_path / 'MANIFEST.in').write_text('include README.md\ninclude docs/figures/teaser.png\ninclude extra.txt\n')
    with pytest.raises(ValueError, match='extra.txt'):
        package_source.check_manifest(tmp_path, names)
    (tmp_path / 'MANIFEST.in').write_text('include README.md\ninclude docs/figures/teaser.png\n')
    package_source.check_manifest(tmp_path, names)
