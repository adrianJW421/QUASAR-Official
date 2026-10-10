"""Build the source archive from PUBLIC_FILES.txt; generated data and test outputs stay in the caller's run directories."""
import argparse
from pathlib import Path
import tarfile


def check_manifest(root, names):
    included = []
    for line in (root / 'MANIFEST.in').read_text().splitlines():
        command, _, name = line.partition(' ')
        if command != 'include' or not name or ' ' in name:
            raise ValueError(f'MANIFEST.in supports only single-file include lines: {line!r}')
        included.append(name)
    expected = set(names) - {'MANIFEST.in'}
    if len(included) != len(set(included)) or set(included) != expected:
        raise ValueError(f'MANIFEST.in differs from PUBLIC_FILES.txt: missing {sorted(expected - set(included))}, '
                         f'extra {sorted(set(included) - expected)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = (root / 'PUBLIC_FILES.txt').read_text().splitlines()
    if not names or names != sorted(set(names)):
        raise ValueError('PUBLIC_FILES.txt must be a nonempty sorted list of distinct paths')
    check_manifest(root, names)
    for name in names:
        path = root / name
        if name == 'LOCAL_SOURCE_MAP.md':
            raise ValueError('The private source locator cannot enter the source manifest')
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Invalid source member: {name}')
        if any(x in Path(name).parts for x in ['runs', 'upstream', '__pycache__', '.pytest_cache', 'build', 'dist']):
            raise ValueError(f'Generated/external directory entered the source manifest: {name}')
        if path.suffix in ['.pt', '.pth', '.safetensors', '.npz', '.npy', '.pyc']:
            raise ValueError(f'Model/feature/geometry output entered the source manifest: {name}')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def anonymous_owner(info):
        # Tar headers store the local owner; clear it so the archive carries no account names.
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        return info

    with tarfile.open(a.output, 'w:gz') as archive:
        for name in names:
            archive.add(root / name, arcname='quasar/' + name, recursive=False, filter=anonymous_owner)
    print(f'{len(names)} source members: {a.output}')


if __name__ == '__main__':
    main()
