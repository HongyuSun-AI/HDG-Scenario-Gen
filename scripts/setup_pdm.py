import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, **kwargs):
    return subprocess.run(['git', '-C', str(path), *args], check=True, **kwargs)


def main():
    metadata = json.loads((ROOT / 'third_party/pdm_sources.json').read_text())
    for source in metadata['sources']:
        path = ROOT / 'third_party' / source['name']
        patch = ROOT / source['patch']
        if hashlib.sha256(patch.read_bytes()).hexdigest() != source['patch_sha256']:
            raise RuntimeError('Patch checksum mismatch: ' + str(patch))
        if not path.exists():
            path.mkdir(parents=True)
            git(path, 'init')
            git(path, 'remote', 'add', 'origin', source['url'])
        head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
        if head.returncode:
            git(path, 'fetch', '--depth', '1', 'origin', source['commit'])
            git(path, 'checkout', '--detach', 'FETCH_HEAD')
        elif head.stdout.strip() != source['commit']:
            raise RuntimeError('Checkout mismatch: ' + str(path))
        applied = subprocess.run(['git', '-C', str(path), 'apply', '--reverse', '--check', str(patch)],
                                 capture_output=True)
        if applied.returncode:
            git(path, 'apply', '--check', str(patch))
            git(path, 'apply', str(patch))
        print('Ready:', source['name'], source['commit'])


if __name__ == '__main__':
    main()
