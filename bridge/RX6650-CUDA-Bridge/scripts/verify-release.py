"""Verify checksum manifest for a packaged RX 6650 XT release (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path) -> list[str]:
    problems = []
    try:
        manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
        paths = manifest['sha256']
        if not isinstance(paths, dict) or not paths:
            raise ValueError('Expected non-empty sha256 mapping')
        for name, expected in paths.items():
            if not isinstance(name, str) or not isinstance(expected, str):
                problems.append('Invalid manifest entry')
                continue
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                problems.append(f'Missing or unsafe path: {name}')
            elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                problems.append(f'Checksum mismatch: {name}')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f'Invalid manifest: {exc}')
    return problems


if __name__ == '__main__':
    issues = verify(ROOT)
    if issues:
        print('FAIL: ' + '; '.join(issues))
        sys.exit(1)
    print('PASS: packaged project file checksums match MANIFEST.json')
