"""4K memory-bounded PBR test command. Requires numpy and Pillow. CPU only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pbr_4k_backend import render_material


def main():
    p = argparse.ArgumentParser(description='4K streaming PBR material renderer (NumPy CPU)')
    p.add_argument('--preset', choices=['basalt', 'worn_metal'], default='basalt')
    p.add_argument('--size', type=int, default=4096)
    p.add_argument('--tile-rows', type=int, default=256)
    p.add_argument('--seed', type=int, default=6650)
    p.add_argument('--normal-format', choices=['directx', 'opengl'], default='directx')
    p.add_argument('--normal-strength', type=float, default=4.0)
    p.add_argument('--out', type=Path, default=Path('reports/pbr-4k'))
    args = p.parse_args()
    try:
        meta = render_material(args.out, args.size, args.size, tile_rows=args.tile_rows,
                               preset=args.preset, seed=args.seed, normal_format=args.normal_format,
                               strength=args.normal_strength)
        print(json.dumps({'ok': True, 'output': str(args.out.absolute()), **meta}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
