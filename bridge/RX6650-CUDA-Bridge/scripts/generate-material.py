"""Reproducible standard-library PBR generator (GPU not claimed)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from pbr_backend import export_maps, generate_cpu, load_height_png, check_params, PRESETS, FORMATS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Generate tiled procedural PBR maps (CPU oracle)')
    p.add_argument('--preset', choices=PRESETS, default='basalt')
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--height', type=int, default=256)
    p.add_argument('--size', type=int, help='Square output: overrides width and height')
    p.add_argument('--seed', type=int, default=6650)
    p.add_argument('--normal-format', choices=FORMATS, default='directx',
                   help='directx = Y- for Unreal; opengl = Y+ for Godot')
    p.add_argument('--normal-strength', type=float, default=4.0)
    p.add_argument('--height-input', type=Path, help='Optional PNG height map instead of analytical procedural height')
    p.add_argument('--out', type=Path, default=Path('reports/pbr-material'))
    args = p.parse_args(argv)
    start = time.perf_counter()
    try:
        imported = None
        if args.height_input:
            width, height, imported = load_height_png(args.height_input)
        else:
            width = height = args.size if args.size is not None else None
            if width is None:
                width, height = args.width, args.height
        check_params(width, height, args.preset, args.normal_format, args.normal_strength, args.seed)
        maps = generate_cpu(width, height, preset=args.preset, seed=args.seed,
                            normal_format=args.normal_format, strength=args.normal_strength,
                            height_override=imported)
        meta = export_maps(args.out, maps, width, height, preset=args.preset, seed=args.seed,
                           normal_format=args.normal_format, strength=args.normal_strength)
        output = {'stage': 'pbr-cpu', 'ok': True, 'output_dir': str(args.out.resolve()),
                  'roundtrip_verified': meta['roundtrip_verified'], 'orm_verified': meta['orm_verified'],
                  'normal_max_length_error': meta['normal_max_length_error'],
                  'map_count': meta['map_count'], 'maps': list(maps), 'width': width, 'height': height,
                  'normal_format': args.normal_format, 'gpu_compute_confirmed': False,
                  'elapsed_seconds': round(time.perf_counter()-start, 4)}
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({'stage': 'pbr-cpu', 'ok': False, 'error': str(exc)}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
