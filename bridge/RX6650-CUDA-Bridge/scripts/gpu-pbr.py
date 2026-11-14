"""RX 6650 XT (Windows native HIP PyTorch) real-GPU PBR acceptance test.

Requires WMI hardware gate when invoked via run-tests.py.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from gpu_primitives_shim import inspect_torch
from native_backend import check_native_device
from pbr_backend import check_params, export_maps, generate_cpu
from pbr_gpu_backend import compare_cpu_gpu, generate_gpu


def emit(data: dict):
    print(json.dumps(data, indent=2, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Native AMD GPU PBR test; not a CUDA bridge')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--out', type=Path, default=Path('reports/pbr-gpu'))
    args = parser.parse_args(argv)
    try:
        check_params(args.size, args.size, 'basalt', 'directx', 4, 6650)
        if not 3 <= args.iterations <= 20:
            raise ValueError('iterations must be between 3 and 20')
    except ValueError as exc:
        emit({'ok': False, 'stage': 'parameters', 'error': str(exc)})
        return 1
    if platform.system() != 'Windows':
        emit({'ok': False, 'stage': 'platform', 'error': 'Real Windows RX 6650 XT required'})
        return 3
    try:
        import torch
    except (ImportError, OSError) as exc:
        emit({'ok': False, 'stage': 'import', 'error': str(exc)})
        return 3
    try:
        info = inspect_torch(torch)
        status, reason = check_native_device(info)
        if status != 'PASS':
            emit({**info, 'ok': False, 'stage': 'identity', 'error': reason})
            return 3 if status == 'BLOCKED' else 1
        free, total = torch.cuda.mem_get_info(0)
        free_mib = free / 1048576
        if free_mib < 1024 + 128:
            emit({**info, 'ok': False, 'stage': 'memory-unavailable',
                  'error': 'Requires 1 GiB free VRAM reserve plus 128 MiB test headroom'})
            return 3
        cpu = generate_cpu(args.size, args.size, preset='basalt', seed=6650,
                           normal_format='directx', strength=4)
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
        peak_delta = 0
        comparisons = []
        data = None
        seen_tensors = 0
        for _ in range(args.iterations):
            data, count = generate_gpu(torch, args.size, args.size, preset='basalt',
                                       seed=6650, normal_format='directx', strength=4)
            seen_tensors = max(seen_tensors, count)
            comparisons.append(compare_cpu_gpu(cpu, data))
        torch.cuda.synchronize(0)
        allocated_mib = torch.cuda.max_memory_allocated(0) / 1048576
        peak_delta = max(r['max_channel_delta'] for r in comparisons)
        mismatches = sum(r['mismatch_count'] for r in comparisons)
        # Memory check happens before accepting the exported sample; no intentional OOM.
        if allocated_mib > 512:
            raise RuntimeError(f'Tracked torch peak {allocated_mib:.2f} MiB exceeded cap 512 MiB')
        result = export_maps(args.out, data, args.size, args.size, preset='basalt', seed=6650,
                             normal_format='directx', strength=4, backend='native-rocm')
        success = mismatches == 0 and seen_tensors >= 7 and result['roundtrip_verified']
        payload = {**info, 'stage': 'gpu-pbr', 'ok': success,
                   'gpu_compute_confirmed': bool(success), 'gpu_tensor_map_count': seen_tensors,
                   'iterations_completed': args.iterations, 'mismatch_count': mismatches,
                   'max_channel_delta': peak_delta, 'max_accepted_channel_delta': 3,
                   'roundtrip_verified': result['roundtrip_verified'], 'orm_verified': result['orm_verified'],
                   'normal_max_length_error': result['normal_max_length_error'],
                   'width': args.size, 'height': args.size, 'map_count': result['map_count'],
                   'peak_torch_allocated_mib': round(allocated_mib, 4),
                   'free_vram_mib_before': round(free_mib, 2),
                   'elapsed_seconds': round(time.perf_counter() - t0, 5)}
        emit(payload)
        return 0 if success else 5
    except Exception as exc:
        emit({'stage': 'execution', 'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
        return 6


if __name__ == '__main__':
    sys.exit(main())
