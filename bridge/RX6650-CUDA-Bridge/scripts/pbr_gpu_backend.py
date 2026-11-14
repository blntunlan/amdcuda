"""Native HIP PyTorch tensor PBR pipeline. Only called after strict GPU verification.

The torch.cuda namespace also denotes native AMD ROCm devices. This is NOT
CUDA compatibility/ZLUDA and does not silently run PyTorch on CPU.
"""
from __future__ import annotations

from pbr_backend import (MAPS, check_params, height_at, surface_channels,
                         validate_maps)


def generate_tensor_maps(torch, width: int, height: int, *, preset: str = 'basalt', seed: int = 6650,
                         normal_format: str = 'directx', strength: float = 4.0, device: str = 'cpu'):
    """Shared tensor math, separately unit-tested on CPU without a GPU claim."""
    check_params(width, height, preset, normal_format, strength, seed)
    x = torch.arange(width, device=device, dtype=torch.float32) / width
    y = torch.arange(height, device=device, dtype=torch.float32) / height
    v, u = torch.meshgrid(y, x, indexing='ij')
    h = height_at(u, v, preset, seed, torch.sin, torch.cos)
    left, right = torch.roll(h, 1, 1), torch.roll(h, -1, 1)
    top, bottom = torch.roll(h, 1, 0), torch.roll(h, -1, 0)
    avg = (left + right + top + bottom) * .25
    dx = (right - left) * .5 * strength
    dy = (bottom - top) * .5 * strength
    green_sign = -1.0 if normal_format == 'directx' else 1.0
    nx, ny, nz = -dx, -dy * green_sign, torch.ones_like(h)
    n = torch.stack([nx, ny, nz], dim=2)
    n = n / torch.linalg.vector_norm(n, dim=2, keepdim=True)
    base, rough, metal, ao = surface_channels(h, avg, preset,
                                             lambda t: torch.clamp(t, 0, 1))
    groups = {
        'BaseColor': torch.stack(base, dim=2),
        'Normal': (n + 1) / 2,
        'Roughness': rough,
        'Metallic': metal,
        'AO': ao,
        'ORM': torch.stack([ao, rough, metal], dim=2),
        'Height': h,
    }
    return groups


def generate_gpu(torch, width: int, height: int, *, preset: str = 'basalt', seed: int = 6650,
                 normal_format: str = 'directx', strength: float = 4.0):
    groups = generate_tensor_maps(torch, width, height, preset=preset, seed=seed,
                                  normal_format=normal_format, strength=strength, device='cuda:0')
    if set(groups) != set(MAPS) or any(t.device.type != 'cuda' or t.device.index not in (0, None)
                                        for t in groups.values()):
        raise RuntimeError('GPU tensor identity was not preserved for all material maps')
    torch.cuda.synchronize(0)
    data = {}
    for name, t in groups.items():
        raw = torch.clamp(torch.round(t * 255), 0, 255).to(torch.uint8)
        if not raw.is_cuda:
            raise RuntimeError('Unexpected CPU fallback in quantization')
        data[name] = bytes(raw.contiguous().flatten().to('cpu').tolist())
    validate_maps(data, width, height, normal_format=normal_format)
    return data, len(groups)


def compare_cpu_gpu(reference: dict[str, bytes], actual: dict[str, bytes], *, max_per_channel_delta: int = 3) -> dict:
    if set(reference) != set(actual):
        raise ValueError('Map set differs from CPU oracle')
    max_diff = 0
    mismatches = 0
    for name in reference:
        a, b = reference[name], actual[name]
        if len(a) != len(b):
            raise ValueError(f'{name}: different dimensions/channels from CPU oracle')
        for ref, out in zip(a, b):
            diff = abs(ref - out)
            max_diff = max(max_diff, diff)
            mismatches += diff > max_per_channel_delta
    return {'max_channel_delta': max_diff, 'mismatch_count': mismatches,
            'max_accepted_channel_delta': max_per_channel_delta}
