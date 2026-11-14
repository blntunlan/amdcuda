"""Safe RX 6650 XT/ROCm PyTorch preflight; stdlib-only until torch import."""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def classify_backend(name: str, hip_version: str | None, cuda_version: str | None, available: bool, *, target: str = '6650 XT') -> tuple[str, str]:
    """No CPU/NVIDIA/mismatched AMD card can pass as target AMD GPU."""
    if not available:
        return 'BLOCKED', 'PyTorch GPU backend not available'
    if not hip_version:
        return 'BLOCKED', f'Not ROCm/HIP PyTorch (torch.version.cuda={cuda_version!r})'
    if not re.search(r'6650\s*XT', name, flags=re.I):
        return 'BLOCKED', f'GPU mismatch: {name!r}; expected RX {target}'
    return 'READY', f'ROCm backend enumerated target: {name}'


def assess_free_memory(total_gb: float) -> str:
    if total_gb < 20:
        return 'WARNING: LLM2Vec Llama 3 8B CPU encoder may exceed available RAM; consider remote TEXT_ENCODER_URL or RAM upgrade.'
    return 'RAM size does not guarantee encoder will fit; monitor RAM and page file.'


def probe(run_kernel: bool = True) -> dict:
    out = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'status': 'BLOCKED',
        'reason': 'not started',
        'gpu_compute': 'NOT_RUN',
    }
    try:
        import torch
    except Exception as e:
        out['reason'] = f'PyTorch unavailable: {type(e).__name__}: {e}'
        return out
    out.update(torch_version=torch.__version__, torch_hip=str(torch.version.hip or ''), torch_cuda=str(torch.version.cuda or ''))
    try:
        available = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if available else ''
    except Exception as e:
        out['reason'] = f'GPU enumeration exception: {type(e).__name__}: {e}'
        return out
    status, reason = classify_backend(name, torch.version.hip, torch.version.cuda, available)
    out.update(status=status, reason=reason, device=name)
    if status != 'READY' or not run_kernel:
        return out
    try:
        torch.manual_seed(17)
        # Actual device compute; fail on wrong result or driver error.
        a_cpu = torch.randn((128, 128), dtype=torch.float32)
        b_cpu = torch.randn((128, 128), dtype=torch.float32)
        expected = a_cpu @ b_cpu
        torch.cuda.synchronize()
        tic = time.perf_counter()
        actual = (a_cpu.to('cuda:0') @ b_cpu.to('cuda:0')).cpu()
        torch.cuda.synchronize()
        elapsed_ms = round((time.perf_counter() - tic) * 1000, 3)
        max_error = (actual - expected).abs().max().item()
        out.update(gpu_compute='PASS' if torch.allclose(expected, actual, rtol=2e-3, atol=2e-3) else 'FAIL',
                   latency_ms=elapsed_ms, max_abs_error=round(max_error, 6))
        if out['gpu_compute'] != 'PASS':
            out.update(status='FAIL', reason='GPU matmul differs from CPU reference')
        else:
            out.update(status='PASS', reason='RX 6650 XT HIP PyTorch matrix operation verified')
    except Exception as e:
        out.update(status='FAIL', gpu_compute='FAIL', reason=f'GPU execution exception: {type(e).__name__}: {e}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='reports/backend_probe.json')
    ap.add_argument('--no-kernel', action='store_true')
    args = ap.parse_args()
    result = probe(not args.no_kernel)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'REPORT: {out.resolve()}')
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
