"""Native ROCm GPU matmul attestation for RX 6650 XT (Windows only).

Must be invoked behind the independent WMI diagnostic gate from run-tests.py.
JSON output is machine-readable, but is NOT cryptographic GPU execution proof.
No downloads or system configuration changes are made.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time

from native_backend import check_native_device


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def inspect_torch(torch) -> dict:
    """Collect metadata before any tensor allocation."""
    info = {
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()),
        "available": bool(torch.cuda.is_available()),
        "gpu_name": "",
        "gpu_arch": "",
    }
    if info["available"] and info["device_count"] == 1:
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        info["gpu_arch"] = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    return info


def execute_gpu_matmul(torch, *, size: int, iterations: int) -> dict:
    """Executes real GPU work; data generated on CPU, transferred explicitly."""
    torch.manual_seed(6650)
    a_cpu = torch.randn(size, size, dtype=torch.float32, device="cpu")
    b_cpu = torch.randn(size, size, dtype=torch.float32, device="cpu")
    reference = a_cpu @ b_cpu

    a_gpu = a_cpu.to("cuda:0")
    b_gpu = b_cpu.to("cuda:0")
    if not (bool(a_gpu.is_cuda) and bool(b_gpu.is_cuda)):
        raise RuntimeError("Inputs did not transfer to a torch.cuda GPU device")
    # Exclude asynchronous transfers from the timed GPU kernel section.
    torch.cuda.synchronize(0)
    try:
        torch.cuda.reset_peak_memory_stats(0)
    except (AttributeError, RuntimeError):
        pass
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = a_gpu @ b_gpu
        if not bool(result.is_cuda):
            raise RuntimeError("Result tensor unexpectedly fell back to CPU")
    torch.cuda.synchronize(0)
    kernel_time = time.perf_counter() - start
    actual = result.to("cpu")
    max_difference = float((actual - reference).abs().max().item())
    match = bool(torch.allclose(actual, reference, atol=1e-3, rtol=1e-3))
    if not math.isfinite(max_difference):
        match = False

    metrics = {"ok": match, "gpu_compute_confirmed": True, "stage": "matmul",
               "iterations": iterations, "matrix_size": size,
               "gpu_kernel_seconds": round(kernel_time, 6),
               "max_abs_difference": max_difference,
               "atol": 1e-3, "rtol": 1e-3}
    try:
        metrics["peak_torch_allocated_mib"] = round(torch.cuda.max_memory_allocated(0) / (1024**2), 3)
    except (AttributeError, RuntimeError):
        metrics["peak_torch_allocated_mib"] = None
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an RX6650XT native ROCm kernel test")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.iterations < 3 or args.iterations > 1000 or args.size < 16 or args.size > 1024:
        emit({"ok": False, "stage": "parameters", "error": "iterations must be 3..1000 and size 16..1024"})
        return 1
    if platform.system() != "Windows":
        emit({"ok": False, "stage": "platform", "error": "Native Windows target required"})
        return 3
    try:
        import torch
    except (ImportError, OSError) as exc:
        emit({"ok": False, "stage": "import", "error": str(exc)})
        return 3
    try:
        info = inspect_torch(torch)
        status, detail = check_native_device(info)
        if status != "PASS":
            emit({**info, "ok": False, "stage": "identity", "error": detail})
            return 3 if status == "BLOCKED" else 1
        metrics = execute_gpu_matmul(torch, size=args.size, iterations=args.iterations)
        emit({**info, **metrics})
        return 0 if metrics["ok"] else 5
    except Exception as exc:
        emit({"ok": False, "stage": "execution", "error": f"{type(exc).__name__}: {exc}"})
        return 6


if __name__ == "__main__":
    sys.exit(main())
