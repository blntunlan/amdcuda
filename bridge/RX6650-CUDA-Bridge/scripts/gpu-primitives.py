"""Windows-native ROCm GPU primitive acceptance suite for RX 6650 XT.

Only runs after the WMI identity gate in run-tests.py (manual direct calls do not
provide independent host identity). Does not use ZLUDA or simulate the GPU.
Prints one JSON object to stdout, uses nonzero status on any error.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time

from native_backend import check_native_device
from primitives_backend import check_parameters


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def op_result(name: str, elapsed_seconds: float, *, correct: bool, gpu_tensor_verified: bool,
              iterations: int = 1, **extras) -> dict:
    return {"name": name, "status": "PASS" if (correct and gpu_tensor_verified) else "FAIL",
            "elapsed_seconds": round(elapsed_seconds, 6), "iterations": iterations,
            "correct": bool(correct), "gpu_tensor_verified": bool(gpu_tensor_verified), **extras}


def inspect_torch(torch) -> dict:
    info = {"torch_version": str(getattr(torch, "__version__", "unknown")),
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "available": bool(torch.cuda.is_available()), "gpu_name": "", "gpu_arch": ""}
    if info["available"] and info["device_count"] == 1:
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        info["gpu_arch"] = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    return info


def compare(torch, actual, expected, *, atol: float, rtol: float) -> tuple[bool, float, int]:
    """Full tensor comparison on CPU, including NaN/Inf rejection."""
    if actual.shape != expected.shape:
        return False, math.inf, -1
    actual = actual.to("cpu")
    expected = expected.to("cpu")
    finite = bool(torch.isfinite(actual).all().item()) and bool(torch.isfinite(expected).all().item())
    if not finite:
        return False, math.inf, -1
    diff = (actual - expected).abs()
    mismatch_count = int((~torch.isclose(actual, expected, atol=atol, rtol=rtol)).sum().item())
    return mismatch_count == 0, float(diff.max().item()), mismatch_count


def run_suite(torch, *, iterations: int, vector_size: int, matrix_size: int, transfer_mib: int) -> dict:
    device = "cuda:0"  # PyTorch ROCm intentionally uses torch.cuda API
    torch.manual_seed(6650)
    operations = []
    start = time.perf_counter()
    zeros = torch.zeros(4096, dtype=torch.float32, device=device)
    zeros_total = float(zeros.sum().item())  # synchronize and verify an actual GPU operation
    torch.cuda.synchronize(0)
    operations.append(op_result("allocation", time.perf_counter() - start,
                                correct=zeros_total == 0.0, gpu_tensor_verified=bool(zeros.is_cuda)))

    # One controlled transfer allocation, no unbounded stress allocations.
    count = transfer_mib * 1024 * 1024 // 4
    src = (torch.arange(count, dtype=torch.int32, device="cpu") % 251)
    start = time.perf_counter()
    uploaded = src.to(device)
    torch.cuda.synchronize(0)
    operations.append(op_result("host_to_device", time.perf_counter() - start,
                                correct=bool(uploaded.is_cuda) and tuple(uploaded.shape) == tuple(src.shape),
                                gpu_tensor_verified=bool(uploaded.is_cuda), transferred_mib=transfer_mib))

    start = time.perf_counter()
    returned = uploaded.to("cpu")
    torch.cuda.synchronize(0)
    operations.append(op_result("device_to_host", time.perf_counter() - start,
                                correct=bool(torch.equal(returned, src)),
                                gpu_tensor_verified=bool(uploaded.is_cuda), transferred_mib=transfer_mib))

    gen = torch.Generator(device="cpu").manual_seed(6650)
    vec_a = torch.randn(vector_size, dtype=torch.float32, generator=gen)
    vec_b = torch.randn(vector_size, dtype=torch.float32, generator=gen)
    vec_ref = vec_a + vec_b
    gpu_a = vec_a.to(device)
    gpu_b = vec_b.to(device)
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    out = None
    for _ in range(iterations):
        out = gpu_a + gpu_b
        if not bool(out.is_cuda):
            raise RuntimeError("Vector output unexpectedly on CPU")
    torch.cuda.synchronize(0)
    duration = time.perf_counter() - start
    ok, max_diff, mismatches = compare(torch, out, vec_ref, atol=1e-5, rtol=1e-5)
    operations.append(op_result("vector_add", duration, correct=ok,
                                gpu_tensor_verified=bool(out.is_cuda), iterations=iterations,
                                elements=vector_size, max_abs_difference=max_diff,
                                mismatch_count=mismatches, atol=1e-5, rtol=1e-5))

    mat_a = torch.randn((matrix_size, matrix_size), dtype=torch.float32, generator=gen)
    mat_b = torch.randn((matrix_size, matrix_size), dtype=torch.float32, generator=gen)
    mat_ref = mat_a @ mat_b
    gpu_mat_a = mat_a.to(device)
    gpu_mat_b = mat_b.to(device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except AttributeError:
        pass
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    mat_out = None
    for _ in range(iterations):
        mat_out = gpu_mat_a @ gpu_mat_b
        if not bool(mat_out.is_cuda):
            raise RuntimeError("Matrix output unexpectedly on CPU")
    torch.cuda.synchronize(0)
    duration = time.perf_counter() - start
    ok, max_diff, mismatches = compare(torch, mat_out, mat_ref, atol=1e-3, rtol=1e-3)
    operations.append(op_result("matmul", duration, correct=ok,
                                gpu_tensor_verified=bool(mat_out.is_cuda), iterations=iterations,
                                elements=matrix_size * matrix_size, max_abs_difference=max_diff,
                                mismatch_count=mismatches, atol=1e-3, rtol=1e-3))

    metrics = {}
    for key, fn in (("peak_torch_allocated_mib", torch.cuda.max_memory_allocated),
                    ("peak_torch_reserved_mib", torch.cuda.max_memory_reserved)):
        try:
            metrics[key] = round(fn(0) / (1024 ** 2), 3)
        except (RuntimeError, AttributeError, TypeError):
            metrics[key] = None
    try:
        free, total = torch.cuda.mem_get_info(0)
        metrics["gpu_free_mib_after"] = round(free / (1024 ** 2), 3)
        metrics["gpu_total_mib"] = round(total / (1024 ** 2), 3)
    except (RuntimeError, AttributeError, TypeError):
        metrics["gpu_free_mib_after"] = None
        metrics["gpu_total_mib"] = None
    success = all(op["status"] == "PASS" for op in operations)
    return {"stage": "primitives", "ok": success, "gpu_compute_confirmed": success,
            "operations": operations, **metrics,
            "parameters": {"iterations": iterations, "vector_size": vector_size,
                           "matrix_size": matrix_size, "transfer_mib": transfer_mib}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native RX 6650 XT ROCm GPU primitive acceptance suite")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--vector-size", type=int, default=262144)
    parser.add_argument("--matrix-size", type=int, default=256)
    parser.add_argument("--transfer-mib", type=int, default=4)
    args = parser.parse_args(argv)
    problems = check_parameters(args.iterations, args.vector_size, args.matrix_size, args.transfer_mib)
    if problems:
        emit({"ok": False, "stage": "parameters", "error": "; ".join(problems)})
        return 1
    if platform.system() != "Windows":
        emit({"ok": False, "stage": "platform", "error": "Windows RX 6650 XT hardware required"})
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
        metrics = run_suite(torch, iterations=args.iterations, vector_size=args.vector_size,
                            matrix_size=args.matrix_size, transfer_mib=args.transfer_mib)
        emit({**info, **metrics})
        return 0 if metrics["ok"] else 5
    except Exception as exc:
        emit({"ok": False, "stage": "execution", "error": f"{type(exc).__name__}: {exc}"})
        return 6


if __name__ == "__main__":
    sys.exit(main())
