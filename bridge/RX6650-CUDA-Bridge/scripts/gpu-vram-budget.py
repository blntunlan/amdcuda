"""Native Windows RX 6650 XT v0.5 streamed GPU matmul acceptance test.

Requires an independently validated Windows adapter identity when run through
run-tests.py. This standalone process executes real HIP PyTorch GPU tensors;
it neither emulates CUDA nor pretends that CPU fallback passed GPU verification.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time

from gpu_primitives_shim import inspect_torch  # local import-safe shim
from memory_backend import (MemoryBudgetExceeded, MemoryConfig, MemoryUnavailable,
                            RetryExhausted, check_allocation_request, execute_with_retries,
                            plan_memory, tile_tasks, valid_config)
from native_backend import check_native_device


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


def _gpu_checked(tensor) -> None:
    if tensor.is_cuda is not True or tensor.device.type != "cuda" or tensor.device.index not in (0, None):
        raise RuntimeError("Tiled intermediate unexpectedly used CPU or a different GPU")


def _compute_once(torch, a_cpu, b_cpu, plan, *, budget_mib: float):
    """CPU-resident source/target; only bounded microbatch tiles live on GPU."""
    batch, dim, _ = a_cpu.shape
    result = torch.empty((batch, dim, dim), dtype=torch.float32, device="cpu")
    seen_gpu = False
    for s, e, i, i2, j, j2 in tile_tasks(batch, dim, plan.tile_size, plan.microbatch_size):
        out = torch.zeros((e - s, i2 - i, j2 - j), dtype=torch.float32, device="cuda:0")
        _gpu_checked(out)
        for k in range(0, dim, plan.tile_size):
            k2 = min(k + plan.tile_size, dim)
            a = a_cpu[s:e, i:i2, k:k2].to("cuda:0")
            b = b_cpu[s:e, k:k2, j:j2].to("cuda:0")
            _gpu_checked(a)
            _gpu_checked(b)
            product = torch.bmm(a, b)
            _gpu_checked(product)
            out.add_(product)
            seen_gpu = True
            # No surprise allocations allowed past the *tracked* cap.
            if torch.cuda.memory_allocated(0) / 1048576 > budget_mib + .01:
                raise MemoryBudgetExceeded("Tracked PyTorch tensors exceeded VRAM budget")
            del a, b, product
        result[s:e, i:i2, j:j2].copy_(out.to("cpu"))
        del out
        if torch.cuda.memory_allocated(0) / 1048576 > budget_mib + .01:
            raise MemoryBudgetExceeded("Torch active tensors exceeded cap after readback")
    torch.cuda.synchronize(0)
    return result, seen_gpu


def run_memory_suite(torch, config: MemoryConfig) -> dict:
    """Run >=20 actual streamed matmul passes and compare each pass to CPU."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    initial = plan_memory(config, free_mib=free_bytes / 1048576, total_mib=total_bytes / 1048576)

    # Keep reference matrices on system memory.  No artificial device allocation
    # above the limit; rejecting budget+1 happens *before* requesting VRAM.
    guard_rejected = not check_allocation_request(initial.effective_budget_mib + 1,
                                                 initial.effective_budget_mib)
    generator = torch.Generator(device="cpu").manual_seed(6650)
    a_cpu = torch.randn((config.batch_size, config.matrix_size, config.matrix_size),
                        generator=generator, dtype=torch.float32)
    b_cpu = torch.randn((config.batch_size, config.matrix_size, config.matrix_size),
                        generator=generator, dtype=torch.float32)
    reference = torch.bmm(a_cpu, b_cpu)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except (AttributeError, RuntimeError):
        pass

    def cleanup() -> None:
        torch.cuda.empty_cache()

    def work(plan):
        torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        max_diff = 0.0
        mismatches = 0
        gpu_verified = True
        completed = 0
        for _ in range(config.iterations):
            output, seen_gpu = _compute_once(torch, a_cpu, b_cpu, plan,
                                              budget_mib=plan.effective_budget_mib)
            gpu_verified &= seen_gpu
            finite = bool(torch.isfinite(output).all().item())
            if not finite:
                raise RuntimeError("GPU produced nonfinite outputs")
            delta = (output - reference).abs()
            max_diff = max(max_diff, float(delta.max().item()))
            mismatches += int((~torch.isclose(output, reference, rtol=1e-3, atol=1e-3)).sum().item())
            completed += 1
            if mismatches:
                raise RuntimeError("GPU tiled result differed from CPU oracle")
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - start
        allocated_mib = torch.cuda.max_memory_allocated(0) / 1048576
        if allocated_mib > plan.effective_budget_mib + .01:
            raise MemoryBudgetExceeded("Peak tracked PyTorch allocation exceeded budget")
        try:
            reserved = torch.cuda.max_memory_reserved(0) / 1048576
        except (RuntimeError, AttributeError):
            reserved = None
        return {"iterations_completed": completed, "gpu_tensor_verified": gpu_verified,
                "mismatch_count": mismatches, "max_abs_difference": max_diff,
                "elapsed_seconds": round(elapsed, 6),
                "peak_torch_allocated_mib": round(allocated_mib, 5),
                "peak_torch_reserved_mib": round(reserved, 5) if reserved is not None else None}

    measurements, final_plan, attempts = execute_with_retries(initial, work, cleanup)
    free_after, _ = torch.cuda.mem_get_info(0)
    return {
        "stage": "vram-budget", "ok": True, "gpu_compute_confirmed": bool(measurements["gpu_tensor_verified"]),
        "config": vars(config), "plan": final_plan.to_dict(), "attempts": attempts,
        "oom_recoveries": len(attempts) - 1, "safe_oversize_rejected": guard_rejected,
        "gpu_free_mib_after": round(free_after / 1048576, 5), **measurements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RX 6650 XT safe VRAM tile/batch test (Windows HIP PyTorch)")
    parser.add_argument("--budget-mib", type=int, default=512)
    parser.add_argument("--reserve-mib", type=int, default=1024)
    parser.add_argument("--matrix-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    config = MemoryConfig(**vars(args))
    errors = valid_config(config)
    if errors:
        emit({"ok": False, "stage": "parameters", "error": "; ".join(errors)})
        return 1
    if platform.system() != "Windows":
        emit({"ok": False, "stage": "platform", "error": "Target Windows RX 6650 XT required"})
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
        try:
            metrics = run_memory_suite(torch, config)
        except MemoryUnavailable as exc:
            emit({**info, "ok": False, "stage": "memory-unavailable", "error": str(exc)})
            return 3
        except RetryExhausted as exc:
            emit({**info, "ok": False, "stage": "oom-recovery-exhausted", "error": str(exc),
                  "attempts": exc.attempts})
            return 6
        emit({**info, **metrics})
        return 0 if metrics["ok"] else 5
    except Exception as exc:
        emit({"ok": False, "stage": "execution", "error": f"{type(exc).__name__}: {exc}"})
        return 6


if __name__ == "__main__":
    sys.exit(main())
