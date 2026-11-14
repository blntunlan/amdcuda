"""RX 6650 XT VRAM policy, safe streaming plan and strict evidence checks.

No GPU dependencies here.  Allocation estimates are deliberately conservative
*for the four explicit FP32 tile tensors*, not guarantees about the driver,
allocator pools, rocBLAS workspace or other Windows processes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, TypeVar

from native_backend import check_native_device

MAX_BUDGET_MIB = 6144           # No more than 6 GiB, never all nominal 8 GiB
MIN_TILE = 16
MIN_ITERATIONS = 20
SAFETY_FIXED_MIB = 32           # Estimated allocator/library margin, not measured VRAM
SAFETY_TENSOR_FACTOR = 8        # At least 4 explicit tensors plus temporary slack


@dataclass(frozen=True)
class MemoryConfig:
    budget_mib: int = 512
    reserve_mib: int = 1024
    matrix_size: int = 128
    tile_size: int = 64
    batch_size: int = 4
    microbatch_size: int = 2
    iterations: int = 20


def valid_config(config: MemoryConfig) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, MemoryConfig):
        return ["Expected MemoryConfig"]
    limits = (
        ("budget_mib", config.budget_mib, 64, MAX_BUDGET_MIB),
        ("reserve_mib", config.reserve_mib, 256, 4096),
        ("matrix_size", config.matrix_size, 32, 1024),
        ("tile_size", config.tile_size, MIN_TILE, 512),
        ("batch_size", config.batch_size, 1, 16),
        ("microbatch_size", config.microbatch_size, 1, 16),
        ("iterations", config.iterations, MIN_ITERATIONS, 100),
    )
    for field, value, low, high in limits:
        if type(value) is not int or not low <= value <= high:
            errors.append(f"{field} must be an integer in {low}..{high}")
    if type(config.tile_size) is int and type(config.matrix_size) is int and config.tile_size > config.matrix_size:
        errors.append("tile_size must not exceed matrix_size")
    if type(config.microbatch_size) is int and type(config.batch_size) is int and config.microbatch_size > config.batch_size:
        errors.append("microbatch_size must not exceed batch_size")
    if type(config.tile_size) is int and config.tile_size >= MIN_TILE and config.tile_size & (config.tile_size - 1):
        errors.append("tile_size must be a power of 2")
    return errors


@dataclass(frozen=True)
class MemoryPlan:
    effective_budget_mib: float
    gpu_free_mib_before: float
    gpu_total_mib: float
    requested_budget_mib: int
    reserve_mib: int
    tile_size: int
    microbatch_size: int
    estimated_tile_peak_mib: float
    adjustments: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryUnavailable(RuntimeError):
    """Target GPU had insufficient safe free memory to start."""


class MemoryBudgetExceeded(RuntimeError):
    """PyTorch's live peak would exceed the project's user budget."""


class RetryExhausted(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


def _finite_positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def estimate_tile_peak_mib(tile: int, microbatch: int) -> float:
    """Host->GPU A/B, accumulate, product; 2x extra temporary estimate.

    Driver-level usage may exceed this estimate. We additionally inspect real
    torch memory counters during the test and allow the runner to shrink.
    """
    return SAFETY_FIXED_MIB + (SAFETY_TENSOR_FACTOR * microbatch * tile * tile * 4) / (1024 * 1024)


def check_allocation_request(request_mib: Any, effective_budget_mib: Any) -> bool:
    """Pre-allocation guard; does not attempt GPU allocation."""
    return (_finite_positive(request_mib) and _finite_positive(effective_budget_mib)
            and request_mib <= effective_budget_mib)


def _shrink(tile_size: int, microbatch: int) -> tuple[int, int] | None:
    if microbatch > 1:
        return tile_size, max(1, microbatch // 2)
    if tile_size > MIN_TILE:
        return max(MIN_TILE, tile_size // 2), 1
    return None


def tile_tasks(batch_size: int, dim: int, tile: int, microbatch: int):
    """Yield every disjoint output block, including non-divisible edge tiles."""
    for start in range(0, batch_size, microbatch):
        end = min(batch_size, start + microbatch)
        for row in range(0, dim, tile):
            row_end = min(dim, row + tile)
            for col in range(0, dim, tile):
                yield start, end, row, row_end, col, min(dim, col + tile)


def plan_memory(config: MemoryConfig, *, free_mib: float, total_mib: float) -> MemoryPlan:
    errors = valid_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    if not _finite_positive(free_mib) or not _finite_positive(total_mib) or free_mib > total_mib:
        raise ValueError("Invalid live GPU free/total memory measurement")
    # Identity validated by the preceding HIP + Windows identity stages.
    if total_mib < 4096 or total_mib > 16384:
        raise MemoryUnavailable("Reported GPU VRAM is implausible for RX 6650 XT")
    budget = min(float(config.budget_mib), max(0.0, free_mib - config.reserve_mib))
    if budget < SAFETY_FIXED_MIB + 1:
        raise MemoryUnavailable("Insufficient GPU free VRAM after preserving the reserve")
    tile, micro = config.tile_size, config.microbatch_size
    adjusted = 0
    while estimate_tile_peak_mib(tile, micro) > budget:
        result = _shrink(tile, micro)
        if result is None:
            raise MemoryUnavailable("Even minimum GPU tiles exceed the safe memory budget")
        tile, micro = result
        adjusted += 1
    return MemoryPlan(
        effective_budget_mib=round(budget, 5), gpu_free_mib_before=round(free_mib, 5),
        gpu_total_mib=round(total_mib, 5), requested_budget_mib=config.budget_mib,
        reserve_mib=config.reserve_mib, tile_size=tile, microbatch_size=micro,
        estimated_tile_peak_mib=round(estimate_tile_peak_mib(tile, micro), 5), adjustments=adjusted,
    )


def shrink_plan(plan: MemoryPlan) -> MemoryPlan | None:
    result = _shrink(plan.tile_size, plan.microbatch_size)
    if result is None:
        return None
    tile, micro = result
    return replace(plan, tile_size=tile, microbatch_size=micro,
                   estimated_tile_peak_mib=round(estimate_tile_peak_mib(tile, micro), 5),
                   adjustments=plan.adjustments + 1)


def is_oom_error(exc: BaseException) -> bool:
    """Narrowly classify common PyTorch ROCm and CUDA OOM exceptions."""
    if isinstance(exc, MemoryBudgetExceeded):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return ("outofmemory" in name or "out of memory" in msg
            or "hiperroroutofmemory" in msg or "hip_error_out_of_memory" in msg)


T = TypeVar("T")


def execute_with_retries(initial: MemoryPlan, work: Callable[[MemoryPlan], T],
                         cleanup: Callable[[], None]) -> tuple[T, MemoryPlan, list[dict[str, Any]]]:
    """On OOM / observed budget violation, clear cache and reduce work.

    Unrelated RuntimeError errors are *not* swallowed. On last failure stop
    instead of attempting a known-oversize allocation.
    """
    attempts: list[dict[str, Any]] = []
    plan = initial
    while True:
        if not check_allocation_request(plan.estimated_tile_peak_mib, plan.effective_budget_mib):
            raise MemoryBudgetExceeded("Unsafe memory estimate reached execute_with_retries")
        try:
            value = work(plan)
            attempts.append({"status": "PASS", "tile_size": plan.tile_size,
                             "microbatch_size": plan.microbatch_size})
            return value, plan, attempts
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            attempts.append({"status": "RETRIED", "tile_size": plan.tile_size,
                             "microbatch_size": plan.microbatch_size,
                             "reason": "GPU OOM or live budget limit"})
            new_plan = shrink_plan(plan)
            if new_plan is None:
                raise RetryExhausted("No smaller tile or microbatch remains", attempts) from exc
        # The caught exception's traceback must be released before empty_cache,
        # otherwise its frames may still hold GPU allocations.
        cleanup()
        plan = new_plan


def evaluate_vram_budget(payload: dict[str, Any]) -> tuple[str, str]:
    """Reject missing/fabricated-looking/inconsistent device & budget evidence.

    As with earlier stages this is a *validator*, not a substitute for running
    the actual Windows GPU script with WMI/ROCm identity checks.
    """
    if not isinstance(payload, dict) or payload.get("stage") != "vram-budget":
        return "FAIL", "Missing v0.5 GPU memory suite result"
    device_status, reason = check_native_device(payload)
    if device_status != "PASS":
        return device_status, reason
    if payload.get("ok") is not True or payload.get("gpu_compute_confirmed") is not True:
        return "FAIL", "Incomplete or failed GPU tiled compute"
    config = payload.get("config")
    if not isinstance(config, dict):
        return "FAIL", "Missing requested memory policy"
    try:
        policy = MemoryConfig(**config)
    except TypeError:
        return "FAIL", "Invalid memory policy schema"
    problems = valid_config(policy)
    if problems:
        return "FAIL", "; ".join(problems)
    if (payload.get("iterations_completed") != policy.iterations or
            type(payload.get("iterations_completed")) is not int or
            payload.get("gpu_tensor_verified") is not True or
            payload.get("safe_oversize_rejected") is not True):
        return "FAIL", "Insufficient iterations, missing GPU tensor or missing safe rejection"
    if payload.get("mismatch_count") != 0 or type(payload.get("mismatch_count")) is not int:
        return "FAIL", "CPU/GPU numerical mismatch"
    error = payload.get("max_abs_difference")
    if (type(error) not in (int, float) or not math.isfinite(error) or error < 0):
        return "FAIL", "Nonfinite/invalid numerical error"
    elapsed = payload.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        return "FAIL", "Invalid execution time"
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        return "FAIL", "Missing live memory plan"
    for k in ("effective_budget_mib", "gpu_free_mib_before", "gpu_total_mib",
              "estimated_tile_peak_mib"):
        if not _finite_positive(plan.get(k)):
            return "FAIL", f"Invalid live memory metric: {k}"
    budget, free, total = (plan["effective_budget_mib"],
                           plan["gpu_free_mib_before"], plan["gpu_total_mib"])
    if (plan.get("requested_budget_mib") != policy.budget_mib or
            plan.get("reserve_mib") != policy.reserve_mib or
            budget > policy.budget_mib or
            budget > max(0, free - policy.reserve_mib) + .01 or
            budget > MAX_BUDGET_MIB or free > total or
            plan["estimated_tile_peak_mib"] > budget or
            type(plan.get("tile_size")) is not int or
            type(plan.get("microbatch_size")) is not int or
            plan["tile_size"] > policy.tile_size or
            plan["tile_size"] < MIN_TILE or
            plan["microbatch_size"] > policy.microbatch_size or
            plan["microbatch_size"] < 1):
        return "FAIL", "Unsafe or inconsistent runtime memory plan"
    peak = payload.get("peak_torch_allocated_mib")
    if (type(peak) not in (int, float) or not math.isfinite(peak) or peak < 0):
        return "FAIL", "Missing real torch peak allocated metric"
    if peak > budget + .01:
        return "FAIL", "Observed torch allocation exceeded memory budget"
    attempts = payload.get("attempts")
    if (not isinstance(attempts, list) or len(attempts) < 1 or
            not isinstance(attempts[-1], dict) or attempts[-1].get("status") != "PASS" or
            any(not isinstance(item, dict) or item.get("status") != "RETRIED" for item in attempts[:-1])):
        return "FAIL", "No final successful OOM recovery attempt"
    if payload.get("oom_recoveries") != len(attempts) - 1:
        return "FAIL", "OOM recovery count inconsistent"
    return "PASS", f"{policy.iterations} GPU tiled FP32 runs; tracked peak {peak:.2f}/{budget:.2f} MiB"
