"""Pure validation and safety rules for native RX 6650 XT GPU primitives.

This code is intentionally independent of PyTorch. It validates evidence supplied
by the separately executed Windows/GPU process. It cannot independently prove
which physical hardware executed a kernel; the host WMI gate is also required.
"""
from __future__ import annotations

import math
from typing import Any

from native_backend import check_native_device

REQUIRED_OPERATIONS = ("allocation", "host_to_device", "device_to_host", "vector_add", "matmul")
MIN_COMPUTE_ITERATIONS = 20


def check_parameters(iterations: int, vector_size: int, matrix_size: int, transfer_mib: int) -> list[str]:
    errors: list[str] = []
    for name, value, lower, upper in (
        ("iterations", iterations, MIN_COMPUTE_ITERATIONS, 200),
        ("vector_size", vector_size, 1024, 2_097_152),
        ("matrix_size", matrix_size, 16, 1024),
        ("transfer_mib", transfer_mib, 1, 32),
    ):
        if type(value) is not int or value < lower or value > upper:
            errors.append(f"{name} must be an integer in {lower}..{upper}")
    return errors


def validate_operation(operation: dict[str, Any], expected_name: str) -> list[str]:
    problems: list[str] = []
    if not isinstance(operation, dict) or operation.get("name") != expected_name:
        return [f"missing/wrong operation: {expected_name}"]
    if operation.get("status") != "PASS":
        problems.append(f"{expected_name}: operation did not PASS")
    if operation.get("gpu_tensor_verified") is not True:
        problems.append(f"{expected_name}: GPU tensor was not confirmed")
    if operation.get("correct") is not True:
        problems.append(f"{expected_name}: comparison failed")
    elapsed = operation.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        problems.append(f"{expected_name}: invalid elapsed_seconds")
    count = operation.get("iterations")
    minimum = MIN_COMPUTE_ITERATIONS if expected_name in ("vector_add", "matmul") else 1
    if type(count) is not int or count < minimum:
        problems.append(f"{expected_name}: expected at least {minimum} iterations")
    if expected_name in ("vector_add", "matmul"):
        difference = operation.get("max_abs_difference")
        if (type(difference) not in (int, float) or not math.isfinite(difference)
                or difference < 0):
            problems.append(f"{expected_name}: invalid max_abs_difference")
        error_count = operation.get("mismatch_count")
        if type(error_count) is not int or error_count != 0:
            problems.append(f"{expected_name}: mismatched CPU/GPU elements")
    return problems


def evaluate_primitives(payload: dict[str, Any]) -> tuple[str, str]:
    """Fail closed for incomplete, incompatible, or incorrect evidence."""
    if not isinstance(payload, dict) or payload.get("stage") != "primitives":
        return "FAIL", "Missing GPU primitive suite evidence"
    status, detail = check_native_device(payload)
    if status != "PASS":
        return status, detail
    if payload.get("gpu_compute_confirmed") is not True or payload.get("ok") is not True:
        return "FAIL", "GPU primitive suite was not completed successfully"
    operations = payload.get("operations")
    if not isinstance(operations, list) or len(operations) != len(REQUIRED_OPERATIONS):
        return "FAIL", "Expected all five primitive operations exactly once"
    problems = [p for operation, name in zip(operations, REQUIRED_OPERATIONS)
                for p in validate_operation(operation, name)]
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", "Native GPU allocation, H2D/D2H, vector addition and matmul validated"
