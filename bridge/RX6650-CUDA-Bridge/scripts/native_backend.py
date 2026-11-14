"""Strict native ROCm validation rules for this one RX 6650 XT.

Pure helpers deliberately accept normal dictionaries, enabling offline tests without
installing PyTorch or suggesting that a mock GPU is a verified physical GPU.
"""
from __future__ import annotations

import math
import re
from typing import Any

TARGET_NAME = re.compile(r"\bRX\s*6650\s*XT\b", re.IGNORECASE)
ARCH_NAME = re.compile(r"\bgfx1032(?:\b|:)", re.IGNORECASE)


def check_native_device(info: dict[str, Any]) -> tuple[str, str]:
    """Fail closed if torch cannot independently identify the native AMD backend."""
    if not isinstance(info, dict):
        return "FAIL", "Runtime inventory must be a JSON object"
    if not info.get("hip_version"):
        return "BLOCKED", "PyTorch lacks ROCm/HIP support; NVIDIA CUDA or CPU build is not accepted"
    count = info.get("device_count")
    if type(count) is not int or count < 1:
        return "BLOCKED", "No ROCm GPUs visible"
    if count != 1:
        return "BLOCKED", f"{count} GPU devices visible; explicit binding not implemented"
    if info.get("available") is not True:
        return "BLOCKED", "ROCm runtime reports GPU unavailable"
    name = str(info.get("gpu_name") or "")
    arch = str(info.get("gpu_arch") or "")
    if not TARGET_NAME.search(name):
        return "BLOCKED", f"Runtime GPU is not explicitly identified as RX 6650 XT: {name!r}"
    if arch and not ARCH_NAME.search(arch):
        return "BLOCKED", f"Runtime GPU architecture is not gfx1032: {arch!r}"
    return "PASS", "Single native ROCm RX 6650 XT runtime device identified"


def evaluate_native_math(payload: dict[str, Any], *, min_iterations: int = 3) -> tuple[str, str]:
    """Validate structured evidence; result must be from live hardware gate."""
    if not isinstance(payload, dict) or payload.get("stage") != "matmul":
        return "FAIL", "No GPU matmul evidence"
    status, reason = check_native_device(payload)
    if status != "PASS":
        return status, reason
    if payload.get("gpu_compute_confirmed") is not True or payload.get("ok") is not True:
        return "FAIL", "Kernel did not finish with correct GPU output"
    iterations = payload.get("iterations")
    if type(iterations) is not int or iterations < min_iterations:
        return "FAIL", f"Expected at least {min_iterations} GPU matmul iterations"
    max_diff = payload.get("max_abs_difference")
    if type(max_diff) not in (int, float) or not math.isfinite(max_diff) or max_diff < 0:
        return "FAIL", "Invalid or nonfinite numerical error"
    elapsed = payload.get("gpu_kernel_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        return "FAIL", "Missing/nonfinite execution duration"
    return "PASS", f"{iterations} native ROCm matmuls verified; max_abs_difference={max_diff}"


def numeric_accuracy(values: list[float], *, reference: list[float], atol: float = 1e-3,
                     rtol: float = 1e-3) -> bool:
    """Tiny dependency-free numeric oracle used only by unit tests and fixtures."""
    return (len(values) == len(reference) and len(values) > 0 and
            all(math.isfinite(a) and math.isfinite(b) and
                abs(a - b) <= atol + rtol * abs(b) for a, b in zip(values, reference)))
