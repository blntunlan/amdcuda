"""Pure verification rules for the RX 6650 XT single-card test harness.

Statuses are PASS, FAIL or BLOCKED; BLOCKED is never reported as PASS.
No global GPU state is changed by this module.
"""
from __future__ import annotations

import re
import math
from typing import Any

from native_backend import check_native_device, evaluate_native_math
from primitives_backend import evaluate_primitives
from memory_backend import evaluate_vram_budget

TARGET_PATTERN = re.compile(r"\bRX\s*6650\s*XT\b", re.IGNORECASE)


def validate_profile(profile: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if profile.get("windows_only") is not True:
        problems.append("Only Windows target is allowed")
    if profile.get("llvm_target") != "gfx1032":
        problems.append("Unexpected LLVM target")
    if profile.get("vram_gb_nominal") != 8:
        problems.append("Unexpected VRAM target")
    if profile.get("device_allowlist") != ["RX 6650 XT"]:
        problems.append("Target allowlist must contain exactly RX 6650 XT")
    expected = {
        "no_driver_modification",
        "no_global_cuda_dll_overrides",
        "no_automatic_binary_downloads",
        "no_auto_hsa_override",
    }
    for field in expected:
        if profile.get("safety", {}).get(field) is not True:
            problems.append(f"Required safety setting is false: {field}")
    return problems


def evaluate_diagnostic(diagnostic: dict[str, Any]) -> tuple[str, str]:
    """Require the correct Windows machine with an independently named card."""
    if not isinstance(diagnostic, dict):
        return "FAIL", "Invalid diagnostic JSON"
    gpus = diagnostic.get("gpus") or []
    if not isinstance(gpus, list):
        return "FAIL", "GPU list must be an array"
    names = [str(gpu.get("name", "")) for gpu in gpus if isinstance(gpu, dict)]
    matched = [name for name in names if TARGET_PATTERN.search(name)]
    if not matched:
        return "BLOCKED", "Windows did not independently detect RX 6650 XT"
    if len(matched) != 1:
        return "FAIL", "Multiple RX 6650 XT adapters found; selection must be explicit"
    # Avoid an accidental CUDA implementation running on a second NVIDIA card.
    if any("nvidia" in name.casefold() for name in names):
        return "BLOCKED", "Another NVIDIA GPU detected; cannot attribute GPU work to RX 6650 XT"
    if diagnostic.get("target_detected") is not True:
        return "FAIL", "Diagnostic target_detected flag disagrees with the adapter list"
    return "PASS", matched[0]


def evaluate_smoke_result(returncode: int, payload: dict[str, Any] | None) -> tuple[str, str]:
    """Classify actual native ROCm matmul evidence, never legacy/CPU-only JSON."""
    if not isinstance(payload, dict):
        return "FAIL", "GPU smoke-test did not emit valid JSON"
    if payload.get("stage") in ("platform", "import", "device", "identity"):
        return "BLOCKED", str(payload.get("error") or payload.get("detail") or payload["stage"])
    if payload.get("stage") == "matmul":
        if returncode != 0:
            return "FAIL", str(payload.get("error") or f"GPU matmul returned {returncode}")
        return evaluate_native_math(payload)
    return "FAIL", str(payload.get("error") or "GPU execution or numerical comparison failed")


def evaluate_primitive_result(returncode: int, payload: dict[str, Any] | None) -> tuple[str, str]:
    """Classify native GPU primitives without treating a missing backend as success."""
    if not isinstance(payload, dict):
        return "FAIL", "Primitive test did not emit valid JSON"
    if payload.get("stage") in ("platform", "import", "identity"):
        return "BLOCKED", str(payload.get("error") or payload["stage"])
    if payload.get("stage") == "primitives":
        if returncode != 0:
            return "FAIL", str(payload.get("error") or f"GPU primitives returned {returncode}")
        return evaluate_primitives(payload)
    return "FAIL", str(payload.get("error") or "GPU primitive suite failed")



def evaluate_vram_result(returncode: int, payload: dict[str, Any] | None) -> tuple[str, str]:
    """Memory test can be BLOCKED for missing headroom, but OOM exhaustion FAILS."""
    if not isinstance(payload, dict):
        return "FAIL", "Memory suite did not return valid JSON"
    if payload.get("stage") in ("platform", "import", "identity", "memory-unavailable"):
        return "BLOCKED", str(payload.get("error") or payload["stage"])
    if payload.get("stage") == "vram-budget":
        if returncode != 0:
            return "FAIL", str(payload.get("error") or f"Memory suite returned {returncode}")
        return evaluate_vram_budget(payload)
    return "FAIL", str(payload.get("error") or "GPU memory suite was unsuccessful")


def overall_status(stages: list[dict[str, Any]]) -> str:
    if any(stage.get("status") == "FAIL" for stage in stages):
        return "FAIL"
    if any(stage.get("status") == "BLOCKED" for stage in stages):
        return "BLOCKED"
    return "PASS"


def evaluate_pbr_cpu_result(returncode: int, payload: dict[str, Any] | None) -> tuple[str, str]:
    """Check an actually generated/re-imported CPU sample, not a synthetic GPU result."""
    if not isinstance(payload, dict) or payload.get('stage') != 'pbr-cpu' or returncode != 0:
        return 'FAIL', str((payload or {}).get('error') or 'PBR CPU export failed')
    if payload.get('ok') is not True or payload.get('roundtrip_verified') is not True or payload.get('orm_verified') is not True:
        return 'FAIL', 'PBR integrity/roundtrip evidence is missing'
    if (payload.get('gpu_compute_confirmed') is not False or
        type(payload.get('map_count')) is not int or payload['map_count'] != 7 or
        type(payload.get('width')) is not int or payload['width'] < 8 or
        type(payload.get('height')) is not int or payload['height'] < 8):
        return 'FAIL', 'Invalid PBR CPU report shape or GPU claim'
    if set(payload.get('maps') or []) != {'BaseColor','Normal','Roughness','Metallic','AO','ORM','Height'}:
        return 'FAIL', 'PBR maps incomplete'
    norm = payload.get('normal_max_length_error')
    if type(norm) not in (int, float) or not math.isfinite(norm) or not 0 <= norm <= .015:
        return 'FAIL', 'PBR normal vectors invalid'
    return 'PASS', '7 PBR maps exported/reimported; normal and ORM checks passed (CPU only)'


def evaluate_pbr_gpu_result(returncode: int, payload: dict[str, Any] | None) -> tuple[str, str]:
    """Fail-closed RX 6650 XT native GPU material test report acceptance."""
    if not isinstance(payload, dict):
        return 'FAIL', 'Missing GPU PBR result'
    if payload.get('stage') in ('platform','import','identity','memory-unavailable'):
        return 'BLOCKED', str(payload.get('error') or payload['stage'])
    if payload.get('stage') != 'gpu-pbr' or returncode != 0:
        return 'FAIL', str(payload.get('error') or 'GPU PBR execution failed')
    status, detail = check_native_device(payload)
    if status != 'PASS':
        return status, detail
    if payload.get('ok') is not True or payload.get('gpu_compute_confirmed') is not True:
        return 'FAIL', 'No verified GPU texture generation'
    if type(payload.get('iterations_completed')) is not int or payload['iterations_completed'] < 3:
        return 'FAIL', 'Insufficient GPU PBR repeats'
    if type(payload.get('gpu_tensor_map_count')) is not int or payload['gpu_tensor_map_count'] < 7:
        return 'FAIL', 'GPU tensors missing'
    if payload.get('roundtrip_verified') is not True or payload.get('orm_verified') is not True:
        return 'FAIL', 'GPU PBR maps failed PNG/ORM roundtrip'
    if type(payload.get('map_count')) is not int or payload['map_count'] != 7:
        return 'FAIL', 'Incomplete GPU PBR map set'
    if type(payload.get('mismatch_count')) is not int or payload['mismatch_count'] != 0:
        return 'FAIL', 'GPU textures disagree with CPU oracle'
    delta = payload.get('max_channel_delta')
    if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0 or delta > 3:
        return 'FAIL', 'GPU/CPU per-channel delta is too large'
    norm = payload.get('normal_max_length_error')
    if type(norm) not in (int, float) or not math.isfinite(norm) or norm < 0 or norm > .015:
        return 'FAIL', 'GPU normal is not normalized'
    peak = payload.get('peak_torch_allocated_mib')
    if type(peak) not in (int, float) or not math.isfinite(peak) or peak <= 0 or peak > 512:
        return 'FAIL', 'Invalid or excessive tracked GPU allocation'
    return 'PASS', f"{payload['iterations_completed']} RX 6650 XT GPU PBR runs agree with CPU; max delta={delta}"
