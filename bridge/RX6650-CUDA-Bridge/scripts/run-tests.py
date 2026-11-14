"""Run unit tests and opt-in Windows RX6650XT hardware verification.

Examples:
    python scripts/run-tests.py --tier unit
    python scripts/run-tests.py --tier all
    python scripts/run-tests.py --tier all --json-out reports/custom.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from verification import evaluate_diagnostic, evaluate_smoke_result, evaluate_primitive_result, evaluate_vram_result, evaluate_pbr_cpu_result, evaluate_pbr_gpu_result, overall_status, validate_profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "verification.json"


def call(argv: list[str], timeout: int) -> tuple[int, str, str]:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def load_json_output(raw: str) -> dict | None:
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else None
    except (ValueError, TypeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="RX 6650 XT verification gate")
    parser.add_argument("--tier", choices=("unit", "all"), default="unit")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--gpu-python", type=Path, default=None,
                        help="Isolated ROCm PyTorch interpreter (default: .venv-gfx1032/Scripts/python.exe if present)")
    args = parser.parse_args()

    stages = []
    t0 = time.perf_counter()
    profile_path = ROOT / "config" / "gfx1032.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        problems = validate_profile(profile)
        stages.append({"name": "target-profile", "status": "FAIL" if problems else "PASS",
                       "detail": "; ".join(problems) if problems else "Strict gfx1032 target and safety rules verified"})
    except (OSError, ValueError) as exc:
        stages.append({"name": "target-profile", "status": "FAIL", "detail": str(exc)})

    try:
        rc, output, err = call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], timeout=60)
        stages.append({"name": "unit-tests", "status": "PASS" if rc == 0 else "FAIL",
                       "detail": (err or output)[-6000:], "exit_code": rc})
    except (OSError, subprocess.TimeoutExpired) as exc:
        stages.append({"name": "unit-tests", "status": "FAIL", "detail": str(exc)})

    # Real procedural PBR CPU smoke (PNG bytes are written and decoded again).
    try:
        rc, stdout, stderr = call([sys.executable, str(ROOT / "scripts" / "generate-material.py"),
                                   "--size", "64", "--out", str(ROOT / "reports" / "pbr-cpu-smoke")], timeout=60)
        payload = load_json_output(stdout)
        status, detail = evaluate_pbr_cpu_result(rc, payload)
        stages.append({"name": "pbr-cpu", "status": status, "detail": detail,
                       "exit_code": rc, "metrics": payload or {"stderr": stderr[-2000:]}})
    except (OSError, subprocess.TimeoutExpired) as exc:
        stages.append({"name": "pbr-cpu", "status": "FAIL", "detail": str(exc)})

    if args.tier == "all":
        if platform.system() != "Windows":
            stages.append({"name": "windows-diagnostic", "status": "BLOCKED",
                           "detail": "Actual Windows RX 6650 XT machine required; host is " + platform.system()})
            stages.append({"name": "gpu-matmul", "status": "BLOCKED",
                           "detail": "Diagnostic did not run on the target Windows GPU"})
            stages.append({"name": "gpu-primitives", "status": "BLOCKED",
                           "detail": "Diagnostic did not run on the target Windows GPU"})
            stages.append({"name": "gpu-vram-budget", "status": "BLOCKED",
                           "detail": "Diagnostic did not run on the target Windows GPU"})
        else:
            shell = shutil.which("powershell") or shutil.which("pwsh")
            if not shell:
                stages.append({"name": "windows-diagnostic", "status": "BLOCKED",
                               "detail": "PowerShell is missing"})
                stages.append({"name": "gpu-matmul", "status": "BLOCKED",
                               "detail": "Diagnostic unavailable"})
                stages.append({"name": "gpu-primitives", "status": "BLOCKED", "detail": "Diagnostic unavailable"})
                stages.append({"name": "gpu-vram-budget", "status": "BLOCKED", "detail": "Diagnostic unavailable"})
            else:
                diag_path = ROOT / "reports" / "diagnostic.json"
                # Clear stale diagnostic so it cannot accidentally pass the new test.
                diag_path.unlink(missing_ok=True)
                try:
                    rc, stdout, stderr = call([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                               str(ROOT / "scripts" / "diagnose.ps1"), "-OutFile", str(diag_path)], timeout=45)
                    if not diag_path.is_file():
                        status, detail = "FAIL", f"Diagnostic returned {rc} without report: {stderr or stdout}"
                    else:
                        data = json.loads(diag_path.read_text(encoding="utf-8-sig"))
                        status, detail = evaluate_diagnostic(data)
                        if rc not in (0, 2):
                            status, detail = "FAIL", f"Diagnostic exited unexpectedly with {rc}: {stderr or stdout}"
                    stages.append({"name": "windows-diagnostic", "status": status, "detail": detail})
                except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    stages.append({"name": "windows-diagnostic", "status": "FAIL", "detail": str(exc)})

                if stages[-1]["status"] != "PASS":
                    stages.append({"name": "gpu-matmul", "status": "BLOCKED", "detail": "GPU identity gate not passed"})
                    stages.append({"name": "gpu-primitives", "status": "BLOCKED", "detail": "GPU identity gate not passed"})
                    stages.append({"name": "gpu-vram-budget", "status": "BLOCKED", "detail": "GPU identity gate not passed"})
                else:
                    try:
                        default_env = ROOT / ".venv-gfx1032" / "Scripts" / "python.exe"
                        gpu_python = (str(args.gpu_python.resolve()) if args.gpu_python else
                                      str(default_env) if default_env.exists() else sys.executable)
                        rc, stdout, stderr = call([gpu_python, str(ROOT / "scripts" / "smoke-test-torch.py")], timeout=180)
                        payload = load_json_output(stdout)
                        status, detail = evaluate_smoke_result(rc, payload)
                        stages.append({"name": "gpu-matmul", "status": status, "detail": detail,
                                       "exit_code": rc, "metrics": payload or {"stderr": stderr[-2000:]}})
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        stages.append({"name": "gpu-matmul", "status": "FAIL", "detail": str(exc)})

                    if stages[-1]["status"] != "PASS":
                        stages.append({"name": "gpu-primitives", "status": "BLOCKED",
                                       "detail": "Baseline GPU matmul not verified"})
                    else:
                        try:
                            rc, stdout, stderr = call(
                                [gpu_python, str(ROOT / "scripts" / "gpu-primitives.py")], timeout=180)
                            payload = load_json_output(stdout)
                            status, detail = evaluate_primitive_result(rc, payload)
                            stages.append({"name": "gpu-primitives", "status": status, "detail": detail,
                                           "exit_code": rc, "metrics": payload or {"stderr": stderr[-2000:]}})
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            stages.append({"name": "gpu-primitives", "status": "FAIL", "detail": str(exc)})
                    if stages[-1]["status"] != "PASS":
                        stages.append({"name": "gpu-vram-budget", "status": "BLOCKED",
                                       "detail": "Prerequisite native GPU primitive suite did not PASS"})
                    else:
                        try:
                            rc, stdout, stderr = call(
                                [gpu_python, str(ROOT / "scripts" / "gpu-vram-budget.py")], timeout=300)
                            payload = load_json_output(stdout)
                            status, detail = evaluate_vram_result(rc, payload)
                            stages.append({"name": "gpu-vram-budget", "status": status, "detail": detail,
                                           "exit_code": rc, "metrics": payload or {"stderr": stderr[-2000:]}})
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            stages.append({"name": "gpu-vram-budget", "status": "FAIL", "detail": str(exc)})

    if args.tier == "all":
        preconditions = (platform.system() == "Windows" and
                         all(any(s["name"] == stage and s["status"] == "PASS" for s in stages)
                             for stage in ("target-profile", "unit-tests", "pbr-cpu", "windows-diagnostic",
                                           "gpu-matmul", "gpu-primitives", "gpu-vram-budget")))
        if not preconditions:
            stages.append({"name": "gpu-pbr", "status": "BLOCKED",
                           "detail": "Native gfx1032 GPU prerequisite and PBR CPU suite must PASS first"})
        else:
            try:
                default_env = ROOT / ".venv-gfx1032" / "Scripts" / "python.exe"
                gpu_python = (str(args.gpu_python.resolve()) if args.gpu_python else
                              str(default_env) if default_env.exists() else sys.executable)
                rc, stdout, stderr = call([gpu_python, str(ROOT / "scripts" / "gpu-pbr.py"),
                                           "--out", str(ROOT / "reports" / "pbr-gpu")], timeout=180)
                payload = load_json_output(stdout)
                status, detail = evaluate_pbr_gpu_result(rc, payload)
                stages.append({"name": "gpu-pbr", "status": status, "detail": detail,
                               "exit_code": rc, "metrics": payload or {"stderr": stderr[-2000:]}})
            except (OSError, subprocess.TimeoutExpired) as exc:
                stages.append({"name": "gpu-pbr", "status": "FAIL", "detail": str(exc)})

    result = {
        "project": "RX6650-CUDA-Bridge", "profile_version": "0.6.1", "tier": args.tier,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host_os": platform.system(), "target_gpu": "RX 6650 XT", "llvm_target": "gfx1032",
        "overall": overall_status(stages), "hardware_verified": (args.tier == "all" and platform.system() == "Windows" and
                              all(any(s["name"] == name and s["status"] == "PASS" for s in stages)
                                  for name in ("target-profile", "unit-tests", "pbr-cpu", "windows-diagnostic", "gpu-matmul", "gpu-primitives", "gpu-vram-budget", "gpu-pbr"))),
        "elapsed_seconds": round(time.perf_counter() - t0, 3), "stages": stages,
    }
    report_path = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for stage in stages:
        detail = stage['detail']
        if stage['name'] == 'unit-tests':
            detail = 'see JSON report for full test listing'
        print(f"[{stage['status']}] {stage['name']}: {detail}")
    print(f"RESULT: {result['overall']}; hardware verified: {result['hardware_verified']}; report: {report_path}")
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 3}[result["overall"]]


if __name__ == "__main__":
    raise SystemExit(main())
