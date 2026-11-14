# Changelog

## v0.6.1 — 2026-11-13

- Added a memory-bounded, 256-row NumPy CPU streaming exporter that renders **4096×4096** BaseColor, Normal, Roughness, Metallic, AO, ORM, Height PNG textures, without changing the strict 1MP reference backend.
- Added high-resolution normal-gradient scaling; matched the old CPU oracle exactly at ≤256px and tested partition-invariant tile edges.
- Rendered 4K basalt/DirectX and worn-metal/OpenGL materials, 14 actual PNG files, all independently decoded, hashed, validated for normal-unit length and exact ORM channel packing. Verified near-seam channel difference ≤1 LSB in both axes.
- Added standalone 4K asset verifier, 4K test documentation and additional unit tests; measured peak RSS about 290 MiB per process on this Linux test environment. **No physical RX 6650 XT GPU was available.**

## v0.6.0 — 2026-11-13

- Created deterministic, cyclically tileable procedural basalt and worn-metal height sources with two different PBR recipes (not AI image generation).
- CPU reference generates BaseColor, Normal, Roughness, Metallic, AO, combined ORM (R=AO/G=Roughness/B=Metallic), and Height, with DirectX Y− / OpenGL Y+ tangent-space normal toggle.
- Implemented bounded 8-bit PNG writer plus strict reader, CRC checks, re-import roundtrip, normalized tangent-space normal validation, byte-perfect ORM correctness, optional Pillow-based external grayscale height PNG import, and metadata sidecar.
- Added native HIP PyTorch GPU texture equations (`pbr_gpu_backend.py`), an opt-in three-run GPU verifier that checks every map byte against CPU (≤3/255 per channel), and ≤512 MiB tracked PyTorch allocation limit after Windows/HIP RX 6650 XT identification.
- Added explicit CPU-PBR and GPU-PBR stages to the JSON test runner, fail-closed report validators, negative tests and test artifacts; GPU test remains BLOCKED on non-target hardware.
- Added material wiring instructions for Unreal Engine (DirectX normals, ORM) and Godot (OpenGL normals, separate maps), plus reproducible procedural sample packages.
- **No RX 6650 XT physical tests run in this environment. This milestone provides a working CPU generator and a candidate native GPU implementation, not CUDA emulation or an AI texture model.**


## v0.5.0 — 2026-11-13

- Added bounded `gfx1032` VRAM policy with 6 GiB ceiling on **tracked PyTorch allocations**, 512 MiB default, and 1 GiB free-memory reserve; no deliberate large GPU allocations.
- Added live free/total GPU memory queries, memory-headroom gating, CPU-resident streaming operands and results, and configurable matrix tiles/microbatches.
- Added GPU batched tiled FP32 matmul acceptance: ≥20 full passes and independent full CPU oracle on **every** pass, with GPU tensor confirmation, numerical tolerance, timing and live PyTorch peak memory readings.
- Added fail-closed OOM/budget retry strategy (shrink microbatch, then tile) and an oversized-request preallocation guard; unrelated runtime errors are never treated as OOM.
- Added strict GPU memory report validator, new `gpu-vram-budget` stage after v0.3/v0.4 gates, and no-GPU tests for edge tiles, planner, simulated OOM and invalid results.
- **No physical RX 6650 XT testing performed in this build container**. Software unit tests must pass; hardware test remains BLOCKED until run on the specified Windows card. This is not a CUDA driver or universal application compatibility layer.

## v0.4.0 — 2026-11-13

- Implemented five native HIP/PyTorch GPU primitive acceptance tests: zero-filled GPU allocation, host-to-device transfer, verified device-to-host readback, FP32 vector addition and FP32 matmul.
- Added full CPU numerical oracles (per-element comparisons), at least 20 consecutive GPU compute iterations, explicit GPU device checks, and synchronized timings.
- Added strict offline-evaluable evidence contract: rejects missing/wrong/duplicate operations, CPU fallback, wrong GPU, nonfinite timings, bad math, insufficient iterations, and incomplete runs.
- Extended `--tier all` to run primitives after existing Windows hardware identity and baseline GPU-matmul gates; skips with `BLOCKED` if a prerequisite is unavailable.
- Added hardware safety bounds on workloads (bounded tensor sizes and transfers), and optional GPU allocation/reservation/free-memory observations.
- Hardware ROCm primitive execution **not yet performed on the user's Windows RX 6650 XT**; all physical compute remains BLOCKED in this build environment.


## v0.3.0 — 2026-11-13

- Added GPU-backend attestation: refuses CPU-only or NVIDIA CUDA PyTorch, a runtime GPU with the wrong name/architecture, and multiple ambiguous runtime GPUs.
- Added explicit 20-iteration native ROCm matmul benchmark (FP32 CPU reference, HIP version, elapsed GPU kernel time, optional PyTorch peak allocation).
- Added a separate, opt-in Windows Python 3.12 environment helper; no automatic binary downloads or global driver changes.
- Added Windows install and recovery guide for TheRock `[device-gfx1032]` Python packages and `--gpu-python` interpreter selection.
- Strengthened hardware PASS classification: legacy/incomplete JSON results cannot be accepted as verified GPU compute.
- Added SHA-256 packaged-file verification, including tests for tampering and directory traversal.
- Verified in this build environment: 47/47 offline tests PASS; native GPU and Windows hardware checks BLOCKED because the target GPU is not present.
- No CUDA API emulation or finished application integration is claimed.

## v0.2.0 — 2026-11-13

- Defined detailed v0.2 → v1.0 milestones, repeatable acceptance criteria and rollback policy.
- Added pure Python gate/status logic: PASS, FAIL and BLOCKED (hardware missing is never PASS).
- Added Windows hardware diagnostics + PyTorch math orchestration with machine-readable JSON report.
- Added 21 unit tests covering target lock, safety controls, mismatched GPUs, numerical failures and CLI.
- Added GitHub Actions example for automated Windows/Linux software tests on push and pull requests.
- Verified locally: 21/21 offline tests PASS; hardware gate correctly BLOCKED on Linux build host; **RX 6650 XT hardware has not been tested**.

## v0.1.0

- Initial graphics card profile, Windows diagnostic, PyTorch GPU smoke test and experimental ZLUDA launcher.
