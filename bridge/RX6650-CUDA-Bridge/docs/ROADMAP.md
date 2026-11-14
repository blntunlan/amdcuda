# RX6650-CUDA-Bridge — Development roadmap & test policy

Date: 2026-11-13 · Device scope: **Sapphire RX 6650 XT, 8 GB, Navi 23, `gfx1032`, native Windows**.

## Delivery principle

Each change is one small, reversible increment. For every increment:

1. Record hypothesis, expected outcome and affected components.
2. Run the **unit gate** (`python scripts/run-tests.py --tier unit`). Fix failures before proceeding.
3. On the actual Windows RX 6650 XT machine run the **hardware gate** (`python scripts/run-tests.py --tier all` (both baseline GPU matmul **and** primitive suite must PASS)).
4. Compare GPU outputs to known CPU/reference outputs, repeat, and record timings, VRAM and versions.
5. Produce a JSON report under `reports/`; do not use the word *verified* when a stage says `BLOCKED`.
6. If a new feature fails, revert or repair only that increment and rerun regression tests. Publish a release only with completed required gates. No automatic patching of Windows drivers, system CUDA DLLs or registry.

### Test statuses
- **PASS:** actually executed and met measurable acceptance criteria.
- **FAIL:** executed and failed, or test infrastructure produced invalid/uninterpretable data.
- **BLOCKED:** required GPU, compatible software or Windows platform missing. **Never silently promote to PASS.**
- **NOT STARTED:** feature implementation and its tests not yet created.

## Milestones / acceptance matrix

| Milestone | What we will build | Required tests, with explicit gates | Current |
|---|---|---|---|
| **v0.2 – Test harness** | Deterministic unit gate, hardware runner, JSON results, target allowlist | All unit tests PASS; Windows GPU test never silently passes on non-target PC; report includes blocked reason | **Software tests PASS; hardware BLOCKED here** |
| **v0.3 – Native compute** | Optional local Python 3.12 venv, TheRock gfx1032 install guide, strict native ROCm runtime identification, repeated FP32 GPU matmul and memory/time metrics | Windows WMI identifies RX 6650 XT; HIP PyTorch selects the same one visible GPU; 20 FP32 GPU matmuls match CPU; no CPU fallback | **Implementation + 47 offline tests PASS; physical RX 6650 XT gate BLOCKED here** |
| **v0.4 – CUDA/HIP primitives** | Tensor allocation, copying, vector add, matrix multiplication; only necessary compatibility functions | FP32 reference within `rtol=1e-3, atol=1e-3` for matmul; vector tolerance `1e-5`; 20 consecutive GPU iterations; H2D/D2H transfer verified by CPU readback; failures produce nonzero exit | **Implemented; offline tests PASS, RX 6650 XT hardware BLOCKED** |
| **v0.5 – VRAM budget** | Configurable tile sizes and microbatches, CPU-resident source/target, OOM recovery, preallocation refusal | ≤6 GiB *tracked* PyTorch allocations, default 512 MiB + 1 GiB free reserve; 20 streamed FP32 runs CPU-checked; oversize refused without intentionally crashing GPU | **Software implemented / unit PASS; physical GPU BLOCKED** |
| **v0.6 – PBR material pipeline** | Two real procedural PBR presets, import height PNG, generate 7 maps, ORM and DirectX/OpenGL normals; optional native ROCm GPU route | PNG readback, valid dimensions/channels/normals/ORM, periodic height, CPU/Torch math oracle; actual GPU 3 repeated runs within ≤3 channel values per pixel and ≤512 MiB tracked allocation | **CPU pipeline + offline tests PASS; native RX 6650 XT GPU gate BLOCKED** |
| **v0.6.1 – 4K PBR stress test** | 4096² memory-bounded CPU strip exporter for two PBR presets | 14 actual 4K PNGs, full decoded checksum, exact ORM/normal/mode/dimensions, seam verification and regression | **CPU 4K outputs PASS; physical GPU BLOCKED** |
| **v0.7 – AI texture** | One small, source-available diffusion/texture model using native AMD path, tiling/offload as needed | Fixed prompt/seed/config, valid outputs, 3 repeated inference runs, peak memory and duration captured; visual quality inspected manually | NOT STARTED |
| **v0.8 – AI image-to-3D** | Pilot one model with manageable 8 GB memory use; Blender export/import | Non-empty geometry, finite vertices/normals, valid UVs, texture integrity, import/export roundtrip; no unbounded VRAM growth | NOT STARTED |
| **v0.9 – Per-app ZLUDA** | Isolated loader + only required CUDA API substitutions for **one identified real application** | Does app really execute GPU kernels on 6650 XT? numerical parity and dependency-specific tests; crashes logged; no system DLL replacement | NOT STARTED |
| **v1.0 – Integrated personal toolkit** | One-command repeatable pipeline, rollback, version pinning and benchmark reports | All required unit + hardware + end-to-end application tests PASS; stable repeat runs; CPU/GPU comparisons documented, no invented performance boost | NOT STARTED |

The throughput/speed targets will be set **after** a reproducible baseline on the actual GPU. We will not impose a synthetic GPU faster-than-CPU gate on tiny workloads, which can be dominated by PCIe / startup overhead.

## Sequencing, constraints and possible fallback decisions

- **Prefer TheRock native ROCm for open-source ML models**; current TheRock list includes Windows `gfx1032` as release-ready, but this is **not identical** to AMD's mainstream HIP SDK Windows device table. TheRock itself cautions about maturity.
- **Only use ZLUDA where the target CUDA-only application requires it.** ZLUDA cannot be assumed to implement all CUDA libraries, especially ML-related or graphics APIs.
- On a genuine backend incompatibility: capture full versions and error logs, try a separately pinned previous backend inside a virtual environment, then assess a targeted patch. **Never blanket-spoof another GPU via an automatic HSA override.**
- If the model exceeds 8 GB, use tiled inference, reduced resolution, quantization or CPU offload only when the specific model supports them and correctness/quality is tested. No universal 8 GB guarantee.
- The existing `launch-zluda.ps1` is a prototype launcher. Its actual CLI compatibility and GPU activity are **not** yet established and must be tested during v0.9.

## Baseline recorded on 2026-11-13

This ChatGPT build environment is Linux and does **not** provide the user's Sapphire RX 6650 XT. We can run Python unit tests, JSON validation, compilation and packaging checks here. In v0.3, 47 offline unit tests passed; v0.4 adds strict acceptance tests for 5 primitives, including simulated negative cases for wrong GPU, CPU PyTorch, CUDA-only PyTorch, multiple GPUs, bad math and legacy JSON results. These unit tests do **not** emulate GPU correctness. We **cannot** honestly claim Windows PowerShell GPU discovery, HIP kernels or ZLUDA execution passed on the user's card.

### Commands

```powershell
# 1. Pure software checks, safe on any machine
python .\scripts\run-tests.py --tier unit

# 2. Only on the actual Windows 6650 XT machine; GPU must be installed/configured
python .\scripts\run-tests.py --tier all --gpu-python .\.venv-gfx1032\Scripts\python.exe

# 3. Inspect generated report
Get-Content .\reports\verification.json
```

Exit codes: `0` PASS, `1` FAIL, `3` BLOCKED. A hardware test can only become PASS on a verified target machine.

## Sources verified 2026-11-13

- https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md
- https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md
- https://github.com/ROCm/TheRock/blob/main/RELEASES.md
- https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html
- https://github.com/vosen/ZLUDA

## Continuous software checks

`.github/workflows/unit-tests.yml` runs the offline unit gate on every **push** and **pull request**, on both Windows and Linux GitHub runners, *if the project is committed to a GitHub repository*. GitHub-hosted runners do **not** contain the user's RX 6650 XT, so the hardware gate remains a separate mandatory local check before promoting GPU-related features.

## v0.5 delivery and acceptance — added 2026-11-13

- **No CUDA DLL modifications** or per-app compatibility claims. This is native ROCm/HIP PyTorch GPU computation.
- Default: `--budget-mib 512 --reserve-mib 1024 --tile-size 64 --batch-size 4 --microbatch-size 2 --matrix-size 128 --iterations 20`.
- GPU inputs and outputs are tiled from CPU memory. This is **streamed compute**, not a general automatic offloader for arbitrary AI models.
- Before any GPU allocation, the plan reads real `torch.cuda.mem_get_info()`, preserves reserve headroom, and rejects unsafe requests. It reduces microbatch first, then tile size on a narrowly classified OOM or observed tracked-budget violation.
- 20 complete streamed FP32 batched matmuls are compared to CPU for every iteration, `rtol=atol=1e-3`, including nonfinite checks.
- Hardware PASS requires Windows independent WMI identification, ROCm `gfx1032`, the v0.3 and v0.4 checks, nonzero live-memory metrics, ≤ configured torch-allocated limit, and 20 complete iterations. `torch.cuda.max_memory_reserved()` and actual total GPU memory are reported separately; driver library allocations are **not covered** by torch-allocated peak.
- The oversized-request test is **preallocation refusal** (configured budget+1 MiB): it never intentionally allocates beyond the safe budget or forces device OOM.
- v0.5 unit tests use simulated failures and a CPU-only edge-tile oracle; these must **not** be mistaken for actual 6650 XT kernel execution.
- GPU results in `reports/verification.json`, stage `gpu-vram-budget`; no hardware-tested release claim until the user's machine returns all six PASS.

```powershell
python .\scripts\run-tests.py --tier all --gpu-python .\.venv-gfx1032\Scripts\python.exe
# optional direct test, bypasses independent Windows WMI gate:
.\.venv-gfx1032\Scripts\python.exe .\scripts\gpu-vram-budget.py --budget-mib 512 --reserve-mib 1024
```

## v0.6 acceptance and truthfulness addendum — 2026-11-13

- Deterministic seed 6650, two material presets, bounded 8–2048 dimension and ≤1,048,576 pixels; no unsafe GPU-sized unbounded request.
- Export and re-import 7 actual PNGs. Use 8-bit RGB BaseColor/Normal/ORM and 8-bit grayscale Roughness/Metallic/AO/Height. `material.json` records channels and color-space assumptions.
- Normal length maximum 8-bit quantization deviation ≤0.015. DirectX Y− and OpenGL Y+ differ solely in encoded green channel. ORM R/G/B must be exactly AO/Roughness/Metallic (byte-for-byte).
- Strict pure-Python CPU reference and negative tests (corrupted PNG, wrong resolution/normal/ORM, invalid seed, size bomb and bad GPU evidence). Optional torch *CPU tensor math* provides cross-implementation numerical reference **without claiming GPU support**.
- GPU PBR test requires independently successful Windows RX 6650 XT WMI check, native ROCm runtime, v0.3/0.4/0.5 tests and CPU PBR gate. Repeats 3 times. Compares all pixels/channels against CPU baseline with tolerance ≤3/255 and zero above-tolerance differences. Reports runtime, actual hip version, and tracked torch peak memory ≤512 MiB; no CPU fallback permitted.
- **This build host has no physical RX 6650 XT. Hardware stage is BLOCKED. CPU reference is implemented but is not a CUDA or ZLUDA implementation.**

```powershell
python .\scripts\run-tests.py --tier unit
python .\scripts\run-tests.py --tier all --gpu-python .\.venv-gfx1032\Scripts\python.exe
python .\scripts\generate-material.py --preset basalt --size 256 --normal-format directx --out .\reports\pbr-basalt
```
