# RX6650-CUDA-Bridge — Personal `gfx1032` toolkit (v0.6.1)

**Only target:** Sapphire Radeon RX 6650 XT (RDNA2 / Navi 23 / `gfx1032`), 8 GB VRAM, native Windows.

This is a diagnostics + verification **prototype**, **not** a CUDA driver or a functioning compatibility implementation. It does **not** automatically install AMD/NVIDIA drivers, download binaries, edit the registry, or modify the system CUDA DLLs.

## v0.6.1 addition: 4096×4096 PBR texture stress test (CPU only)

The previously strict 1 megapixel reference is unchanged. A new memory-bounded NumPy renderer generates **real 4K textures** as 7 PNG maps, with a separate independent image verifier. `docs/PBR_4K_TEST.md` has Windows commands and results. Tested procedural examples are included in `reports/materials/`.

```powershell
python -m pip install -r .\requirements-4k.txt
python .\scripts\generate-material-4k.py --size 4096 --preset basalt --normal-format directx --out .\reports\materials\basalt_directx_4096
python .\scripts\verify-4k-assets.py .\reports\materials\basalt_directx_4096 --report .\reports\4k-verification.json
```

**Scope:** v0.6.1 CPU output validated at 4K; native AMD GPU and CUDA compatibility **not validated**. This is not an AI photo-texture generator. Normal-map gradient strength is corrected at high resolution relative to a 256px baseline. The code requires NumPy and Pillow, but does not download or modify GPU drivers.

## Project files

- `docs/ROADMAP.md` — incremental roadmap, acceptance and test policy for every version.
- `config/gfx1032.json` — strict RX 6650 XT profile and safe-operation flags.
- `scripts/verification.py` — pure target identity and test status classification helpers.
- `scripts/native_backend.py` — fail-closed gfx1032 / RX 6650 XT native runtime and numerical validation.
- `scripts/run-tests.py` — automated **PASS / FAIL / BLOCKED** gate, saved JSON report.
- `tests/test_verification.py` — runnable platform-independent unit tests.
- `.github/workflows/unit-tests.yml` — runs the offline unit suite automatically on GitHub push/PR once added to a repository; no GPU claims.
- `scripts/diagnose.ps1` — read-only Windows hardware/driver/HIP/ZLUDA inventory.
- `scripts/new-native-env.ps1` — opt-in isolated Python 3.12 environment creation (no downloads).
- `docs/INSTALL_WINDOWS.md` — Windows setup and manually installed TheRock ROCm PyTorch instructions.
- `scripts/verify-release.py` — SHA-256 packaged-file integrity check.
- `scripts/smoke-test-torch.py` — native HIP backend check, explicit GPU identity, 20 GPU FP32 matrix multiplications with CPU reference and metrics.
- `scripts/gpu-primitives.py` — second GPU acceptance test: GPU allocation, 4 MiB host-to-device transfer and verified readback, 20 vector additions, 20 matrix multiplications; CPU oracle and memory/timing metrics.
- `scripts/primitives_backend.py` — independent, offline-checkable contract for all five primitives. Incomplete/mismatched/CPU-only GPU evidence is rejected.
- `scripts/memory_backend.py` — RX 6650 XT-only VRAM planner, tile/microbatch scheduler, OOM recovery and strict result validation (stdlib-only).
- `scripts/gpu-vram-budget.py` — live Windows HIP GPU test: 20 streamed/batched FP32 matmuls compared to the full CPU oracle, with observed PyTorch memory peaks and safe oversize rejection (no forced hardware OOM).
- `scripts/gpu_primitives_shim.py` — import-safe ROCm runtime inventory shared by the v0.5 script.
- `tests/test_memory_backend.py` — no-GPU tests for numeric edge tiles, budget clamps, safe refusal, simulated OOM recovery and negative GPU evidence.
- `tests/test_primitives_backend.py` — positive and negative acceptance tests including NaN, wrong device, wrong operation and corrupted result.
- `scripts/pbr_backend.py`, `scripts/generate-material.py` — CPU PBR reference, PNG writer/reader and validation.
- `scripts/pbr_gpu_backend.py`, `scripts/gpu-pbr.py` — optional native AMD HIP PyTorch texture generation and CPU parity test (requires real hardware).
- `docs/PBR_WORKFLOW.md` — material hookup and map conventions.
- `scripts/launch-zluda.ps1` — experimental per-app launcher, **not yet hardware verified**.
- `docs/architecture.md` — execution strategies and non-goals.
- `examples/` — build-host reference logs; they are **not** results from the user's GPU.

## v0.6 PBR textures (working CPU reference, GPU validation pending)

Generate **seven real PNG maps** from deterministic periodic procedural height:
`BaseColor` (sRGB), `Normal`, `Roughness`, `Metallic`, `AO`, `ORM` (R=AO, G=Roughness, B=Metallic), and `Height` (all linear/non-color except BaseColor).
Normal maps are normalized (up to 8-bit quantization), PNG files are decoded again after writing, and ORM channels are compared byte-for-byte. Presets: `basalt` and `worn_metal`; tileable math; exact seed reproducibility. This CPU reference is useful independently of the experimental GPU backend.

```powershell
# Unreal: DirectX Y- tangent normal; packed ORM for UE (AO, roughness, metallic)
python .\scripts\generate-material.py --preset basalt --size 256 --seed 6650 --normal-format directx --out .\reports\basalt-unreal

# Godot: OpenGL Y+ tangent normal; separate roughness and metallic maps available
python .\scripts\generate-material.py --preset worn_metal --size 256 --seed 6650 --normal-format opengl --out .\reports\metal-godot

# Optional height-map import; Pillow provides broad PNG interoperability
python .\scripts\generate-material.py --height-input .\reports\basalt-unreal\Height.png --preset basalt --normal-format directx --out .\reports\height-reimport
```

The output folder contains PNGs and `material.json` (conventions, parameters, validation results). With no Pillow installed, the bundled strict PNG reader supports **only 8-bit filter-0 grayscale/RGB PNGs created by this toolkit**; for ordinary externally exported PNG inputs, `python -m pip install Pillow` in the interpreter that runs the tool. Image resolution is deliberately bounded to 1,048,576 pixels. Source textures are synthetic procedural patterns, **not AI model output or photogrammetry**. Geometry displacement, actual physically calibrated materials, node graphs, baking from meshes, roughness measurement and commercial-quality visual fidelity are outside v0.6.

Optional *native AMD GPU computation* is implemented separately in `scripts/gpu-pbr.py` and `scripts/pbr_gpu_backend.py`. It runs only on **native ROCm PyTorch** after Windows WMI + HIP `RX 6650 XT` checks, executes 3 texture-generation passes and compares **every byte** to the pure CPU oracle (≤3 levels error allowed). It rejects missing/wrong GPU, excess tracked PyTorch VRAM and invalid PNG roundtrips. All kernel execution on the actual 6650 XT remains `BLOCKED` until tested there. This is **not** a CUDA compatibility implementation.

See `docs/PBR_WORKFLOW.md` for material wiring in Blender, Godot and Unreal.

## Every change: run two gates

```powershell
# Safe, CPU-only software checks (all platforms)
python .\scripts\run-tests.py --tier unit

# Physical machine verification; only on a Windows PC with RX 6650 XT
python .\scripts\run-tests.py --tier all
```

Results go to `reports/verification.json`. A unit-test success **never** means GPU execution is verified. v0.5 adds the `gpu-vram-budget` hardware stage after v0.4 primitives; v0.6 adds a mandatory real CPU PBR export stage and a native `gpu-pbr` stage. On a machine without the target Windows GPU the hardware gate reports **BLOCKED** (exit code 3), not PASS. Never call a version GPU-tested until both the device identity and GPU math checks pass.

### Standalone diagnostics

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\diagnose.ps1
```

Produces `reports/diagnostic.json`. No administrator privileges or driver modifications needed.

### Native GPU testing (v0.3–v0.5 hardware acceptance pending)

A compatible *separately installed* gfx1032 ROCm-enabled PyTorch distribution is necessary. See `docs/INSTALL_WINDOWS.md` for the isolated environment instructions, then run:

```powershell
python .\scripts\run-tests.py --tier all --gpu-python .\.venv-gfx1032\Scripts\python.exe
```

PyTorch exposes the `torch.cuda` API for ROCm, too. Its name alone does not imply a NVIDIA GPU. We require a HIP-enabled PyTorch build and exactly one runtime RX 6650 XT GPU, then check independent Windows identity, explicit CUDA-device allocation (ROCm uses this device name), 20 real operations and numerical agreement in the v0.3 gate, followed by the v0.4 primitive suite, the v0.5 streaming memory gate and v0.6 GPU-PBR gate. The default memory policy targets 512 MiB of live PyTorch allocations and preserves 1 GiB of reported free VRAM; the hard configurable tracked-tensor cap is 6 GiB. The OS driver and kernel workspaces have additional physical VRAM use that these counters do not fully bound. ZLUDA is deliberately excluded from this native backend gate; its per-app experiments remain a separate, later milestone.

## Known limits

- AMD mainstream Windows HIP SDK and TheRock have **different support status** for `gfx1032`; check the specific versions before installing. TheRock Windows support is still evolving.
- 8 GB is nominal VRAM, not available AI model capacity. The v0.5 6 GiB maximum is a **tracked PyTorch allocation budget**, not a physical VRAM guarantee or full-system limit. Library workspaces, allocator cache, OS and other processes may use additional memory. Default active cap is 512 MiB with a 1 GiB free-memory reserve; untrusted free/total readings cause an error or BLOCKED status.
- `launch-zluda.ps1` remains a prototype; its compatibility has not been checked against a specific release and target application.
- No full CUDA, OptiX or cuDNN compatibility claims, nor automatic `HSA_OVERRIDE_GFX_VERSION` spoofing.
- **Local v0.6 offline software tests pass. Actual RX 6650 XT GPU hardware validation remains BLOCKED until run on the Windows target. Run the bundled unit test command to see the current count.**

Sources: [TheRock supported GPUs](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md), [TheRock Windows status](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md), [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), [AMD HIP system requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html), [ZLUDA](https://github.com/vosen/ZLUDA).
