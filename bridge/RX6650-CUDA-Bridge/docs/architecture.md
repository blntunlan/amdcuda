# Design and verification — RX 6650 XT only

## 0. Scope
Windows native, RDNA2 Navi 23 `gfx1032`, 8 GB VRAM. Personal use only. No cross-GPU generalization.

## 1. Native AMD path (preferred for source-available model code)
Target: ROCm/HIP/PyTorch backend compiled for gfx1032. Verify exact distribution and library compatibility. Check GPU identity and run deterministic CPU-vs-GPU numerical tests. Do not treat a successful `torch.cuda.is_available()` as a complete validation.

## 2. CUDA compatibility path
Target: ZLUDA for unmodified CUDA applications (driver/runtime subset). Test under a per-app launcher, with per-app logs and isolated dependencies. Patch only the needed behavior for a specific real workload. Avoid broad undefined 'CUDA fully supported' claims.

## 3. Acceptance gates
1. Real RX 6650 XT identified by OS and runtime.
2. Runtime and kernels launch with no CPU fallback.
3. CPU/GPU numerical output agrees within documented tolerance.
4. v0.5 tracks PyTorch allocated VRAM under a controlled budget (≤6 GiB configured, default 512 MiB), preserves free-memory headroom, and reports reserved memory. Physical driver allocations are not globally constrained.
5. Repeated execution and application-specific feature tests are stable.
6. Optional performance benchmark vs CPU with identical inputs (no fabricated gains).

## 4. Workload priorities
- P0: GEMM/vector operations via native ROCm backend, correct FP32 output.
- P1: GPU texturing/AI material inference using a small tested model.
- P2: AI image-to-3D inference that fits VRAM or uses controlled CPU offload.
- P3: CUDA-only third-party app through ZLUDA when native port is not possible.

## Non-goals
- NVIDIA CUDA driver or CUDA kernel implementation from scratch.
- Universal OptiX/CUDA compatibility.
- Unverified performance promises.
- Automated system-wide driver and DLL modifications.

## v0.5 VRAM plan

- CPU-resident FP32 source/reference/output; only small contiguous tile/microbatch tensors travel to `cuda:0` (ROCm alias).
- `memory_backend.py` plans free-memory headroom before GPU work. It uses a deliberately padded estimate for four tile operands/results plus a fixed margin; **this is not a driver/rocBLAS workspace guarantee**.
- `gpu-vram-budget.py` runs actual tiled batched matrix products and checks output against independent full CPU matmul on every iteration.
- Recovery policy: shrink microbatch first; if already 1, halve the tile until 16. Never retry unrelated kernel/runtime errors. On exhausted fallback, FAIL; on no initial memory, BLOCKED.
- Real PyTorch peak allocated is enforced. PyTorch peak reserved is reported but may exceed this allocation budget because of caching; OS and third-party GPU memory need separate profiling on the physical PC.


## v0.6 PBR material architecture

`height_at()` is a deterministic 2D periodic analytical function, shared between Python `math` CPU reference and PyTorch vector expression. A toroidal (wraparound) central-difference normal derivative generates normalized tangent-space normals. CPU and HIP compute the same scalar map formulas, then quantize to PNG pixels. Periodicity is mathematical at the UV edges (derivatives are periodic); the boundary pixel values are not necessarily identical, which is correct for adjacent texels of a continuous repeating texture.

The v0.6 runtime has two paths:

1. `generate-material.py`: functional dependency-free CPU reference; can create and roundtrip 7 PNGs now. The reference produces output even when Windows HIP is unavailable. This path does not use CUDA, HIP or the GPU.
2. `gpu-pbr.py`: separately gated Windows gfx1032-only native ROCm PyTorch computations; not enabled as an automatic fallback. Pure CPU reference comparison plus WMI/RX identity is mandatory. `torch.cuda` is PyTorch's normal AMD ROCm device API, not a CUDA-emulation proof. Results only become verified after a real local Windows GPU run.

Neither path performs Blender mesh baking, texture-to-material neural inference, physics-based BRDF measurement, nor automatic material assignment. The generated patterns are suitable for testing map plumbing; visual style production requires further work.
