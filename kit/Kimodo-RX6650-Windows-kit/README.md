# Kimodo / RX 6650 XT — Windows experimental bring-up kit

This is **not an NVIDIA CUDA implementation** or a tested AMD Kimodo port. It is a narrowly scoped installer, diagnostic, and CLI launcher for the **RX 6650 XT (gfx1032)** on **native Windows**. The official Kimodo source and model weights are downloaded directly from their upstream repositories by the user's PC; **this ZIP does not contain them**.

## What we verified

- Repository source: https://github.com/nv-tlabs/kimodo
- Kimodo selects `torch.cuda` if available. ROCm-enabled PyTorch exposes the same public `torch.cuda` API.
- Official `TEXT_ENCODER_DEVICE=cpu` lowers *GPU* use to under ~3 GB, but LLM2Vec uses Llama 3 8B and can exceed a 16 GB RAM PC.
- TheRock lists `gfx1032` for its Windows builds. AMD's standard Windows HIP SDK does **not** officially support RX 6650 XT. This experimental distinction matters.
- `--no-postprocess` skips the optional native C++ MotionCorrection stage, allowing a build-free initial test.

## Prerequisites

1. Windows 10/11, updated Radeon Adrenalin drivers. Install Python **3.11 x64** and Git for Windows; ensure `py -3.11` works in PowerShell. Python 3.11 is a starting point only; actual AMD wheels may require another interpreter version.
2. Free disk space for Python, ROCm/PyTorch, multiple model downloads, and caches (plan for **tens of GB**).
3. Hugging Face account with access granted to **meta-llama/Meta-Llama-3-8B-Instruct**. Accept the model license on Hugging Face and create a personal access token; **never share your token or include it in reports**.
4. Because the system has 16 GB RAM, local text encoding may fail. Use `-TextEncoderUrl` to connect an existing permitted text-encoder service, or explicitly opt into low-memory local test `-AllowLowRAM` understanding possible OOM or heavy pagefile swapping.

## Commands (PowerShell, inside this kit directory)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Mode gpu
powershell -NoProfile -ExecutionPolicy Bypass -File .\doctor.ps1
```

If the experimental AMD torch install fails, stop and inspect the error. Do **not** install NVIDIA CUDA wheels or pretend that CPU fallback is GPU acceleration. To test the official CPU-only Kimodo pipeline (very slow and likely RAM-limited), create a different workspace:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Mode cpu -Workspace "$HOME\kimodo-cpu"
```

Hugging Face authentication (don't paste token into logs):

```powershell
& "$HOME\kimodo-rx6650\.venv\Scripts\hf.exe" auth login
```

After granting Llama access and a GPU `PASS` in `doctor.ps1`:

```powershell
# Recommended if you have an independently running Kimodo-compatible text encoder API:
powershell -NoProfile -ExecutionPolicy Bypass -File .\generate.ps1 -TextEncoderUrl "http://127.0.0.1:9550/"

# Riskier: full text encoder on your own 16 GB RAM:
powershell -NoProfile -ExecutionPolicy Bypass -File .\generate.ps1 -AllowLowRAM
```

The command outputs `first_motion.npz` and `first_motion.bvh` under `~/kimodo-rx6650/kimodo/output/` on success. The sample uses Kimodo-SOMA-RP-v1.2, 3 sec, 20 diffusion steps, 1 sample, no motion postprocessing; quality may be lower than defaults. The repo downloads the checkpoint on demand.

**Optional demo:** run `install.ps1 -Mode gpu -WithDemo`; this fetches extra demo packages, but motion postprocessing is still excluded. The demo has **not** been made reliable for this initial port; CLI BVH is the acceptance target.

## Strict pass conditions

- `backend_probe.json` `status=PASS` AND `gpu_compute=PASS`, correct device string, real GPU matmul numerically matches CPU.
- `generate.ps1` produces both NPZ and BVH with nonzero sizes, without `--no-postprocess` being removed.
- Loading/export into Blender or Unreal is a separate acceptance test; neither is claimed tested here.
- If GPU is `BLOCKED`, AMD Kimodo inference is **not verified** even if CPU tests pass.

## Roadmap

0. Environment & gated model access (login manually, no secrets in reports).
1. TheRock gfx1032 PyTorch real tensor test (doctor).
2. Kimodo CLI import & dependency check.
3. CPU local or external text encoder, generate 1 sec motion.
4. 3 sec and then longer motions, monitor VRAM/RAM and repeatability.
5. BVH → Blender/Unreal retargeting and foot correction if C++ extension can be built.
6. Optional interactive demo and constrained motion UI; only after CLI verified.

Sources: [Kimodo](https://github.com/nv-tlabs/kimodo), [Kimodo installation](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html), [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), [AMD Windows support table](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html).
