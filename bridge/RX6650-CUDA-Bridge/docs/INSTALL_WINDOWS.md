# RX 6650 XT / gfx1032 — Windows native ROCm setup (v0.5)

This walkthrough is **opt-in** and designed for a single local computer. It does not modify or replace Windows GPU drivers, system CUDA DLLs, or registry settings. It does **not** create a working CUDA-compatible driver. The first objective is to execute a real ROCm PyTorch GPU operation.

TheRock lists `gfx1032` as sanity-tested/release-ready on Windows, but its Windows compute stack remains evolving. Do not assume every PyTorch package, kernel, or model works on an RX 6650 XT.

## Step 1 — Verify Windows GPU identity

From this repository's root, open Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\diagnose.ps1
```

Inspect `reports/diagnostic.json`. You should see the `RX 6650 XT` device in `gpus`. WMI adapter RAM may be truncated; do not use this number as the VRAM truth.

## Step 2 — Create an isolated Python 3.12 environment

Install Python 3.12 with the Windows `py` launcher if it isn't installed already. Then:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\new-native-env.ps1
```

This creates `.venv-gfx1032` under the repository, refuses to overwrite an existing environment, and **does not install anything from the internet**. If you prefer, create the environment directly using `py -3.12 -m venv .venv-gfx1032`.

## Step 3 — Install gfx1032-enabled ROCm PyTorch only inside the venv

After reviewing the upstream release guide, manually execute:

```powershell
.\.venv-gfx1032\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1032]"
```

The index and `[device-gfx1032]` extra are from the upstream TheRock multi-arch install model. You may need to choose a compatible released version or the nightly channel depending on upstream availability. If installing fails, **keep the error** instead of installing CPU-only PyTorch from another index or using an automatic GPU architecture override. This command intentionally installs only `torch` and its declared dependencies, not `torchvision` or other AI frameworks.

Before recording a passing baseline, pin the packages used on your machine:

```powershell
.\.venv-gfx1032\Scripts\python.exe -m pip freeze > reports\rocm-packages-local.txt
```

## Step 4 — Execute the two-tier verification

```powershell
python .\scripts\run-tests.py --tier unit
python .\scripts\run-tests.py --tier all --gpu-python .\.venv-gfx1032\Scripts\python.exe
```

A successful v0.5 hardware run must show `PASS` on all **six** stages: target profile, software tests, WMI identity, real GPU matmul and the primitive suite (`gpu-primitives`). The GPU test checks 20 consecutive matmuls and validates the final FP32 output against a CPU result, as well as recording PyTorch/HIP versions, GPU architecture, duration and PyTorch-allocated memory. Software PASS alone is **not** hardware validation. An unavailable backend produces BLOCKED and a nonzero exit status.

The result is in `reports/verification.json`. Share that file and `reports/diagnostic.json` when working through compatibility issues; review them first because Windows PNP device IDs can reveal hardware identifiers. Do not upload the package freeze if private package sources/tokens are present.

## Interpretation and recovery

| Outcome | Meaning / next action |
| --- | --- |
| `PASS` on all | Native ROCm PyTorch executed a small GPU compute operation; general CUDA interoperability is **not** established |
| `BLOCKED` at WMI | Wrong GPU, multiple ambiguous GPUs, or no RX 6650 XT was detected |
| `BLOCKED` at matmul identity | CPU-only PyTorch, other GPU, or unsupported ROCm backend |
| `FAIL` at matmul | Kernel crashed, GPU fallback was detected, or CPU/GPU values did not agree |
| `FAIL` at unit tests | Stop; fix regression before attempting more hardware tests |

Never copy compatibility DLLs to Windows system directories or replace `rocblas` libraries globally. Any experimental backend modifications must be confined to the local venv or per-application folder. Each new feature gets its own tests and a failure report.

## References (checked 2026-11-13)

- https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md
- https://github.com/ROCm/TheRock/blob/main/RELEASES.md
- https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md

## v0.4 – GPU primitives acceptance details

`python scripts/run-tests.py --tier all --gpu-python .\.venv-gfx1032\Scripts\python.exe` now launches **two separate real GPU programs**, in order. `gpu-primitives.py` only starts after the v0.3 `gpu-matmul` stage has passed.

The primitive runner uses `torch.cuda` (PyTorch uses the same API for native HIP), and does **not** use NVIDIA CUDA or ZLUDA. Workloads are intentionally small and fixed: GPU zero-allocation with synchronized sum, 4 MiB CPU→GPU transfer, GPU→CPU round-trip equality, 20 262,144-element FP32 vector additions, and 20 256×256 FP32 matmuls. It compares every returned element with the CPU reference using elementwise `torch.isclose` (`rtol/atol=1e-5/1e-5` for vectors, `1e-3/1e-3` for matmul). All operations are GPU-device checked and synchronized.

Detailed per-operation results are stored under the `gpu-primitives.metrics` object inside `reports/verification.json`, with duration, mismatch count, maximum absolute difference, and available PyTorch memory usage. Memory readings are diagnostic, **not** proof of 8 GiB physical VRAM availability; a controlled 6 GiB budget and OOM strategy belong to v0.5.

For direct manual testing on the verified Windows computer, you may run:

```powershell
.\.venv-gfx1032\Scripts\python.exe .\scripts\gpu-primitives.py --iterations 20 --vector-size 262144 --matrix-size 256 --transfer-mib 4
```

Direct invocation skips the **independent Windows WMI identity gate**, so only the full `run-tests.py --tier all` report should be used as hardware acceptance evidence. Do not claim a fully CUDA-compatible API/driver from a successful ROCm primitive test.

## v0.5 — tracked VRAM budget & streaming work

`gpu-vram-budget.py` is launched **only after** all previous Windows identity, 20 matmul and primitive stages pass. Its defaults are intentionally light: 512 MiB PyTorch-allocated budget, 1 GiB minimum reported free-VRAM reserve, 128x128 matrices, tiles 64, batch of 4, microbatch of 2, and 20 passes. It computes full CPU references and re-checks every iteration. No globally installed drivers or DLLs are changed.

Customize the direct standalone test only if necessary:

```powershell
.\.venv-gfx1032\Scripts\python.exe .\scripts\gpu-vram-budget.py --budget-mib 512 --reserve-mib 1024 --tile-size 64 --microbatch-size 2 --batch-size 4 --matrix-size 128 --iterations 20
```

`--budget-mib` is bounded to **64..6144**; higher values fail **before** importing PyTorch. This is a bound on **tracked PyTorch allocation**, not all graphics driver/rocBLAS physical use. `--reserve-mib` preserves free memory for desktop/apps; lowering it is not recommended. A genuine OOM causes a retry with a smaller microbatch and then tile size. Exhaustion yields FAIL. Insufficient safe free memory yields BLOCKED. The safe oversized-request check does not intentionally trigger OOM on the physical card.

Run `python scripts/run-tests.py --tier all --gpu-python .\.venv-gfx1032\Scripts\python.exe` for an acceptance report combining all six gates. Upload `reports/verification.json` and optionally `reports/diagnostic.json` after inspecting device identifiers. GPU execution is unverified until the actual RX 6650 XT reports PASS.
