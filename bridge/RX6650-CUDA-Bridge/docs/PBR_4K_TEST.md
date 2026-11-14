# v0.6.1 — 4096×4096 PBR material production and verification

## Scope and status

This is a **real CPU render / PNG asset validation**. It is **not** a GPU/RX 6650 XT test or a claim that CUDA/ZLUDA is functional. The previous v0.6 `generate-material.py` intentionally limits images to 1,048,576 pixels. v0.6.1 adds a separate NumPy CPU strip renderer, preserving existing tests and safety checks.

- Texture dimensions: 4096×4096, 8-bit RGB or L PNG, 7 maps per material.
- Presets: `basalt` with `directx` normal map, and `worn_metal` with `opengl` normal map.
- PBR equations: same as v0.6, with high-resolution normal-gradient normalization relative to a 256-pixel baseline to prevent visually flat 4K normals.
- Memory efficiency: 256-row tiles (16 passes per output), one halo row at each tile edge, independent zlib-compressed PNG scanline streams.
- Verification: full source-versus-decoded pixel SHA256, file SHA256, PNG dimensions/mode, exact ORM packing, normal length, edge tilability (max across opposing boundary values is 2/255), complete output metadata.
- Color: BaseColor in sRGB, others linear data. ORM: R=AO, G=Roughness, B=Metallic.
- Quality limitation: generated procedural patterns are smooth, periodic analytic textures, **not** photorealistic microdetail or image-to-material AI reconstruction. 4K resolution does not create extra physical detail by itself.

## Windows steps (Python 3.12+)

From extracted project root:

```powershell
python -m pip install -r .\requirements-4k.txt
python .\scripts\generate-material-4k.py --size 4096 --tile-rows 256 --preset basalt --normal-format directx --out .\reports\materials\basalt_directx_4096
python .\scripts\generate-material-4k.py --size 4096 --tile-rows 256 --preset worn_metal --normal-format opengl --out .\reports\materials\worn_metal_opengl_4096
python .\scripts\verify-4k-assets.py .\reports\materials\basalt_directx_4096 .\reports\materials\worn_metal_opengl_4096 --report .\reports\4k-verification.json
python .\scripts\run-tests.py --tier unit
```

For actual hardware testing of ROCm, follow `docs/INSTALL_WINDOWS.md`. No actual GPU kernel is executed by this 4K renderer. Hardware remains `BLOCKED` until verified on the target physical Windows RX 6650 XT.

## Test environment record (2026-11-13)

Build host: Linux, no RX 6650 XT; 2× 4096×4096 sets generated. 14 PNG files passed independent decoded checksum / color-mode / normal / ORM / seam verification. Each build had 16 256px tiles. Approximate generation+PNG readback per material: 3.7–4.4 seconds (CPU tool self-reported); full shell processes approximately 4.8–5.3 seconds; measured peak resident set roughly 290 MiB. These are **not** measurements from the user's PC.

See `reports/4k-verification.json` and each `reports/materials/*_4k_run.json` for per-material results.
