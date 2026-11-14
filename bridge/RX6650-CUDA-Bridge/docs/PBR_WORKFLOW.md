# v0.6 PBR Workflow — Blender / Unreal Engine / Godot

All commands run from the extracted `RX6650-CUDA-Bridge` project root. They require ordinary Python 3.12+ and **do not** need AMD drivers for CPU generation.

## Generate test materials

```powershell
python .\scripts\generate-material.py --preset basalt --size 256 --seed 6650 --normal-format directx --out .\reports\basalt-ue
python .\scripts\generate-material.py --preset worn_metal --size 256 --seed 6650 --normal-format opengl --out .\reports\metal-godot
```

Each folder includes `BaseColor.png`, `Normal.png`, `Roughness.png`, `Metallic.png`, `AO.png`, `ORM.png`, `Height.png`, and `material.json`. The generator uses analytically repeating height, not an AI neural net, and has no mesh-based baking. It can be used offline on a non-AMD machine as a reproducible reference.

## Unreal Engine

- Use **DirectX normal format** (`--normal-format directx`): green channel points Y−. In Unreal's material graph, BaseColor RGB → Base Color, Normal RGB → Normal, ORM R → Ambient Occlusion, ORM G → Roughness, ORM B → Metallic.
- Use sRGB **enabled** for BaseColor only; **disabled** for Normal/ORM/Roughness/Metallic/AO/Height. Let Unreal recognize the normal texture's compression as `Normalmap`; use mask compression for ORM when appropriate.
- Sample all three texture maps with the same UV and tiling. Height is optional (requires an appropriate displacement/parallax workflow and is not directly connected as a base PBR pin).
- These are procedural sample textures; evaluate visuals and adjust material scale, texture filtering and normal strength in engine.

## Godot 4

- Use **OpenGL normal format** (`--normal-format opengl`): green channel Y+.
- In a StandardMaterial3D, assign BaseColor, normal, roughness, metallic and optional AO in their respective texture slots. If using custom shaders, ORM R=AO, G=Roughness, B=Metallic.
- BaseColor is color/sRGB. All other maps are linear data textures. Use matching UV scale and `Repeat` texture sampling for the seamless material.
- Godot normal map settings can flip the green channel; do **not** enable an additional Y-flip if using the correct `opengl` export.

## Blender

- BaseColor PNG via Image Texture in `sRGB` → Principled BSDF `Base Color`.
- `Normal`, `Roughness`, `Metallic` and `AO` via Image Texture set to `Non-Color`. Use **OpenGL normal** with a Normal Map node → Principled `Normal`.
- Feed Roughness → Roughness, Metallic → Metallic. AO may be multiplied into BaseColor for a particular stylization, but physical workflows typically keep AO separate or apply it in the correct renderer pipeline.
- Height via Bump node or true displacement as appropriate; it is optional. Do not mix DirectX green orientation into Blender's normal-node input without flipping Y.

## Import your own grayscale PNG height map

```powershell
python -m pip install Pillow
python .\scripts\generate-material.py --height-input .\height-source.png --preset basalt --normal-format directx --out .\reports\height-import
```

If Pillow is unavailable, only the toolkit's own grayscale 8-bit PNG encoding is accepted. Imported heights produce normals, AO, color, roughness and metallic via the same formulas. This is an experimental height-based PBR conversion, not reconstruction of physically measured surfaces.

## Hardware acceptance (Windows RX 6650 XT only)

```powershell
python .\scripts\run-tests.py --tier all --gpu-python .\.venv-gfx1032\Scripts\python.exe
```

`reports/verification.json` must show `pbr-cpu=PASS` **and** `gpu-pbr=PASS` (plus all earlier Windows/HIP gates) before v0.6 GPU support can be claimed. The GPU test uses native ROCm PyTorch, not ZLUDA/CUDA binary compatibility. Hardware unavailable means `BLOCKED`, **not** `PASS`.
