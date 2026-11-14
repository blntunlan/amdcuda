# blender -b -P verify_fbx.py -- <file.fbx|glb> <motion.npz> [render.png]
# Re-imports the exported file and compares joint world positions against the NPZ.
import sys

import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
path, npz = a[0], a[1]
png = a[2] if len(a) > 2 else None
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 30
if path.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)
else:
    bpy.ops.import_scene.gltf(filepath=path)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
m = np.load(npz)
pj = m["posed_joints"]
F = pj.shape[0]
names = [b.name for b in arm.pose.bones]
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimodo_names import NAMES  # noqa: E402  (77 joint order)

act = arm.animation_data.action if arm.animation_data else None
print("action", act.name if act else None, "range", tuple(act.frame_range) if act else None)
worst = 0
start = round(act.frame_range[0]) if act else 0
for f in [0, F // 2, F - 1]:
    bpy.context.scene.frame_set(start + f)
    e = []
    for j, n in enumerate(NAMES):
        if n in arm.pose.bones:
            p = arm.matrix_world @ arm.pose.bones[n].head
            e.append(np.linalg.norm(np.array([p.x, p.z, -p.y]) - pj[f, j]))
    worst = max(worst, max(e))
    print(f"frame {f}: joint max err {max(e):.5f} m  ({len(e)} joints)")
print("VERDICT", "PASS" if worst < 1e-3 else "FAIL", worst)

if png:
    sc = bpy.context.scene
    try:
        sc.render.engine = "BLENDER_WORKBENCH"
    except TypeError:
        pass
    sc.render.resolution_x, sc.render.resolution_y = 960, 540
    hip = arm.pose.bones["Hips"]
    bpy.context.scene.frame_set(start + F // 2)
    c = arm.matrix_world @ hip.head
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    cam.location = (c.x + 2.6, c.y - 3.4, 1.4)
    d = (c - cam.location)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    bpy.ops.mesh.primitive_plane_add(size=30)
    sc.render.filepath = png
    bpy.ops.render.render(write_still=True)
