# blender -b -P probe_bvh.py -- <bvh> <motion.npz> <skin_standard.npz>
import sys
import bpy
import numpy as np

bvh, npz, skin = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_anim.bvh(filepath=bvh, global_scale=0.01, rotate_mode="NATIVE", update_scene_fps=True,
                        update_scene_duration=True, axis_forward="-Z", axis_up="Y")
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
m = np.load(npz)
s = np.load(skin)
names = [str(n) for n in s["rig_joint_names"]]
bones = [b.name for b in arm.data.bones]
print("bones", len(bones), "rig", len(names), "missing:", [n for n in names if n not in bones][:10])
print("frames", bpy.context.scene.frame_start, bpy.context.scene.frame_end, "fps", bpy.context.scene.render.fps)

def y_up(v):  # Blender Z-up -> Kimodo Y-up
    return np.array([v[0], v[2], -v[1]])

bind_t = s["bind_rig_transform"][:, :3, 3]
rest_err = []
for j, n in enumerate(names):
    if n in arm.data.bones:
        rest_err.append(np.linalg.norm(y_up(arm.matrix_world @ arm.data.bones[n].head_local) - bind_t[j]))
print("rest vs bind joint pos err (m): max %.4f mean %.4f" % (max(rest_err), np.mean(rest_err)))
print("bind rot identity? max |R-I| =", float(np.abs(s["bind_rig_transform"][:, :3, :3] - np.eye(3)).max()))

pj = m["posed_joints"]
for f in [0, pj.shape[0] // 2, pj.shape[0] - 1]:
    bpy.context.scene.frame_set(f + 1)
    errs = [np.linalg.norm(y_up(arm.matrix_world @ arm.pose.bones[n].head) - pj[f, j])
            for j, n in enumerate(names) if n in arm.pose.bones]
    print("frame %d pose joint err (m): max %.4f" % (f, max(errs)))
