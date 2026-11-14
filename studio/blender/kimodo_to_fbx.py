"""Kimodo BVH -> skinned SOMA character (Blender) -> FBX/GLB.

blender -b --factory-startup -P kimodo_to_fbx.py -- --bvh X.bvh --npz X.npz --skin skin_standard.npz
        --out X.fbx [--glb X.glb] [--blend X.blend] [--no-mesh] [--verify]

Steps:
 1. Import the BVH (source armature). Pose is exact relative to the NPZ.
 2. Build a new "SOMA" armature whose rest pose = skin bind_rig_transform (via K_j correction).
 3. Copy Transforms (world) from source -> SOMA, bake, delete the source.
 4. Bind the SOMA mesh with skin weights (bind pose = rest pose, so LBS is exact).
 5. Optional: compare vertex by vertex against Kimodo's NumPy LBS.
"""
import argparse
import json
import sys

import bpy
import numpy as np
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--bvh", required=True)
ap.add_argument("--npz", required=True)
ap.add_argument("--skin", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--glb")
ap.add_argument("--blend")
ap.add_argument("--name", default="SOMA")
ap.add_argument("--no-mesh", action="store_true")
ap.add_argument("--verify", action="store_true")
args = ap.parse_args(argv)

# Kimodo Y-up -> Blender Z-up: (x, y, z) -> (x, -z, y)
C = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def to_np(m):
    return np.array([list(r) for r in m], dtype=np.float64)


def rigid(m):
    """Remove scale/shear from the rotation part (SVD)."""
    u, _, vt = np.linalg.svd(m[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    out = m.copy()
    out[:3, :3] = r
    return out


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_anim.bvh(filepath=args.bvh, global_scale=0.01, rotate_mode="NATIVE", update_scene_fps=True,
                        axis_forward="-Z", axis_up="Y")
src = next(o for o in bpy.data.objects if o.type == "ARMATURE")
src.name = "BVH_src"

motion = np.load(args.npz)
skin = np.load(args.skin)
names = [str(n) for n in skin["rig_joint_names"]]
J = len(names)
G = motion["global_rot_mats"].astype(np.float64)  # [F, J, 3, 3]
Pj = motion["posed_joints"].astype(np.float64)  # [F, J, 3]
F = Pj.shape[0]
scene = bpy.context.scene
scene.frame_start, scene.frame_end = 1, F  # the BVH importer always places frames at 1..F


def kimodo_world(f, j):
    a = np.eye(4)
    a[:3, :3] = G[f, j]
    a[:3, 3] = Pj[f, j]
    return C @ a


def blender_world(j):
    return to_np(src.matrix_world @ src.pose.bones[names[j]].matrix)


# K_j = inv(A_j) @ P_j  (should be constant over time)
scene.frame_set(1)
K = [np.linalg.inv(kimodo_world(0, j)) @ blender_world(j) for j in range(J)]
scene.frame_set(F)
k_drift = max(np.abs(np.linalg.inv(kimodo_world(F - 1, j)) @ blender_world(j) - K[j]).max() for j in range(J))
print(f"[k6] K_j drift over time: {k_drift:.2e}")

bind = skin["bind_rig_transform"].astype(np.float64)
rest = [rigid(C @ bind[j] @ K[j]) for j in range(J)]

# Hierarchy: parent of each joint from rig_joint_connections (parent, child)
parent = {int(c): int(p) for p, c in skin["rig_joint_connections"]}

arm_data = bpy.data.armatures.new(args.name)
arm = bpy.data.objects.new(args.name, arm_data)
scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
ebones = {}
for j, n in enumerate(names):
    eb = arm_data.edit_bones.new(n)
    eb.head = (0, 0, 0)
    eb.tail = (0, 0.05, 0)
    eb.matrix = Matrix(rest[j].tolist())
    # Bone length: distance to the first child, otherwise 3 cm
    kids = [c for c, p in parent.items() if p == j]
    eb.length = max(0.02, min(0.3, np.linalg.norm(rest[kids[0]][:3, 3] - rest[j][:3, 3]))) if kids else 0.03
    eb.matrix = Matrix(rest[j].tolist())  # length can change the matrix; set it again
    ebones[j] = eb
for c, p in parent.items():
    ebones[c].parent = ebones[p]
bpy.ops.object.mode_set(mode="POSE")
for j, n in enumerate(names):
    pb = arm.pose.bones[n]
    pb.rotation_mode = "QUATERNION"
    con = pb.constraints.new("COPY_TRANSFORMS")
    con.target = src
    con.subtarget = n
bpy.ops.object.mode_set(mode="OBJECT")

# Bake (visual transform), then remove constraints
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.pose.select_all(action="SELECT")
bpy.ops.nla.bake(frame_start=1, frame_end=F, only_selected=True, visual_keying=True,
                 clear_constraints=True, bake_types={"POSE"})
bpy.ops.object.mode_set(mode="OBJECT")
arm.animation_data.action.name = f"{args.name}_anim"
bpy.data.objects.remove(src, do_unlink=True)
for a in list(bpy.data.actions):
    if a.users == 0:
        bpy.data.actions.remove(a)


def action_fcurves(action):
    if hasattr(action, "layers") and len(action.layers):
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    yield from cb.fcurves
    elif hasattr(action, "fcurves"):
        yield from action.fcurves


# Move to frame 0 = t 0 (otherwise FBX/glTF shift by 1/30 s)
for fc in action_fcurves(arm.animation_data.action):
    for kp in fc.keyframe_points:
        kp.co.x -= 1
        kp.handle_left.x -= 1
        kp.handle_right.x -= 1
    fc.update()
scene.frame_start, scene.frame_end = 0, F - 1

mesh_obj = None
if not args.no_mesh:
    verts = (C[:3, :3] @ skin["bind_vertices"].astype(np.float64).T).T
    faces = skin["faces"].tolist()
    me = bpy.data.meshes.new(args.name + "_body")
    me.from_pydata(verts.tolist(), [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    mesh_obj = bpy.data.objects.new(args.name + "_body", me)
    scene.collection.objects.link(mesh_obj)
    groups = [mesh_obj.vertex_groups.new(name=n) for n in names]
    idx, w = skin["lbs_indices"], skin["lbs_weights"]
    for v in range(idx.shape[0]):
        for k in range(idx.shape[1]):
            if w[v, k] > 0:
                groups[idx[v, k]].add([v], float(w[v, k]), "ADD")
    mesh_obj.parent = arm
    mod = mesh_obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    mat = bpy.data.materials.new("SOMA_grey")
    mat.diffuse_color = (0.72, 0.75, 0.8, 1)
    me.materials.append(mat)

report = {"frames": F, "fps": scene.render.fps, "k_drift": k_drift}

if args.verify and mesh_obj is not None:
    # Kimodo LBS (NumPy) vs Blender evaluated mesh
    bv = skin["bind_vertices"].astype(np.float64)
    bv_h = np.concatenate([bv, np.ones((bv.shape[0], 1))], 1)
    inv_bind = np.linalg.inv(bind)
    errs = {}
    for f in sorted({0, F // 3, 2 * F // 3, F - 1}):
        aff = np.stack([kimodo_world(f, j) for j in range(J)]) @ np.stack([C] * J) @ inv_bind  # Blender world
        # bind verts are in Kimodo coords -> aff takes kimodo-bind -> blender world; use C on bind side:
        aff = np.stack([kimodo_world(f, j) @ inv_bind[j] for j in range(J)])
        ref = np.einsum("vkab,vb->vka", aff[idx][..., :3, :], bv_h)
        ref = (ref * w[..., None]).sum(1)
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = mesh_obj.evaluated_get(dg)
        em = ev.to_mesh()
        got = np.array([(mesh_obj.matrix_world @ v.co)[:] for v in em.vertices])
        ev.to_mesh_clear()
        errs[f] = float(np.linalg.norm(got - ref, axis=1).max())
    report["vertex_err_max_m"] = errs
    print("[k6] vertex max error per frame (m):", errs)

scene.frame_set(0)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.fbx(filepath=args.out, use_selection=False, add_leaf_bones=False, bake_anim=True,
                         bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                         bake_anim_simplify_factor=0.0, apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL",
                         armature_nodetype="NULL", primary_bone_axis="Y", secondary_bone_axis="X")
if args.glb:
    bpy.ops.export_scene.gltf(filepath=args.glb, export_format="GLB", export_animations=True)
if args.blend:
    bpy.ops.wm.save_as_mainfile(filepath=args.blend)
print("[k6] REPORT " + json.dumps(report))
