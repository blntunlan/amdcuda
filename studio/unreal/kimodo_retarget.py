"""Kimodo SOMA FBX -> UE import -> IK Rig (SOMA + Manny) -> IK Retargeter -> retargeted Manny animation.

uh py kimodo_retarget.py -Arg <config.json>
config: {"fbx": "...fbx", "name": "walk_wave", "dest": "/Game/Kimodo",
         "target_mesh": "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple"}
"""
import json

import unreal

with open(UH_ARG, encoding="utf-8") as fh:  # noqa: F821 (UH_ARG is provided by uh)
    cfg = json.load(fh)  # noqa: F821 (UH_ARG is provided by uh)
dest = cfg.get("dest", "/Game/Kimodo")
name = cfg["name"]
log = []
at = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
SRC, TGT = unreal.RetargetSourceOrTarget.SOURCE, unreal.RetargetSourceOrTarget.TARGET


def find(folder, cls):
    out = []
    for p in eal.list_assets(folder, recursive=True, include_folder=False):
        a = eal.load_asset(p)
        if isinstance(a, cls):
            out.append(a)
    return out


# ---- 1. Import (Interchange) ----
src_folder = f"{dest}/{name}"
if not find(src_folder, unreal.SkeletalMesh) or cfg.get("reimport"):
    t = unreal.AssetImportTask()
    t.filename = cfg["fbx"]
    t.destination_path = src_folder
    t.automated = True
    t.replace_existing = True
    t.save = True
    at.import_asset_tasks([t])
    log.append({"imported": [str(p) for p in t.get_editor_property("imported_object_paths")]})
src_mesh = find(src_folder, unreal.SkeletalMesh)[0]
src_anims = find(src_folder, unreal.AnimSequence)
bounds = src_mesh.get_bounds()
log.append({"src_mesh": src_mesh.get_path_name(), "anims": [a.get_path_name() for a in src_anims],
            "bounds_cm": [round(bounds.box_extent.x * 2, 1), round(bounds.box_extent.y * 2, 1), round(bounds.box_extent.z * 2, 1)],
            "anim_len_s": [round(a.get_play_length(), 3) for a in src_anims]})

# ---- 2-3. IK Rigs + Retargeter (SOMA skeleton is always the same -> reuse; rebuild on request) ----
rig_dir = f"{dest}/Rigs"
rtg_path = f"{rig_dir}/RTG_SOMA_to_Manny"
tgt_mesh = eal.load_asset(cfg.get("target_mesh", "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple"))
have_all = all(eal.does_asset_exist(f"{rig_dir}/{n}") for n in ("IK_SOMA", "IK_Manny_Kimodo", "RTG_SOMA_to_Manny"))
if have_all and not cfg.get("rebuild_rigs"):
    rtg = eal.load_asset(rtg_path)
    log.append({"rigs": "reused"})
else:
    # dependents first: RTG references the rigs, otherwise the deletion is silently refused and create_asset opens a modal
    for n in ("RTG_SOMA_to_Manny", "IK_SOMA", "IK_Manny_Kimodo"):
        if eal.does_asset_exist(f"{rig_dir}/{n}") and not eal.delete_asset(f"{rig_dir}/{n}"):
            raise RuntimeError(f"could not delete {rig_dir}/{n} (still referenced?)")
    def make_rig(asset_name, mesh):
        rig = at.create_asset(asset_name, f"{dest}/Rigs", unreal.IKRigDefinition, unreal.IKRigDefinitionFactory())
        c = unreal.IKRigController.get_controller(rig)
        c.set_skeletal_mesh(mesh)
        return rig, c


    tgt_rig, tc = make_rig("IK_Manny_Kimodo", tgt_mesh)
    auto_t = tc.apply_auto_generated_retarget_definition()
    def chains(c):
        return {str(ch.chain_name): (str(c.get_retarget_chain_start_bone(ch.chain_name)), str(c.get_retarget_chain_end_bone(ch.chain_name)))
                for ch in c.get_retarget_chains()}


    tgt_chains = chains(tc)
    log.append({"manny_auto": auto_t, "manny_root": str(tc.get_retarget_root()), "manny_chains": tgt_chains})

    src_rig, sc = make_rig("IK_SOMA", src_mesh)
    auto_s = sc.apply_auto_generated_retarget_definition()
    if not auto_s:
        # Manual chains with the same names as Manny's so Exact mapping works.
        sc.set_retarget_root("Hips")
        want = {
            "Spine": ("Spine1", "Chest"), "Neck": ("Neck1", "Neck2"), "Head": ("Head", "Head"),
            "LeftClavicle": ("LeftShoulder", "LeftShoulder"), "RightClavicle": ("RightShoulder", "RightShoulder"),
            "LeftArm": ("LeftArm", "LeftHand"), "RightArm": ("RightArm", "RightHand"),
            "LeftLeg": ("LeftLeg", "LeftToeBase"), "RightLeg": ("RightLeg", "RightToeBase"),
            "LeftThumb": ("LeftHandThumb1", "LeftHandThumb3"), "RightThumb": ("RightHandThumb1", "RightHandThumb3"),
        }
        for side in ("Left", "Right"):
            for f_ue, f_soma in (("Index", "Index"), ("Middle", "Middle"), ("Ring", "Ring"), ("Pinky", "Pinky")):
                want[f"{side}{f_ue}Metacarpal"] = (f"{side}Hand{f_soma}1", f"{side}Hand{f_soma}1")
                want[f"{side}{f_ue}"] = (f"{side}Hand{f_soma}2", f"{side}Hand{f_soma}4")
        # Take only the chains Manny has; match its leg chain split (e.g. separate Foot chain).
        for cname, (s, e) in want.items():
            if cname in tgt_chains:
                sc.add_retarget_chain(cname, s, e, "")
        if "LeftFoot" in tgt_chains:
            for side in ("Left", "Right"):
                sc.set_retarget_chain_end_bone(f"{side}Leg", f"{side}Foot")
                sc.add_retarget_chain(f"{side}Foot", f"{side}Foot", f"{side}ToeBase", "")
    if auto_s:
        # Fill gaps in the auto definition: spine up to Chest, neck, metacarpal (SOMA X1 = metacarpal).
        have = {str(ch.chain_name) for ch in sc.get_retarget_chains()}
        fixes = {"Spine": ("Spine1", "Chest"), "Neck": ("Neck1", "Neck2")}
        for side in ("Left", "Right"):
            for f in ("Index", "Middle", "Ring", "Pinky"):
                fixes[f"{side}{f}"] = (f"{side}Hand{f}2", f"{side}Hand{f}4")
                fixes[f"{side}{f}Metacarpal"] = (f"{side}Hand{f}1", f"{side}Hand{f}1")
        for cname, (s, e) in fixes.items():
            if cname not in tgt_chains:
                continue
            if cname in have:
                sc.set_retarget_chain_start_bone(cname, s)
                sc.set_retarget_chain_end_bone(cname, e)
            else:
                sc.add_retarget_chain(cname, s, e, "")
    src_chains = chains(sc)
    log.append({"soma_auto": auto_s, "soma_root": str(sc.get_retarget_root()), "soma_chains": src_chains,
                "unmapped_manny_chains": sorted(set(tgt_chains) - set(src_chains))})

    # ---- 3. Retargeter ----
    rtg = at.create_asset("RTG_SOMA_to_Manny", f"{dest}/Rigs", unreal.IKRetargeter, unreal.IKRetargetFactory())
    rc = unreal.IKRetargeterController.get_controller(rtg)
    rc.set_ik_rig(SRC, src_rig)
    rc.set_ik_rig(TGT, tgt_rig)
    rc.set_preview_mesh(SRC, src_mesh)
    rc.set_preview_mesh(TGT, tgt_mesh)
    rc.add_default_ops()
    rc.assign_ik_rig_to_all_ops(SRC, src_rig)
    rc.assign_ik_rig_to_all_ops(TGT, tgt_rig)
    rc.auto_map_chains(unreal.AutoMapChainType.EXACT, True)
    rc.auto_align_all_bones(TGT)
    log.append({"ops": [str(rc.get_op_name(i)) for i in range(rc.get_num_retarget_ops())]})


# ---- 3b. Manny foot fix (applied every run; does not rely on the asset being saved) ----
# SOMA's foot angle copied FK onto Manny leaves Manny's ball ~2 cm in the floor when the foot is flat.
# Measured (tune_feet.py, moonwalk): local Z −12° → flat ball 0.8–0.9 cm = Manny's own reference 0.8 cm.
yaw = float(cfg.get("foot_offset_yaw", -12.0))
rc_fix = unreal.IKRetargeterController.get_controller(rtg)
for b in ("foot_l", "foot_r"):
    rc_fix.set_rotation_offset_for_retarget_pose_bone(b, unreal.Rotator(0.0, 0.0, yaw).quaternion(), unreal.RetargetSourceOrTarget.TARGET)
rtg.modify()
log.append({"foot_offset_yaw": yaw})

# ---- 4. Batch retarget ----
inputs = unreal.IKRetargetBatchOperationInputs()
inputs.assets_to_retarget = [unreal.AssetRegistryHelpers.create_asset_data(a) for a in src_anims]
inputs.source_mesh = src_mesh
inputs.target_mesh = tgt_mesh
inputs.ik_retarget_asset = rtg
inputs.target_path = f"{dest}/Manny"
inputs.suffix = "_Manny"
inputs.overwrite_existing_files = True
inputs.include_referenced_assets = False
new = unreal.IKRetargetBatchOperation.run_batch_retarget(inputs)
new_paths = [str(a.package_name) for a in new]
anims_out = [eal.load_asset(p) for p in new_paths]
log.append({"retargeted": new_paths,
            "len_s": [round(a.get_play_length(), 3) for a in anims_out if isinstance(a, unreal.AnimSequence)]})
eal.save_directory(dest, only_if_is_dirty=True, recursive=True)
eal.save_asset(rtg.get_path_name(), only_if_is_dirty=False)
result = log
