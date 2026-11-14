"""Find foot_l/foot_r retarget-pose rotation offsets that make Manny's flat foot sit like its own reference.
cfg: {"src_mesh","src_anim","tgt_mesh","candidates":[[roll,pitch,yaw],...], "apply": {"foot_l":[r,p,y],"foot_r":[r,p,y]} (optional)}"""
import json

import unreal

with open(UH_ARG, encoding="utf-8") as fh:  # noqa: F821
    cfg = json.load(fh)
eal = unreal.EditorAssetLibrary
P = unreal.AnimPoseExtensions
W = unreal.AnimPoseSpaces.WORLD
opts = unreal.AnimPoseEvaluationOptions()
T = unreal.RetargetSourceOrTarget.TARGET
rtg = unreal.load_asset("/Game/Kimodo/Rigs/RTG_SOMA_to_Manny")
rc = unreal.IKRetargeterController.get_controller(rtg)
src_mesh, src_anim = unreal.load_asset(cfg["src_mesh"]), unreal.load_asset(cfg["src_anim"])
tgt_mesh = unreal.load_asset(cfg["tgt_mesh"])
TMP = "/Game/Kimodo/_tune"


def retarget(path, suffix):
    inp = unreal.IKRetargetBatchOperationInputs()
    inp.assets_to_retarget = [unreal.AssetRegistryHelpers.create_asset_data(src_anim)]
    inp.source_mesh, inp.target_mesh, inp.ik_retarget_asset = src_mesh, tgt_mesh, rtg
    inp.target_path, inp.suffix, inp.overwrite_existing_files = path, suffix, True
    inp.include_referenced_assets = False
    out = unreal.IKRetargetBatchOperation.run_batch_retarget(inp)
    return unreal.load_asset(str(out[0].package_name))


def pct(xs, q):
    s = sorted(xs)
    return s[int(q * (len(s) - 1))]


def measure(anim):
    res = {}
    n = int(anim.get_play_length() * 30)
    z = {b: [] for b in ("foot_l", "ball_l", "foot_r", "ball_r")}
    for i in range(0, n, 2):
        pose = P.get_anim_pose_at_time(anim, i / 30, opts)
        for b in z:
            z[b].append(P.get_bone_pose(pose, b, W).translation.z)
    for s in ("l", "r"):
        res[s] = {"ball_flat": round(pct(z[f"ball_{s}"], 0.1), 2), "ball_min": round(min(z[f"ball_{s}"]), 2),
                  "foot_flat": round(pct(z[f"foot_{s}"], 0.1), 2)}
    return res


def set_feet(rl, rr):
    rc.set_rotation_offset_for_retarget_pose_bone("foot_l", unreal.Rotator(*rl).quaternion(), T)
    rc.set_rotation_offset_for_retarget_pose_bone("foot_r", unreal.Rotator(*rr).quaternion(), T)


out = {}
if "apply" in cfg:
    set_feet(cfg["apply"]["foot_l"], cfg["apply"]["foot_r"])
    eal.save_asset(rtg.get_path_name())
    out["applied"] = cfg["apply"]
else:
    out["baseline"] = measure(retarget(TMP, "_base"))
    for c in cfg["candidates"]:
        set_feet(c, c)
        out[str(c)] = measure(retarget(TMP, "_t"))
    set_feet([0, 0, 0], [0, 0, 0])
    eal.delete_directory(TMP)
result = out
