"""Compare source (SOMA) and target (Manny) animations: hips path, hand/foot heights."""
import json

import unreal

with open(UH_ARG, encoding="utf-8") as fh:  # noqa: F821
    cfg = json.load(fh)
src = unreal.load_asset(cfg["src_anim"])
tgt = unreal.load_asset(cfg["tgt_anim"])
P = unreal.AnimPoseExtensions
W = unreal.AnimPoseSpaces.WORLD
opts = unreal.AnimPoseEvaluationOptions()

pairs = {"hips": ("Hips", "pelvis"), "hand_r": ("RightHand", "hand_r"), "hand_l": ("LeftHand", "hand_l"),
         "foot_l": ("LeftFoot", "foot_l"), "foot_r": ("RightFoot", "foot_r"), "head": ("Head", "head")}


def sample(anim, which):
    n = 24
    L = anim.get_play_length()
    rows = []
    for i in range(n + 1):
        t = L * i / n
        pose = P.get_anim_pose_at_time(anim, t, opts)
        row = {"t": round(t, 3)}
        for k, names in pairs.items():
            tr = P.get_bone_pose(pose, names[which], W)
            v = tr.translation
            row[k] = [round(v.x, 1), round(v.y, 1), round(v.z, 1)]
        rows.append(row)
    return rows


S, T = sample(src, 0), sample(tgt, 1)


def travel(rows):
    a, b = rows[0]["hips"], rows[-1]["hips"]
    return round(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5, 1)


def corr(key, axis):
    xs = [r[key][axis] for r in S]
    ys = [r[key][axis] for r in T]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return round(num / den, 3) if den else None


result = {
    "len": [round(src.get_play_length(), 3), round(tgt.get_play_length(), 3)],
    "hips_travel_cm": [travel(S), travel(T)],
    "hips_height_mean": [round(sum(r["hips"][2] for r in S) / len(S), 1), round(sum(r["hips"][2] for r in T) / len(T), 1)],
    "hand_r_peak_z": [max(r["hand_r"][2] for r in S), max(r["hand_r"][2] for r in T)],
    "head_peak_z": [max(r["head"][2] for r in S), max(r["head"][2] for r in T)],
    "foot_min_z": [min(min(r["foot_l"][2], r["foot_r"][2]) for r in S), min(min(r["foot_l"][2], r["foot_r"][2]) for r in T)],
    "corr": {"hips_x": corr("hips", 0), "hips_y": corr("hips", 1), "hand_r_z": corr("hand_r", 2),
             "hand_l_z": corr("hand_l", 2), "foot_l_z": corr("foot_l", 2), "foot_r_z": corr("foot_r", 2)},
}
