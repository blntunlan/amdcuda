"""Moonwalk foot quality on source (SOMA) and target (Manny), UE world space (Z up, cm).
cfg: {"src_anim":..., "tgt_anim":..., "tgt_mesh": "...SKM_Manny_Simple"}"""
import json

import unreal

with open(UH_ARG, encoding="utf-8") as fh:  # noqa: F821
    cfg = json.load(fh)
P = unreal.AnimPoseExtensions
W = unreal.AnimPoseSpaces.WORLD
opts = unreal.AnimPoseEvaluationOptions()
RIGS = {
    "src": (cfg["src_anim"], {"pelvis": "Hips", "foot_l": "LeftFoot", "ball_l": "LeftToeBase", "foot_r": "RightFoot", "ball_r": "RightToeBase"}),
    "tgt": (cfg["tgt_anim"], {"pelvis": "pelvis", "foot_l": "foot_l", "ball_l": "ball_l", "foot_r": "foot_r", "ball_r": "ball_r"}),
}


def track(anim, names, fps=30):
    n = int(round(anim.get_play_length() * fps))
    out = {k: [] for k in names}
    for i in range(n):
        pose = P.get_anim_pose_at_time(anim, i / fps, opts)
        for k, b in names.items():
            v = P.get_bone_pose(pose, b, W).translation
            out[k].append((v.x, v.y, v.z))
    return out


def pct(xs, q):
    s = sorted(xs)
    return s[int(q * (len(s) - 1))]


def analyse(tr, fps=30):
    pel = tr["pelvis"]
    T = len(pel)
    # direction of travel (pelvis), facing is the opposite (moonwalk)
    dx, dy = pel[-1][0] - pel[0][0], pel[-1][1] - pel[0][1]
    L = (dx * dx + dy * dy) ** 0.5
    back = (dx / L, dy / L)  # unit vector of backwards motion
    res = {"travel_cm": round(L, 1)}
    lifts = {}
    for s in ("l", "r"):
        f, b = tr[f"foot_{s}"], tr[f"ball_{s}"]
        fz = [p[2] for p in f]
        flat_h = pct(fz, 0.1)
        lifts[s] = [z - flat_h for z in fz]
        res[f"ball_{s}_min_cm"] = round(min(p[2] for p in b), 1)
        res[f"foot_{s}_min_cm"] = round(min(fz), 1)
        res[f"ball_{s}_flat_cm"] = round(pct([p[2] for p in b], 0.1), 1)
    # alternating heels
    a, c = lifts["l"], lifts["r"]
    ma, mc = sum(a) / T, sum(c) / T
    num = sum((x - ma) * (y - mc) for x, y in zip(a, c))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mc) ** 2 for y in c)) ** 0.5
    res["heel_anti_corr"] = round(num / den, 2) if den else None
    # glide (flat foot back speed / body back speed) and planted ball slip (heel raised: ball should be stationary)
    body = [((pel[i + 1][0] - pel[i][0]) * back[0] + (pel[i + 1][1] - pel[i][1]) * back[1]) * fps for i in range(T - 1)]
    num = den = 0.0
    slips = []
    for s in ("l", "r"):
        f, b, lift = tr[f"foot_{s}"], tr[f"ball_{s}"], lifts[s]
        for i in range(T - 1):
            v = ((f[i + 1][0] - f[i][0]) * back[0] + (f[i + 1][1] - f[i][1]) * back[1]) * fps
            if lift[i] < 2.0 and body[i] > 10:
                num += v
                den += body[i]
            if lift[i] > 5.0:  # heel up -> planted on the ball
                bv = ((b[i + 1][0] - b[i][0]) ** 2 + (b[i + 1][1] - b[i][1]) ** 2) ** 0.5 * fps
                slips.append(bv)
    res["glide_ratio"] = round(num / den, 2) if den else None
    res["planted_ball_slip_cm_s"] = round(sorted(slips)[len(slips) // 2], 1) if slips else None  # median
    return res


result = {}
for key, (path, names) in RIGS.items():
    result[key] = analyse(track(unreal.load_asset(path), names))
# Manny reference pose: ball height when standing flat (bone height, not the sole)
mesh = unreal.load_asset(cfg["tgt_mesh"])
ref = P.get_reference_pose(mesh.skeleton) if hasattr(P, "get_reference_pose") else None
if ref is not None:
    result["manny_ref_ball_l_cm"] = round(P.get_bone_pose(ref, "ball_l", W).translation.z, 1)
    result["manny_ref_foot_l_cm"] = round(P.get_bone_pose(ref, "foot_l", W).translation.z, 1)
