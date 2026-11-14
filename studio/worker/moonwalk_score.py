"""Moonwalk-specific foot analysis: python moonwalk_score.py <library_dir> <name> [<name>...]

SOMA 77 joint indices: LeftFoot 69 / LeftToeBase 70 / LeftToeEnd 71, RightFoot 74 / RightToeBase 75 / RightToeEnd 76.
"""
import json
import sys

import numpy as np

L_ANK, L_TOE, L_TIP, R_ANK, R_TOE, R_TIP = 69, 70, 71, 74, 75, 76
FPS = 30


def score(path):
    d = np.load(path)
    pj = d["posed_joints"].astype(np.float64)
    T = len(pj)
    h = d["global_root_heading"]
    face = np.stack([h[:, 1], h[:, 0]], 1)  # (fx, fz)
    root = d["smooth_root_pos"][:, [0, 2]] if "smooth_root_pos" in d.files else pj[:, 0, [0, 2]]
    mface = face.mean(0) / np.linalg.norm(face.mean(0))
    back_m = -float((root[-1] - root[0]) @ mface)  # positive = moved backwards

    feet = pj[:, [L_ANK, L_TOE, L_TIP, R_ANK, R_TOE, R_TIP], 1]
    low_l, low_r = feet[:, :3].min(1), feet[:, 3:].min(1)
    ground = np.minimum(low_l, low_r)
    contact = float((ground < 0.04).mean())  # at least one foot on the ground
    airborne = int(((low_l > 0.06) & (low_r > 0.06)).sum())
    penetration_cm = float(min(0.0, feet.min()) * 100)

    # heel lift = ankle height above its "flat" height (10th percentile)
    lift_l = pj[:, L_ANK, 1] - np.percentile(pj[:, L_ANK, 1], 10)
    lift_r = pj[:, R_ANK, 1] - np.percentile(pj[:, R_ANK, 1], 10)
    anti = float(np.corrcoef(lift_l, lift_r)[0, 1])  # moonwalk: alternating → negative
    diff = lift_l - lift_r
    alternations = int((np.diff(np.sign(diff[np.abs(diff) > 0.02])) != 0).sum())

    # backward speed of the flat foot (on the ground, heel down, nearly stationary vertically),
    # as a ratio of the body's backward speed. Normal walk ~0 (planted foot is still), moonwalk ~1 (foot glides).
    rv = np.diff(root, axis=0) * FPS
    body_back = -(rv * face[:-1]).sum(1)
    num, den = 0.0, 0.0
    for ank, low, lift in ((L_ANK, low_l, lift_l), (R_ANK, low_r, lift_r)):
        v3 = np.diff(pj[:, ank], axis=0) * FPS
        back = -(v3[:, [0, 2]] * face[:-1]).sum(1)
        flat = (low[:-1] < 0.025) & (lift[:-1] < 0.02) & (np.abs(v3[:, 1]) < 0.15) & (body_back > 0.1)
        num += back[flat].sum()
        den += body_back[flat].sum()
    glide = num / den if den > 0 else 0.0

    sp = np.linalg.norm(np.diff(pj, axis=0), axis=-1).max(1) * FPS
    jerk = float(np.percentile(np.abs(np.diff(sp)) * FPS, 99))
    return {
        "frames": T, "back_m": round(back_m, 2), "contact": round(contact, 3), "airborne": airborne,
        "penetration_cm": round(penetration_cm, 1), "heel_anti_corr": round(anti, 2), "alternations": alternations,
        "glide_ratio": round(float(glide), 2), "jerk_p99": round(jerk, 1),
    }


if __name__ == "__main__":
    lib = sys.argv[1]
    rows = {}
    for n in sys.argv[2:]:
        try:
            rows[n] = score(f"{lib}/{n}.npz")
        except FileNotFoundError:
            rows[n] = None
    print(json.dumps(rows, indent=1))
