"""Ground-contact foot IK: bend the toe joint so toe tips don't go below the floor (body unchanged).

On the ball of the foot (heel raised), a real foot bends the toes upward; generated motion usually tilts the
whole foot instead, so the toe tip penetrates. Per frame and per foot: if ToeEnd y < floor_y, rotate ToeBase
about the ball in the vertical plane of the toe until the tip sits on the floor. ToeEnd is a leaf, so
only ToeBase/ToeEnd globals and ToeBase local change.
"""
import numpy as np

# SOMA77: (Foot, ToeBase, ToeEnd)
FEET = ((69, 70, 71), (74, 75, 76))


def _rodrigues(axis, th):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def fix_toes(posed_joints, global_rot_mats, local_rot_mats, floor_y=0.0):
    """Arrays [T, 77, ...]; modified in place. Returns per-foot count of corrected frames and max bend (deg)."""
    up = np.array([0.0, 1.0, 0.0])
    stats = []
    for foot, ball, tip in FEET:
        n, max_deg = 0, 0.0
        for f in range(len(posed_joints)):
            b, t = posed_joints[f, ball], posed_joints[f, tip]
            if t[1] >= floor_y:
                continue
            w = t - b
            L = np.linalg.norm(w)
            axis = np.cross(w, up)
            if L < 1e-6 or np.linalg.norm(axis) < 1e-6:
                continue
            cur = np.arcsin(np.clip(w[1] / L, -1, 1))
            tgt = np.arcsin(np.clip((floor_y - b[1]) / L, -1, 1))
            R = _rodrigues(axis, tgt - cur)
            posed_joints[f, tip] = b + R @ w
            global_rot_mats[f, ball] = R @ global_rot_mats[f, ball]
            global_rot_mats[f, tip] = R @ global_rot_mats[f, tip]
            local_rot_mats[f, ball] = global_rot_mats[f, foot].T @ global_rot_mats[f, ball]
            n += 1
            max_deg = max(max_deg, float(np.degrees(tgt - cur)))
        stats.append({"frames": n, "max_bend_deg": round(max_deg, 1)})
    return {"left": stats[0], "right": stats[1]}
