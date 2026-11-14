"""Studio timeline + constraint spec -> Kimodo inputs, and measuring how well constraints were followed.

Job fields (all optional except the prompt/segments):
  segments: [{"prompt": "...", "duration": 3.0}, ...]   (overrides prompt/duration)
  constraints: [
    {"type": "root_path", "points": [[x, z], ...], "every": 5}        # meters, Kimodo Y-up world
    {"type": "pose", "source": "<library item>", "source_frame": 40,  # full-body keyframe
     "time": 2.0, "place": "path"|"origin"|"source"}
    {"type": "effector", "effector": "right-hand"|"left-hand"|"right-foot"|"left-foot",
     "source": "...", "source_frame": 10, "time": 1.0, "place": ...}
  ]
"""
import os

import numpy as np
import torch

from kimodo.constraints import (
    FullBodyConstraintSet,
    LeftFootConstraintSet,
    LeftHandConstraintSet,
    RightFootConstraintSet,
    RightHandConstraintSet,
    Root2DConstraintSet,
)
from kimodo.skeleton import SOMASkeleton30

EFFECTORS = {
    "left-hand": (LeftHandConstraintSet, "LeftHand"),
    "right-hand": (RightHandConstraintSet, "RightHand"),
    "left-foot": (LeftFootConstraintSet, "LeftFoot"),
    "right-foot": (RightFootConstraintSet, "RightFoot"),
}
MAX_TOTAL_SEC = 20.0


def _nonempty(lst):
    return [c for c in lst if c is None or not hasattr(c, "frame_indices") or len(c.frame_indices) > 0]


def install_empty_constraint_fix(model):
    """Kimodo bug: in multi-prompt, crop_move also passes along constraints that are EMPTY in a segment,
    and create_conditions crashes on reshape. Filter empties at the two entry points (no upstream change)."""
    import kimodo.model.kimodo_model as km

    rep_cls = type(model.motion_rep)
    if not getattr(rep_cls, "_studio_patched", False):
        orig = rep_cls.create_conditions_from_constraints

        def patched(self, constraints_lst, *a, **k):
            return orig(self, _nonempty(constraints_lst), *a, **k)

        rep_cls.create_conditions_from_constraints = patched
        rep_cls._studio_patched = True
    if not getattr(km.post_process_motion, "_studio_patched", False):
        orig_pp = km.post_process_motion

        def pp(local_rot_mats, root_positions, contacts, skeleton, constraint_lst=None, *a, **k):
            if constraint_lst:
                constraint_lst = [_nonempty(c) if isinstance(c, list) else c for c in constraint_lst]
                if not isinstance(constraint_lst[0], list):
                    constraint_lst = _nonempty(constraint_lst)
            return orig_pp(local_rot_mats, root_positions, contacts, skeleton, constraint_lst, *a, **k)

        pp._studio_patched = True
        km.post_process_motion = pp


def parse_segments(job, fps):
    segs = job.get("segments") or [{"prompt": job["prompt"], "duration": job.get("duration", 3.0)}]
    texts, frames = [], []
    for s in segs:
        p = str(s["prompt"]).strip()
        if not p:
            raise ValueError("boş segment prompt'u")
        if not p.endswith("."):
            p += "."
        d = float(s.get("duration", 3.0))
        if not 0.5 <= d <= 10.0:
            raise ValueError(f"segment süresi 0.5-10 sn olmalı: {d}")
        texts.append(p)
        frames.append(int(round(d * fps)))
    if sum(frames) / fps > MAX_TOTAL_SEC:
        raise ValueError(f"toplam süre en fazla {MAX_TOTAL_SEC} sn")
    return texts, frames


def resample_path(points, n):
    """Resample a polyline to n points at equal arc-length spacing."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("kök yolu en az 2 nokta ister")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        raise ValueError("kök yolu uzunluğu sıfır")
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], 1), float(cum[-1])


def facing_heading(facing, span, f0, T):
    """Facing direction -> Kimodo global_root_heading (cos a, sin a), a = atan2(fx, fz).
    facing: "travel" (direction of travel), "reverse" (backwards to travel: moonwalk/walking backwards), or angle in degrees (0 = +z, 90 = +x)."""
    if isinstance(facing, (int, float)) or (isinstance(facing, str) and facing.lstrip("-").replace(".", "").isdigit()):
        a = np.full(T, np.radians(float(facing)))
    else:
        if facing not in ("travel", "reverse"):
            raise ValueError(f"bilinmeyen bakış: {facing}")
        d = np.gradient(span, axis=0)
        k = max(3, len(span) // 20)  # smooth the corners a little
        d = np.stack([np.convolve(d[:, i], np.ones(k) / k, mode="same") for i in (0, 1)], 1)
        if facing == "reverse":
            d = -d
        a_span = np.arctan2(d[:, 0], d[:, 1])
        a = np.concatenate([np.full(f0, a_span[0]), a_span, np.full(T - f0 - len(span), a_span[-1])])
    return np.stack([np.cos(a), np.sin(a)], 1)


def _rot_y(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


SOMA77_FEET = {"left": (69, 70), "right": (74, 75)}  # (Foot/ankle, ToeBase/ball)


def moonwalk_feet(c, pj_ref, gr_ref, dense_path, dense_heading, f0, f1, fps):
    """Moonwalk foot pattern -> per-frame full poses (77 joints) and the foot constrained in that frame.
    Planted foot: heel raised (rotated by heel_deg about the ball), ball stationary in world.
    Gliding foot: flat, sliding back at ~2v. The model handles the ±swap_gap frames around each swap."""
    S = max(4, int(round(float(c.get("step", 0.5)) * fps)))
    theta = np.radians(float(c.get("heel_deg", 35)))
    gap = int(c.get("swap_gap", 2))
    every = max(1, int(c.get("every", 2)))
    drop = float(c.get("pelvis_drop", 0.02))
    # canonical: pelvis xz = 0, facing +z
    h = c.get("_ref_heading")
    a_ref = np.arctan2(h[1], h[0])  # heading = (cos a, sin a)
    Rc = _rot_y(-a_ref)
    P0 = (pj_ref - [pj_ref[0, 0], 0, pj_ref[0, 2]]) @ Rc.T
    G0 = np.einsum("ij,njk->nik", Rc, gr_ref)
    span = dense_path[f0:f1 + 1]
    v = float(np.linalg.norm(np.diff(span, axis=0), axis=1).mean() * fps)
    d = v * S / fps  # relative stride per step
    poses, which = [], []
    for f in range(f0, f1 + 1):
        k, r = divmod(f - f0, S)
        u = r / S
        if r < gap or r > S - 1 - gap or (f - f0) % every:
            continue
        glide = "left" if k % 2 == 0 else "right"
        P, G = P0.copy(), G0.copy()
        P[0, 1] -= drop
        for side, (ank, ball) in SOMA77_FEET.items():
            dz = (d / 2 - d * u) if side == glide else (-d / 2 + d * u)
            chain = [ank, ball, ank + 2]  # + ToeEnd (not constrained in the 30-joint model, moved for consistency)
            P[chain, 2] += dz
            if side != glide:  # heel up: rotate the ankle about the ball, toes stay flat
                Rx = _rot_x(theta)
                P[ank] = P[ball] + Rx @ (P[ank] - P[ball])
                G[ank] = Rx @ G[ank]
        b = np.arctan2(dense_heading[f, 1], dense_heading[f, 0])
        Rw = _rot_y(b)
        Pw = P @ Rw.T
        Pw[:, [0, 2]] += dense_path[f]
        Gw = np.einsum("ij,njk->nik", Rw, G)
        poses.append((f, Pw, Gw, glide))
    return poses, {"stride_m": round(d, 3), "speed_ms": round(v, 2), "step_frames": S}


def build(job, model, library_dir, device):
    """Returns texts, num_frames, constraint_lst, targets (for measurement), info."""
    fps = model.fps
    texts, frames = parse_segments(job, fps)
    T = sum(frames)
    skel = model.skeleton
    sl = skel.get_skel_slice(skel.somaskel77) if isinstance(skel, SOMASkeleton30) else slice(None)
    cons, targets, info = [], [], {"total_frames": T}
    dense_path = dense_heading = None

    specs = job.get("constraints") or []
    for c in specs:
        if c.get("type") == "root_path":
            # Time range: the path applies only between [start, end] s; outside it the motion is free.
            f0 = int(np.clip(round(float(c.get("start", 0.0)) * fps), 0, T - 2))
            f1 = int(np.clip(round(float(c.get("end", T / fps)) * fps), f0 + 1, T)) - 1
            n = f1 - f0 + 1
            span, length = resample_path(c["points"], n)
            # dense_path is also used to place keyframes; outside the range, clamp to the nearest end.
            dense_path = np.concatenate([np.repeat(span[:1], f0, 0), span, np.repeat(span[-1:], T - 1 - f1, 0)])
            every = max(1, int(c.get("every", 5)))
            idx = np.unique(np.concatenate([np.arange(f0, f1 + 1, every), [f1]]))
            heading = facing_heading(c.get("facing"), span, f0, T) if c.get("facing") not in (None, "", "free") else None
            dense_heading = heading
            cons.append(Root2DConstraintSet(
                skel, torch.tensor(idx), torch.tensor(dense_path[idx], dtype=torch.float32, device=device),
                global_root_heading=None if heading is None else torch.tensor(heading[idx], dtype=torch.float32, device=device)))
            tgt = {"type": "root_path", "frames": idx.tolist(), "xz": dense_path[idx].tolist()}
            if heading is not None:
                tgt["heading"] = heading[idx].tolist()
            targets.append(tgt)
            info["path_length_m"] = round(length, 2)
            info["path_range_s"] = [round(f0 / fps, 2), round((f1 + 1) / fps, 2)]
            info["path_speed_ms"] = round(length / (n / fps), 2)

    for c in specs:
        if c.get("type") != "moonwalk":
            continue
        if dense_path is None or dense_heading is None:
            raise ValueError("moonwalk, bakışı ayarlı (facing) bir kök yolu ister")
        ref = str(c.get("ref", "moonwalk-a-s1-pp"))
        if not ref.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"geçersiz referans: {ref}")
        d = np.load(os.path.join(library_dir, ref + ".npz"))
        rf = int(c.get("ref_frame", 5))
        c = dict(c, _ref_heading=d["global_root_heading"][rf].tolist())
        m0 = int(np.clip(round(float(c.get("start", 0.0)) * fps), 0, T - 2))
        m1 = int(np.clip(round(float(c.get("end", T / fps)) * fps), m0 + 1, T)) - 1
        poses, minfo = moonwalk_feet(c, d["posed_joints"][rf].astype(np.float64), d["global_rot_mats"][rf].astype(np.float64),
                                     dense_path, dense_heading, m0, m1, fps)
        fr = torch.tensor([p[0] for p in poses])
        P = torch.tensor(np.stack([p[1][sl] for p in poses]), dtype=torch.float32, device=device)
        R = torch.tensor(np.stack([p[2][sl] for p in poses]), dtype=torch.float32, device=device)
        root2d = torch.tensor(dense_path[fr.numpy()], dtype=torch.float32, device=device)
        cons.append(LeftFootConstraintSet(skel, fr, P, R, root2d))
        cons.append(RightFootConstraintSet(skel, fr, P.clone(), R.clone(), root2d.clone()))
        targets.append({"type": "feet", "frames": fr.tolist(), "joints": [69, 74],
                        "pos": [[p[1][69].tolist(), p[1][74].tolist()] for p in poses]})
        info["moonwalk"] = dict(minfo, constrained_frames=len(poses))

    for c in specs:
        if c.get("type") not in ("pose", "effector"):
            continue
        src = str(c["source"])
        if not src.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"geçersiz kaynak adı: {src}")
        d = np.load(os.path.join(library_dir, src + ".npz"))
        pj, gr = d["posed_joints"].astype(np.float32), d["global_rot_mats"].astype(np.float32)
        sf = int(np.clip(int(c.get("source_frame", 0)), 0, len(pj) - 1))
        f = int(np.clip(round(float(c.get("time", 0)) * fps), 0, T - 1))
        pos, rot = pj[sf].copy(), gr[sf]
        place = c.get("place") or ("path" if dense_path is not None else "origin")
        root = pos[0].copy()  # Hips (77 and 30: index 0)
        if place == "path" and dense_path is not None:
            pos[:, [0, 2]] += dense_path[f] - root[[0, 2]]
        elif place == "origin":
            pos[:, [0, 2]] -= root[[0, 2]]
        P = torch.tensor(pos[sl][None], device=device)
        R = torch.tensor(rot[sl][None], device=device)
        fi = torch.tensor([f])
        if c["type"] == "pose":
            cons.append(FullBodyConstraintSet(skel, fi, P, R))
            targets.append({"type": "pose", "frame": f, "joints": pos.tolist(), "label": f"{src}#{sf}"})
        else:
            cls, joint = EFFECTORS[c["effector"]]
            cons.append(cls(skel, fi, P, R, None))
            j77 = list(skel.somaskel77.bone_order_names).index(joint)
            targets.append({"type": "effector", "effector": c["effector"], "frame": f, "joint": j77,
                            "pos": pos[j77].tolist(), "label": f"{src}#{sf}"})
    return texts, frames, cons, targets, info


def measure(output, targets):
    """How closely the generated motion followed each constraint (cm)."""
    pj = output["posed_joints"][0]
    root = output["smooth_root_pos"][0] if "smooth_root_pos" in output else pj[:, 0]
    res = []
    for t in targets:
        if t["type"] == "root_path":
            e = np.linalg.norm(root[t["frames"]][:, [0, 2]] - np.asarray(t["xz"]), axis=1) * 100
            r = {"type": "root_path", "mean_cm": round(float(e.mean()), 1), "max_cm": round(float(e.max()), 1)}
            if "heading" in t and "global_root_heading" in output:
                h = output["global_root_heading"][0][t["frames"]]
                g = np.asarray(t["heading"])
                ang = np.degrees(np.abs(np.arctan2(h[:, 1] * g[:, 0] - h[:, 0] * g[:, 1], (h * g).sum(1))))
                r["heading_mean_deg"] = round(float(ang.mean()), 1)
                r["heading_max_deg"] = round(float(ang.max()), 1)
            res.append(r)
        elif t["type"] == "pose":
            e = np.linalg.norm(pj[t["frame"]] - np.asarray(t["joints"]), axis=1) * 100
            res.append({"type": "pose", "frame": t["frame"], "label": t["label"],
                        "mean_cm": round(float(e.mean()), 1), "max_cm": round(float(e.max()), 1)})
        elif t["type"] == "feet":
            e = np.linalg.norm(pj[t["frames"]][:, t["joints"]] - np.asarray(t["pos"]), axis=-1) * 100
            res.append({"type": "feet", "mean_cm": round(float(e.mean()), 1), "max_cm": round(float(e.max()), 1)})
        else:
            e = float(np.linalg.norm(pj[t["frame"], t["joint"]] - np.asarray(t["pos"])) * 100)
            res.append({"type": "effector", "effector": t["effector"], "frame": t["frame"], "label": t["label"], "err_cm": round(e, 1)})
    return res
