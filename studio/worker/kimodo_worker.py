"""Kalıcı Kimodo worker'ı.

Modeli (ve LLM2Vec metin kodlayıcıyı) BİR KEZ yükler, sonra stdin'den satır satır
JSON iş okur, üretir, çıktıyı doğrular ve stdout'a satır satır JSON sonuç yazar.

Protokol (JSON Lines):
  stdin  -> {"id": "abc", "prompt": "A person walks forward.", "duration": 3.0,
             "steps": 20, "seed": 42, "out_dir": "C:/.../library", "name": "walk"}
  stdout -> {"event": "ready", ...} | {"event": "progress", ...} |
            {"event": "done", "id": ..., "ok": true, "files": {...}, "checks": {...}} |
            {"event": "error", "id": ..., "message": ...}
Sadece "@@" ile başlayan stdout satırları protokoldür; diğer her şey log'dur.
"""

import json
import os
import sys
import time
import traceback

PROTO = "@@"


def emit(obj):
    sys.stdout.write(PROTO + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def rss_gib():
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**30, 2)
    except Exception:
        return None


def validate(output, n_frames_expected, fps):
    import numpy as np

    pj = output["posed_joints"][0]
    checks = {}
    checks["frames"] = int(pj.shape[0])
    checks["frames_ok"] = pj.shape[0] == n_frames_expected
    finite = all(
        np.isfinite(v).all() for v in output.values() if hasattr(v, "dtype") and v.dtype.kind == "f"
    )
    checks["finite"] = bool(finite)
    root = output["root_positions"][0]
    checks["root_travel_m"] = round(float(np.linalg.norm(root[-1] - root[0])), 3)
    # Kare başına en büyük eklem yer değiştirmesi -> hız (m/s). >12 m/s patlama/titreme demek.
    vel = np.linalg.norm(np.diff(pj, axis=0), axis=-1) * fps
    checks["max_joint_speed_ms"] = round(float(vel.max()), 2)
    checks["speed_ok"] = checks["max_joint_speed_ms"] < 12.0
    h = pj[..., 1]
    checks["height_range_m"] = [round(float(h.min()), 3), round(float(h.max()), 3)]
    checks["ok"] = checks["frames_ok"] and checks["finite"] and checks["speed_ok"]
    return checks


def write_view_bin(path, global_rot_mats, posed_joints):
    """Tarayıcı için: float32 [F, J, 12] = 3x3 dünya rotasyonu (satır sıralı) + dünya konumu."""
    import numpy as np

    F, J = posed_joints.shape[:2]
    arr = np.concatenate([global_rot_mats.reshape(F, J, 9), posed_joints.reshape(F, J, 3)], axis=-1)
    arr.astype("<f4").tofile(path)


def export_skin(lib_dir):
    """Placeholder SOMA gövdesini (kimodo skin_standard.npz) tarayıcı formatına bir kez çevirir."""
    import numpy as np

    from kimodo.assets import skeleton_asset_path

    bin_path = os.path.join(lib_dir, "_skin.bin")
    if os.path.exists(bin_path):
        return
    d = np.load(str(skeleton_asset_path("somaskel77") / "skin_standard.npz"))
    inv_bind = np.linalg.inv(d["bind_rig_transform"].astype(np.float64)).astype("<f4")
    parts = [
        ("vertices", d["bind_vertices"].astype("<f4")),
        ("faces", d["faces"].astype("<u4")),
        ("lbs_indices", d["lbs_indices"].astype("<u4")),
        ("lbs_weights", d["lbs_weights"].astype("<f4")),
        ("inv_bind", inv_bind),
        ("connections", d["rig_joint_connections"].astype("<u4")),
    ]
    layout, off = {}, 0
    with open(bin_path, "wb") as f:
        for key, a in parts:
            b = a.tobytes()
            layout[key] = {"offset": off, "shape": list(a.shape), "dtype": a.dtype.str}
            f.write(b)
            off += len(b)
    layout["joint_names"] = [str(n) for n in d["rig_joint_names"]]
    with open(os.path.join(lib_dir, "_skin.json"), "w", encoding="utf-8") as f:
        json.dump(layout, f)


def main():
    t0 = time.time()
    import torch

    from kimodo import load_model
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots
    from kimodo.tools import seed_everything

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import studio_constraints
    import foot_fix
    import numpy as np

    model_name = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1.2")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and os.environ.get("KIMODO_ALLOW_CPU") != "1":
        emit({"event": "fatal", "message": "GPU yok (torch.cuda.is_available()=False). Sessiz CPU fallback kapalı."})
        return 2

    model = load_model(model_name, device=device, default_family="Kimodo")
    studio_constraints.install_empty_constraint_fix(model)
    skeleton = model.skeleton
    if isinstance(skeleton, SOMASkeleton30):
        skeleton = skeleton.somaskel77.to(device)
    emit(
        {
            "event": "ready",
            "model": model_name,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None,
            "fps": model.fps,
            "load_sec": round(time.time() - t0, 1),
            "rss_gib": rss_gib(),
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job_id = None
        try:
            job = json.loads(line)
            if job.get("cmd") == "quit":
                break
            if job.get("cmd") == "ping":
                emit({"event": "pong", "rss_gib": rss_gib()})
                continue
            if job.get("cmd") == "export_skin":
                os.makedirs(job["out_dir"], exist_ok=True)
                export_skin(job["out_dir"])
                emit({"event": "skin_ready", "dir": job["out_dir"]})
                continue
            if job.get("cmd") == "convert":
                # Eski üretilmiş NPZ'yi görüntüleyici formatına çevir.
                import numpy as np

                d = np.load(job["npz"])
                write_view_bin(job["view"], d["global_rot_mats"], d["posed_joints"])
                emit({"event": "converted", "view": job["view"], "frames": int(d["posed_joints"].shape[0])})
                continue
            job_id = job["id"]
            steps = int(job.get("steps", 50))
            seed = job.get("seed")
            postprocess = bool(job.get("postprocess", False))
            out_dir = job["out_dir"]
            name = job.get("name") or job_id
            os.makedirs(out_dir, exist_ok=True)
            texts, frames, constraint_lst, targets, cinfo = studio_constraints.build(job, model, out_dir, device)
            n_frames = sum(frames)
            prompt = " ".join(texts)
            duration = round(n_frames / model.fps, 3)

            emit({"event": "progress", "id": job_id, "stage": "generating"})
            t1 = time.time()
            if seed is not None:
                seed_everything(int(seed))
            output = model(
                texts,
                frames,
                constraint_lst=constraint_lst,
                num_denoising_steps=steps,
                num_samples=1,
                multi_prompt=True,
                num_transition_frames=5,
                post_processing=postprocess,
                return_numpy=True,
            )
            gen_sec = time.time() - t1
            ground = None
            if job.get("ground_fix", True):
                # toe IK: bend the toes instead of letting the toe tip go below the floor (body unchanged)
                for k in ("posed_joints", "global_rot_mats", "local_rot_mats"):
                    output[k] = np.array(output[k], dtype=np.float32)
                ground = foot_fix.fix_toes(output["posed_joints"][0], output["global_rot_mats"][0], output["local_rot_mats"][0])
            checks = validate(output, n_frames, model.fps)
            if ground is not None:
                checks["ground_fix"] = ground
            if targets:
                checks["constraints"] = studio_constraints.measure(output, targets)

            npz_path = os.path.join(out_dir, name + ".npz")
            bvh_path = os.path.join(out_dir, name + ".bvh")
            single = {
                k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v)
                for k, v in output.items()
            }
            save_kimodo_npz(npz_path, single)
            joints_pos = torch.from_numpy(output["posed_joints"][0]).to(device)
            joints_rot = torch.from_numpy(output["global_rot_mats"][0]).to(device)
            local = global_rots_to_local_rots(joints_rot, skeleton)
            save_motion_bvh(
                bvh_path, local, joints_pos[:, skeleton.root_idx, :], skeleton=skeleton, fps=model.fps
            )
            meta = {
                "id": job_id,
                "prompt": prompt,
                "duration": duration,
                "steps": steps,
                "seed": seed,
                "postprocess": postprocess,
                "ground_fix": ground is not None,
                "segments": [{"prompt": t, "duration": round(f / model.fps, 3)} for t, f in zip(texts, frames)],
                "constraints": job.get("constraints") or [],
                "constraint_info": cinfo,
                "model": model_name,
                "fps": model.fps,
                "gen_sec": round(gen_sec, 2),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checks": checks,
            }
            with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            view_path = os.path.join(out_dir, name + ".view.bin")
            write_view_bin(view_path, output["global_rot_mats"][0], output["posed_joints"][0])
            files = {"npz": npz_path, "bvh": bvh_path, "view": view_path}
            for k, p in files.items():
                if os.path.getsize(p) == 0:
                    raise RuntimeError(f"{k} boş yazıldı: {p}")
            emit(
                {
                    "event": "done",
                    "id": job_id,
                    "ok": checks["ok"],
                    "files": files,
                    "checks": checks,
                    "gen_sec": meta["gen_sec"],
                    "rss_gib": rss_gib(),
                    "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device != "cpu" else None,
                }
            )
        except Exception as e:
            emit({"event": "error", "id": job_id, "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
