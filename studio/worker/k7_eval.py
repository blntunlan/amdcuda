"""K7 A/B: same prompt + seed, post_processing off/on -> foot skate metrics.

python k7_eval.py <out.json>
"""
import json
import sys
import time

import numpy as np
import torch

from kimodo import load_model
from kimodo.metrics.foot_skate import FootSkateFromContacts, FootSkateFromHeight
from kimodo.skeleton import SOMASkeleton30
from kimodo.tools import seed_everything

PROMPTS = [
    ("A person walks forward.", 4.0),
    ("A person runs in a circle.", 5.0),
    ("A person walks forward and waves with the right hand.", 4.0),
    ("A person walks backwards slowly.", 4.0),
]
SEED, STEPS = 7, 50

device = "cuda:0"
model = load_model("Kimodo-SOMA-RP-v1.2", device=device, default_family="Kimodo")
skel = model.skeleton.somaskel77 if isinstance(model.skeleton, SOMASkeleton30) else model.skeleton
fps = model.fps
m_contact = FootSkateFromContacts(skeleton=skel, fps=fps)
m_height = FootSkateFromHeight(skeleton=skel, fps=fps)


def metrics(out):
    pj = torch.from_numpy(out["posed_joints"]).float()  # [1, T, J, 3]
    fc = torch.from_numpy(out["foot_contacts"]).float()
    L = torch.tensor([pj.shape[1]])
    r = {}
    for m in (m_contact, m_height):
        for k, v in m._compute(posed_joints=pj, foot_contacts=fc, lengths=L).items():
            r[k] = round(float(v.reshape(-1)[0]), 4)
    # extra: lowest foot joint (penetration) and contact-frame share
    fidx = skel.foot_joint_idx
    r["min_foot_y"] = round(float(pj[0, :, fidx, 1].min()), 4)
    r["contact_ratio"] = round(float((fc > 0.5).float().mean()), 3)
    return r


rows = []
for prompt, dur in PROMPTS:
    row = {"prompt": prompt, "duration": dur}
    outs = {}
    for pp in (False, True):
        seed_everything(SEED)
        t0 = time.time()
        out = model([prompt], [int(dur * fps)], constraint_lst=[], num_denoising_steps=STEPS, num_samples=1,
                    multi_prompt=True, num_transition_frames=5, post_processing=pp, return_numpy=True)
        row[f"sec_{'on' if pp else 'off'}"] = round(time.time() - t0, 2)
        row["on" if pp else "off"] = metrics(out)
        outs[pp] = out
    # how much the motion changed (should be small: correction, not new motion)
    d = np.linalg.norm(outs[True]["posed_joints"] - outs[False]["posed_joints"], axis=-1)
    row["joint_shift_mean_cm"] = round(float(d.mean() * 100), 2)
    row["joint_shift_max_cm"] = round(float(d.max() * 100), 2)
    row["finite"] = bool(np.isfinite(outs[True]["posed_joints"]).all())
    rows.append(row)
    print(json.dumps(row), flush=True)

json.dump(rows, open(sys.argv[1], "w"), indent=2)
