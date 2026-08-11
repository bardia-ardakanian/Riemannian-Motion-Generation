"""
Multi-seed error bars for the headline cells. The HumanML3D protocol replicates
generation many times and reports mean +- CI; every number in the report so far is
a single seed, so it is unknown whether e.g. 0.16 vs 0.19 is a real difference.

Each model is evaluated at its own best (steps, guidance) and at the shared 8-step
budget, over several generation seeds.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from rmg_eval import gather_test, prep_refs, eval_model, load_model
sys.path.insert(0, os.path.expanduser("~/riemannian-vfm/motion_real"))
from eval_official import OfficialEvaluator

SEEDS = [0, 1, 2, 3, 4]
# run -> [(steps, guidance, label), ...]
CELLS = {
    "rmg_base":     [(128, 3.0, "peak"), (8, 4.0, "nfe8")],
    "base_vel_ext": [(16, 3.0, "peak"), (8, 2.0, "nfe8")],
    "base_6d":      [(16, 3.0, "peak"), (8, 3.0, "nfe8")],
    "base_ep_int":  [(32, 1.0, "peak"), (8, 1.0, "nfe8")],
    "base_ep_ext":  [(16, 2.0, "peak"), (8, 2.0, "nfe8")],
}
out = "report_sweep/multiseed.json"
dev = "cuda"
ev = OfficialEvaluator(dev)
raws, toks, gtlens, real263 = gather_test(1024)
res = json.load(open(out)) if os.path.exists(out) else {}
t0 = time.time()
# references depend on the seed only through the diversity subsample; rebuild per seed
refs_cache = {}
for run, cells in CELLS.items():
    p = f"runs/{run}/model.pth"
    if not os.path.exists(p):
        print(f"[skip] {run}"); continue
    model, flow, c = load_model(p, "ema", dev)
    res.setdefault(run, {})
    for steps, g, label in cells:
        res[run].setdefault(label, {})
        for sd in SEEDS:
            if str(sd) in res[run][label]:
                continue
            if sd not in refs_cache:
                refs_cache[sd] = prep_refs(ev, raws, toks, real263, sd)
            m = eval_model(ev, model, flow, gtlens, refs_cache[sd], guidance=g,
                           steps=steps, gen_batch=32, seed=sd, solver="euler")
            res[run][label][str(sd)] = m
            json.dump(res, open(out, "w"), indent=2)
            print(f"[{time.time()-t0:5.0f}s] {run:14s} {label:5s} nfe={steps:<4} g={g} seed={sd} "
                  f"FID={m['fid']:6.3f} R@3={m['R_top3']:.3f}", flush=True)
    del model, flow; torch.cuda.empty_cache()

print("\n=== summary: mean +- std over seeds ===", flush=True)
for run, cells in res.items():
    for label, seeds in cells.items():
        f = np.array([v["fid"] for v in seeds.values()])
        r = np.array([v["R_top3"] for v in seeds.values()])
        print(f"  {run:14s} {label:5s} n={len(f)}  FID {f.mean():6.3f} +- {f.std(ddof=1):.3f}   "
              f"R@3 {r.mean():.3f} +- {r.std(ddof=1):.3f}", flush=True)
print("DONE", flush=True)
