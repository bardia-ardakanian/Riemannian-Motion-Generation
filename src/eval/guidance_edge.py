"""
Check the guidance-grid edge effect: the intrinsic baseline's best guidance at low
NFE sits at the EDGE of the grid we swept (g=3.0) and was still improving there,
while the extrinsic arm's optimum is INTERIOR (g=2.0). If so, the intrinsic arm was
scored at an unexplored boundary exactly where the reported gap is largest.
Extends the sweep upward for both arms. Merges into report_sweep/sweep.json.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from rmg_eval import gather_test, prep_refs, eval_model, load_model
sys.path.insert(0, os.path.expanduser("~/riemannian-vfm/motion_real"))
from eval_official import OfficialEvaluator

GRID = [("rmg_base", [4, 8, 16], [4.0, 5.0, 6.0]),
        ("base_vel_ext", [4, 8, 16], [1.0, 4.0, 5.0])]

dev = "cuda"
outpath = "report_sweep/sweep.json"
ev = OfficialEvaluator(dev)
raws, toks, gtlens, real263 = gather_test(1024)
refs = prep_refs(ev, raws, toks, real263, 0)
res = json.load(open(outpath)) if os.path.exists(outpath) else {}
t0 = time.time()
for run, nfes, gs in GRID:
    model, flow, c = load_model(f"runs/{run}/model.pth", "ema", dev)
    res.setdefault(run, {"step": c.get("step", "?")})
    for s in nfes:
        res[run].setdefault(str(s), {})
        for g in gs:
            if str(g) in res[run][str(s)]:
                continue
            m = eval_model(ev, model, flow, gtlens, refs, guidance=g, steps=s,
                           gen_batch=32, seed=0, solver="euler")
            res[run][str(s)][str(g)] = m
            json.dump(res, open(outpath, "w"), indent=2)
            print(f"[{time.time()-t0:5.0f}s] {run:14s} nfe={s:<3} g={g:<4} FID={m['fid']:7.3f} R@3={m['R_top3']:.3f}", flush=True)
    del model, flow; torch.cuda.empty_cache()
print("DONE", flush=True)
