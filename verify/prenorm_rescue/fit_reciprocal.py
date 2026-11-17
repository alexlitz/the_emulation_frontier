"""Sup relative error of the 1D compensators (fitlib) on the real token-RMS ranges (rms_range.py):
for each model, the worst norm (largest max/min, position 0 excluded) and the median norm, on the
min-max range and the p0.1-p99.9 range. Errors are re-verified on a finer grid, in float64 and in
float32. For OLMo-2 also the post-norm inputs (sublayer outputs) and the best fixed scale (k = 0).

Run: PYENV_VERSION=3.12.6 python verify/prenorm_rescue/fit_reciprocal.py [model-substring]\nWrites results/fit_<model>.json per model.
"""
import glob
import json
import os

import numpy as np

import fitlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
KS = [1, 2, 4, 8, 16, 32]


def curve(lo, hi):
    out = {"D": hi / lo, "k0_fixed_scale": F.const_err(hi / lo)}
    for fam in ("glu", "relu"):
        for k in KS:
            if fam == "glu" and k == 32:
                continue                        # numerically meaningless (coefficients ~1e10+)
            c = F.Comp(fam, k, lo, hi)
            out[f"{fam}_k{k}"] = {"lp": float(c.err), "f64": F.verify(c, lo, hi),
                                   "f32": F.verify(c, lo, hi, np.float32),
                                   "max_abs_coef": float(np.max(np.abs(c.c)))}
    return out


def main(only=None):
    res = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "rms_*.json"))):
        if only and only not in fn:
            continue
        r = json.load(open(fn))
        norms = r["norms"]
        is_post = lambda n: "olmo-2" in r["model"].lower() and ("post_" in n)
        pre = {n: v for n, v in norms.items() if not is_post(n)}
        ranked = sorted(pre, key=lambda n: pre[n]["no_pos0"]["D_minmax"])
        pick = {"worst": ranked[-1], "median": ranked[len(ranked) // 2]}
        m = {}
        for tag, n in pick.items():
            s = pre[n]["no_pos0"]
            m[tag] = {"norm": n, "minmax": curve(s["min"], s["max"]),
                      "p0.1-p99.9": curve(s["p0.1"], s["p99.9"])}
            print(r["model"], tag, n, "D", round(s["D_minmax"], 2), flush=True)
        if "olmo-2" in r["model"].lower():
            post = {n: v["no_pos0"] for n, v in norms.items() if is_post(n)}
            m["post_norm_inputs"] = {n: {"D_minmax": s["D_minmax"], "D_p": s["D_p"],
                                         "k0_fixed_scale_minmax": F.const_err(s["D_minmax"]),
                                         "k0_fixed_scale_p": F.const_err(s["D_p"])}
                                     for n, s in post.items()}
        res[r["model"]] = m
        tag = os.path.basename(fn)[4:-5]
        json.dump(m, open(os.path.join(HERE, "results", f"fit_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
