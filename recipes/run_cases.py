"""Run a framework reference over the validation cases (recipes/cases.py) inside that framework's venv.

    <venv>/bin/python -m recipes.run_cases FRAMEWORK CASES_DIR [--kind universal|llama] [--only ...] [--json OUT]

Prints one line per case and writes {case: compare() result} to --json. Python 3.8-compatible (DyNet)."""
import argparse
import importlib
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recipes.compare import compare  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("framework")
    ap.add_argument("cases")
    ap.add_argument("--kind", default="universal")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--redo", action="store_true", help="rerun cases already passing in --json")
    a = ap.parse_args(argv)
    mod = importlib.import_module("recipes.frameworks.%s.%s" % (a.framework, a.kind))
    root = os.path.join(a.cases, a.kind)
    names = sorted(os.listdir(root))
    if a.only:
        names = [n for n in names if n in a.only]
    res = {}
    if a.json and os.path.isfile(a.json) and not a.redo:
        res = json.load(open(a.json))                    # resume: keep the passing results
        res = {k: v for k, v in res.items() if v.get("pass")}
    for n in names:
        if n in res:
            continue
        d = os.path.join(root, n)
        ref = os.path.join(d, "ref.npz")
        if not os.path.isfile(ref):
            continue
        t0 = time.time()
        try:
            m = mod.load_model(d)
            r = np.load(ref)
            ids = r["ids"].tolist()
            pre = m.prefill_logits(ids)
            new, steps = m.generate(ids, len(r["new"]))
            out = os.path.join(d, "out-%s.npz" % a.framework)
            np.savez(out, ids=np.array(ids), new=np.array(new), prefill=np.asarray(pre, np.float32),
                     steps=np.stack(steps))
            c = compare(ref, out)
        except Exception as e:
            c = {"pass": False, "error": "%s: %s" % (type(e).__name__, e)}
            traceback.print_exc()
        c["seconds"] = round(time.time() - t0, 1)
        res[n] = c
        print("%-4s %-34s %s" % ("ok" if c.get("pass") else "FAIL", n,
                                 c.get("error") or "max|d| %.2e  decode %.2e  greedy %s" % (
                                     c["max_abs_diff"], c["decode_max_abs_diff"] or 0, c["greedy_identical"])),
              flush=True)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
    print("%d/%d pass" % (sum(1 for c in res.values() if c.get("pass")), len(res)))


if __name__ == "__main__":
    main()
