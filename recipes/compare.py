"""Compare a recipe run (runtime.run_cli --npz) with the transformers reference (recipes.reference).

pass        (strict, the spec): fp32 prompt logits within TOL = 1e-4 of transformers' fp32 logits, and the
            greedy continuation identical.
pass_floor  (fp32-noise criterion, when the reference has fp64 logits): the greedy continuation identical,
            and the run within max(TOL, FLOOR_X x transformers' own fp32 error) of the exact (fp64) logits.
            Two fp32 implementations with different kernels differ by about their own rounding error. At a
            massive-activation position (position 0 of SmolLM2), transformers' fp32 is itself 1.3e-4 from
            the fp64 value, and on GPT-2's large logits it is 1.0e-4: there, 1e-4 between two fp32 runs is
            below the noise."""
import numpy as np

TOL = 1e-4
N_GREEDY = 32
FLOOR_X = 4.0


def compare(ref_path, run_path, tol=TOL):
    r, o = np.load(ref_path), np.load(run_path)
    assert r["ids"].tolist() == o["ids"].tolist(), "different prompt ids"
    pre_r, pre_o = r["prefill"].astype(np.float64), o["prefill"].astype(np.float64)
    V = min(pre_r.shape[-1], pre_o.shape[-1])          # a host may pad the vocabulary
    diff = np.abs(pre_r[..., :V] - pre_o[..., :V])
    d = float(diff.max())
    n = min(len(r["new"]), len(o["new"]))
    same = r["new"][:n].tolist() == o["new"][:n].tolist() and len(o["new"]) >= min(N_GREEDY, len(r["new"]))
    steps_r, steps_o = r["steps"], o["steps"]
    k = min(len(steps_r), len(steps_o))
    first = next((i for i in range(n) if r["new"][i] != o["new"][i]), n)
    kk = min(k, first + 1)                              # decode logits are comparable up to the first divergence
    dd = float(np.abs(steps_r[:kk, :V].astype(np.float64) - steps_o[:kk, :V]).max()) if kk else None
    res = {"max_abs_diff": d, "max_abs_diff_pos1": float(diff[1:].max()) if len(diff) > 1 else 0.0,
           "logit_scale": float(np.abs(pre_r).max()), "greedy_tokens": int(n),
           "greedy_identical": bool(same), "first_divergence": None if same else int(first),
           "decode_max_abs_diff": dd, "pass": bool(d <= tol and same)}
    if "prefill64" in r.files:
        p64 = r["prefill64"].astype(np.float64)[..., :V]
        res["hf32_fp64_floor"] = float(np.abs(pre_r[..., :V] - p64).max())
        res["fp64_max_abs_diff"] = float(np.abs(pre_o[..., :V] - p64).max())
        res["pass_floor"] = bool(same and res["fp64_max_abs_diff"] <= max(tol, FLOOR_X * res["hf32_fp64_floor"]))
    return res


if __name__ == "__main__":
    import sys
    print(compare(sys.argv[1], sys.argv[2]))
