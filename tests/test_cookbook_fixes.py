"""The cookbook constructions on REAL transformers modules / checkpoints, fp64 and fp32:

  * LayerNorm subset RMSNorm by the whole-model fold (P on every residual writer), on the stock
    Pythia-14m (untied) and OPT-125m (tied -> untied) checkpoints;
  * dense subset MoE through a zeroed router on the real Mixtral / Qwen2-MoE / DeepSeek-V3
    blocks, with the picked experts and their weights PROBED (topk tie-break);
  * reserved-channel carrying (CLA, sequential >= parallel) under a real pre-norm: not exact on
    stock Llama layers, exact with a per-group norm; parallel >= sequential at 2x depth exact on
    stock GPT-NeoX layers.

Run:  python -m pytest tests/test_cookbook_fixes.py -q
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

F64, F32 = torch.float64, torch.float32


# --- LayerNorm -> RMSNorm fold ------------------------------------------------------------

def _cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
        return try_to_load_from_cache(name, "config.json") is not None
    except Exception:
        return False


@pytest.mark.parametrize("name,tied", [("EleutherAI/pythia-14m", False), ("facebook/opt-125m", True)])
def test_layernorm_folds_into_rmsnorm_on_a_whole_model(name, tied):
    if not _cached(name):
        pytest.skip(f"{name} not in the local HF cache")
    from norm.layernorm_rmsnorm import fold_check
    r = fold_check(name, F64)
    assert r["untied"] is tied
    assert r["logit_err"] < 1e-9, r                      # exact (fp64 noise)
    # without an lm_head bias the final beta is a per-vocab logit shift W_U beta: NOT exact
    assert fold_check(name, F64, lm_head_bias=False)["logprob_err"] > 1.0
    # fp32: same order as the checkpoint's own fp32-vs-fp64 error (pythia-14m 1.2e-2, opt 1.8e-5)
    r32 = fold_check(name, F32)
    assert r32["logit_err"] < (5e-2 if "pythia" in name else 1e-4), r32


# --- dense -> MoE via a zeroed router -----------------------------------------------------

def test_zeroed_mixtral_router_picks_a_tie_broken_set_not_0_to_k():
    from moe.zero_router import check
    r = check("mixtral", F64)
    assert r["err"] < 1e-12, r
    assert set(r["weights"]) == set(r["picked"]) and all(abs(w - 0.5) < 1e-12 for w in r["weights"].values())
    if sorted(r["picked"]) != [0, 1]:                    # CPU torch picks [5, 6]: dealing into 0..k-1 drops the FFN
        naive = check("mixtral", F64, picked=[0, 1])
        assert naive["err"] > 0.5 * naive["scale"]


@pytest.mark.parametrize("dtype,tol", [(F64, 1e-12), (F32, 5e-6)])
def test_dense_into_real_moe_blocks_exact(dtype, tol):
    from moe.zero_router import check
    rows = [check("mixtral", dtype), check("mixtral", dtype, mode="replicate"),
            check("qwen2_moe", dtype), check("qwen2_moe", dtype, norm_topk_prob=True)]
    if dtype == F32:                                     # HF DeepseekV3MoE only runs in fp32
        rows += [check("deepseek_v3", dtype), check("deepseek_v3", dtype, norm_topk_prob=False)]
    for r in rows:
        assert r["err"] < tol, r


def test_router_weight_scales_per_family():
    from moe.zero_router import check
    w = lambda r: sorted(v for k, v in r["weights"].items() if k != "shared")
    assert w(check("mixtral", F64)) == [0.5, 0.5]                                   # 1/k  -> down x k
    q = check("qwen2_moe", F64)                                                       # norm_topk_prob=False
    assert w(q) == [0.125, 0.125] and q["weights"]["shared"] == 0.5                   # 1/E; sigmoid(0)
    d = check("deepseek_v3", F32)                                                     # sigmoid, rsf=2.5, k=4
    assert all(abs(v - 2.5 / 4) < 1e-6 for v in w(d)) and d["weights"]["shared"] == 1.0
    d2 = check("deepseek_v3", F32, norm_topk_prob=False)
    assert all(abs(v - 2.5 * 0.5) < 1e-6 for v in w(d2))                              # rsf * sigmoid(0)


# --- carrying through a real pre-norm -----------------------------------------------------

@pytest.mark.parametrize("dtype", [F64, F32])
def test_cla_carry_is_not_exact_under_stock_prenorm(dtype):
    from block.prenorm_carry import cla
    err, scale = cla(dtype)
    assert err > 1e-3 * scale                            # stock single-rms norm rescales the copy
    err_g, _ = cla(dtype, group_norm=True)
    assert err_g < 1e-6                                  # exact with a per-group norm (fp32 softmax in HF)


@pytest.mark.parametrize("dtype", [F64, F32])
def test_seq_hosting_parallel_is_not_exact_under_stock_prenorm(dtype):
    from block.prenorm_carry import seq_hosts_parallel
    err, scale = seq_hosts_parallel(dtype)
    assert err > 1e-3 * scale
    assert seq_hosts_parallel(dtype, group_norm=True)[0] < 1e-6


@pytest.mark.parametrize("dtype", [F64, F32])
def test_parallel_hosts_sequential_at_2x_depth_on_stock_neox(dtype):
    from block.prenorm_carry import parallel_hosts_seq_2x_depth
    assert parallel_hosts_seq_2x_depth(dtype)[0] < 1e-12
