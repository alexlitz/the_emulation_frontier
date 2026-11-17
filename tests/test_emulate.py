"""elephantine/emulate.py: the written host checkpoint computes the target.

Real pairs (downloaded as needed): the target's weights and only the host's config.json go in; the
checkpoint is reloaded from disk with AutoModelForCausalLM (the host's own stock class) and compared with
the target in fp32 and in fp64 (both models' internal fp32 casts lifted): max |delta logits| at fp64
rounding and identical greedy continuations. Synthetic pairs (tiny random models of each family, saved
locally) cover the family-specific folds: biases, qk-norm, Gemma's (1 + w) norms, embedding scale,
sandwich norms and softcaps, Granite's multipliers, Phi-3's fused projections, sliding windows, the MoE
hosts (tied / shared / ERNIE's split over the selected experts), MoE targets (router copy, duplicated
experts, ERNIE's selection bias keeping spare experts out), LayerNorm streams (the mean-carrying padding
and the [h; -h] mirror), parallel blocks (GPT-NeoX, StableLM, Falcon, Cohere), partial / interleaved RoPE,
learned positions (GPT-2, OPT), a gain-free norm host (OLMo) and Falcon-H1's zeroed Mamba branch. Real
targets also go into synthetic hosts of the families whose real configs are too big to write here.
Refusals: a non-edge, an edge that needs a limit / regime, and exact edges outside the script's scope.

Run:  pytest tests/test_emulate.py -q
"""
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from elephantine.emulate import (APPROX_EDGE, NOT_AN_EDGE, OUT_OF_SCOPE, Refused, emulate, fetch_config,
                                 support)

TOK = "HuggingFaceTB/SmolLM2-135M"          # tokenizer for the synthetic models (vocab 49152)


def _scratch():
    """Big checkpoints go next to the HF cache (EMULATE_TEST_DIR overrides), not into a RAM-backed /tmp."""
    base = os.environ.get("EMULATE_TEST_DIR")
    if not base:
        from huggingface_hub import constants
        base = os.path.join(constants.HF_HOME, "emulate-tests")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(dir=base)


def _run(target, host, fp64_rel, fp32_greedy=True, **kw):
    out = _scratch()
    try:
        r = emulate(target, host, os.path.join(out, "host"), fp64=True, log=lambda *a: None, **kw)
        v = r["verification"]
        if os.environ.get("EMULATE_TEST_LOG"):                 # the measured agreement, one JSON line per pair
            row = {"target": target, "host": str(host)}
            for prec in ("fp32", "fp64"):
                row[prec] = {k: v[prec][k] for k in ("max_abs_diff", "rel", "argmax_match")}
                row[prec]["greedy_identical"] = all(g["identical"] for g in v[prec]["greedy"])
            with open(os.environ["EMULATE_TEST_LOG"], "a") as f:
                f.write(json.dumps(row) + "\n")
        assert v["fp64"]["rel"] < fp64_rel, v["fp64"]
        assert all(g["identical"] for g in v["fp64"]["greedy"]), v["fp64"]
        if fp32_greedy:
            assert all(g["identical"] for g in v["fp32"]["greedy"]), v["fp32"]
        assert v["fp32"]["rel"] < 1e-4, v["fp32"]
        cfg = json.loads(Path(out, "host", "config.json").read_text())
        from elephantine.emulate import STOCK_ALIASES
        arch = fetch_config(host)["architectures"]
        assert "auto_map" not in cfg and cfg["architectures"] == [STOCK_ALIASES.get(a, a) for a in arch]
        assert Path(out, "host", "README.md").exists() and Path(out, "host", "plan.json").exists()
        return r, cfg
    finally:
        shutil.rmtree(out, ignore_errors=True)
        gc.collect()


LIGHT = dict(prompts=["The capital of France is", "def fibonacci(n):"], new_tokens=8)   # the 2B hosts in fp64


def _host_dir(root, name, mt, kw):
    """A host config.json only (a stock config of family mt): the script never reads the host's weights."""
    from transformers import AutoConfig, AutoModelForCausalLM
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    cfg = AutoConfig.for_model(mt, **kw)
    cfg.architectures = [AutoModelForCausalLM._model_mapping[type(cfg)].__name__]
    cfg.save_pretrained(d)
    return d


def _stock_target_dir(root, repo):
    """repo's files (linked) with its config's remote-code keys dropped (auto_map, seq_aux): the checkpoint as the
    stock class reads it. (The relation reads a remote modeling_deepseek + seq_aux + noaux_tc config as the
    first-release 0-filled group limit, group_mask_zero_fill; the test runs the stock -inf class.)"""
    from huggingface_hub import snapshot_download
    src = snapshot_download(repo)
    d = os.path.join(root, "target-" + repo.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(src):
        if f != "config.json" and not f.endswith(".py"):
            os.symlink(os.path.realpath(os.path.join(src, f)), os.path.join(d, f))
    cfg = json.loads(Path(src, "config.json").read_text())
    for k in ("auto_map", "seq_aux"):
        cfg.pop(k, None)
    Path(d, "config.json").write_text(json.dumps(cfg, indent=2))
    return d


# --- real pairs ------------------------------------------------------------------------------------------

def test_smollm2_135m_in_360m():
    """Same family, every capacity axis padded: hidden 576 -> 960 (eps and gains), 30 -> 32 layers, 9 -> 15
    query heads and 3 -> 5 KV heads, FFN 1536 -> 2560."""
    r, cfg = _run("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", 1e-12)
    assert r["edge"]["exact"] and cfg["vocab_size"] == 49152
    assert cfg["rms_norm_eps"] == pytest.approx(1e-5 * 576 / 960)


def test_qwen25_05b_gqa_in_qwen15_18b_mha():
    """GQA (14 queries on 2 KV heads) -> MHA (16 KV heads): each KV head tiled over 7 host heads; head_dim
    64 -> 128 (RoPE pair i on host pair 2 i, the same theta); the q/k/v biases carried."""
    r, cfg = _run("Qwen/Qwen2.5-0.5B", "Qwen/Qwen1.5-1.8B", 1e-12)
    assert any(s["slot"] == "kv" and s["target"] == "GQA" and s["host"] == "MHA" for s in r["edge"]["slots"])
    assert cfg["rope_theta"] == 1000000.0 and cfg["vocab_size"] == 151936


def test_smollm2_135m_in_granite_cross_family():
    """Llama into Granite: the host's embedding / residual / attention / logit multipliers folded."""
    r, cfg = _run("HuggingFaceTB/SmolLM2-135M", "ibm-granite/granite-3.3-2b-base", 1e-12, **LIGHT)
    assert cfg["model_type"] == "granite" and cfg["residual_multiplier"] == fetch_config(
        "ibm-granite/granite-3.3-2b-base")["residual_multiplier"]


def test_qwen3_06b_in_17b_qk_norm():
    """qk-norm on both sides with the hidden width doubled: one eps serves the hidden norms and the qk-norm."""
    _run("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", 1e-12, **LIGHT)


def test_sliding_window_target_in_mistral_host():
    """A sliding-window (1024) Mistral in a bigger Mistral whose window is set to the target's. head_dim
    26 -> 48 is not an integer ratio: the host's rotary takes a per-pair longrope table (theta 1), so inv_freq is
    the target's bit for bit (v2 rescaled theta, agreeing to ~1e-7)."""
    r, cfg = _run("Felladrin/Minueza-32M-Base", "OuteAI/Lite-Mistral-150M-v2-Instruct", 1e-12)
    assert cfg["sliding_window"] == 1024 and cfg["rope_scaling"]["rope_type"] == "longrope"


# --- real pairs: the families and paths added after v1 ------------------------------------------------------

def test_pythia_70m_in_160m():
    """GPT-NeoX (Pythia): LayerNorm stream padded 512 -> 768 with the mean carried in the new dims, parallel
    residual, fused per-head qkv, partial RoPE (16 of 64 dims), biased ungated GeLU MLP; 6 -> 12 layers."""
    r, cfg = _run("EleutherAI/pythia-70m", "EleutherAI/pythia-160m", 1e-12)
    assert r["construction"]["stream"] == "mean" and cfg["layer_norm_eps"] == pytest.approx(1e-5 * 512 / 768)


def test_pythia_70m_in_1b():
    """head_dim 64 -> 256 with rotary_pct 0.25: the target's 8 RoPE pairs go on host pairs 0, 4, 8, ... (same
    theta, bit-identical inv_freq), its 48 unrotated dims on the host's unrotated dims."""
    r, cfg = _run("EleutherAI/pythia-70m", "EleutherAI/pythia-1b", 1e-12)
    assert cfg["rotary_emb_base"] == 10000 and "bit-identical" in r["construction"]["lines"][3]


def test_gpt2_in_gpt2_medium():
    """GPT-2: learned absolute positions copied (lifted like every writer), Conv1D weights, LayerNorm stream
    768 -> 1024, 12 -> 24 layers, 12 -> 16 heads, the tied head through the final norm's gain and bias."""
    _run("openai-community/gpt2", "openai-community/gpt2-medium", 1e-12)


def test_opt_125m_in_13b():
    """OPT: learned positions (offset 2), q scaled inside q_proj, LayerNorm with a fixed eps (1e-5): the stream
    is carried x sqrt(D / d) so that eps / a^2 is the target's at the host's width."""
    r, cfg = _run("facebook/opt-125m", "facebook/opt-1.3b", 1e-12, **LIGHT)
    assert any("fixed" in n for n in r["notes"])


def test_ernie_03b_in_qwen25_3b():
    """ERNIE-4.5 as a target: interleaved RoPE pairs (2i, 2i + 1) onto the host's rotate_half pairs, the host's
    q/k/v biases at 0."""
    _run("baidu/ERNIE-4.5-0.3B-PT", "Qwen/Qwen2.5-3B", 1e-12, **LIGHT)


def test_granite4_350m_in_1b():
    """Granite-4.0 (granitemoehybrid, all attention layers, RoPE): the fused shared_mlp and every Granite
    multiplier on both sides."""
    _run("ibm-granite/granite-4.0-350m", "ibm-granite/granite-4.0-1b", 1e-12, **LIGHT)


def test_qwen3_moe_2_experts_in_4():
    """An MoE target: 2 experts, top-1 -> 4 experts, top-2. The router rows are copied and each target expert
    duplicated: the softmax over the copies sums back to the target's gate and the top-2 picks both copies."""
    r, _ = _run("huihui-ai/Huihui-MoE-0.8B-2E", "DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder", 1e-12, **LIGHT)
    assert set(r["construction"]["ffn"].values()) == {"moe"} and set(r["construction"]["expert_copies"].values()) == {2}


def test_kv_head_packing_llama_in_mistral():
    """KV-head packing: 4 KV heads of 6 dims (2 queries each) in 3 host KV heads of 64 dims with 4 queries each: two
    target KV heads side by side per host head, each block rotating at the target's frequencies through a longrope
    table (short = long factor per pair, theta 1: bit-identical inv_freq); the host's own window (4096 >= its
    context, so the relation reads it as full) removed (the window lever)."""
    r, cfg = _run("delphi-suite/v0-llama2-100k", "crumb/nano-mistral", 1e-12)
    assert cfg["rope_scaling"]["rope_type"] == "longrope" and cfg["sliding_window"] is None
    assert any(len(v) == 2 for v in r["construction"]["kv_heads"].values())


def test_kv_head_packing_smollm2_in_qwen25_3b():
    """SmolLM2-135M (3 KV heads x 3 queries, head_dim 64) in Qwen2.5-3B (2 KV heads x 8 queries, head_dim 128)."""
    r, cfg = _run("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-3B", 1e-12, **LIGHT)
    assert cfg["rope_scaling"]["rope_type"] == "longrope"


def test_sliding_target_in_llama_mask_edge():
    """A sliding-window (1024) Mistral on a Llama host, which has no window setting: the relation's mask edge -- the
    host attends fully, the window is the caller's attention mask, and on prompts shorter than the window none is
    needed. The README says so."""
    r, _ = _run("Felladrin/Minueza-32M-Base", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1e-6, **LIGHT)
    assert any("caller's attention mask" in n for n in r["notes"])


def test_gpt_neo_tinystories_1m_in_3m():
    """GPT-Neo (TinyStories): unscaled attention scores, learned positions, alternating global / local (256)
    layers, the host's attention_types set to the target's."""
    r, cfg = _run("roneneldan/TinyStories-1M", "roneneldan/TinyStories-3M", 1e-12)
    assert cfg["window_size"] == 256


def test_tiny_gpt2_in_starcoder_mqa():
    """GPT-2 (2 heads of 1 dim) in GPT-BigCode (tiny_starcoder_py, one fused Linear c_attn, multi-query): KV-head
    packing puts both target K / V heads side by side in the host's one KV head (learned positions: no rotary, so
    the packing needs no table)."""
    _run("sshleifer/tiny-gpt2", "bigcode/tiny_starcoder_py", 1e-12)


def test_lfm2_350m_in_700m():
    """LFM2: the gated short-conv layers and the qk-normed attention layers, each on the host's own kind."""
    _run("LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", 1e-12, **LIGHT)


def _run_limit(target, host, **kw):
    """A LIMIT construction (the relation labels the edge a limit): the high-agreement standard -- top-1 >= 99% and
    >= 90% of greedy continuations identical, in fp32 and fp64 -- and the note says LIMIT."""
    out = _scratch()
    try:
        r = emulate(target, host, os.path.join(out, "host"), fp64=True, log=lambda *a: None, **kw)
        for prec in ("fp32", "fp64"):
            v = r["verification"][prec]
            assert v["argmax_match"] >= 0.99, v
            assert sum(g["identical"] for g in v["greedy"]) >= 0.9 * len(v["greedy"]), v
        assert not r["edge"]["exact"] and any(n.startswith("LIMIT") for n in r["notes"])
        assert "LIMIT" in Path(out, "host", "README.md").read_text()
        return r
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_limit_relu_opt_in_gelu_starcoder():
    """ReLU (OPT) on a GeLU-tanh host (GPT-BigCode): GeLU(lam u) / lam -> ReLU(u), lam = 2^20."""
    _run_limit("yujiepan/opt-tiny-2layers-random", "bigcode/tiny_starcoder_py")


def test_limit_relu_synthetic(synth_root):
    """OPT (ReLU) in a stock GPT-2 config (erf GeLU): the same limit."""
    t = _model_dir(synth_root, "lim-t", "opt", OP)
    h = _model_dir(synth_root, "lim-h", "gpt2", dict(G2B, activation_function="gelu"), weights=False)
    _run_limit(t, h, prompts=["The capital of France is", "def f(x):"], new_tokens=8)


@pytest.mark.parametrize("hmt,hkw", [("qwen3", {}), ("olmoe", dict(num_experts=4, num_experts_per_tok=2,
                                                                      intermediate_size=112)),
                                      ("lfm2", dict(layer_types=["full_attention"] * 3, pad_token_id=None,
                                                    block_auto_adjust_ff_dim=False))],
                         ids=["qwen3_head", "olmoe_proj", "lfm2_head"])
def test_eps_lever_qknorm_off(hmt, hkw, synth_root):
    """The eps lever (a LIMIT the relation labels from round 2, pass 5; the round2-p4 relation here does not make
    these pairs edges, so the gate is skipped): a plain Llama on a host whose q / k RMS norm is switched off by
    scaling q_proj / k_proj by s and the gain by sqrt(eps) / s, rho <= 2^-60: at fp64 rounding."""
    t = _model_dir(synth_root, "lever-t-" + hmt, "llama", _kw("llama", {}, SMALL))
    h = _model_dir(synth_root, "lever-h-" + hmt, hmt, _kw(hmt, hkw, BIG), weights=False)
    from elephantine import emulate as em
    orig = em.plan_construction
    em.plan_construction = lambda tr, hr, limit=False: orig(tr, hr, limit=True)
    try:
        r = emulate(t, h, os.path.join(synth_root, "lever-o-" + hmt), fp64=True, prompts=["The capital of France is",
                    "def f(x):"], new_tokens=8, log=lambda *a: None, check_relation=False)
    finally:
        em.plan_construction = orig
    v = r["verification"]
    assert any(n.startswith("LIMIT") and "eps lever" in n for n in r["notes"])
    assert v["fp64"]["rel"] < 1e-12 and all(g["identical"] for g in v["fp64"]["greedy"]), v["fp64"]
    assert all(g["identical"] for g in v["fp32"]["greedy"]), v["fp32"]


def test_eps_lever_is_not_an_edge_under_round2_p4():
    """Under this branch's relation (round2-p4) qk-norm no <- head is still the refused regime edge."""
    with pytest.raises(Refused) as e:
        emulate("Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-1.7B", None, plan_only=True, log=lambda *a: None)
    assert e.value.code == NOT_AN_EDGE


def test_mamba_130m_in_370m():
    """Mamba-1 in a bigger Mamba-1: stream 768 -> 1024, inner 1536 -> 2048, dt rank 48 -> 64 zero-padded, 24 -> 48
    layers (every other one the identity)."""
    r, cfg = _run("state-spaces/mamba-130m-hf", "state-spaces/mamba-370m-hf", 1e-12, **LIGHT)
    assert cfg["layer_norm_epsilon"] == pytest.approx(1e-5 * 768 / 1024)


TINY = "Maykeye/TinyLLama-v0"                 # a real Llama (64 wide, 8 layers, 16 heads of 4)
STORIES_MOE = "ggml-org/stories15M_MOE"      # a real Mixtral (288 wide, 4 experts, top-2)
HOSTS = {   # stock configs of the families whose real configs are too big to write in a test
    # (the hosts' heads untied like the targets': round 2 reads a tied head as a feature a target without one lacks)
    "cohere": (TINY, "cohere", dict(vocab_size=32000, hidden_size=192, intermediate_size=320, num_hidden_layers=16,
                                    num_attention_heads=16, num_key_value_heads=16, logit_scale=0.25,
                                    tie_word_embeddings=False,
                                    max_position_embeddings=2048, rope_theta=10000.0)),
    "falcon_h1": (TINY, "falcon_h1", dict(vocab_size=32000, hidden_size=96, intermediate_size=320, num_hidden_layers=9,
                                          num_attention_heads=16, num_key_value_heads=16, head_dim=8, mamba_d_ssm=64,
                                          mamba_n_heads=4, mamba_d_head=16, mamba_d_state=8, mamba_n_groups=1,
                                          mamba_chunk_size=8, attention_in_multiplier=0.7, key_multiplier=0.3,
                                          attention_out_multiplier=0.2, mlp_multipliers=[0.5, 0.25],
                                          embedding_multiplier=3.0, lm_head_multiplier=0.125, rope_theta=10000.0,
                                          max_position_embeddings=2048, tie_word_embeddings=False)),
    "seed_oss": (TINY, "seed_oss", dict(vocab_size=32000, hidden_size=128, intermediate_size=320, num_hidden_layers=9,
                                        num_attention_heads=16, num_key_value_heads=16, head_dim=8, attention_bias=True,
                                        max_position_embeddings=2048, rope_theta=10000.0)),
    # v3: small real checkpoints of the families added: OLMo-2 / 3 (post-norm only, whole-projection qk-norm; OLMo-3's
    # sliding layers), Bamba (Mamba-2: each target channel held twice), PhiMoE, Seed-OSS, and a GLM-4.5-MoE test model
    # (128 experts, top-8, sigmoid) in 256 experts top-16 of half the width: each expert as two copies holding a slice
    "olmo2_real": ("optimum-intel-internal-testing/tiny-random-olmo2", "olmo2", dict(
        vocab_size=100352, hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=6,
        num_key_value_heads=6, max_position_embeddings=4096, rope_theta=500000, rms_norm_eps=1e-6,
        tie_word_embeddings=False, pad_token_id=100277, eos_token_id=100257)),
    "olmo3_real": ("yujiepan/olmo-3-tiny-random", "olmo3", dict(
        vocab_size=100278, hidden_size=16, head_dim=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=8,
        num_key_value_heads=4, layer_types=["sliding_attention", "full_attention", "full_attention"], sliding_window=4096,
        max_position_embeddings=65536, rope_theta=500000, rms_norm_eps=1e-6, tie_word_embeddings=False,
        pad_token_id=100277, eos_token_id=100257)),
    "bamba_real": ("yujiepan/bamba-tiny-random", "bamba", dict(
        vocab_size=128256, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, attn_layer_indices=[1], mamba_n_heads=16, mamba_d_head=8, mamba_n_groups=1,
        mamba_d_state=12, mamba_d_conv=4, mamba_expand=4, mamba_chunk_size=16, max_position_embeddings=4096,
        tie_word_embeddings=False, pad_token_id=0, bos_token_id=128000, eos_token_id=128001)),
    "seed_oss_real": ("yujiepan/seed-oss-tiny-random", "seed_oss", dict(
        vocab_size=155136, hidden_size=16, head_dim=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=8,
        num_key_value_heads=4, attention_bias=True, max_position_embeddings=524288, rope_theta=10000000.0,
        tie_word_embeddings=False, pad_token_id=1, eos_token_id=2, bos_token_id=0)),
    # a tiny real DeepSeek-V3 (yarn RoPE, interleaved, group-limited sigmoid routing over 256 experts): latents x 2
    "deepseek_v3_real": ("yujiepan/deepseek-v3-tiny-random", "deepseek_v3", dict(
        vocab_size=129280, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=24,
        moe_intermediate_size=16, n_routed_experts=256, num_experts_per_tok=8, n_shared_experts=2,
        first_k_dense_replace=1, n_group=8, topk_group=4, routed_scaling_factor=2.5, norm_topk_prob=True,
        max_position_embeddings=163840, rope_theta=10000, tie_word_embeddings=False, bos_token_id=0, eos_token_id=1,
        rope_scaling={"beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 1.0, "mscale_all_dim": 1.0,
                      "original_max_position_embeddings": 4096, "type": "yarn"})),
    "glm4_moe_real": ("np-cr/testing-glm4-moe", "glm4_moe", dict(
        vocab_size=151552, hidden_size=32, head_dim=8, intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, partial_rotary_factor=0.5, moe_intermediate_size=704, n_routed_experts=256,
        num_experts_per_tok=16, n_shared_experts=2, first_k_dense_replace=1, attention_bias=True, rope_theta=1000000.0,
        max_position_embeddings=131072, pad_token_id=None)),
    "olmo": (TINY, "olmo", dict(vocab_size=32000, hidden_size=192, intermediate_size=320, num_hidden_layers=9,
                                num_attention_heads=16, num_key_value_heads=16, max_position_embeddings=2048,
                                tie_word_embeddings=False, rope_theta=10000.0)),
    "stablelm": (TINY, "stablelm", dict(vocab_size=32000, hidden_size=256, intermediate_size=320, num_hidden_layers=9,
                                        num_attention_heads=16, num_key_value_heads=16, partial_rotary_factor=0.25,
                                        use_qkv_bias=True, max_position_embeddings=2048, rope_theta=10000.0)),
    "ernie_moe_split": (TINY, "ernie4_5_moe", dict(vocab_size=32000, hidden_size=96, intermediate_size=320,
                                                   num_hidden_layers=9, num_attention_heads=16, num_key_value_heads=16,
                                                   head_dim=8, moe_intermediate_size=64, moe_k=4, moe_num_experts=8,
                                                   moe_num_shared_experts=0, moe_layer_start_index=0,
                                                   max_position_embeddings=2048, rope_theta=10000.0,
                                                   tie_word_embeddings=False)),
    "granitemoe_tied": (TINY, "granitemoe", dict(vocab_size=32000, hidden_size=128, intermediate_size=256,
                                                 num_hidden_layers=9, num_attention_heads=16, num_key_value_heads=16,
                                                 num_local_experts=4, num_experts_per_tok=2, embedding_multiplier=12.0,
                                                 residual_multiplier=0.22, attention_multiplier=0.1, logits_scaling=8.0,
                                                 max_position_embeddings=2048, rope_theta=10000.0)),
    "mixtral_in_ernie_spare": (STORIES_MOE, "ernie4_5_moe", dict(
        vocab_size=32000, hidden_size=384, intermediate_size=1024, num_hidden_layers=7, num_attention_heads=8,
        num_key_value_heads=8, head_dim=48, moe_intermediate_size=768, moe_k=4, moe_num_experts=12,
        moe_num_shared_experts=0, moe_layer_start_index=0, max_position_embeddings=256, rope_theta=1000000.0,
        tie_word_embeddings=False)),
}


@pytest.mark.parametrize("case", list(HOSTS))
def test_real_target_in_synthetic_host(case, tmp_path):
    """A real target in a stock config of each host family: Cohere (parallel, one shared LayerNorm, the [h; -h]
    mirror, 2 layers per target layer, logit_scale), Falcon-H1 (Mamba-2 branch zeroed, muP multipliers folded),
    OLMo (no norm gains: they go into the readers; fixed eps), StableLM (the mirror, partial RoPE), ERNIE-4.5-MoE
    (the FFN dealt over the 4 experts its selection bias picks), Granite-MoE (every expert = the FFN), and a real
    Mixtral in an ERNIE MoE host with 4 spare experts (copies x 2, top-2 -> top-4, the spare ones never
    selected)."""
    target, mt, kw = HOSTS[case]
    if case == "deepseek_v3_real":
        target = _stock_target_dir(str(tmp_path), target)
    r, _ = _run(target, _host_dir(str(tmp_path), case, mt, kw), 1e-12)
    modes = set(r["construction"]["ffn"].values())
    want = {"ernie_moe_split": {"split"}, "granitemoe_tied": {"tied"}, "mixtral_in_ernie_spare": {"moe"},
            "glm4_moe_real": {"dense", "moe"}, "deepseek_v3_real": {"dense", "moe"}, "bamba_real": {"dense"}}
    assert modes == want.get(case, {"dense"}), modes


@pytest.mark.parametrize("target,host,code,needle", [
    ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B", NOT_AN_EDGE, "does not emulate"),      # kv 192 > 128
    ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", NOT_AN_EDGE, "does not emulate"),               # 14 > 12 heads
    # the edge-case audit made these three non-edges of the relation itself (they were an approximate edge
    # and two v1 scope refusals): the qk-norm regime and dropped q / k / v biases are measured not payable
    # (block/fold_agreement.py), and KV heads are compared by count (3 KV heads in 2 groups of 8). Round 2,
    # attention / norm pass: the qk-norm value is its variant ("head"); KV-head packing made SmolLM2-135M <-
    # Qwen2.5-3B an exact edge again (2 x 64 <= 128 per head, a longrope table), which v3 builds
    # (round 2, pass 5: the eps lever switches the Qwen3 host's qk-norm off, so the q / k / v bias is the wall left)
    ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-1.7B", NOT_AN_EDGE, "bias_qkv<-none"),
    ("Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-1.7B", NOT_AN_EDGE, "bias_qkv<-none"),
    # (fr-positions, fp_table_length: OPT-125m's 2048-row table does not fit GPT-2's 1024 rows -- was APPROX_EDGE,
    # "limit", ReLU <- GeLU)
    ("facebook/opt-125m", "openai-community/gpt2", NOT_AN_EDGE, "does not emulate"),
    # ALiBi <- RoPE was an approximate edge; round 3 walls it on falcon-rw's q / k / v bias and its ungated FFN
    ("tiiuae/falcon-rw-1b", "abeja/gpt-neox-japanese-2.7b", NOT_AN_EDGE, "bias_qkv<-none"),
    # (fr-positions, fp_table_length: an 8192-row learned table does not fit GPT-2-medium's 1024 rows -- the relation's
    # wall now, the script's "learned positions" refusal before)
    ("bigcode/tiny_starcoder_py", "openai-community/gpt2-medium", NOT_AN_EDGE, "pos learned_abs<-learned_abs"),
])
def test_refused(target, host, code, needle):
    with pytest.raises(Refused) as e:
        emulate(target, host, None, plan_only=True, log=lambda *a: None)
    assert e.value.code == code
    assert needle in str(e.value) and "REFUSED" in e.value.text
    from elephantine.emulate import main
    assert main([target, host, "--plan-only"]) == code


# --- synthetic pairs: one per family fold -----------------------------------------------------------------

V = 49152
SMALL = dict(vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, max_position_embeddings=256, tie_word_embeddings=True)
BIG = dict(vocab_size=V, hidden_size=96, intermediate_size=160, num_hidden_layers=3, num_attention_heads=6,
           num_key_value_heads=3, head_dim=32, max_position_embeddings=256, tie_word_embeddings=False)


def _model_dir(root, name, mt, kw, weights=True):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    d = os.path.join(root, name)
    cfg = AutoConfig.for_model(mt, **kw)
    if not weights:
        os.makedirs(d, exist_ok=True)
        cfg.architectures = [AutoModelForCausalLM._model_mapping[type(cfg)].__name__]
        cfg.save_pretrained(d)
        return d
    torch.manual_seed(len(name))
    m = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.08)
    m.save_pretrained(d)
    AutoTokenizer.from_pretrained(TOK).save_pretrained(d)
    return d


SW = dict(sliding_window=4)
LT3 = ["sliding_attention", "full_attention", "sliding_attention"]
# full configs (no SMALL / BIG merge) for the families whose head_dim is hidden / heads
NX = dict(vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
          max_position_embeddings=256, tie_word_embeddings=False)
NXB = dict(vocab_size=V, hidden_size=96, intermediate_size=160, num_hidden_layers=4, num_attention_heads=6,
           max_position_embeddings=256, tie_word_embeddings=False)
G2 = dict(vocab_size=V, n_embd=64, n_layer=2, n_head=4, n_positions=256)
G2B = dict(vocab_size=V, n_embd=96, n_layer=3, n_head=6, n_positions=300)
OP = dict(vocab_size=V, hidden_size=64, ffn_dim=96, num_hidden_layers=2, num_attention_heads=4, word_embed_proj_dim=64,
          max_position_embeddings=256)
FA = dict(vocab_size=V, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, ffn_hidden_size=96,
          max_position_embeddings=256, alibi=False, bias=False, tie_word_embeddings=True)
FAB = dict(vocab_size=V, hidden_size=96, num_hidden_layers=4, num_attention_heads=6, ffn_hidden_size=160,
           max_position_embeddings=256, alibi=False, bias=False, tie_word_embeddings=True)
GM = dict(vocab_size=V, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
          max_position_embeddings=256, embedding_multiplier=3.0, residual_multiplier=0.5, logits_scaling=2.0)
GMB = dict(vocab_size=V, hidden_size=96, num_hidden_layers=3, num_attention_heads=6, num_key_value_heads=3,
           max_position_embeddings=256, embedding_multiplier=12.0, residual_multiplier=0.22, logits_scaling=8.0,
           tie_word_embeddings=False)
G4 = dict(GMB, position_embedding_type="rope", num_local_experts=0, attention_multiplier=0.015625)
ERM = dict(moe_intermediate_size=48, moe_k=2, moe_num_experts=8, moe_num_shared_experts=0, moe_layer_start_index=1,
           use_bias=False)
MX = dict(sliding_window=None)
DS = dict(vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
          num_key_value_heads=4, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16, moe_intermediate_size=32,
          n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1, n_group=1,
          topk_group=1, max_position_embeddings=256, tie_word_embeddings=True, rope_scaling=None)
GL = dict(moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
          partial_rotary_factor=0.5, n_group=1, topk_group=1)
CASES = {
    "llama>llama": ("llama", {}, "llama", {}),
    "llama>qwen2": ("llama", {}, "qwen2", {}),
    "qwen2>qwen2": ("qwen2", {}, "qwen2", {}),
    "llama_bias>llama_bias": ("llama", dict(attention_bias=True, mlp_bias=True), "llama", dict(attention_bias=True, mlp_bias=True)),
    "granite>llama": ("granite", dict(embedding_multiplier=3.0, residual_multiplier=0.3, attention_multiplier=0.11,
                                      logits_scaling=2.5), "llama", {}),
    "llama>granite": ("llama", {}, "granite", dict(embedding_multiplier=12.0, residual_multiplier=0.22,
                                                   attention_multiplier=0.015625, logits_scaling=8.0,
                                                   tie_word_embeddings=True)),
    "qwen3>qwen3": ("qwen3", {}, "qwen3", {}),
    "gemma>gemma": ("gemma", {}, "gemma", {}),
    "gemma2>gemma2": ("gemma2", dict(SW, num_hidden_layers=2, layer_types=LT3[:2], query_pre_attn_scalar=16),
                      "gemma2", dict(SW, layer_types=LT3, query_pre_attn_scalar=24)),
    "gemma3>gemma3": ("gemma3_text", dict(SW, layer_types=LT3[:2], rope_local_base_freq=500.0),
                      "gemma3_text", dict(SW, layer_types=LT3, rope_local_base_freq=900.0)),
    "phi3>phi3": ("phi3", dict(pad_token_id=None), "phi3", dict(pad_token_id=None)),
    "llama>phi3": ("llama", {}, "phi3", dict(pad_token_id=None)),
    "mistral_sliding>mistral": ("mistral", SW, "mistral", dict(sliding_window=None)),
    "mistral_sliding>qwen2": ("mistral", SW, "qwen2", {}),
    "llama>mixtral": ("llama", {}, "mixtral", dict(intermediate_size=112, num_local_experts=4, num_experts_per_tok=2,
                                                   sliding_window=None)),
    "llama>qwen2_moe_shared": ("llama", {}, "qwen2_moe", dict(head_dim=16, moe_intermediate_size=48, shared_expert_intermediate_size=128,
                                                              num_experts=4, num_experts_per_tok=2)),
    "qwen3>qwen3_moe_tied": ("qwen3", {}, "qwen3_moe", dict(moe_intermediate_size=112, num_experts=4,
                                                           num_experts_per_tok=2, norm_topk_prob=False)),
    # ERNIE-4.5: interleaved RoPE; MoE host with a selection bias (tied, split over the selected experts, shared)
    "llama>ernie": ("llama", {}, "ernie4_5", {}),
    "ernie>ernie": ("ernie4_5", {}, "ernie4_5", {}),
    "llama>ernie_moe_tied": ("llama", {}, "ernie4_5_moe", dict(ERM, moe_intermediate_size=112)),
    "llama>ernie_moe_split": ("llama", {}, "ernie4_5_moe", dict(ERM, moe_intermediate_size=64)),
    "llama>ernie_moe_shared": ("llama", {}, "ernie4_5_moe", dict(ERM, moe_num_shared_experts=3)),
    # MoE targets: router copy, experts duplicated (m = k_h / k_t), spare experts kept out by a selection bias
    "mixtral>mixtral": ("mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=2),
                        "mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=2)),
    "mixtral>mixtral_x2": ("mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=1),
                           "mixtral", dict(MX, num_local_experts=8, num_experts_per_tok=2)),
    "mixtral>ernie_moe_spare": ("mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=2, intermediate_size=48),
                                "ernie4_5_moe", dict(ERM, moe_num_experts=12, moe_k=4, moe_layer_start_index=0)),
    "qwen2_moe>qwen2_moe": ("qwen2_moe", dict(head_dim=16, moe_intermediate_size=32, shared_expert_intermediate_size=48,
                                              num_experts=4, num_experts_per_tok=2),
                            "qwen2_moe", dict(head_dim=16, moe_intermediate_size=48, shared_expert_intermediate_size=64,
                                              num_experts=4, num_experts_per_tok=2)),
    "qwen3_moe>qwen3_moe_x2": ("qwen3_moe", dict(moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                                                 norm_topk_prob=False),
                               "qwen3_moe", dict(moe_intermediate_size=48, num_experts=8, num_experts_per_tok=4,
                                                 norm_topk_prob=False)),
    "ernie_moe>ernie_moe": ("ernie4_5_moe", dict(ERM, moe_num_shared_experts=1),
                            "ernie4_5_moe", dict(ERM, moe_intermediate_size=64, moe_num_shared_experts=2)),
    "granitemoe>granitemoe_x2": ("granitemoe", dict(GM, num_local_experts=4, num_experts_per_tok=2, intermediate_size=32),
                                 "granitemoe", dict(GMB, num_local_experts=8, num_experts_per_tok=4, intermediate_size=48)),
    "granitemoe>mixtral": ("granitemoe", dict(GM, num_local_experts=4, num_experts_per_tok=2, intermediate_size=32),
                           "mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=2)),
    "llama>granitemoe": ("llama", {}, "granitemoe", dict(GMB, num_local_experts=4, num_experts_per_tok=2,
                                                         intermediate_size=112)),
    # LayerNorm streams, parallel blocks, partial RoPE, learned positions
    "neox>neox": ("gpt_neox", NX, "gpt_neox", NXB),
    "neox_seq>neox_par": ("gpt_neox", dict(NX, use_parallel_residual=False), "gpt_neox", NXB),
    "neox_full>neox_partial": ("gpt_neox", dict(NX, rotary_pct=1.0), "gpt_neox", dict(NXB, rotary_pct=0.5, hidden_size=192)),
    "gpt2>gpt2": ("gpt2", G2, "gpt2", G2B),
    "gpt2>gpt2_inv_layer_scale": ("gpt2", G2, "gpt2", dict(G2B, scale_attn_by_inverse_layer_idx=True)),
    "opt>opt": ("opt", OP, "opt", dict(OP, hidden_size=96, ffn_dim=160, num_hidden_layers=3, num_attention_heads=6,
                                       word_embed_proj_dim=96, max_position_embeddings=300)),
    "stablelm_par>stablelm_par": ("stablelm", dict(NX, num_key_value_heads=2, use_parallel_residual=True, use_qkv_bias=True),
                                  "stablelm", dict(NXB, num_key_value_heads=3, use_parallel_residual=True, use_qkv_bias=True)),
    "stablelm>stablelm_par": ("stablelm", dict(NX, num_key_value_heads=2, use_qkv_bias=True),
                              "stablelm", dict(NXB, num_key_value_heads=3, use_parallel_residual=True, use_qkv_bias=True)),
    "llama>stablelm_mirror": ("llama", {}, "stablelm", dict(NXB, hidden_size=192, num_key_value_heads=3,
                                                            partial_rotary_factor=0.5, num_hidden_layers=3)),
    "glm>glm": ("glm", dict(partial_rotary_factor=0.5, attention_bias=True, pad_token_id=None),
                "glm", dict(partial_rotary_factor=0.5, attention_bias=True, pad_token_id=None)),
    "llama>glm": ("llama", {}, "glm", dict(partial_rotary_factor=0.5, head_dim=64, pad_token_id=None)),
    "olmo>olmo": ("olmo", dict(NX, num_key_value_heads=4), "olmo", dict(NXB, num_key_value_heads=6)),
    "llama>olmo_mirror": ("llama", {}, "olmo", dict(NXB, hidden_size=192, num_key_value_heads=6, num_hidden_layers=3)),
    "falcon_seq_bias>falcon_mha": ("falcon", dict(FA, multi_query=False, parallel_attn=False, bias=True),
                                   "falcon", dict(FAB, multi_query=False, parallel_attn=True, bias=True)),
    "falcon_new1>falcon_new1": ("falcon", dict(FA, new_decoder_architecture=True, num_kv_heads=2, num_ln_in_parallel_attn=1),
                                "falcon", dict(FAB, new_decoder_architecture=True, num_kv_heads=3, num_ln_in_parallel_attn=1)),
    "neox>falcon_new": ("gpt_neox", dict(NX, rotary_pct=1.0), "falcon", dict(FAB, new_decoder_architecture=True,
                                                                            num_kv_heads=6, bias=True,
                                                                            tie_word_embeddings=False)),
    "falcon_new>neox": ("falcon", dict(FA, new_decoder_architecture=True, num_kv_heads=2, activation="gelu"),
                        "gpt_neox", dict(NXB, rotary_pct=1.0)),
    # (the relation reads Falcon-7B's multi_query as MHA, so this one is checked without its gate)
    "falcon7b>falcon_new": ("falcon", dict(FA, multi_query=True, parallel_attn=True, new_decoder_architecture=False),
                            "falcon", dict(FAB, new_decoder_architecture=True, num_kv_heads=2), False),
    # v3: KV-head packing (a longrope table repeating the target's frequencies per packed block), the window lever
    # (a sliding host's window removed for a full target), a no-window host for a sliding target (the mask edge,
    # exact on any context <= the window), and MoE hosts without a selection bias: a constant residual channel
    # selects the experts that hold a dense FFN (+R) or keeps spare experts out (-R); copies holding slices of an
    # expert (m e_h >= e_t); renormalisation set where the code reads it
    "llama>llama_kv_pack": ("llama", {}, "llama", dict(num_attention_heads=4, num_key_value_heads=1, head_dim=32)),
    "llama>qwen2_kv_pack_nope_dims": ("llama", dict(num_attention_heads=4, num_key_value_heads=4, head_dim=16),
                                      "qwen2", dict(num_attention_heads=8, num_key_value_heads=2, head_dim=32)),
    "llama>mistral_window_lever": ("llama", {}, "mistral", dict(sliding_window=256)),
    "llama>qwen2_window_lever": ("llama", {}, "qwen2", dict(use_sliding_window=True, sliding_window=256,
                                                             max_window_layers=1)),
    "mistral_sliding>llama_mask": ("mistral", dict(sliding_window=64), "llama", {}),
    "llama>mixtral_split_const": ("llama", {}, "mixtral", dict(MX, intermediate_size=48, num_local_experts=4,
                                                               num_experts_per_tok=2)),
    "llama>qwen2_moe_split_shared": ("llama", {}, "qwen2_moe", dict(head_dim=16, moe_intermediate_size=32,
                                                                    shared_expert_intermediate_size=32, num_experts=4,
                                                                    num_experts_per_tok=2)),
    "mixtral>mixtral_spare_const": ("mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=2),
                                    "mixtral", dict(MX, num_local_experts=6, num_experts_per_tok=2)),
    "qwen3_moe>qwen3_moe_slices_spare": ("qwen3_moe", dict(moe_intermediate_size=32, num_experts=4, num_experts_per_tok=1,
                                                           norm_topk_prob=True),
                                         "qwen3_moe", dict(moe_intermediate_size=16, num_experts=10,
                                                           num_experts_per_tok=2, norm_topk_prob=True)),
    "qwen3_moe>qwen3_moe_set_norm": ("qwen3_moe", dict(moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                                                       norm_topk_prob=True),
                                     "qwen3_moe", dict(moe_intermediate_size=48, num_experts=6, num_experts_per_tok=2,
                                                       norm_topk_prob=False)),
    "granitemoe>granitemoe_spare": ("granitemoe", dict(GM, num_local_experts=4, num_experts_per_tok=2, intermediate_size=32),
                                    "granitemoe", dict(GMB, num_local_experts=6, num_experts_per_tok=2,
                                                       intermediate_size=48)),
    "llama>seed_oss": ("llama", {}, "seed_oss", dict(attention_bias=True, attention_out_bias=True, mlp_bias=True)),
    "seed_oss>seed_oss": ("seed_oss", dict(attention_bias=True), "seed_oss", dict(attention_bias=True)),
    # GLM-4.5 MoE (sigmoid router, selection bias, routed_scaling_factor, group-limited routing, dense first layers)
    "llama>glm4_moe_split": ("llama", {}, "glm4_moe", dict(GL, n_shared_experts=1, n_group=2, topk_group=1,
                                                          norm_topk_prob=False, routed_scaling_factor=2.5)),
    "glm4_moe>glm4_moe_x2_spare": ("glm4_moe", dict(GL, n_routed_experts=4, num_experts_per_tok=1, use_qk_norm=True,
                                                    routed_scaling_factor=1.5, norm_topk_prob=False),
                                   "glm4_moe", dict(GL, moe_intermediate_size=16, n_routed_experts=10,
                                                    num_experts_per_tok=2, n_shared_experts=2, use_qk_norm=True,
                                                    routed_scaling_factor=2.5, norm_topk_prob=False)),
    "glm4_moe>glm4_moe_groups": ("glm4_moe", dict(GL, n_routed_experts=4, num_experts_per_tok=2, n_group=2, topk_group=1),
                                 "glm4_moe", dict(GL, n_routed_experts=4, num_experts_per_tok=2, n_group=4, topk_group=2,
                                                  moe_intermediate_size=48)),
    # PhiMoE: LayerNorm, SparseMixer top-2 (a zero router weighs the two picks 1/E and 1/(E-1))
    "llama>phimoe_tied_mirror": ("llama", {}, "phimoe", dict(NXB, hidden_size=192, num_key_value_heads=3, intermediate_size=112,
                                                              num_local_experts=4, sliding_window=None, attention_bias=True)),
    "llama>phimoe_kv_pack": ("llama", {}, "phimoe", dict(NXB, hidden_size=128, num_attention_heads=4, num_key_value_heads=1,
                                                          intermediate_size=112, num_local_experts=4, sliding_window=None,
                                                          attention_bias=True)),
    "phimoe>phimoe": ("phimoe", dict(NX, num_key_value_heads=2, num_local_experts=4, sliding_window=None, attention_bias=True),
                      "phimoe", dict(NXB, num_key_value_heads=3, num_local_experts=4, sliding_window=None,
                                     attention_bias=True, lm_head_bias=True)),
    # post-norm-only blocks and the whole-projection qk-norm (OLMo-2 / 3, OLMoE, FlexOlmo)
    "olmo2>olmo2": ("olmo2", {}, "olmo2", {}),
        "olmo3>olmo3": ("olmo3", dict(SW, layer_types=LT3[:2]), "olmo3", dict(SW, layer_types=LT3)),
    "olmoe>olmoe_x2": ("olmoe", dict(intermediate_size=32, num_experts=4, num_experts_per_tok=1, norm_topk_prob=True),
                       "olmoe", dict(intermediate_size=48, num_experts=8, num_experts_per_tok=2, norm_topk_prob=False)),
    "flex_olmo>flex_olmo": ("flex_olmo", dict(intermediate_size=32, num_experts=4, num_experts_per_tok=2, pad_token_id=None),
                            "flex_olmo", dict(intermediate_size=48, num_experts=4, num_experts_per_tok=2,
                                              pad_token_id=None)),
    # hosts with multipliers / extra branches
    # GPT-Neo: unscaled scores, learned positions, global / local layers (attention_types and window_size set)
    "gpt_neo>gpt_neo": ("gpt_neo", dict(vocab_size=V, hidden_size=64, num_layers=2, num_heads=4, max_position_embeddings=256,
                                        attention_types=[[["global", "local"], 1]], window_size=4, intermediate_size=96),
                        "gpt_neo", dict(vocab_size=V, hidden_size=96, num_layers=3, num_heads=6, max_position_embeddings=300,
                                        attention_types=[[["global"], 3]], window_size=256, intermediate_size=160)),
    "gpt_neo_global>gpt2": ("gpt_neo", dict(vocab_size=V, hidden_size=64, num_layers=2, num_heads=4,
                                            max_position_embeddings=256, attention_types=[[["global"], 2]]),
                            "gpt2", G2B),
    # GPT-BigCode (MQA and MHA fused c_attn) and Helium (interleaved RoPE)
    "bigcode_mqa>bigcode_mqa": ("gpt_bigcode", dict(vocab_size=V, n_embd=64, n_layer=2, n_head=4, n_positions=256,
                                                    multi_query=True),
                                "gpt_bigcode", dict(vocab_size=V, n_embd=96, n_layer=3, n_head=6, n_positions=300,
                                                    multi_query=True)),
    "bigcode_mha>bigcode_mqa_pack": ("gpt_bigcode", dict(vocab_size=V, n_embd=32, n_layer=2, n_head=2, n_positions=256,
                                                         multi_query=False),
                                     "gpt_bigcode", dict(vocab_size=V, n_embd=96, n_layer=3, n_head=3, n_positions=300,
                                                         multi_query=True)),
    "gpt2>bigcode_mha": ("gpt2", G2, "gpt_bigcode", dict(vocab_size=V, n_embd=96, n_layer=3, n_head=6, n_positions=300,
                                                       multi_query=False)),
    "helium>helium": ("helium", dict(head_dim=16), "helium", dict(head_dim=16)),
    "llama>helium": ("llama", {}, "helium", dict(head_dim=16)),
    # Apertus: xIELU (per-layer alpha_p / alpha_n copied), qk-norm, ungated MLP
    "apertus>apertus": ("apertus", {}, "apertus", {}),
    # LFM2: gated short-conv layers (hidden channels zero-padded, taps right-aligned) and qk-normed attention layers
    "lfm2>lfm2": ("lfm2", dict(layer_types=["conv", "full_attention"], conv_L_cache=3, block_auto_adjust_ff_dim=False,
                               pad_token_id=None),
                  "lfm2", dict(layer_types=["conv", "full_attention", "conv"], conv_L_cache=4, block_auto_adjust_ff_dim=True,
                               intermediate_size=240, pad_token_id=None)),
    # DeepSeek-V3 MLA: latents held m times (the hard-coded latent-norm eps sees the same rms), heads / nope / v
    # zero-padded, the shared rotary key's pairs on pairs 2 i; its MoE (sigmoid router, selection bias, groups)
    "deepseek_v3>deepseek_v3": ("deepseek_v3", dict(DS, q_lora_rank=16, kv_lora_rank=16),
                                "deepseek_v3", dict(DS, hidden_size=96, num_attention_heads=6, num_key_value_heads=6,
                                                    q_lora_rank=32, kv_lora_rank=48, qk_nope_head_dim=12,
                                                    qk_rope_head_dim=16, v_head_dim=20, num_hidden_layers=3,
                                                    moe_intermediate_size=48, n_routed_experts=8,
                                                    num_experts_per_tok=4, tie_word_embeddings=False)),
    "deepseek_v3_noq>deepseek_v3": ("deepseek_v3", dict(DS, q_lora_rank=None, kv_lora_rank=16, rope_interleave=False),
                                    "deepseek_v3", dict(DS, q_lora_rank=None, kv_lora_rank=32, num_hidden_layers=3,
                                                        rope_interleave=True, tie_word_embeddings=False)),
    # Mamba-1: inner channels, state, dt rank and conv taps zero-padded
    "mamba>mamba": ("mamba", dict(vocab_size=V, hidden_size=64, state_size=8, num_hidden_layers=2, conv_kernel=3,
                                  time_step_rank=8, use_bias=True, tie_word_embeddings=True),
                    "mamba", dict(vocab_size=V, hidden_size=96, state_size=12, num_hidden_layers=3, conv_kernel=4,
                                  time_step_rank=12, use_bias=True, tie_word_embeddings=False)),
    # hybrid hosts: the target's layers on the host's attention layers, the Mamba-2 layers zeroed
    "llama>bamba": ("llama", {}, "bamba", dict(num_hidden_layers=5, attn_layer_indices=[1, 3], mamba_n_heads=6,
                                               mamba_d_head=32, mamba_d_state=8, mamba_chunk_size=8)),
    "llama>granite4_hybrid": ("llama", {}, "granitemoehybrid", dict(G4, shared_intermediate_size=128, num_hidden_layers=5,
                                                                   layer_types=["mamba", "attention", "mamba", "attention",
                                                                                "mamba"],
                                                                   mamba_n_heads=6, mamba_d_head=32, mamba_d_state=8,
                                                                   mamba_chunk_size=8)),
    "llama>granite4": ("llama", {}, "granitemoehybrid", dict(G4, shared_intermediate_size=128, layer_types=["attention"] * 3,
                                                            mamba_n_heads=6)),
    "granite4>granite4": ("granitemoehybrid", dict(G4, shared_intermediate_size=96, layer_types=["attention"] * 2,
                                                   num_hidden_layers=2, mamba_n_heads=4, embedding_multiplier=3.0, residual_multiplier=0.5,
                                                   attention_multiplier=0.1, logits_scaling=2.0),
                          "granitemoehybrid", dict(G4, shared_intermediate_size=128, layer_types=["attention"] * 3,
                                                   mamba_n_heads=6)),
    "llama>cohere": ("llama", {}, "cohere", dict(hidden_size=192, num_hidden_layers=4, num_attention_heads=6,
                                                 num_key_value_heads=3, logit_scale=0.25)),
    "llama>falcon_h1": ("llama", {}, "falcon_h1", dict(mamba_d_ssm=64, mamba_n_heads=4, mamba_d_head=16, mamba_d_state=8,
                                                       mamba_n_groups=1, mamba_chunk_size=8, attention_in_multiplier=0.7,
                                                       key_multiplier=0.3, attention_out_multiplier=0.2,
                                                       mlp_multipliers=[0.5, 0.25], embedding_multiplier=3.0,
                                                       lm_head_multiplier=0.125)),
}


# PhiMoE builds its cos / sin from torch.arange(dtype=float32) inside the rotary forward, which the fp64 lift does not
# reach: against a target whose rotary runs at fp64 the gap is that fp32 phase (~1e-9), not the construction
# (phimoe>phimoe, both sides fp32 there, is at 1e-12)
FP64_TOL = {"llama>phimoe_tied_mirror": 1e-8, "llama>phimoe_kv_pack": 1e-8}


@pytest.fixture(scope="module")
def synth_root():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _kw(mt, kw, base):
    """A synthetic config: merged with SMALL / BIG unless complete; no head_dim for the classes that ignore it."""
    from elephantine.emulate import HD_FROM_HIDDEN
    kw = kw if "vocab_size" in kw else {**base, **kw}
    return {k: v for k, v in kw.items() if not (k == "head_dim" and mt in HD_FROM_HIDDEN)}


@pytest.mark.parametrize("case", list(CASES))
def test_synthetic_family(case, synth_root):
    tmt, tkw, hmt, hkw, *gate = CASES[case]
    t = _model_dir(synth_root, case + "-t", tmt, _kw(tmt, tkw, SMALL))
    h = _model_dir(synth_root, case + "-h", hmt, _kw(hmt, hkw, BIG), weights=False)
    out = os.path.join(synth_root, case + "-out")
    r = emulate(t, h, out, fp64=True, prompts=["The capital of France is", "def f(x):"], new_tokens=8,
                log=lambda *a: None, check_relation=gate[0] if gate else True)
    v = r["verification"]
    if os.environ.get("EMULATE_TEST_LOG"):
        with open(os.environ["EMULATE_TEST_LOG"], "a") as f:
            f.write(json.dumps({"synthetic": case, "fp64_rel": v["fp64"]["rel"], "fp32_rel": v["fp32"]["rel"],
                                "greedy": all(g["identical"] for g in v["fp64"]["greedy"])}) + "\n")
    assert v["fp64"]["rel"] < FP64_TOL.get(case, 1e-12), (case, v["fp64"])
    assert all(g["identical"] for g in v["fp64"]["greedy"]), v["fp64"]
    shutil.rmtree(out, ignore_errors=True)


def test_rope_ratio_not_integer(synth_root):
    """rotated dims 4 -> 6 (GPT-NeoX, rotary_pct 0.25, head_dim 16 -> 24): a per-pair longrope table, exact."""
    t = _model_dir(synth_root, "ratio-t", "gpt_neox", NX)
    h = _model_dir(synth_root, "ratio-h", "gpt_neox", dict(NXB, hidden_size=144), weights=False)
    r = emulate(t, h, os.path.join(synth_root, "ratio-out"), fp64=True, prompts=["def f(x):"], new_tokens=8,
                log=lambda *a: None)
    v = r["verification"]
    assert v["fp64"]["rel"] < 1e-12 and all(g["identical"] for g in v["fp64"]["greedy"]), v["fp64"]


def test_rope_ratio_theta_overflow_regression(synth_root):
    """Spot check (verify/spot/SPOT.md): head_dim 6 -> 128 is a ratio 21.33; rescaling theta (theta^(128 / 126) here,
    theta^(hd_h / hd_t) in the spot check's build) can overflow fp32 in the host's inv_freq and zero every pair past
    the first. The per-pair table (theta 1) is exact; a host without one refuses an overflowing theta."""
    t = _model_dir(synth_root, "ovf-t", "llama", dict(SMALL, hidden_size=12, num_attention_heads=2,
                                                      num_key_value_heads=2, head_dim=6, intermediate_size=24))
    h = _model_dir(synth_root, "ovf-h", "llama", dict(BIG, num_attention_heads=6, num_key_value_heads=6, head_dim=128,
                                                      hidden_size=96), weights=False)
    r = emulate(t, h, os.path.join(synth_root, "ovf-out"), fp64=True, prompts=["def f(x):", "The capital of"],
                new_tokens=8, log=lambda *a: None)
    v = r["verification"]
    assert v["fp64"]["rel"] < 1e-12 and all(g["identical"] for g in v["fp64"]["greedy"]), v["fp64"]
    cfg = json.loads(Path(synth_root, "ovf-out", "config.json").read_text())
    assert cfg["rope_theta"] == 1.0 and cfg["rope_scaling"]["rope_type"] == "longrope"
    from elephantine.emulate import _rope_plan, family_of, resolved
    tr, hr = fetch_config(t), fetch_config(h)
    tc, hc = resolved(tr), resolved(hr)
    out = _rope_plan(tc, family_of(tr, "target"), hc, family_of(hr, "host"), hr, 6, 128, [0, 21, 42], 85.0, {}, [])
    assert isinstance(out, list) and "overflows fp32" in out[0]


def test_generation_config_copied(synth_root):
    """Spot check (verify/spot/SPOT.md): the target's generation_config.json -- here eos ids that differ from its
    config's -- goes into the written checkpoint, so generate() stops as the target does."""
    from transformers import GenerationConfig
    t = _model_dir(synth_root, "gen-t", "llama", _kw("llama", {}, SMALL))
    GenerationConfig(bos_token_id=1, eos_token_id=[2, 7, 11], pad_token_id=0, max_new_tokens=5).save_pretrained(t)
    h = _model_dir(synth_root, "gen-h", "llama", _kw("llama", {}, BIG), weights=False)
    out = os.path.join(synth_root, "gen-out")
    emulate(t, h, out, verify_it=False, log=lambda *a: None)
    g = GenerationConfig.from_pretrained(out)
    assert g.eos_token_id == [2, 7, 11] and g.max_new_tokens == 5 and g.pad_token_id == 0


def test_generation_config_real_dpp_gpt_in_qwen15():
    """The spot check's case: dikiyplayerpig/dpp-gpt-V2.2-Pro-260m in Qwen/Qwen1.5-0.5B, greedy text identical with
    each model's own generation defaults."""
    from transformers import GenerationConfig
    r, _ = _run("dikiyplayerpig/dpp-gpt-V2.2-Pro-260m", "Qwen/Qwen1.5-0.5B", 1e-12, **LIGHT)
    want = GenerationConfig.from_pretrained("dikiyplayerpig/dpp-gpt-V2.2-Pro-260m")
    assert r["generation_config"]["eos_token_id"] == want.eos_token_id


def test_rope_real_unit_test_llama_in_salamandra():
    """The spot check's real cases: MaxJeblick/llama2-0b-unit-test (head_dim 6) in BSC-LT/salamandra-2b (128) and in
    AI4PD/ProtGPT3-MSA (64): the per-pair table, bit-identical inv_freq."""
    for host in ("BSC-LT/salamandra-2b", "AI4PD/ProtGPT3-MSA"):
        r = emulate("MaxJeblick/llama2-0b-unit-test", host, None, plan_only=True, log=lambda *a: None)
        assert r["supported"]
    from elephantine.emulate import plan_construction
    con = plan_construction(fetch_config("MaxJeblick/llama2-0b-unit-test"), fetch_config("BSC-LT/salamandra-2b"))
    assert "bit-identical" in con.rope_note and con.config_changes["rope_theta"] == 1.0


@pytest.mark.parametrize("tmt,tkw,hmt,hkw,needle", [
    # a target mixing full and sliding layers on a host with no window setting (one sliding target is the mask edge)
    ("qwen2", dict(use_sliding_window=True, sliding_window=4, max_window_layers=1), "llama", {}, "no window setting"),
    ("qwen2", {}, "llama", {}, "bias"),                             # q/k/v biases, none on the host
    ("gemma2", dict(query_pre_attn_scalar=16), "gemma2", dict(query_pre_attn_scalar=24, final_logit_softcapping=20.0),
     "softcap"),                                                   # a softcap value the host may not change
    ("llama", dict(intermediate_size=240), "mixtral", dict(intermediate_size=112, num_local_experts=4,
                                                           num_experts_per_tok=2, sliding_window=None), "FFN"),
    # an MoE target needs an MoE host whose routing reproduces its gates
    ("mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=2), "llama", {}, "dense"),
    ("mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=2), "mixtral",
     dict(MX, num_local_experts=6, num_experts_per_tok=2, hidden_size=64), "no selection bias"),   # no free dim
    ("mixtral", dict(MX, num_local_experts=4, num_experts_per_tok=2), "mixtral",
     dict(MX, num_local_experts=8, num_experts_per_tok=3), "multiple"),                # top-k 3 vs 2
    ("qwen3_moe", dict(moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2, norm_topk_prob=False),
     "ernie4_5_moe", dict(ERM, moe_num_experts=8, moe_k=4, moe_layer_start_index=0), "renormalised"),
    # the stream maps: a LayerNorm target on an RMSNorm host; the mirror needs 2d
    ("gpt_neox", NX, "llama", {}, "LayerNorm target on an RMSNorm host"),
    ("llama", {}, "stablelm", dict(NXB, num_key_value_heads=3), "mirror"),
    # blocks: a parallel target on a sequential host
    ("gpt_neox", NX, "gpt_neox", dict(NXB, use_parallel_residual=False), "parallel-block target"),
    # families refused outright, with the reason
    ("llama", {}, "deepseek_v3", dict(q_lora_rank=None, kv_lora_rank=64, qk_rope_head_dim=16, qk_nope_head_dim=16,
                                      v_head_dim=32, n_routed_experts=4, num_experts_per_tok=2), "MLA"),
    ("llama", {}, "olmo2", {}, "post-norm-only"),
    ("olmo2", {}, "llama", {}, "post-norm-only"),
    ("olmoe", dict(num_experts=4, num_experts_per_tok=2), "qwen3_moe", dict(num_experts=4, num_experts_per_tok=2),
     "qk-norm: target variant proj"),
    ("falcon", dict(FA, alibi=True), "falcon", dict(FAB, alibi=True), "ALiBi"),
    ("granitemoehybrid", dict(G4, layer_types=["mamba", "attention"], mamba_n_heads=4), "llama", {}, "Mamba-2"),
])
def test_synthetic_refusals(tmt, tkw, hmt, hkw, needle, synth_root):
    tag = f"ref-{tmt}-{hmt}-{needle.split()[0]}"
    t = _model_dir(synth_root, tag + "-t", tmt, _kw(tmt, tkw, SMALL), weights=False)
    h = _model_dir(synth_root, tag + "-h", hmt, _kw(hmt, hkw, BIG), weights=False)
    rs = support(fetch_config(t), fetch_config(h))
    assert rs and any(needle in r for r in rs), rs


def test_out_dir_is_deleted_only_when_this_tool_wrote_it(tmp_path, monkeypatch):
    """--out never deletes a directory it did not write: an existing non-empty directory without the marker, a
    file, the current directory (or a parent), the home directory, "/" and a source checkout are refused; an empty
    directory or an earlier output (marker present) is reused. The vendored tef_recipes copy behaves the same."""
    import sys
    from elephantine import emulate as em
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "recipes" / "tef_recipes"))
    from tef_recipes import emulate as vend
    for mod in (em, vend):
        base = tmp_path / mod.__name__
        base.mkdir()
        monkeypatch.chdir(base)
        keep = base / "project"
        keep.mkdir()
        (keep / "notes.txt").write_text("mine")
        with pytest.raises(mod.OutDirError, match="not an output"):
            mod.prepare_out(keep)
        assert (keep / "notes.txt").read_text() == "mine"
        (base / "afile").write_text("x")
        with pytest.raises(mod.OutDirError, match="not a directory"):
            mod.prepare_out(base / "afile")
        for bad in (".", base, base.parent, "/", "~", Path.home(), mod.ROOT):
            with pytest.raises(mod.OutDirError, match="refusing"):
                mod.prepare_out(bad)
        repo = base / "checkout"
        (repo / ".git").mkdir(parents=True)
        with pytest.raises(mod.OutDirError, match="refusing"):
            mod.prepare_out(repo)
        assert (base / "afile").exists() and (repo / ".git").is_dir()
        fresh = mod.prepare_out(base / "new" / "out")             # missing: created and marked
        assert (fresh / mod.OUT_MARKER).is_file()
        (fresh / "model.safetensors").write_text("old")
        again = mod.prepare_out(fresh)                             # an earlier output: emptied and re-marked
        assert sorted(p.name for p in again.iterdir()) == [mod.OUT_MARKER]
        empty = base / "empty"
        empty.mkdir()
        assert (mod.prepare_out(empty) / mod.OUT_MARKER).is_file()


def test_cli_refuses_a_foreign_out_dir(tmp_path, capsys):
    """The command line exits with a usage error, before downloading anything, when --out is not its own."""
    from elephantine import emulate as em
    (tmp_path / "keep.txt").write_text("mine")
    with pytest.raises(SystemExit) as e:
        em.main(["a/target", "a/host", "--out", str(tmp_path)])
    assert e.value.code == 2 and "not an output of this tool" in capsys.readouterr().err
    assert (tmp_path / "keep.txt").read_text() == "mine"
