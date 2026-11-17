"""Tiny random-weight validation cases for the framework references, built in the repo's environment.

universal cases: every stock-family case of tests/test_stock_loaders.py (CASES: a small stock transformers
model with random weights, loaded into the universal host by its from_* loader). Each is exported in the
tef-ucore format; the reference is the torch UniversalElephantine itself in fp64 (which the stock-loader tests
check against the family's own transformers code), prompt logits and a greedy continuation.

llama cases: small random Llama-style configs (one per feature: GQA, biases, qk-norm, windows, RoPE
scalings) saved as HF checkpoints; the reference is transformers.

    python -m recipes.cases OUT_DIR [--only CASE ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "recipes", "tef_recipes"))
os.environ.setdefault("USE_TF", "0")

T_PROMPT, N_NEW = 12, 16

LB = dict(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3, num_attention_heads=6,
          num_key_value_heads=2, max_position_embeddings=128)
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}
L3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
      "original_max_position_embeddings": 16}
LLAMA_CASES = {
    "llama-gqa": ("LlamaConfig", "LlamaForCausalLM", dict(LB)),
    "llama-biases-tied": ("LlamaConfig", "LlamaForCausalLM", dict(LB, attention_bias=True, mlp_bias=True,
                                                                  tie_word_embeddings=True)),
    "llama-llama3-rope": ("LlamaConfig", "LlamaForCausalLM", dict(LB, rope_scaling=L3)),
    "llama-linear-rope": ("LlamaConfig", "LlamaForCausalLM", dict(LB, rope_scaling={"rope_type": "linear", "factor": 2.0})),
    "llama-yarn": ("LlamaConfig", "LlamaForCausalLM", dict(LB, rope_scaling=YARN)),
    "mistral-window": ("MistralConfig", "MistralForCausalLM", dict(LB, sliding_window=4)),
    "ministral-layer-types": ("MinistralConfig", "MinistralForCausalLM",
                              dict(LB, head_dim=8, sliding_window=4,
                                   layer_types=["sliding_attention", "full_attention", "sliding_attention"])),
    "qwen2-window": ("Qwen2Config", "Qwen2ForCausalLM", dict(LB, use_sliding_window=True, sliding_window=4,
                                                             max_window_layers=1)),
    "qwen3-qknorm-yarn": ("Qwen3Config", "Qwen3ForCausalLM", dict(LB, head_dim=16, rope_scaling=YARN)),
}


def _ids(vocab, T=T_PROMPT, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, vocab, size=T).tolist()


def _greedy(fwd, ids, n):
    """fwd(list of ids) -> (T, V) numpy logits; repeated full forwards (no cache)."""
    seq = list(ids)
    for _ in range(n):
        seq.append(int(np.argmax(fwd(seq)[-1])))
    full = fwd(seq[:-1])
    T = len(ids)
    return {"ids": np.array(ids), "new": np.array(seq[T:]), "prefill": full[:T], "steps": full[T - 1:]}


def _stock(table):
    def build(case):
        from tests import test_stock_loaders as S
        cfg_name, model_name, loader, kw, opt = getattr(S, table)[case]
        return S._build(cfg_name, model_name, kw, opt), loader
    return build


def _x_mamba(case=None):
    from transformers import MambaConfig, MambaForCausalLM
    from tests.test_universal_elephantine import _rand_
    import torch
    cfg = MambaConfig(vocab_size=97, hidden_size=32, state_size=8, num_hidden_layers=3, expand=2, conv_kernel=4,
                      time_step_rank=4, residual_in_fp32=False, use_mambapy=False)
    return _rand_(MambaForCausalLM(cfg).to(torch.float64).eval()), "from_mamba"


def _x_lfm2(case=None):
    from transformers import Lfm2Config, Lfm2ForCausalLM
    from tests.test_universal_elephantine import _rand_
    import torch
    cfg = Lfm2Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, conv_L_cache=3, conv_bias=True, block_auto_adjust_ff_dim=False,
                     layer_types=["conv", "full_attention", "conv", "full_attention"], attn_implementation="eager")
    torch.manual_seed(0)
    return _rand_(Lfm2ForCausalLM(cfg).to(torch.float64).eval()), "from_lfm"


def _x_qwen3next(case=None):
    from tests.test_universal_elephantine import _qwen3next
    return _qwen3next()[0], "from_qwen3next"


def _x_gemma1(case=None):
    from transformers import GemmaConfig, GemmaForCausalLM
    from tests.test_universal_elephantine import _rand_
    import torch
    cfg = GemmaConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=1, head_dim=12, hidden_act="gelu_pytorch_tanh", attn_implementation="eager")
    return _rand_(GemmaForCausalLM(cfg).to(torch.float64).eval()), "from_gemma"


def _x_qwen35(case=None):                                         # transformers 5 (run with --tf5)
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q, configuration_qwen3_5 as qc
    cfg = qc.Qwen3_5TextConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                               linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4,
                               intermediate_size=48)
    t = q.Qwen3_5ForCausalLM(cfg).to(torch.float64).eval()
    for l in t.model.layers:                              # CPU torch paths (fla kernels are GPU Triton)
        g = getattr(l, "linear_attn", None)
        if g is not None:
            g.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
            g.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
            g.causal_conv1d_fn = None
    torch.manual_seed(0)
    with torch.no_grad():
        for p_ in t.parameters():
            p_.add_(torch.randn_like(p_) * 0.05)
    return t, "from_qwen35"


def _x_gpt2(case=None):
    from transformers import GPT2Config, GPT2LMHeadModel
    from tests.test_universal_elephantine import _rand_
    import torch
    cfg = GPT2Config(vocab_size=97, n_embd=48, n_layer=3, n_head=4, n_positions=32, attn_implementation="eager")
    return _rand_(GPT2LMHeadModel(cfg).to(torch.float64).eval()), "from_gpt2"


def _x_gpt_neox(case=None):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from tests.test_universal_elephantine import _rand_
    import torch
    cfg = GPTNeoXConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3, num_attention_heads=6,
                        rotary_pct=0.25, hidden_act="gelu", attn_implementation="eager")
    return _rand_(GPTNeoXForCausalLM(cfg).to(torch.float64).eval()), "from_gpt_neox"


def _x_apertus(case=None):
    from transformers import ApertusConfig, ApertusForCausalLM
    from tests.test_universal_elephantine import _rand_
    import torch
    cfg = ApertusConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3, num_attention_heads=6,
                        num_key_value_heads=2, attn_implementation="eager")
    return _rand_(ApertusForCausalLM(cfg).to(torch.float64).eval()), "from_apertus"


EXTRA = {"gpt2": _x_gpt2, "gpt_neox": _x_gpt_neox, "apertus": _x_apertus, "mamba": _x_mamba, "lfm2": _x_lfm2, "qwen3_next": _x_qwen3next, "gemma-geglu": _x_gemma1}
EXTRA_TF5 = {"qwen3_5-text": _x_qwen35}


def universal_case(case, out_dir, builder=None):
    import torch
    import elephantine.universal as U
    from tef_recipes.ucore_export import save
    t, loader = (builder or _stock("CASES"))(case)
    host = getattr(U, loader)(t).eval()
    d = os.path.join(out_dir, "universal", case)
    from tef_recipes.ucore_export import Unsupported
    try:
        meta = save(d, host, extra={"case": case, "loader": loader, "model_type": t.config.model_type})
    except Unsupported as e:                              # a branch outside the tef-ucore core: recorded
        return {"case": case, "loader": loader, "model_type": t.config.model_type, "refused": str(e)}
    vocab = host.cfg.vocab
    ids = _ids(min(vocab, 97))

    def fwd(seq):
        with torch.no_grad():
            return host(torch.tensor([seq])).double()[0].numpy()
    ref = _greedy(fwd, ids, N_NEW)
    np.savez(os.path.join(d, "ref.npz"), **ref)
    return {"case": case, "dir": d, "loader": loader, "model_type": t.config.model_type, "n_params": meta["n_params"]}


def llama_case(case, out_dir):
    import torch
    import transformers as tr
    cfg_name, model_name, kw = LLAMA_CASES[case]
    torch.manual_seed(0)
    cfg = getattr(tr, cfg_name)(**kw, attn_implementation="eager")
    m = getattr(tr, model_name)(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.add_(torch.randn_like(p) * 0.05)
    d = os.path.join(out_dir, "llama", case)
    m.save_pretrained(d, safe_serialization=True)
    ids = _ids(97)

    def fwd(seq):
        with torch.no_grad():
            return m(torch.tensor([seq]), use_cache=False).logits[0].double().numpy()
    ref = _greedy(fwd, ids, N_NEW)
    np.savez(os.path.join(d, "ref.npz"), **ref)
    return {"case": case, "dir": d, "model_type": cfg.model_type}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--kind", choices=["universal", "llama", "both"], default="both")
    ap.add_argument("--tf5", action="store_true", help="the transformers-5 cases (run with transformers 5.18 on PYTHONPATH)")
    a = ap.parse_args(argv)
    if a.tf5:                                            # transformers 5.18 unpacked on PYTHONPATH (see TF5_PRELUDE)
        import types
        m = types.ModuleType("transformers.dependency_versions_check")
        m.dep_version_check = lambda *x, **k: None
        sys.modules["transformers.dependency_versions_check"] = m
        sys.modules["fla"] = None
    import torch
    torch.set_num_threads(2)
    todo = []
    if a.tf5:
        from tests.test_stock_loaders import TF5_CASES
        todo += [("universal", c, _stock("TF5_CASES")) for c in TF5_CASES]
        todo += [("universal", c, b) for c, b in EXTRA_TF5.items()]
    else:
        from tests.test_stock_loaders import CASES
        if a.kind in ("llama", "both"):
            todo += [("llama", c, None) for c in LLAMA_CASES]
        if a.kind in ("universal", "both"):
            todo += [("universal", c, None) for c in CASES]
            todo += [("universal", c, b) for c, b in EXTRA.items()]
    if a.only:
        todo = [t for t in todo if t[1] in a.only]
    if a.tf5:
        a.kind = "universal"
    index_path = os.path.join(a.out, "cases.json")
    index = json.load(open(index_path)) if os.path.isfile(index_path) else {}
    for kind, c, b in todo:
        try:
            r = llama_case(c, a.out) if kind == "llama" else universal_case(c, a.out, b)
            r["kind"] = kind
            print("ok  ", kind, c, flush=True)
        except Exception as e:
            r = {"case": c, "kind": kind, "error": f"{type(e).__name__}: {e}"}
            print("FAIL", kind, c, r["error"], flush=True)
            if not str(e).startswith(("layer", "model", "ffn")):
                traceback.print_exc()
        index[f"{kind}/{c}"] = r
        os.makedirs(a.out, exist_ok=True)
        json.dump(index, open(index_path, "w"), indent=1)


if __name__ == "__main__":
    main()
