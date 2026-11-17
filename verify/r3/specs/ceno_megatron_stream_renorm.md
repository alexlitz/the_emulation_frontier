# ceno_megatron_stream_renorm
live 12 / placed 12 (ceno 12); direction over-credit; batch a4_norms

## 1. What the code computes
(CladeTeam/CENO-1.3B-A450M-1m/modeling_ceno.py, snapshot 6806109b…)
- CENOBlock builds `self.norm` (input norm; None for Mamba blocks under `mamba_in_proj_layernorm`, true in every snapshot config) and, for attention / mlp / moe blocks, a second `pre_mlp_norm = CENORMSNorm(hidden, eps=layer_norm_epsilon)` (:938-961).
- mamba block: `residual = x; out = residual + mamba(norm(x))` (:983-997): plain pre-norm.
- attention block: `hidden_after_attn = residual + attn_out; hidden_states = self.pre_mlp_norm(hidden_after_attn)` -- "final output = pre_mlp_layernorm(x + attention(input_layernorm(x)))", no second residual (:999-1025). The stream leaving the block is a normed one: post-LN.
- mlp block: `input_ln_out = self.norm(x); residual = input_ln_out; out = residual + mlp(pre_mlp_norm(input_ln_out))` (:1027-1035); the dense CENOMLP adds a third norm `fc1_norm` on its input (use_fc1_norm=True at :956, built at :1068 / :1082; experts use_fc1_norm=False, :1152).
- moe block: `residual = input_ln_out = self.norm(x); out = residual + moe(pre_mlp_norm(input_ln_out))` (:1037-1062): the residual is the NORMED input -- the stream is replaced by norm(x).
- Every snapshot hybrid_override_pattern has '*' (attention) and 'E' (MoE) blocks (e.g. 'MEM*EMEME'), so every CENO model renormalises its stream several times. p5 classifies CENO as nemotron_h-like hybrid: block sequential, feats rf_mamba2 (NEMOTRON_H_TYPES includes "ceno", p5:model_graph.py:455); its qkv_input qk-norm is already read (QKNORM_VARIANT_BY_TYPE "ceno": "qkv_input", :1886).
- The stock Nemotron-H block (NemotronHBlock) is plain pre-norm: CENO is not.

## 2. Classifier change
- reads: model_type ceno; hybrid_override_pattern / layers_block_type.
- new: FEATURES `post_ln` (existing key, ("wall", "wall")), set in `_feats` via a new POST_LN_BY_CODE = {"ceno"} when any attention / moe / mlp block exists.
- detection:
```python
def detect_ceno_megatron_stream_renorm(cfg, model_type):
    if model_type != "ceno":
        return None
    lm = lm_of(cfg)
    pat = lm.get("hybrid_override_pattern")
    lbt = lm.get("layers_block_type")
    if isinstance(pat, str) and any(c in pat for c in "*E-"):
        return "post_ln"
    if isinstance(lbt, list) and any(str(x) in ("attention", "moe", "mlp") for x in lbt):
        return "post_ln"
    return None
```
- apply note: p5:model_graph.py `_feats`, the post_ln test (~1824: `if (mt == "opt" and ...) or mt == "openai-gpt"`): add `or (mt in POST_LN_BY_CODE and <pattern has * / E / ->)`. Table POST_LN_BY_CODE beside POST_ONLY_TYPES (~1975). The fc1_norm triple norm is subsumed (no separate key).

## 3. Relation rule
- target side: a CENO target needs a host that replaces its stream by a normed one at the same blocks: only post_ln hosts (OPT-350m, GPT-1, MiniMax after norm_placement_keys_unread) -- none of them hybrid Mamba-2, so no host in practice.
- host side: a CENO host cannot keep an un-normed stream through its attention / MoE blocks: it plays no pre-norm target.
- kind: wall both ways.
- cost: none.
- free by the config rules: none; the block layout is code (pattern letters are a feature switch, not free).
- theorem: needs theorem (post_ln has no Lean row; FEATURES text "a pre-norm host cannot replace its stream by a normed one"). Shape of the statement: no pre-norm stack reproduces x -> N(x + f(N x)) for all x (the output's RMS is pinned to ~1 per token; a pre-norm block's output RMS varies with |x|).

## 4. Test fixtures
- flag: all 12 CENO repos (80M / 300M / 600M / 1.3B x base / 131k / 1m) -> "post_ln".
- not flag: agurung/cobalt-seeded-rl-base-ramp25-stoppen-gen4k-ep2-ncp5-n3nc-base-s3 (nemotron_h: same hybrid pattern grammar, plain pre-norm NemotronHBlock).
- scan: exactly the 12 CENO repos.

## Verdict
real. Fewer edges: the 12 CENO points lose every host (as targets) and every target (as hosts) outside the post_ln class; today they are credited with Nemotron-H-style hybrid partners (Nemotron-H / Bamba / Granite-4.0-H hosts and targets).
