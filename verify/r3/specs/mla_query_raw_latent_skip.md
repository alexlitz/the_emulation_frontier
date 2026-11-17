# mla_query_raw_latent_skip
live 2 / placed 2 (axk2 2); direction over-credit; batch a8_mla_dsv4

## 1. What the code computes
- tf5:models/axk2/modular_axk2.py:285-291 (`AXK2Attention.__init__`): no q_b_proj; `q_a_proj: hidden -> q_lora_rank`, `q_a_layernorm = AXK2RMSNorm(q_lora_rank)` (eps 1e-6 class default, modeling_axk2.py:53), `q_gate_proj = Linear(2 * q_lora_rank -> heads * (qk_head_dim + v_head_dim), bias=False)`.
- forward :327-331: `q_compressed = q_a_proj(x)`; `q_resid = q_a_layernorm(q_compressed)`; `query_gate = q_gate_proj(cat([q_resid, q_compressed]))`, split per head into the query (qk_head_dim) and an elementwise gate (v_head_dim).
- So q = W1 RMSNorm(q_a x) + W2 q_a x: the query is the plain-MLA normed-latent query PLUS a linear, un-normalised rank-q_lora_rank term. The gate half reads the same concat: `attn * sigmoid(gate)` at :396 (the out_gate feature, already mapped via attention_output_gate).
- KV side is DeepSeek-V3.2 MLA (kv_a_layernorm, kv_b_proj, interleaved RoPE, DSA indexer: :293-304, :337-345).
- p5 today: kv = "MLA", q_latent = q_lora_rank (1536), feats out_gate + dsa (p5:model_graph.py:2255-2258, :3315); nothing marks the raw-latent term, so any MLA host with q_latent >= 1536, out_gate and the dims is credited with A.X-K2.
- skt/A.X-K1 (axk1) has the plain q_b_proj (no q_gate_proj in modular_axk1.py: grep empty).

## 2. Classifier change
- reads: model_type axk2 (code fact), kv_lora_rank and q_lora_rank (the latent path exists)
- new: FEATURES key `"mla_q_raw"`, in `_feats`
- detection:
```python
def detect_mla_query_raw_latent_skip(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "axk2" and lm.get("kv_lora_rank") and num(lm.get("q_lora_rank")) > 0:
        return True
    return None
```
- apply note: p5:model_graph.py: add to the `FEATURES` dict (next to "out_gate", :1602) `"mla_q_raw": ("wall", "free", "MLA query that also reads the un-normed q latent (A.X-K2 q_gate_proj over cat(q_a_layernorm(q_a x), q_a x)): a plain MLA query sees only the RMS-normed latent; an A.X-K2 host zeroes the raw half")`, and in `_feats` (after the out_gate block, :1747) `if mt == "axk2" and d.get("kv_lora_rank") and _num(d.get("q_lora_rank")) > 0: f.add("mla_q_raw")`.

## 3. Relation rule
- target side: a target with mla_q_raw needs a host with mla_q_raw (and the usual MLA latent rule: host q_latent >= target's, kv_latent, mla_nope / mla_v)
- host side: an mla_q_raw host plays plain MLA (q_latent > 0) by zeroing W2 (the raw-latent columns of q_gate_proj): exact. Not credited (and not proposed): playing a q_latent = 0 MLA target through W2 alone, which needs q_lora_rank >= min(hidden, target q_cap) and is not met by any snapshot pair.
- kind: wall (target); exact (host)
- cost: none
- free by the config rules: neither; the raw-latent path is code (no switch). The host direction is a weight setting (zero columns), not a config value
- theorem: `bias_off`-style zeroing is trivial (host side, exact: W2 = 0 gives the plain query). needs theorem (target side): "a pre-norm plain-MLA query q = W1 N(A x) cannot equal W1' N(A' x) + W2 A' x for W2 A' != 0 on every input" (the normed term is degree 0 along rays of A' x, the raw term degree 1: same ray argument as `mla_value_ray` / `mla_latent_norm_not_affine`)

## 4. Test fixtures
- flag: skt/A.X-K2-NVFP4 -> true; Baekpica/A.X-K2-VQ-GGUF -> true (both axk2 configs)
- not flag: skt/A.X-K1 (axk1 parent, plain q_b_proj); deepseek-ai/DeepSeek-V3.2 (the base class DeepseekV32Attention); deepseek-ai/DeepSeek-V3; dots-studio/dots3-note-prev (MLA + out gate from x, not from the latent); lightseekorg/kimi-k2.5-eagle3-mla (Kimi MLA)

## Verdict
real (over-credit): A.X-K2 as a target loses every non-axk2 MLA host it is credited from today (out_gate MLA hosts with q_latent >= 1536, kv_latent >= 512: Kimi-style gated MLA hosts, a few); as a host nothing changes (free). Fewer edges, roughly 0-10 pairs (2 targets).
