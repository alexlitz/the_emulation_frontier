# falcon_alibi_slope_scale
live 2 / placed 29 (bloom 18, falcon 11); direction both; batch a5_positions

## 1. What the code computes
- Falcon `tf5:models/falcon/modeling_falcon.py`:
  - `:187`: slopes are `slopes.bfloat16()`, and `alibi = bf16(s)·arange`.
  - `:780-785` (FalconModel.forward): the 4-D mask becomes `alibi / sqrt(hidden_size // num_heads)` (masked).
  - SDPA path (`:380-387`): `softmax(q·k/sqrt(hd) + alibi/sqrt(hd))`, so the effective slopes are bf16(S(n))/sqrt(head_dim).
  - Eager path (`:405-407`): `(q·k + alibi)·inv_norm_factor + attention_mask`, and the mask already holds alibi/sqrt(hd). Eager therefore applies 2·alibi/sqrt(hd), an implementation inconsistency. The relation follows SDPA, the default implementation.
- BLOOM (`tf5:models/bloom/modeling_bloom.py`, hunk f458a8d2145b) computes q·k/sqrt(d) + alibi, which is S(n) unscaled. MPT and Baichuan-13B are unscaled too.
- round2-p5:
  - `alibi_slopes(n, bias_max, bf16)` / `alibi_nests` (p5:model_graph.py:1175-1206) model Falcon as bf16(S(n)) with no 1/sqrt(hd).
  - `_alibi_dims` (`:1209-1222`) has no scale field.
  - Effect: a Falcon ALiBi config with n ≤ 8 heads (bf16 exact) is claimed to nest with BLOOM / MPT of ≥ n heads, and the reverse, although every Falcon slope is sqrt(hd) times smaller. Not fixed in p5.
- Snapshot:
  - tiiuae/falcon-rw-1b: 32 heads, hd 64, slopes bf16(S(32))/8, so about 2^-3.25 ... 2^-11. The 12 slopes below 2^-8 are in no BLOOM / MPT set.
  - fxmarty/tiny-testing-falcon-alibi: 2 heads, hd 16, slopes {2^-4, 2^-8}/4 = {2^-6, 2^-10}. p5 currently nests this with BLOOM's {2^-4, 2^-8} sets.

## 2. Classifier change
- reads: model_type falcon with alibi true; hidden_size, num_attention_heads.
- new: dims field `alibi_scale` (float; 1.0 for BLOOM / MPT / Baichuan, 1/sqrt(head_dim) for Falcon), in `_alibi_dims`. `alibi_slopes(n, bias_max, bf16, scale)` multiplies every slope by scale after the bf16 rounding, and `alibi_nests` compares the 4-tuples.
- detection:
```python
def detect_falcon_alibi_slope_scale(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "falcon" or not lm.get("alibi"):
        return None
    nh = num(lm.get("num_attention_heads") or lm.get("n_head"))
    hd = num(lm.get("hidden_size")) // nh if nh else 0
    return 1.0 / hd ** 0.5 if hd else None
```
- apply note:
  - `_alibi_dims`: add `alibi_scale=(1/sqrt(hidden//n) if mt == "falcon" else 1.0)`, with alibi_scale=0 in the no-ALiBi dict.
  - `alibi_slopes`: take a scale argument, `s = [x*scale for x in s]` after `rb`.
  - emul_matrix's ALiBi <- ALiBi block (`:4463-4479`): pass the 4-tuple `(heads, max, bf16, scale)` to alibi_nests.
  - `_point_key_dims`: add "alibi_scale".
  - Keep ALIBI_REL_TOL 1e-6; the scale is exact in fp32 for power-of-two hd.

## 3. Relation rule
- target side: a Falcon ALiBi target needs a host whose slope set contains {bf16(S(n))/sqrt(hd)}. Only a Falcon ALiBi host with the same head_dim, and ≥ the target's heads where bf16 sets nest, qualifies. BLOOM, MPT and Baichuan hosts are walled. The scale does not fold into weights: the bias is added outside q·k, and the host's slopes are computed from its head count.
- host side: a Falcon ALiBi host plays only Falcon-ALiBi targets of the same scale. BLOOM / MPT targets are walled. NoPE targets are still fine through alibi_off (the alibi flag gives RoPE with inv_freq 0).
- kind: wall both ways (exact only for the same scale and nesting sets).
- cost: none.
- free by the config rules: Falcon's `alibi` flag is a feature switch, already priced by alibi_off (it gives RoPE, not other slopes). head_dim is a shape. Nothing here is free.
- theorem: `alibiSlopes_mono`, `heads_inject` (nesting by head count, same scale). Needs theorem: `alibi_scaled_not_mem`, i.e. for c = 1/sqrt(hd) < 1 with hd a power of 4, c·S(n) ⊄ S(m) for any m when n ≥ 1 (the largest slope c·2^(-8/N) < 2^(-8/M) fails for some slope). Compare `alibi_falcon_slope_not_mem` (the bf16 half).

## 4. Test fixtures
- flag: tiiuae/falcon-rw-1b -> 0.125 (hd 64); fxmarty/tiny-testing-falcon-alibi -> 0.25 (hd 16)
- not flag:
  - tiiuae/falcon-7b (falcon, alibi off: RoPE)
  - bigscience/bloom-560m (unscaled ALiBi)
  - replit/replit-code-v1-3b (MPT ALiBi)

## Verdict
Real, both directions, small.
- The tiny Falcon-ALiBi point loses its claimed BLOOM / MPT hosts (and is junk-like anyway).
- falcon-rw-1b already fails BLOOM nesting through bf16 (n = 32 > 8), so its edges barely change.
- Net: a handful of ALiBi <- ALiBi edges disappear.
