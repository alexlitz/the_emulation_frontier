# attn_res_keys_unread
live 4 / placed 4 (arloop 2, gpts3_hybrid 1, klm 1); direction over-credit; batch a12_extra_inputs

## 1. What the code computes
- GPT-S3-3M (AxiomicLabs/GPT-S3-3M/modeling_gpts3.py): `AttnResSources` stores every source and its `rsqrt(mean(s^2) + attn_res_eps)` (:313-326); `AttnResAggregator`: learned query (zeros) * key-norm weight, `softmax_i(<q, s_i> * r_i)` over ALL earlier sources, output `sum_i w_i s_i` (:327-349); block: `h = attn(ln_1(attn_res_1(sources)))`, append h, append `mlp(ln_2(attn_res_2(sources)))` -- no residual sum (:351-371); model: `sources = [wte(x)]`, final `hidden = attn_res_f(sources)` (:409, :480-484). Every sublayer output is its own source.
- KLM-1.5 (MinimaLabs/KLM-1.5/modeling_klm.py): `AttnResSources.mix` = softmax of `query . rms(source)` over the sources, the current mixer output joining as `partial` for the FFN (:277-308); queries `attn_res_query * attn_res_norm`, 2L + 1 consumers (:318-320, :337); sources: embedding, n-gram memory, then one per layer `out + ffn` (:338-351); readout `norm(mix(queries[-1]))` (:352).
- ARLoop (osunlp/ARLoop-1.2B/modeling_arloop.py; 0.6B identical): `_route` = softmax over `[blocks..., partial]` of `<v, q * w> * rsqrt(mean v^2 + eps)`, keeping `values[0]` (the embedding) and the last `sw_blocks` blocks (:131-148); a new block starts every `attnres_block_size // 2` layers (:196-198); per-sublayer queries (:112-116, :194-203), final route (:206). Snapshot `attnres_block_size` 18 (0.6B) / 24 (1.2B), `sw_blocks` 3.
- p5 `_feats` attn_res keys: `attn_res_block_size`, `block_attn_res_block_size`, `attn_res_boundaries`, `attention_residuals` (p5:model_graph.py:1768-1770). GPT-S3 has only `attn_res_eps`, ARLoop spells `attnres_block_size`, KLM has none.
- PreNormLoop (arloop_baseline) is the plain-residual control: `h = h + attn(...)`, `h = h + mlp(...)` (osunlp/PreNormLoop-0.6B/modeling_baseline.py:169-170) -- no AttnRes.

## 2. Classifier change
- reads: model_type; ARLoop `attnres_block_size`, `sw_blocks`
- new: FEATURES `attn_res` (existing) for these types; dims `attn_res_sig` = [sublayers per source block, window in blocks (0 = all)], in `_feats` / `_pass5_dims`
- detection:
```python
def detect_attn_res_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "gpts3_hybrid":
        return [1, 0]
    if model_type == "klm":
        return [2, 0]
    if model_type == "arloop":
        bs = num(lm.get("attnres_block_size"))
        return [bs, num(lm.get("sw_blocks"))] if bs > 0 else None
    return None
```
- apply note: in `_feats` (p5:model_graph.py:1768) add `or _num(d.get("attnres_block_size")) > 0 or mt in ("gpts3_hybrid", "klm")`. Record `attn_res_sig` in `_pass5_dims` and require equality on attn_res pairs in emul_matrix's feats mask (the feature is wall/wall, so only attn_res <-> attn_res pairs survive, and they are exact only for the same source partition and window). The existing key path (Kimi `attn_res_block_size`, kotodama boundaries, alice_ai, cyrah) needs its sig too; its unit (layers vs sublayers) is NOT checked here -- until it is, give those `[-1, -1]` (equal only to themselves).

## 3. Relation rule
- target side: a host with attn_res and the same sig (same source partition, same window, a softmax readout); a plain residual is no softmax mix.
- host side: wall -- the depth softmax cannot be set to a plain sum (weights sum to 1; the zero query gives the mean, not the sum).
- kind: wall both ways; exact attn_res <- attn_res with equal sig (same code).
- cost: none.
- free by the config rules: eps free; block size / window change the sources (not free).
- theorem: OPEN (PROOFS.md "hyper_conn, attn_res (both)").

## 4. Test fixtures
- flag: AxiomicLabs/GPT-S3-3M -> [1, 0]; MinimaLabs/KLM-1.5 -> [2, 0]; osunlp/ARLoop-0.6B -> [18, 3]; osunlp/ARLoop-1.2B -> [24, 3].
- not flag: osunlp/PreNormLoop-0.6B (plain residual control); moonshotai/Kimi-Linear-48B-A3B-Instruct, LuxiaSL/kotodama-tsumugi-1b, AMAImedia/AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF, Lilgarca/my-model-v01 (already attn_res by keys); summerAIO/Cyber-9B-Qwen-AllGDN-Engram.
- scan: exactly the 4 live repos.

## Verdict
real. Fewer edges, both sides: GPT-S3, KLM, ARLoop-0.6B / 1.2B stop playing and being played by every plain-residual model (4 models; GPT-S3 / KLM are small hosts with many targets today). With the sig, no new attn_res <-> attn_res edges appear between the variants.
