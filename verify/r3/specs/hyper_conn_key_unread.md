# hyper_conn_key_unread
live 2 / placed 2 (needle 2); direction over-credit; batch a8_mla_dsv4

## 1. What the code computes
(ext:needle/gh__cactus-compute__cactus = $NOCODE_CACHE/needle/gh__cactus-compute__cactus/, the training / reference code; the hub configs are descriptive: `extras.mhc_lanes: 4`, hub__Cactus-Compute__needle2/config.json:22, needle3/config.json:46)
- config.py:42 `mhc_lanes: int = 4` (TransformerConfig default; the run config comes from checkpoint metadata, checkpoints.py:72)
- architecture.py:477-491 (Stack): `n = mhc_lanes`, per-layer phi_pre / phi_post (L, n d, n), phi_res (L, n d, n^2), biases (b_res init 4 I, :129-130), scalars a_* (0.01), and FIXED code constants `pre_off = 8 * onehot(l mod n) - 4`, `post_off = -4 * (1 - onehot)`; :492 the residual is broadcast to n lanes; :541 `x = mean(x, axis=2)` before the final ZC-RMSNorm.
- _ScanBody (per layer): nx = weight-free rms of the flattened lanes (`_rms_unit`, :112), u = sum sigmoid(a_pre nx phi_pre + b_pre + pre_off) * lanes; lanes <- Sinkhorn_20(a_res nx phi_res + b_res) @ lanes + 2 sigmoid(a_post nx phi_post + b_post + post_off) * (Block(u) - u).
- architecture.py:121-126 `_sinkhorn(logits, iters=20)`: log-domain row then column logsumexp, exp at the end.
- SAN (ext:san/hub__kenpeter123__x-small/san_model.py, "faithful PyTorch port of needle"): :53 `mhc_lanes: int = 4`, :445-461 the same parameters and lane offsets, :145 `_sinkhorn(logits, iters=14)`; its config carries `mhc_lanes` at the top level (config.json:21).
- p5 today: `_feats` tests hc_mult / hc_count / mhc / mhc_enabled / mhc_expansion_rate / hc_rate / hc_magnitude (p5:model_graph.py:1757-1760): neither `extras.mhc_lanes` (needle) nor top-level `mhc_lanes` (san) is read, so both get no hyper_conn feature and are credited to / from single-stream models.

## 2. Classifier change
- reads: extras.mhc_lanes (needle; code default 4 when absent), mhc_lanes (san; default 4)
- new: FEATURES "hyper_conn" (existing) + the hyper_conn variant "needle" / "san" (hyper_conn_variant), in `_feats`
- detection:
```python
def detect_hyper_conn_key_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "needle":
        ex = lm.get("extras") if isinstance(lm.get("extras"), dict) else {}
        n = num(ex.get("mhc_lanes")) if "mhc_lanes" in ex else 4
        return n if n > 1 else None
    if model_type == "san":
        n = num(lm.get("mhc_lanes")) if "mhc_lanes" in lm else 4
        return n if n > 1 else None
    return None
```
- apply note: p5:model_graph.py `_feats` hyper_conn test (:1757): add `or (mt == "needle" and _num((d.get("extras") or {}).get("mhc_lanes", 4)) > 1) or (mt == "san" and _num(d.get("mhc_lanes", 4)) > 1)`; extend the FEATURES "hyper_conn" description with needle / SAN lanes. With hyper_conn_variant applied, HC_VARIANT_BY_TYPE gives "needle" / "san".

## 3. Relation rule
- target side: a needle / SAN target needs a hyper_conn host of the same variant (only needle hosts for needle; SAN only itself: 14 vs 20 Sinkhorn iterations are code constants, not config)
- host side: a needle host plays no single-stream target (the lanes and fixed lane offsets cannot be collapsed to one stream exactly: the Sinkhorn comb reaches the identity only as a limit and pre / post are sigmoids, FEATURES :1609 rationale)
- kind: wall both ways
- cost: none
- free by the config rules: none (lane count is a shape; iterations are code constants here)
- theorem: needs theorem (hyper_conn is OPEN in PROOFS.md "Walls of the edge-case audit": "hyper_conn, attn_res (both) | -- | OPEN")

## 4. Test fixtures
- flag: Cactus-Compute/needle2 -> 4; Cactus-Compute/needle3 -> 4 (the live); kenpeter123/x-small -> 4 (SAN, the same code ported, top-level mhc_lanes: a third carrier the scan finds)
- not flag: deepseek-ai/DeepSeek-V4-Flash (hc_mult, already read); PeetPedro/qwen4_exp-8b-... (hc_count, already read); Advisio/Radha-9B (no HC); Motif-Technologies/Motif-3 (mhc_enabled, already read)

## Verdict
real (over-credit): needle2 / needle3 / SAN gain hyper_conn, so every edge they have with single-stream models disappears (both directions; they are small GQA models, so as targets they are played by many hosts today). Fewer edges: on the order of hundreds of pairs (3 small targets x the hosts that fit them, plus their host edges); no new edges.
