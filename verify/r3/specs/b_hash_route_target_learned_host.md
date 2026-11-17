# b_hash_route_target_learned_host
live 20 / placed 17 (hash targets: deepseek_v4 19, memt_deepseek_v4 1; all scoring_func sqrtsoftplus); direction over-wall (claimed); batch bd

## 1. What the code computes
- Hash layer (tf5:models/deepseek_v4/modeling_deepseek_v4.py:1110-1121): `logits = F.linear(flat, self.weight)`, `scores = self.score_fn(logits)` with `score_fn = ACT2FN[config.scoring_func]` (:1107), `indices = self.tid2eid[input_ids]`, `weights = scores.gather(1, indices)`, `weights / (weights.sum(-1) + 1e-20)`, times `routed_scaling_factor`. Class default `scoring_func = "sqrtsoftplus"` (configuration_deepseek_v4.py:151); every snapshot hash target uses it.
- The learned router a host has (tf5 TopKRouter :1083-1090, or any softmax top-k router): `topk(score(l) + bias)` then gather + renormalise.
- Construction (PROOFS.md Round-2 rules, "hash_route: hash target <- learned-router host"): extra embedding channels hold a k-hot code of each token's hashed experts per hash layer (no sublayer writes them; their fixed norm k c^2 is absorbed by `const_norm_carry`); the layer-l router reads its code at scale lambda > twice the logit range, so the top-k is exactly tid2eid[token] (`topk_code_select`, `code_margin`), on bounded contexts. The selected logits all carry the same shift lambda c: a renormalised SOFTMAX gate is shift-invariant (`renorm_softmax_shift`), so the weights are the target's; sqrt(softplus) is not (0.4208 / 0.5792 at (0, 1) vs 0.4939 / 0.5061 at (20, 21)).
- p5 relation: `FEATURES["hash_route"] = ("wall", "wall", ...)` (p5:model_graph.py:1710-1713); router score is priced by moe_why (MOE_PRICED_FEATS :2105).

## 2. Classifier change
- reads: the hash-target fact (b_hash_route_host's detector) and the target's gate score (`scoring_func`, V4 default sqrtsoftplus) and renormalisation (V4: always, :1120).
- new: none applied (keep the wall: no snapshot target qualifies). Ready rule: FEATURES["hash_route"] target side becomes conditional -- a dims field `hash_score` (the target's score, "" if not hash) read by emul_matrix: target hash_route with hash_score == "softmax" and renorm, on a learned-router host whose own score is renormalised softmax, is exact at a cost.
- detection:
```python
def detect_b_hash_route_target_learned_host(cfg, model_type):
    if not detect_b_hash_route_host(cfg, model_type):
        return None
    lm = lm_of(cfg)
    sf = str(lm.get("scoring_func") or ("sqrtsoftplus" if model_type in _V4_TYPES else "softmax")).lower()
    return sf
```
- apply note: only if a softmax hash target appears: p5:model_graph.py `feats_ok` (:2108) cannot express a pair condition; add a per-pair mask beside the feature check in emul_matrix (:4234) that clears the hash_route target wall when T.hash_score == "softmax", T renorm, H score softmax renorm, and charges `add_h += n_hash_layers * E_t`.

## 3. Relation rule
- target side: hash target on a learned-router MoE host: selection exact (code channels), weights exact iff both scores are renormalised softmax; V4's sqrtsoftplus: OPEN (no construction, no impossibility proof).
- host side: unchanged (b_hash_route_host).
- kind: exact (softmax, renorm, bounded context) | wall (sqrtsoftplus / sigmoid: OPEN).
- cost: hidden + n_hash_layers x E_t channels (one k-hot block per hash layer: tables differ per layer; V4-Flash 3 x 256 = 768, V4-Pro 3 x 384); the host's router must have E_h >= m E_t and the MoE rules as usual.
- free by the config rules: weights only (embedding channels, router columns); no config switch.
- theorem: `topk_code_select`, `code_margin`, `renorm_softmax_shift` (PROOFS.md Round-2 rules); needs theorem for V4: a construction (or impossibility) for a sqrtsoftplus renormalised gate under a common logit shift -- e.g. a per-expert compensating code value is impossible since the shift must dominate the logit range.

## 4. Test fixtures
- flag: deepseek-ai/DeepSeek-V4-Flash-0731, deepseek-ai/DeepSeek-V4-Pro, MemTensor/MemT-1.0-Flash, endless-frontier/Fx-Bio -> "sqrtsoftplus" (hash targets whose score keeps the rule OPEN).
- not flag: HuggingFaceTB/nanowhale-100m-base (num_hash_layers 0); deepseek-ai/DeepSeek-V4.1-Flash (no hash routing); deepseek-ai/DeepSeek-V3, Qwen/Qwen3-30B-A3B (learned routers only).

## Verdict
non-issue for the snapshot: the exact rule needs a renormalised-softmax hash target and every hash target is
V4 sqrtsoftplus (OPEN), which also carries hyper_conn (target-side wall). 0 edges; keep the wall, record the rule.
