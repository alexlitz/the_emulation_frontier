# backout_residual
live 1 / placed 1 (gaggle 1); direction over-credit; batch a4_norms
(the detector also flags the 7 nanochat_gpt repos: the same backout path under config backout_layer, missed by the sweep)

## 1. What the code computes
- Gaggle (castorini/gaggle-nanochat-pretrained-climbmix-20260924/modeling_gaggle.py): `resid_lambdas` (ones), `x0_lambdas` (zeros), `smear_lambda`, and `backout_lambda = nn.Parameter(0.2 * torch.ones(1))` (:164-168), unconditionally. Forward: `x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0` (:297); `backout_layer = self.config.n_layer // 2` (:293); `if i == backout_layer: x_backout = x` (:307-308); `x = x - self.backout_lambda * x_backout` (:309-310); then `x = rmsnorm(x)` (:311) and lm_head.
- nanochat_gpt (jkminder/d16_optimized_537m_seed5/modeling_nanochat_gpt.py): "backout: subtract a scaled mid-layer residual before the final norm" (:20); `if config.backout_layer is not None: self.backout_lambda = nn.Parameter(torch.zeros(1))` (:343-344); `x_backout = x` at `i == backout_layer` (:477, :492-493), `x = x - self.backout_lambda * x_backout` (:494-496). Every snapshot nanochat_gpt config sets backout_layer (6 / 8 / 10, e.g. config.json:10). Init 0, but trained.
- The readout sees x_L - lambda x_m. The x0 / resid lambdas and smear are already FEATURES value_emb (p5 `_feats` ~1755: use_x0_lambdas / use_smear / value_embedding_layers); nothing names the backout path.
- Expanding, x_L - lambda x_m = (1 - lambda) x_m + sum_{i >= m} w_i: the early writers need weight (1 - lambda) at the readout but 1 for every layer >= m reading them; one stream cannot carry both without extra channels every later pre-norm sees.
- Not present: transformers nanochat (tf5:models/nanochat/modeling_nanochat.py, no backout), sn38-nanochrono (modeling_nanochrono.py, none).

## 2. Classifier change
- reads: model_type gaggle (code); nanochat_gpt `backout_layer`.
- new: FEATURES key `backout` ("wall", "free"), set in `_feats`.
- detection:
```python
def detect_backout_residual(cfg, model_type):
    if model_type == "gaggle":
        return True
    if model_type == "nanochat_gpt" and get(cfg, "backout_layer") is not None:
        return True
    return None
```
- apply note: p5:model_graph.py FEATURES (after "value_emb", ~1620); `_feats` beside the value_emb test (~1755): `if mt == "gaggle" or (mt == "nanochat_gpt" and d.get("backout_layer") is not None): f.add("backout")`. Separate from value_emb: value_emb hosts without backout exist (transformers nanochat has neither; other value-residual families).

## 3. Relation rule
- target side: a backout target needs a backout host whose mid layer can be chosen to match: the backout layer index is a config value on nanochat_gpt (free: shape-preserving), fixed at n_layer // 2 on Gaggle (code; matches only when the target's m equals the host's n_layer // 2 after layer alignment).
- host side: backout_lambda = 0 makes the readout x_L: the plain model, exact (nanochat_gpt can also set backout_layer None).
- kind: wall (target side, unless the host has backout) / exact (host side).
- cost: none.
- free by the config rules: lambda is a weight (0 = off); nanochat_gpt's backout_layer is a config switch -- treated like a norm on/off flag (free) for hosts; for targets the path is a feature, not free.
- theorem: host side: needs theorem `backout_off` (lambda = 0 -> x_L; trivial, cf. `value_emb_off`). Target side: OPEN (no construction; same class as the "switch_lora / engram / ple / value_emb target <- plain host" OPEN row, PROOFS "Round-2 rules").

## 4. Test fixtures
- flag: castorini/gaggle-nanochat-pretrained-climbmix-20260924; jkminder/d16_optimized_537m_seed5, jkminder/d12_optimized_286m_seed2, jkminder/d20_optimized_897m_seed8 (backout_layer 8 / 6 / 10) -> True.
- not flag: l0ulan/chinese-babylm-nanochat-d22-step10000 (transformers nanochat: no backout); kingbear329/sn38-r16-c12-step687-alpha05 (sn38-nanochrono: nanochat-derived, no backout).
- scan: gaggle 1 + nanochat_gpt 7.

## Verdict
real, small. Fewer edges: the 8 backout targets (Gaggle, 7 nanochat_gpt) lose value_emb hosts without backout; since every value_emb host in the snapshot that can play them is nanochat_gpt / Gaggle (both backout), the visible change is near 0 today, but it pins the rule for future plain value-residual hosts. Hosts unchanged. Region: recurrent-other (residual stream).
