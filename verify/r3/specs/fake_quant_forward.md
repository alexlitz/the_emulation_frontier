# fake_quant_forward
live 2 / placed 2 (<none> 1, minicpm 1); direction over-credit; batch a3_placement

## 1. What the code computes
- openbmb/BitCPM-CANN-0.5B-unquantized/modeling_minicpm.py:156-162 `LinearQuantizer(nn.Linear)`: every forward uses
  `weight_tensor = self.quantizer(self.weight)`; quant_type "ternary" (:78) -> SteTernaryQuantizer :129-140:
  `s = 1 / mean|W_row|` (clamp 1e-5), `W_q = clamp(round(W s), -1, 1) / s`: each output row is ternary {-a, 0, +a}.
- Used for q / k / v / o (:458-461) and gate / up / down (:369-371): every projection of every layer.
- Same in BitCPM-CANN-8B-unquantized/modeling_minicpm.py:149-155, :866-868, :943-946, and (not in the carriers, same
  code fact) BitCPM-CANN-1B-unquantized / 3B-unquantized/modeling_llama.py:141-147, :328-330, :392-395.
- The configs are byte-identical to the stock ones (BitCPM-CANN-0.5B-unquantized == BitCPM-CANN-0.5B ==
  MiniCPM4-0.5B; 8B-unquantized == BitCPM-CANN-8B; 1B-unquantized differs from BitCPM-CANN-1B only in auto_map /
  transformers_version): the fact is in the modeling file only, so NO config detector can see it.
- p5 has no such feature; classify sees a plain MiniCPM / Llama.

## 2. Classifier change
- reads: the repo id (the config carries nothing)
- new: FEATURES "fake_quant" ("free", "wall"), added by repo id: FAKE_QUANT_REPOS table, at the call sites
- detection:
```python
FAKE_QUANT_REPOS = {"openbmb/BitCPM-CANN-0.5B-unquantized", "openbmb/BitCPM-CANN-1B-unquantized",
                    "openbmb/BitCPM-CANN-3B-unquantized", "openbmb/BitCPM-CANN-8B-unquantized"}
def detect_fake_quant_forward(cfg, model_type):
    rid = (cfg or {}).get("_repo_id")          # snapshot configs carry no repo id: always None on the snapshot
    return True if rid in FAKE_QUANT_REPOS else None
```
- apply note: p5:model_graph.py `build_classes` (:3342) and the models loop in `main` (:5043-5048): after
  `c = classify(d)`, `if name in FAKE_QUANT_REPOS: c["feats"] = "+".join(sorted(set(filter(None,
  c["feats"].split("+"))) | {"fake_quant"}))` (or give classify an optional `name` argument). FEATURES entry
  "fake_quant": ("free", "wall", "weights ternary-quantised inside the forward (BitCPM-CANN *-unquantized
  LinearQuantizer): as a target a fixed linear map q(W) any host stores; as a host it realises only row-ternary
  matrices").

## 3. Relation rule
- target side: free: the effective weights q(W) are a fixed matrix; any host loads q(W)
- host side: wall: each host projection row is {-a, 0, +a}-valued, so a general target row (q / k / v / o / gate /
  up / down) is not realisable; no extra width helps before the nonlinearity / softmax
- kind: exact (target) / wall (host)
- cost: none
- free by the config rules: not a config value (hard-coded quant_type="ternary")
- theorem: target side none needed (weights are free); host side needs theorem: "a row-ternary linear map cannot equal
  a generic row" (a counting / support argument, trivial)

## 4. Test fixtures
- flag (fixture `repo_keyed`: the test passes the config with `_repo_id` set, since the configs carry no repo id):
  openbmb/BitCPM-CANN-{0.5B,1B,3B,8B}-unquantized -> True
- not flag: openbmb/BitCPM-CANN-0.5B, openbmb/BitCPM-CANN-8B, openbmb/BitCPM-CANN-1B (already-quantised checkpoints with
  plain nn.Linear, e.g. BitCPM-CANN-0.5B/modeling_minicpm.py:272-273), openbmb/MiniCPM4-0.5B, openbmb/MiniCPM4-8B

## Verdict
real: 4 points lose their host edges (fewer edges; they were plain MiniCPM / Llama hosts for smaller dense targets,
~4 hosts, a frontier member only if one was a sole cover); their target edges stay. Region: classification.
