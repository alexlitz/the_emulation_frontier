# per_token_early_exit
live 4 / placed 4 (ouro 4); direction over-credit; batch a11_recurrent_loops

## 1. What the code computes
(ByteDance/Ouro-1.4B-Thinking/modeling_ouro.py)
- `OuroModel`: `early_exit_gate = nn.Linear(hidden, 1)` (:561); after each of total_ut_steps passes
  `hidden_states = self.norm(h)`, collected with `gate_list.append(early_exit_gate(h))` (:654-656).
- `OuroForCausalLM.__init__`: `early_exit_step`, `early_exit_threshold` from the config (:682-683).
- forward without labels: exit pdf p_t = sigmoid(g_t) * prod_{s<t}(1 - sigmoid(g_s)), the last step takes the
  remainder (:770-781). If early_exit_step is set: logits of that step (a FIXED depth, :823-827). Else with
  early_exit_threshold: per token, `exit_steps = argmax(cumsum(p) >= threshold)`, never-exceeded -> last step, logits
  = lm_head(h_{exit_step}) (:828-850).
- Snapshot: early_exit_threshold 1.0, no early_exit_step. In exact arithmetic cumsum < 1 until the last step, so the
  last loop is used; in fp32 a token exits at step t < T when its remaining mass rounds away, i.e. sigmoid(g_t) rounds
  to 1 (g_t >~ 17) or the cumulative sum rounds up to 1.0: a data-dependent, per-token depth choice.
- The loss path (labels) uses the expected logits sum_t p_t lm_head(h_t): training only.
- LFM3-RD auto halting (`reasoning_mode` / `reasoning_effort` "auto": exit at the first step with p_ready >= threshold,
  cybertruck32489/LFM3-1.7B-RD/modeling_lfm3_rd.py:1022-1093) is the same kind; both snapshots are "manual" (not flagged).
- p5: no feature; Ouro is read as a plain untied 24 / 48-layer stack.

## 2. Classifier change
- reads: model_type ouro (total_ut_steps, early_exit_threshold, early_exit_step); lfm3_rd (reasoning_mode,
  reasoning_effort).
- new: FEATURES key `early_exit` ("wall", "free"), in `_feats`.
- detection:
```python
def detect_per_token_early_exit(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "ouro" and (num(lm.get("total_ut_steps")) or 4) > 1:
        if lm.get("early_exit_step") is None and lm.get("early_exit_threshold") is not None:
            return "threshold"
        return None
    if model_type == "lfm3_rd" and "auto" in (str(lm.get("reasoning_mode")), str(lm.get("reasoning_effort"))):
        return "act"
    return None
```
- apply note: p5:model_graph.py FEATURES (:1597): `"early_exit": ("wall", "free", "per-token exit at a data-dependent
  loop step (Ouro early_exit_threshold, LFM3-RD auto halting): no fixed-depth host picks a token's readout depth from a
  gate; a host with the gate never exits when its gate bias is very negative")`; `_feats` (:1741) when the detector
  fires.

## 3. Relation rule
- target side: needs a host that selects the readout step per token by the same cumulative-gate rule (another Ouro);
  an unrolled fixed-depth host reads the last step for every token. Wall; a LIMIT candidate: if the measured fraction
  of tokens that exit early is ~0 (or their early-step logits agree with the last step's), Ouro on an unrolled host
  passes the high-agreement standard (top-1 >= 99%, >= 90% identical 32-token greedy, fp32).
- host side: free: an Ouro host sets the gate weight 0 and bias -100 (sigmoid = 3.7e-44): cumsum stays < 1 until the
  last step, so every token reads the last loop (exact in fp32).
- kind: wall target side (limit after measurement); free host side.
- cost: none.
- free by the config rules: early_exit_threshold is a feature switch of the TARGET (not something the host may change
  for it); on the host side the gate is weights.
- theorem: none needed for the host side (fp32 arithmetic); target side needs a measurement (not a theorem).

## 4. Test fixtures
- flag: ByteDance/Ouro-1.4B, -1.4B-Thinking, -2.6B, -2.6B-Thinking -> "threshold" (all live).
- not flag: cybertruck32489/LFM3-1.7B-RD (manual mode); alice-noa-chan/haru (CFRD exit_depths: training-only aux
  losses, inference projects at the final depth); Nanbeige/Nanbeige4.2-3B; tomg-group-umd/huginn-0125.

## Verdict
real. Fewer edges for the 4 Ouro points as targets (with loop_depth_unread they would otherwise get exact
looped <- untied edges to hosts with >= 96 / 192 layers: a few tens of pairs); none lost as hosts. A cheap measurement
(fraction of early exits on Ouro-1.4B) could turn it into a limit edge. Region: recurrent-other (loop).
