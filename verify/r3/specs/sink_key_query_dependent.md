# sink_key_query_dependent
live 1 / placed 1 (iquest_q1 1); direction both; batch a7_attn_exotic

## 1. What the code computes
- ext:iquest_q1/gh__IQuestLab__vllm-iquest-q1/model.py:354-365: with `enable_sink_attention` (getattr default False;
  configs.py:41 default True; snapshot True) a parameter `sink_k` of shape (num_kv_heads, head_dim) is passed to
  IQuestAttention. :393-414: q = q_norm(q) per head, k = k_norm(k), then RoPE on q and k; the sink key is NOT rotated.
- ext:iquest_q1/.../attention.py:66-89: per query head h (kv group h // QUERIES_PER_KV), `sink_logit = (q . s) * SCALE`,
  `out *= 1 / (1 + exp(sink_logit - lse))`: one extra key s with value 0 -- softmax over [real keys, s].
  :185-186 refuses per-head constant sinks together with it. sglang/iquest_q1.py:90-112, 375-423 is the same.
- test_sink_attention.py:271-272 states the reference: `softmax(cat([q k^T, q . s]) )`.
- Contrast: sink=yes (gpt-oss s_aux, MiMo-V2, Flint, p5:model_graph.py:340-351) is a per-head SCALAR logit b_h,
  independent of q. Here the logit is linear in the normed, rotated query: (R_p q) . s.
- A start token at position 0 gives exactly this: the real token at host position p+1 scores (R_{p+1} q) . (R_0 k0);
  with k0 = R_1 s (a constant) this is (R_p q) . s for every p; v0 = 0. No q bias and no zero-frequency pair are
  needed (unlike `sink_sub_plain` / `sink_needs_zero_freq`). A host k-norm writes k0 = g * n(W_k h0); with eps > 0
  ||n(x)||^2 ranges over [0, d), so k0 is reachable iff ||R_1 s / g||^2 < d (true for the trained s unless g is tiny:
  not checked on weights).
- Also seen (not this finding): IQuest's q_norm / k_norm (model.py:350-352, 396-410) classify as qknorm=no in p5.

## 2. Classifier change
- reads: model_type iquest_q1, enable_sink_attention
- new: slot value sink="key" (third value of the sink slot), in classify (sink assignment) + SINK_ORDER
- detection:
```python
def detect_sink_key_query_dependent(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "iquest_q1" and lm.get("enable_sink_attention", False) is True:
        return "key"
    return None
```
- apply note: p5:model_graph.py classify, the `sink = "yes" if (...)` line (~2352): `sink = "key"` first for
  iquest_q1 + enable_sink_attention (drop "enable_sink_attention" from _SINK_KEYS, 632-633). SINK_ORDER (345): add
  "key": [] (no edge from a key host to "no" / "yes"); emul_matrix slot == "sink" (~4194-4204): a "key" target on a
  "no" host takes the start-token rule WITHOUT the q-bias / spare-slot conditions (add_h SINK_EXTRA_HIDDEN only);
  on a "yes" host it is that rule plus the b -> -inf limit (REGIME_PAYABLE ("sink", "no", "yes") already True).

## 3. Relation rule
- target side: a key-sink target needs the start token: k0 = R_1 s_l, v0 = 0 per layer on any host ("no": exact;
  "yes": exact construction + the host's own sinks to -inf, a limit). Today it needs a q bias + a zero-freq pair
  (over-wall).
- host side: a key-sink host cannot remove its sink (s = 0 leaves logit 0: a b = 0 scalar sink; driving q . s -> -inf
  for every q needs a constant query channel, which the pre-norm + q_norm host lacks): wall to "no" targets (today a
  limit: over-credit); wall to "yes" targets except b_h = 0; exact to "key" targets.
- kind: exact (key target <- plain host, start token); limit (key target <- scalar-sink host); wall (key host -> plain /
  scalar-sink target)
- cost: start-token cost only (SINK_EXTRA_HIDDEN = 1 channel); no head-dim spare
- free by the config rules: the start token is the allowed one fixed token; the sink key itself is a weight, not config
- theorem: `sink_host_factor`, `sink_host_ne_plain` (host side, scalar case) | needs theorem: sink_key_start_token
  ((R_{p+1} q) . (R_1 s) = (R_p q) . s; a zero-value key at position 0 = the key sink) and sink_key_host_ne_plain
  (Z / (Z + e^{q.s}) is not 1 for any s when q ranges over a sphere)

## 4. Test fixtures
- flag: IQuestLab/IQuest-Q1 -> "key"
- not flag: openai/gpt-oss-20b (scalar s_aux sinks), XiaomiMiMo/MiMo-V2-Flash (add_swa_attention_sink_bias scalar),
  baidu/ERNIE-5.5-Flash-Base (add_full_attention_sink_bias scalar), Qwen/Qwen3-8B (no sink)

## Verdict
real, both ways. Fewer edges: IQuest-Q1 (a 256-expert MoE host) stops playing plain / scalar-sink targets via the
b -> -inf limit (potentially hundreds of target classes it out-sizes). More edges: IQuest as a target no longer needs a
q-biased host with a spare zero-frequency pair (any GQA-MoE host big enough, once its missing qk-norm is classified).
