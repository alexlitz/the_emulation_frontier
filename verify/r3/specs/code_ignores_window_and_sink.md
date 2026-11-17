# code_ignores_window_and_sink
live 2 / placed 2 (deepseek_v4 2); direction both; batch a6_masks

## 1. What the code computes
- nanowhale-100m / -base run the remote modeling_deepseek_v4.py (auto_map AutoModelForCausalLM
  "modeling_deepseek_v4.DeepseekV4ForCausalLM"), not transformers' DeepSeek-V4.
- Model forward: "always create our own 4D mask", `causal_mask = torch.triu(full(-inf), diagonal=1)`
  (nanowhale-100m-base/modeling_deepseek_v4.py:582-585). It never reads config.sliding_window (128). Every layer is full
  causal attention.
- Attention: `self.attn_sink = nn.Parameter(torch.zeros(num_heads))` is built (:170-171), but "attn_sink bias is ...
  omitted in SDPA path" (:217-219). `F.scaled_dot_product_attention(q, kv, kv, attn_mask=mask, ...)` (:220-225) has no
  sink logit. Identical in nanowhale-100m (:217, :584).
- compress_ratios are all 0 (no kv_compress); hc_mult 4 (the hyper_conn feature still applies).

## 2. Classifier change
- reads: model_type deepseek_v4 + auto_map module modeling_deepseek_v4 (the remote nanowhale code), compress_ratios
- new: window = "full" and sink = "no" for the nanowhale code, in classify (a code override table
  CODE_SLOT_OVERRIDES = {"modeling_deepseek_v4.DeepseekV4ForCausalLM": {"window": "full", "sink": "no"}})
- detection:
```python
def detect_code_ignores_window_and_sink(cfg, model_type):
    if model_type != "deepseek_v4" or "modeling_deepseek_v4.DeepseekV4ForCausalLM" not in auto_map(cfg):
        return None
    lm = lm_of(cfg)
    cr = lm.get("compress_ratios")
    if isinstance(cr, list) and any(num(x) for x in cr):
        return None
    return ["full", "no"]
```
- apply note: classify (p5:model_graph.py ~L2131) sees `outer` (the raw config, which carries auto_map). After the window
  (~L2319-2328) and sink (~L2352) lines, apply the override when the outer auto_map names the nanowhale module. Also set
  `sliding_window=None` in `_normalize` so `_attn_counts` / dims_of count 8 FU and eff_window = max_context. Keying on
  model_type alone (SINK_TYPES deepseek_v4) would keep the 19 transformers-code V4 snapshots right; only these two use
  the remote code.

## 3. Relation rule
- target side: a full-window, sink-free target (was local + sink: it demanded only sliding coverage, and a sink-capable
  host or the sink rescue)
- host side: a full-window, sink-free host (was: sink host -> plain targets only via `sink_to_plain_limit`)
- kind: exact
- cost: none
- free by the config rules: n/a (code facts, not config)
- theorem: none needed for the label. The edges it unlocks are the plain ones (`local_sub_full`; no `sink_host_ne_plain`
  limit)

## 4. Test fixtures
- flag: HuggingFaceTB/nanowhale-100m -> ["full", "no"]; HuggingFaceTB/nanowhale-100m-base -> ["full", "no"]
- not flag: deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4-Pro (transformers code: window + sink real);
  inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B (compressed layers)

## Verdict
Real, both directions, but no snapshot edge changes today. nanowhale keeps hyper_conn (hc_mult 4: wall both ways,
FEATURES ~L1611), and it lacks act_clip / kv_compress, which every other V4 point has. So its edges are walled on other
grounds before window / sink matter. Apply it for correctness; the frontier does not change (2 models). Region:
attention / classification.
