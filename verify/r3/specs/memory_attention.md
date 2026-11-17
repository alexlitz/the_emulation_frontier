# memory_attention
live 5 / placed 5 (cfrd 3, edgerock 1, fabric 1); direction over-credit; batch a7_attn_exotic

## 1. What the code computes
- alice-noa-chan/haru_1.1/model.py:201-225 LocalCausalAttention: causal attention inside fixed chunk_size (64) blocks
  (block-diagonal, not a sliding window); :315-445 CausalSummaryMemory: per chunk, summary_slots (4) learned queries
  softmax-pool the chunk (:335, :367-394), tokens attend to EARLIER chunks' summaries (memory_allowed mask :352-354,
  :436-437) with a per-head `softplus(decay) * chunk_distance` penalty (:431-432), output read_o (:343, :445), scaled
  by sigmoid(memory_gain) (:543, :609). Built in every cell (:503-504); haru / haru_2 the same (haru_2 model.py:515-521
  drops it when cell_attention == "full"; its snapshot says "local").
- tudor-iustin22/Bluebird-1.5-Instruct/fabric_runtime.py:209-231 ChunkSummarizer (summaries_per_chunk 4 learned
  queries pooling each 512-token chunk), :233-262 MemoryAttention (GQA over the summaries of COMPLETED chunks only,
  :203-206, :256-259; no RoPE), :278-296 FabricMemoryBlock `x + g * local + (1 - g) * memory`, g = sigmoid(w . n + b);
  :310-311 on layers i % 3 == 2 when architecture == "fabric".
- rockmatrix-ai/EdgeRock/modeling_edgerock.py:122-322 IndependentGQAMemory on each fixed_memory_swa_layers layer: keys
  written through a residual SwiGLU writer, an admission score selects the top-capacity (1024) keys of all earlier
  chunks (select_topk / merge_topk / exclusive_prefix_topk :51-80), read with a zero pad slot (_gqa_tile :22-31), gate
  `tanh(read_scale) * 2 sigmoid(q.w / sqrt(d) + b)` (:170), added to the attention output.
- A pooled summary key / value is no single token's key / value (the kv_compress argument), and a content-selected
  top-capacity memory is a data-dependent mask; chunk-block-diagonal local attention is a position mask no window
  value gives. p5 classifies all five as plain softmax (haru / Bluebird) or a GDN hybrid (EdgeRock) without it.
- Host side: zeroing the memory output (haru read_o = 0; Bluebird memory o_proj = 0 with the gate's w = 0 and
  local o_proj / sigmoid(b); EdgeRock read_scale = 0) removes the path exactly.

## 2. Classifier change
- reads: model_type cfrd (cell_attention), fabric (architecture), edgerock (fixed_memory_swa_layers,
  independent_memory / fixed_memory_revision)
- new: FEATURES key `mem_attn` ("wall", "free"), in _feats + FEATURES (value kind "summary" / "topk_kv" for the record)
- detection:
```python
def detect_memory_attention(cfg, model_type):
    lm = lm_of(cfg)
    mt = model_type
    if mt == "cfrd":
        return "summary" if _s(lm.get("cell_attention", "local")) != "full" else None
    if mt == "fabric":
        return "summary" if _s(lm.get("architecture", "fabric")) == "fabric" else None
    if mt == "edgerock":
        if lm.get("fixed_memory_swa_layers") and (isinstance(lm.get("independent_memory"), dict)
                                                  or "independent" in _s(lm.get("fixed_memory_revision"))):
            return "topk_kv"
    return None
```
- apply note: p5:model_graph.py `_feats` (after the kv_compress block, ~1806): add "mem_attn" on the detector's tests;
  FEATURES: add "mem_attn" next to "kv_compress". (haru's chunk-local attention also needs window="local"; recorded
  here, not separately keyed.)

## 3. Relation rule
- target side: needs a host with the same memory path (same kind, chunk size, slots / capacity); a plain host
  cannot form pooled or top-capacity-selected keys, nor a block-diagonal chunk mask: wall.
- host side: a memory host zeroes the memory output and plays the plain target: free (exact).
- kind: wall (target side); exact (host side)
- cost: none
- free by the config rules: chunk / capacity sizes are shape values; the memory path is a feature switch. EdgeRock's
  capacity >= context would make the selection keep every earlier key (a DSA-like "dense when context small" reading),
  but its writer / gate still differ from plain attention: not treated as dense.
- theorem: `pool_one` (row "kv_compress", the pooled-key wall's tightness note) | needs theorem: mem_off (zero output
  projection, exact) ; target side OPEN as kv_compress

## 4. Test fixtures
- flag: alice-noa-chan/haru, alice-noa-chan/haru_1.1, alice-noa-chan/haru_2 -> "summary";
  tudor-iustin22/Bluebird-1.5-Instruct -> "summary"; rockmatrix-ai/EdgeRock -> "topk_kv"
- not flag: deepseek-ai/DeepSeek-V4-Flash (kv_compress pools, its own feature), Qwen/Qwen3-8B,
  MemTensor/MemT-1.0-Nano (memt_qwen3_5_moe: "memory" in name, plain)

## Verdict
real. Fewer edges: the 5 models (small softmax / GDN-hybrid targets today, hosted by many larger hosts) lose all
hosts except each other's kind; as hosts they keep plain targets.
