# blocksparse_mask
live 1 / placed 1 (phi3small 1); direction over-credit; batch a6_masks

The detector also flags microsoft/Phi-3-small-8k-instruct (same code and keys). It was not among the sweep's placed
carriers, but it takes the path: 1024 < 8192.

## 1. What the code computes
- Layer i is dense iff `(i + 1) % dense_attention_every_n_layers == 0`, else block-sparse with
  `BlockSparseParams.from_config(config)` (Phi-3-small-8k-instruct/modeling_phi3_small.py:211-221). With
  dense_attention_every_n_layers 2: 16 dense, 16 block-sparse of 32.
- Params (triton_flash_blocksparse_attn.py:51-62): block_size 64, num_local_blocks 16, vert_stride 8,
  homo_head_pattern false (snapshot).
- Mask (triton_flash_blocksparse_attn.py:152-180): per head h, block-causal `q_blk >= k_blk` AND
  (`q_blk - k_blk < local_blocks` OR `(k_blk + h * head_sliding_step + 1) % vert_stride == 0`), with
  head_sliding_step = max(1, vert_stride / n_heads) (:172-175), then token-causal (:180).
- This is a fixed, content-independent, per-head mask: a local band of 16 x 64 = 1024 tokens (block-aligned) plus
  every 8th key block, shifted per head. It equals dense causal attention while T <= 1024, and differs beyond that.
- p5: no key read: window=full, 32 FU.

## 2. Classifier change
- reads: model_type phi3small, dense_attention_every_n_layers, blocksparse_block_size, blocksparse_num_local_blocks
  (+ vert_stride / homo_head_pattern for the pattern identity)
- new: a LAYER_KINDS entry "PU" (fixed sparse pattern, own K / V), with KIND_REACH
  FU -> ("FU", "FT", "LU", "LT", "PU") and "PU" -> ("PU",). FT is unchanged: a K = V layer cannot play an own-V one.
  `_attn_counts` produces it for phi3small's block-sparse layers (16 PU + 16 FU); dims kPU_eff / kPU_param come from
  the LAYER_KINDS loop.
- detection (returns [n_blocksparse, dense-equivalent context]):
```python
def detect_blocksparse_mask(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "phi3small":
        return None
    n = layer_count(lm)
    d = num(lm.get("dense_attention_every_n_layers"))
    bs, lb = num(lm.get("blocksparse_block_size")), num(lm.get("blocksparse_num_local_blocks"))
    if not n or not bs or not lb:
        return None
    k = sum(1 for i in range(n) if not (d and (i + 1) % d == 0))
    mpe = num(lm.get("max_position_embeddings"))
    if not k or (mpe and bs * lb >= mpe):
        return None
    return [k, bs * lb]
```
- apply note:
  - In p5:model_graph.py add "PU" to LAYER_KINDS / KIND_REACH (~L1481) and add "PU" to FU's reach tuple.
  - `_attn_counts` (~L1322): `if mt == "phi3small" and g("blocksparse_block_size"):` kinds = ["full" if (i+1) % d == 0
    else "pattern"]; count "pattern" as kc["PU"].
  - _HALL (`_subsets_hall`) must be rebuilt for the new kind. `_kind_count` / the dims loop pick PU up from LAYER_KINDS.
  - kind_match host_full does not touch PU: no lever, remote code.

## 3. Relation rule
- target side: a PU layer needs an FU host layer (the host masks to the pattern, per head) or the same pattern (PU host)
- host side: a PU layer plays only PU targets. It cannot play full layers (it drops keys beyond 1024 except the stride
  columns) or band-local ones (the stride columns are extra keys).
- kind: exact (target side, mask currency); wall (PU host for FU / LU targets)
- cost: depth only (the host's PU layers do not count for others)
- free by the config rules: the block params are config, but remote code gets no lever. A blocksparse host with
  local_blocks x block >= T would be dense (the DSA-style context caveat). Not taken: DSA is dense only by an
  explicit design decision, and that decision is not extended here.
- theorem: `pattern_sub_full` (FU plays PU); `zeroed_sublayer` (unused PU layers); needs theorem: "the strided
  block-sparse layer is not a band nor full at T > local_blocks x block" (a one-pair witness)

## 4. Test fixtures
- flag: microsoft/Phi-3-small-128k-instruct -> [16, 1024]; microsoft/Phi-3-small-8k-instruct -> [16, 1024]
- not flag: microsoft/Phi-3-mini-4k-instruct, microsoft/Phi-3.5-mini-instruct (phi3, dense), microsoft/Phi-3.5-MoE-instruct

## Verdict
Real, fewer edges for 2 models, as hosts only. A Phi-3-small host now offers 16 FU instead of 32, so it plays full
targets of at most 16 layers (unrolled depth), and plain targets beyond that lose it. As target nothing changes for
non-phi3small hosts (PU needs FU, as FU did). The two Phi-3-small points host each other. Region: attention. If the
user extends the DSA decision to block-sparse ("dense while T <= 1024"), this becomes a no-op.
