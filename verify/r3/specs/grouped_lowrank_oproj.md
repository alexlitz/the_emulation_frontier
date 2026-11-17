# grouped_lowrank_oproj
live 21 / placed 21 (deepseek_v4 21); direction over-credit; batch a7_attn_exotic

## 1. What the code computes
- tf5:models/deepseek_v4/modular_deepseek_v4.py:278-307 DeepseekV4GroupedLinear: weight viewed as (n_groups,
  in_per_group, out_per_group), per-group bmm (:304-307): a block-diagonal linear.
- :697-700 `o_a_proj = GroupedLinear(H * hd / o_groups -> o_groups * o_lora_rank, o_groups)`, `o_b_proj =
  Linear(o_groups * o_lora_rank -> hidden)`; :775-777 `o_b(flatten(o_a(attn.reshape(.., o_groups, -1))))`.
  V4-Flash H 64, hd 512, g 8, r 1024 (each 4096-dim group -> 1024); V4-Pro H 128, g 16, r 1024 (:290-294).
- The o-map is O = W_b . blockdiag(A_1 .. A_g): rank of the part read from group j <= r, and heads of different
  groups meet only after the rank-r bottlenecks. A host forms a target O iff each target group lands inside one
  host group whose rank budget covers it (zero / identity-pad the rest: `fullrank_plays_lowrank`, `lowrank_pad`).
- Same structure in DeepSeek-V4.1 (ext deepseek_v41 inference/model.py:71-76, 634, o_groups 8), memt_deepseek_v4
  (same keys) and ERNIE-5.5 (ext ernie5_5_moe vLLM-ERNIE attention.py:176-185, 230-243: wo_a grouped, n_groups =
  o_groups 8, o_lora_rank 1024).
- p5:model_graph.py dims_of / _pass5_dims have no field for o_groups / o_lora_rank (grep: none).

## 2. Classifier change
- reads: o_groups, o_lora_rank (+ num_attention_heads for heads per group) on the grouped-o types
- new: dims fields `o_groups`, `o_rank` (o_lora_rank), in dims_of (or _pass5_dims)
- detection:
```python
_GROUPED_O_TYPES = ("deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4", "ernie5_5_moe")

def detect_grouped_lowrank_oproj(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in _GROUPED_O_TYPES:
        return None
    g, r = num(lm.get("o_groups")), num(lm.get("o_lora_rank"))
    return f"g{g}r{r}" if g >= 1 and r > 0 else None
```
- apply note: p5:model_graph.py `dims_of` (3077): add `o_groups` / `o_rank` (0 = a dense o_proj); emul_matrix capacity
  check (next to kv_heads_fit): for a grouped host, target heads-per-group H_t / g_t (dense target: H_t, one group)
  must fit host H_h / g_h, k = floor((H_h / g_h) / (H_t / g_t)) target groups per host group, need
  g_h * k >= g_t and r_h >= ceil(g_t / g_h) * r_t (dense target: r_t = min(H_t hd_t, hidden_t)).

## 3. Relation rule
- target side: a grouped target is played exactly by any dense o_proj host (W_o := W_b blockdiag(A)) and by a grouped
  host meeting the packing / rank condition above.
- host side: a grouped host plays a dense-o target only if one host group holds all target heads with
  r_h >= min(H_t hd_t, hidden_t); otherwise a wall (rank deficit).
- kind: exact (where the condition holds); wall (rank / grouping deficit)
- cost: none (a capacity condition)
- free by the config rules: o_groups / o_lora_rank are shape values (not free)
- theorem: `fullrank_plays_lowrank`, `lowrank_pad` (row "full-rank gate plays low-rank; higher rank plays lower")

## 4. Test fixtures
- flag: deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4-Flash-0731 -> "g8r1024"; deepseek-ai/DeepSeek-V4-Pro,
  deepseek-ai/DeepSeek-V4-Pro-0813 -> "g16r1024"; HuggingFaceTB/nanowhale-100m -> "g2r80";
  inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B -> "g2r64"; deepseek-ai/DeepSeek-V4.1-Flash,
  baidu/ERNIE-5.5-Flash-Base, MemTensor/MemT-1.0-Flash -> "g8r1024"; endless-frontier/Fx-Bio -> true
- not flag: deepseek-ai/DeepSeek-V3 (dense o_proj), deepseek-ai/DeepSeek-V3.2, Qwen/Qwen3-8B

## Verdict
real but no frontier change on the snapshot: grouped-o models only play each other (hyper_conn / vnorm walls), and
among them every hosting pair already satisfies the condition where head dims fit (Flash <- Pro, nanowhale <-
Flash, 0.5B <- nanowhale; 0.5B cannot host nanowhale for r 64 < 80, but already fails head_dim 64 < 96). Recorded
for future grouped-o hosts / lifted walls.
