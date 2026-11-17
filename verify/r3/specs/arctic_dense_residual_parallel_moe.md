# arctic_dense_residual_parallel_moe
live 1 / placed 1 (arctic 1); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
(path: Snowflake/snowflake-arctic-instruct/modeling_arctic.py, snapshot be318cae)
- `ArcticMoE`: `is_moe_layer = (layer_id + 1) % moe_layer_frequency == 0` (:920). Snapshot moe_layer_frequency 1, so every layer is MoE: 128 experts x 4864, top-2.
- `ArcticDecoderLayer.__init__`: `parallel_attn_mlp_res = config.parallel_attn_mlp_res and is_moe_layer`. It builds `residual_layernorm` and `residual_mlp = ArcticMLP(is_residual_mlp=True)` (:1042-1050). The residual MLP width is hidden_size, 7168 (`ffn_dim = hidden_dim if is_residual_mlp`, :885).
- forward (:1083-1110): `h = x + attn(n1 x)`; `out = h + residual_mlp(n_r h) + moe(n2 x)`. The MoE reads the normed LAYER INPUT x, in parallel with attention, and the dense SwiGLU reads the post-attention stream.
- Snapshot: parallel_attn_mlp_res true, hidden 7168, 35 layers.
- p5 classifies block = sequential, ffn = moe (no dense path, MoE assumed to read x + attn). Nothing reads parallel_attn_mlp_res.

## 2. Classifier change
- reads: `parallel_attn_mlp_res`, `num_local_experts` (arctic).
- new: FEATURES key `par_moe_res` in `_feats`, plus dims field `res_mlp_w` (= hidden_size) in `dims_of`.
- detection:
```python
def detect_arctic_dense_residual_parallel_moe(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "arctic" and lm.get("parallel_attn_mlp_res") is True and num(lm.get("num_local_experts")) > 1:
        return num(lm.get("hidden_size"))
    return None
```
- apply note: FEATURES (~1597): `"par_moe_res": ("wall", "free", "Arctic: MoE parallel to attention (reads the layer input) plus a dense residual MLP on the post-attention stream")`. Set it in `_feats` for arctic with parallel_attn_mlp_res. `dims_of` (~3200): `res_mlp_w = hidden`, and add `n_layers * hidden` to ffn_units. emul_matrix: a host with par_moe_res gets two constraints. For a MoE target, its usable depth is eff_layers // 2: a pair of host layers plays one sequential MoE layer (layer A attention only; layer B attention zeroed, MoE reading A's output). For a dense target it plays at 1x with the dense width cap res_mlp_w.

## 3. Relation rule
- target side: the MoE output depends on x, not on x + attn(x), and the residual MLP must not see the MoE output. A sequential host would have to carry x past the attention (the parallel <- seq carry, refuted under a pre-norm), and a parallel host has no residual-MLP stage. So it is a wall.
- host side: an Arctic host plays a sequential dense target at 1x by zeroing the MoE (expert down = 0) and using residual_mlp (width <= 7168) as the target's MLP: exact. It plays a sequential MoE target at 2x depth: layer A is the attention (MoE and residual MLP zeroed), and layer B zeroes attention (o_proj = 0) so its MoE reads x + attn, exact (`seq_sub_parallel`). It also plays a parallel-block target whose FFN is a MoE at 1x (residual MLP zeroed).
- kind: target wall; host exact (dense 1x, MoE 2x depth).
- cost: depth 2x for MoE targets; dense width <= hidden_h.
- free by the config rules: parallel_attn_mlp_res is a feature switch (module structure), so it is not free.
- theorem: host `seq_sub_parallel`, `seq_stack_sub_parallel`, `zeroed_sublayer` (PROVED exact). Target wall: `parallel_sub_seq_prenorm_fails` / `prenorm_readout_obstruction` (the carry construction refuted; a general wall is not proved).

## 4. Test fixtures
- flag: Snowflake/snowflake-arctic-instruct -> 7168.
- not flag: HIT-SCIR/Chinese-Mixtral-8x7B (sequential MoE), Qwen/Qwen3-30B-A3B, rombodawg/Everyone-Coder-4x7b-Base (mixtral merge).
- scan: exactly the Arctic repo.

## Verdict
real. Both kinds of change, fewer edges overall. As a target, Arctic loses every host (it currently gets plain sequential MoE hosts with >= 128 x 4864 / top-2 capacity: a few large MoE hosts). As a host, a huge one (480B), it currently plays sequential MoE targets at 1x depth. After the fix it needs 2 layers per target MoE layer (35 layers -> MoE targets of <= 17 layers) and dense targets of width <= 7168 at 1x. That drops a sizeable host fan-out: most MoE targets deeper than 17 layers, on the order of tens to a few hundred edges.
