# ffn_width_keys_unread
live 14 / placed 14 (minimax_m3_vl 4, recurrent_gemma 2, bananamind2_moe 1, gla 1, iquest_q1 1, ivme 1, mima 1,
rwkv5 1, tinymixtral 1, zaya1_vl 1); direction both; batch a2_dims

## 1. What the code computes
- MiniMax-M3 dense layers: `tf5:models/minimax_m3_vl/modular_minimax_m3_vl.py:629` non-MoE layers build
  `MiniMaxM3VLDenseMLP(config, intermediate_size=config.dense_intermediate_size)` (`:103` default 12288; `:340`);
  experts are intermediate_size (3072), shared `:394` shared_intermediate_size. Snapshot: 12288 / 3072.
- IQuest-Q1: `ext:iquest_q1/gh__IQuestLab__vllm-iquest-q1/model.py:451-453` layer 0 (mlp_only_layers [0]) is
  `IQuestQ1DenseMLP(intermediate_size=config.dense_intermediate_size)` = 12288; experts 1536.
- RecurrentGemma: `tf5:models/recurrent_gemma/modeling_recurrent_gemma.py:488-491` `intermediate_size =
  config.intermediate_size // 2`, gate / up / down with bias. Snapshot 15360 -> 7680.
- RWKV-5: `RWKV/v5-Eagle-7B-HF/configuration_rwkv5.py:110` forces `intermediate_size = None`;
  `modeling_rwkv5.py:363-366` width `int((hidden * 3.5) // 32 * 32)` = 14336 (hidden 4096).
- Mima: `root-sys-n/Mima-Pico-15m/modeling_mima.py:108-110` SwiGLU w1 / w3 `n_embd -> ffn_hidden` (1024 saved).
- TinyMixtral: `mikecovlee/tinymixtral-v1.1-1b/modeling_tinymixtral.py:133` experts of
  `config.expert_intermediate_size` (2816 saved). BananaMind-2: `BananaMind/BananaMind-2-MoE/modeling_bananamind2moe.py:146-155`
  expert width `config.expert_intermediate_size` (160).
- Ivme: `IvmeLabs/Ivme-Conversate-v2-Base/modeling_ivme.py:93-94` `inner = int(hidden * ffn_mult * 2 / 3)`, rounded
  up to 8 -> 1024 (hidden 384, ffn_mult 4.0).
- ZAYA1-VL experts: `ext:zaya1_vl/gh__Zyphra__transformers/modeling_zaya1_vl.py:1715-1730` with gated_linear_unit
  `fc1: hidden -> ffn_hidden_size`, `fc2: ffn_hidden_size // 2 -> hidden`: GLU branch 2048 (ffn_hidden_size 4096);
  `:1809-1819` 16 such experts.
- fla GatedMLP (gla, also delta_net / hgrn_bit): `ext:gla/fla/mlp.py:46-50` with intermediate_size None,
  `int(hidden * hidden_ratio * 2 / 3)` rounded up to 256 (gla-1.3B: 5632); `ext:gla/fla/modeling_gla.py:64-67` passes
  `hidden_ratio`, `intermediate_size=config.intermediate_size` (None); same formula `ext:delta_net/fla/mlp.py:49-50`,
  `ext:hgrn_bit/gh__ridgerchu__matmulfreellm/modeling_hgrn_bit.py:47-48`.
- p5 (`p5:model_graph.py` dims_of): dense_inter = `g("intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size",
  "ff_intermediate_size") or 4 * hidden`, halved only for FUSED_GLU_HALF; ew = `g("moe_intermediate_size", ...,
  "ffn_hidden_size", ...) or dense_inter`; a MoE model's dense layers (moe_sig dense_w, ffn_units) use dense_inter =
  intermediate_size. Measured p5 values: recurrentgemma inter 15360, rwkv5 16384, tinymixtral ew 4096 (ffn_active
  8192), ZAYA1 ew 4096, MiniMax-M3 dense 3072, IQuest dense 1536, gla 8192. Not fixed in p5.

## 2. Classifier change
- reads: dense_intermediate_size, ffn_hidden, expert_intermediate_size, ffn_mult, hidden_ratio, gated_linear_unit;
  the recurrent_gemma / rwkv5 code formulas
- new: dims inter / ffn_active / ffn_total / ffn_units / moe_sig (ew, dense_w) values, in dims_of
- detection:
```python
def detect_ffn_width_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    h = _hidden(lm)
    inter = lm.get("intermediate_size")
    if model_type in ("minimax_m3_vl", "minimax_m3_vl_text", "iquest_q1") and num(lm.get("dense_intermediate_size")):
        d = num(lm["dense_intermediate_size"])
        return ["dense_inter", d] if d != num(inter) else None
    if model_type in ("recurrent_gemma", "recurrentgemma") and num(inter):
        return ["inter", num(inter) // 2]
    if model_type == "rwkv5" and inter is None and h:
        return ["inter", int((h * 3.5) // 32 * 32)]
    if model_type == "mima" and num(lm.get("ffn_hidden")):
        return ["inter", num(lm["ffn_hidden"])]
    if model_type in ("tinymixtral", "bananamind2_moe") and num(lm.get("expert_intermediate_size")):
        return ["ew", num(lm["expert_intermediate_size"])]
    if model_type == "ivme" and fnum(lm.get("ffn_mult")) and h:
        w = int(h * fnum(lm["ffn_mult"]) * 2 / 3)
        return ["inter", (w + 7) // 8 * 8]
    if model_type == "zaya1_vl" and lm.get("gated_linear_unit") and num(lm.get("ffn_hidden_size")):
        return ["ew", num(lm["ffn_hidden_size"]) // 2]
    if model_type in ("gla", "delta_net", "hgrn_bit") and inter is None and h:
        w = int(h * (fnum(lm.get("hidden_ratio")) or 4) * 2 / 3)
        return ["inter", 256 * ((w + 255) // 256)]
    return None
```
- apply note: in `dims_of`, right after the `dense_inter = ...` line: the rwkv5 / mima / ivme / fla formulas set
  dense_inter (add "ffn_hidden" to the key list; recurrent_gemma / recurrentgemma join FUSED_GLU_HALF). Add
  "expert_intermediate_size" to the `ew = g(...)` key list, and for zaya1_vl (`gated_linear_unit` and
  ffn_hidden_size) halve ew. For dense_intermediate_size: a separate `dense_w = g("dense_intermediate_size") or
  dense_inter` used for `_moe_sig(..., dense_w, ...)`, the `ffn_units` dense terms (`sub[1] * dense_w`, the
  `_moe_layer_mask` branch) and the dense-first rule, leaving `inter` = intermediate_size.

## 3. Relation rule
- target side: a target needs host FFN units >= its true width per layer kind: the narrower corrections (9 models)
  admit more hosts; MiniMax-M3 x4 / IQuest dense layers (12288) need wider dense host layers: fewer hosts.
- host side: the overstated hosts (recurrentgemma 2x, rwkv5, tinymixtral, ZAYA1 2x, gla, ivme 1.5x, mima) offer
  less: fewer out-edges; MiniMax-M3 / IQuest offer wider dense layers: more out-edges.
- kind: exact (capacity)
- cost: none
- free by the config rules: not free (widths are shape)
- theorem: `ffn_extra_units`, `glu_zero_unit`, `moe_wider_experts` (Capacity rules: ffn_total / ffn_active);
  `dense_sub_moe_shared` (Round-2: dense-first)

## 4. Test fixtures
- flag (all 14 live): BananaMind/BananaMind-2-MoE -> ["ew", 160]; IQuestLab/IQuest-Q1 -> ["dense_inter", 12288];
  IvmeLabs/Ivme-Conversate-v2-Base -> ["inter", 1024]; the 4 MiniMax-M3 / M3.1 -> ["dense_inter", 12288];
  RWKV/v5-Eagle-7B-HF -> ["inter", 14336]; Zyphra/ZAYA1-VL-8B -> ["ew", 2048]; fla-hub/gla-1.3B-100B -> ["inter", 5632];
  google/recurrentgemma-2b, -2b-it -> ["inter", 7680]; mikecovlee/tinymixtral-v1.1-1b -> ["ew", 2816];
  root-sys-n/Mima-Pico-15m -> ["inter", 1024]
- not flag: RWKV/rwkv-4-169m-pile (rwkv: intermediate_size read), fla-hub/rwkv7-0.1B-g1 (hidden_ratio with
  intermediate_size set), baidu/ERNIE-5.5-Flash-Base (gated_linear_unit, not zaya1_vl), techdotus/Ethoric-10M
  (ffn_hidden = intermediate_size), google/gemma-2-2b, Qwen/Qwen3-30B-A3B
- scan: the 14 + fla-hub/delta_net-1.3B-100B and ridger/MMfreeLM-370M (same fla formula; unplaced in p5).

## Verdict
real. Both directions, ~14 models: mostly fewer host edges (9 hosts were credited 1.3-2x their FFN width, as
targets they gain hosts); MiniMax-M3 / IQuest the reverse for their dense layers. Tens to ~100 edges.
