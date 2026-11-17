# hybrid_mla_layers_unread
live 1 / placed 1 (qwen3_5_text 1: sabicap/sabi-noema-mla, outer sabi_mla); direction both; batch a8_mla_dsv4

## 1. What the code computes
(remote code, sabicap/sabi-noema-mla/configuration_sabi_mla.py and modeling_sabi_mla.py, read as text)
- config :30-41 DEFAULT_SHARED_MLA (layers range(3, 64, 4), kv_lora_rank 512, qk_rope_head_dim 64, qk_nope_head_dim 384, v_head_dim 256, enabled True); :57-62 the config's `shared_mla` block is merged over it. Snapshot: enabled true, layers = the 16 full_attention layers (every one), ranks as the defaults.
- decoder :195-201: linear_attention -> Qwen3_5GatedDeltaNet (unchanged); full_attention -> `SabiMlaAttention` if layer_idx in shared_mla.layers, else Qwen3_5Attention.
- attention __init__ :88-101: when enabled, k_proj / v_proj / k_norm are deleted (:93-94); adds hidden_down (hidden -> 512), hidden_rotary (hidden -> 64), kv_b (512 -> heads x (384 + 256)), per-head rotary_query (64 x 64), residual_query (64 x 192), mean_v (heads x 256).
- forward :121-148: `query, gate = q_proj(x).chunk(2)`; `query = q_norm(query)` (Qwen3.5 per-head weighted RMSNorm, q ONLY: no k_norm); `content_q = [q_nope, q_rope @ residual_query_h]`, `rotary_q = RoPE(q_rope @ rotary_query_h)`; `latent = hidden_down(x)` with NO latent norm (:128); `k_rot = RoPE(hidden_rotary(x))` shared by all heads (:129-132); per head K = [W_k,h latent, k_rot], V = W_v,h latent (kv_b rows, :138-145); `attn = (concat_h out_h + mean_v) * sigmoid(gate)` (:147).
- The rope half of the normed query is used twice (unrotated via residual_query, rotated after a learned 64 x 64 mix): no q-norm-then-RoPE head forms that. mean_v is added to softmax-weighted values (rows sum to 1), i.e. a V bias.
- p5 today: classify reads the Qwen3.5 text keys only: kv = "GQA" (4 KV heads), qknorm "head", kv_cap 1024, no kv_latent / mla_rope (p5:model_graph.py:2255-2264, :3315-3316); `shared_mla` is never read.

## 2. Classifier change
- reads: text_config.shared_mla (enabled, layers, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim; DEFAULT_SHARED_MLA for missing keys), layer_types
- new: kv slot "MLA" (or "GQA|MLA" if some full layers are not listed) in classify; dims kv_latent / q_latent 0 / mla_rope / mla_nope / mla_v / kv_heads = heads in dims_of; FEATURES `"mla_qmix"` (wall, wall) and `"bias_qkv"` (mean_v) in _feats; qknorm variant `"q_head"` (q-only per-head norm)
- detection:
```python
def detect_hybrid_mla_layers_unread(cfg, model_type):
    lm = lm_of(cfg)
    sm = lm.get("shared_mla")
    if not isinstance(sm, dict) or sm.get("enabled", True) is False:
        return None
    blk = {**_SABI_DEFAULT, **sm}
    layers = [i for i in (blk.get("layers") or []) if isinstance(i, int)]
    lt = lm.get("layer_types") if isinstance(lm.get("layer_types"), list) else []
    full = [i for i, t in enumerate(lt) if t == "full_attention"]
    if not layers:
        return None
    return {"kv": "MLA" if set(full) <= set(layers) else "GQA|MLA",
            "kv_latent": num(blk["kv_lora_rank"]), "q_latent": 0, "mla_rope": num(blk["qk_rope_head_dim"]),
            "mla_nope": num(blk["qk_nope_head_dim"]), "mla_v": num(blk["v_head_dim"]),
            "mla_layers": len(layers)}
```
- apply note: p5:model_graph.py: a helper `_shared_mla(d)` returning the merged block (or None when absent / enabled false). classify kv chain (:2255): `elif d.get("kv_lora_rank") or _shared_mla(d): kv = "MLA"`. qknorm (:2342): `if _shared_mla(d): qknorm = "q_head"`; add "q_head" to QKNORM_VARIANTS (:224). dims_of (:3089 and :3315-3320): when `_shared_mla(d)`, kv_cap = kv_lora_rank + qk_rope_head_dim (576), q_head = nope + rope, kv_heads = n_heads, q_per_kv = 1, kv_latent / mla_rope / mla_nope / mla_v from the block, q_latent 0. FEATURES: `"mla_qmix": ("wall", "wall", "Sabi shared pre-RoPE MLA: per-head q_norm, then learned rotary_query / residual_query mixes of the normed rope half, no latent norm, no k_norm")`; _feats: `if _shared_mla(d): f |= {"mla_qmix", "bias_qkv"}`.

## 3. Relation rule
- target side: needs a host with mla_qmix (only itself) and a V bias; every MLA host has the latent norm (`mla_latent_norm_not_mha` direction reversed: a normed-latent host cannot drop a norm the target lacks except by the lever, and no host forms the reused normed rope half), every GQA host lacks per-head latent keys
- host side: walled: it cannot play DeepSeek-style MLA targets (no latent norm to give; adding a norm is never free) nor Qwen3.5 GQA targets (no k_norm); shared_mla.enabled = false runs the GQA path but is a feature switch (module choice), not free
- kind: wall both ways
- cost: none
- free by the config rules: none (shared_mla.enabled is a feature switch; ranks are shapes)
- theorem: needs theorem: "a head whose query is N_q(x)-then-(R, Rot) mixes (the normed rope half read both unrotated and rotated after a mix) equals no q-norm-then-RoPE head"; supporting: `mla_rope_sub_mha_partial` (its norm-free K/V side alone is an MHA partial-RoPE head), `knorm_not_plain` (k-norm targets on a k-norm-free host), `mla_latent_norm_not_mha`

## 4. Test fixtures
- flag: sabicap/sabi-noema-mla -> {kv MLA, kv_latent 512, q_latent 0, mla_rope 64, mla_nope 384, mla_v 256, 16 layers}
- not flag: Advisio/Radha-9B, adi274903/Med-Qwen-3.5-4B-Instruct (plain qwen3_5_text), Alennnndy/Qwen3.6-27B-AWQ-4bit, nvidia/Qwen3.6-35B-A3B-NVFP4 (Qwen3.5/3.6 bodies without shared_mla); baidu/ERNIE-5.5-Flash-Base (hybrid MLA keyed differently)

## Verdict
real (both): sabi-noema-mla today plays and is played by Qwen3.5-shaped GQA hybrids (its config is the stock Qwen3.5 one); after the fix it is isolated (MLA kv + mla_qmix wall both ways). Fewer edges, all touching this one model: on the order of tens of pairs (the Qwen3.5 / 3.6 hybrids whose dims it fits either way).
