# gated_prenorm_lowrank
live 9 / placed 9 (grug_moe 7, axk2 2); direction over-credit; batch a4_norms

## 1. What the code computes
- A.X-K2: `AXK2GatedRMSNorm`: `y = self.norm(x)` (AXK2RMSNorm, weighted, rms_norm_eps); `return y * sigmoid(self.mlp(y))` (tf5:models/axk2/modular_axk2.py:195-209); `AXK2GateMLP`: fc1 Linear(hidden -> gated_norm_rank, no bias), SiLU, fc2 Linear(rank -> hidden, no bias) (:185-192); gated_norm_rank default 16 (configuration_axk2.py:132; snapshot 16).
- Placement: `input_layernorm = AXK2GatedRMSNorm` on every layer; `post_attention_layernorm` gated on `mlp_layer_types[i] == "sparse"` layers, plain AXK2RMSNorm on dense ones (modular_axk2.py:402-409); the final norm is plain (:204).
- Grug-MoE: `GrugMoeGatedNorm`: `gate = sigmoid(up_proj(silu(down_proj(h))))`, `h * gate`, rank `GRUG_GATED_NORM_RANK = 128` fixed in code (ext:grug_moe/gh__marin-community__marin/skyrl_train_models_grug_moe.py:54, :402-410); applied after input_layernorm and post_attention_layernorm on every layer (:843-846, :857, :859), after the embedding norm (:893, :974) and after the final norm (:896, :991).
- Both: a data-dependent per-channel gate on the normed sublayer input; p5 reads a plain RMSNorm pre-norm.

## 2. Classifier change
- reads: model_type; axk2 `gated_norm_rank`.
- new: FEATURES key `gated_norm` ("wall", "free"), in `_feats`; dims fields `gnorm_rank` (int) and `gnorm_sites` ("layers" | "layers+emb+final"), in `_pass5_dims`.
- detection:
```python
def detect_gated_prenorm_lowrank(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "axk2":
        return ("layers", num(lm.get("gated_norm_rank")) or 16)
    if model_type == "grug_moe":
        return ("layers+emb+final", 128)
    return None
```
- apply note: p5:model_graph.py FEATURES (after "out_gate", ~1604); `_feats` beside the out_gate test (~1750): `if mt in GATED_NORM_TYPES: f.add("gated_norm")`; `_pass5_dims` out["gnorm_rank"], out["gnorm_sites"]; in `emul_matrix` (feats mask, ~4229) when both have gated_norm require `gnorm_rank_h >= gnorm_rank_t` and sites_h ⊇ sites_t; add both to `_point_key_dims`.

## 3. Relation rule
- target side: a gated-norm target needs a host whose gate sites cover its sites with rank >= its rank (extra rank rows / cols zero; a host site the target lacks is switched off, below). Grug host plays A.X-K2 (128 >= 16, sites ⊇); A.X-K2 host does not play Grug (no embedding / final gates).
- host side: fc2 (up_proj) = 0 gives gate sigmoid(0) = 1/2 at every site; the 1/2 folds into the preceding RMSNorm weight (2 w) -- the plain pre-norm, exact. (Grug's embedding gate: 1/2 folds into embed_norm's weight; untied tables, so also foldable.)
- kind: target side wall (no gated host) / exact (gated host with rank and sites covering); host side exact.
- cost: none (rank padding is free inside the host's gate).
- free by the config rules: gated_norm_rank is shape-changing (gate weights) -- not free; the gate itself is code.
- theorem: host side `out_gate_off` (W_g = 0, sigma = 1/2, weight doubled; PROOFS "Every allowed rule"), `shared_gate_half`; rank `fullrank_plays_lowrank`, `lowrank_pad` (PROOFS "Round-2 rules"). Target side: OPEN (as out_gate / shared_gate target <- ungated host, "Walls of the edge-case audit").

## 4. Test fixtures
- flag: Baekpica/A.X-K2-VQ-GGUF, skt/A.X-K2-NVFP4 -> ("layers", 16); the 7 open-athena Grug / Snowball repos -> ("layers+emb+final", 128) (fixtures use true: the tuple is not JSON).
- not flag: Interchained/Muse-Spark-Open-816B-A42B-GGUF (gates the attention output, not the norm); deepseek-ai/DeepSeek-Math-V2 (deepseek_v32: A.X-K2's parent decoder, plain RMSNorm).
- scan: exactly the 9 live repos.

## Verdict
real. Fewer edges: the 9 points lose every host without a gated norm as targets (A.X-K2 keeps Grug hosts on rank / sites, if the rest of the slots allow); as hosts they are unchanged (exact switch-off). Region: classification / FFN-attention inputs (norm).
