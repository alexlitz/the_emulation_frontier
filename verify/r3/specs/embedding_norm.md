# embedding_norm
live 28 / placed 41 (bloom 18, nanochat_gpt 7, grug_moe 7, mpt 5, muse_glimmer_text 2, muse_glimmer 1, gaggle 1, sn38-nanochrono 1); direction over-credit; batch a4_norms

## 1. What the code computes
- BLOOM: `word_embeddings_layernorm = LayerNorm(embed_dim, eps=layer_norm_epsilon)` (tf5:models/bloom/modeling_bloom.py:425), applied `hidden_states = self.word_embeddings_layernorm(inputs_embeds)` before layer 0 (:499); affine (gain + bias). `tie_word_embeddings: bool = True` class default (tf5:models/bloom/configuration_bloom.py:73): lm_head = E^T reads the UN-normed table.
- MPT (placed as the comparison family): MptModel has no embedding norm (tf5:models/mpt/modeling_mpt.py MptModel.__init__ / forward, hunks bc6050e0619b / d25bf2b27b5d). Not a carrier.
- nanochat_gpt: `x = rms_norm(x)` right after `embed_tokens` (jkminder/d16_optimized_537m_seed5/modeling_nanochat_gpt.py:450-451), `rms_norm = F.rms_norm(x, (x.size(-1),))` weight-free, eps None = finfo eps (:104-107). Untied.
- transformers nanochat: `hidden_states = self.norm(hidden_states)  # Additional norm before the layers` (tf5:models/nanochat/modeling_nanochat.py:406), NanoChatRMSNorm has no weight (:45-54). tie default False (configuration_nanochat.py:72).
- Gaggle: `x = rmsnorm(self.transformer["wte"](input_ids))` (castorini/gaggle-nanochat-pretrained-climbmix-20260924/modeling_gaggle.py:269), weight-free F.rms_norm (:34-36). Untied.
- sn38-nanochrono: `x = _norm(self.embed_tokens(ids))` (ext:sn38-nanochrono/github__chronollm__sn38/modeling_nanochrono.py:170), weight-free (:12-13); separate lm_head (:203).
- Muse-Glimmer: `MuseGlimmerTextNormedEmbedding` = embedding then `MuseGlimmerRMSNorm(eps=norm_eps, with_scale=False)` (tf5:models/muse_glimmer/modular_muse_glimmer.py:792-800), used as `embed_tokens` (:812-814). Snapshot configs untied.
- Grug-MoE: `embed_norm = GrugMoeRMSNorm(hidden, rms_norm_eps)` + `embed_gated_norm = GrugMoeGatedNorm` (ext:grug_moe/gh__marin-community__marin/skyrl_train_models_grug_moe.py:892-893), `hidden_states = self.embed_gated_norm(self.embed_norm(inputs_embeds))` (:974); the config refuses tied embeddings (:319-320).
- RWKV-4 / 5 / 7 (not in the sweep's list, same code fact): layer 0 `pre_ln = nn.LayerNorm(...)` applied first (tf5:models/rwkv/modeling_rwkv.py:332, :342; RWKV/v5-Eagle-7B-HF modeling_rwkv5.py:406, :416; fla rwkv7 `pre_norm` iff `norm_first and layer_idx == 0`, _repos/flash-linear-attention-main/fla/models/rwkv7/modeling_rwkv7.py:125-126). All untied.
- Not an embedding norm: LFM2 `embedding_norm` is the FINAL norm (tf5:models/lfm2/modeling_lfm2.py:528, after the layer loop).
- Key fact: the input to the norm is a table row, so N(E[tok]) is a per-token constant. Untied: it folds into the input table, E' = N(E), exactly. Tied: lm_head uses E, the stream uses N(E) -- one table cannot be both.

## 2. Classifier change
- reads: model_type (code fact; no config key), rwkv7 `norm_first`; the existing tied_emb feature.
- new: dims field `emb_norm` in {"", "ln", "rms_free", "rms", "rms_gated"}, in `_pass5_dims` (no FEATURES key: the rule depends on tied_emb on both sides, which feats_ok's per-feature pairs cannot express).
- detection:
```python
EMB_NORM_BY_CODE = {"bloom": "ln", "rwkv": "ln", "rwkv5": "ln", "rwkv6": "ln",
                    "nanochat_gpt": "rms_free", "nanochat": "rms_free", "gaggle": "rms_free",
                    "sn38-nanochrono": "rms_free", "muse_glimmer": "rms_free", "muse_glimmer_text": "rms_free",
                    "neomme": "rms_free", "inkling": "rms", "grug_moe": "rms_gated"}

def detect_embedding_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "rwkv7":
        return "ln" if lm.get("norm_first", True) is not False else None
    return EMB_NORM_BY_CODE.get(model_type)
```
- apply note: add `EMB_NORM_BY_CODE` next to NORM_BY_CODE (p5:model_graph.py ~463); `_pass5_dims` out["emb_norm"]; in `emul_matrix` (where feats_ok masks are built, ~4229) add a mask: refuse (t, h) when h.emb_norm and not `EMB_NEST(t.emb_norm, h.emb_norm)`, or when t.emb_norm and "tied_emb" in t.feats and "tied_emb" in h.feats and not EMB_NEST(t, h). EMB_NEST: equal, or rms_free <= rms <= rms_gated. Add `emb_norm` to `_point_key_dims` so points split. Replaces nothing (today no rule).

## 3. Relation rule
- target side: an UNTIED target's embedding norm folds into the host's input table (E_h = N_t(E_t)): any host, exact. A TIED target with an emb norm needs an untied host (E_in = N(E), lm_head = E^T) or a tied host with a nesting emb norm.
- host side: a host's emb norm cannot be switched off: it plays only targets whose (folded or own) first residual is N_h of something: targets with a nesting emb-norm kind. rms_gated plays rms (gate fc2 = 0, 1/2 into the weight), rms plays rms_free (weight 1). LN and RMS kinds do not nest.
- kind: target side exact (untied, or tied on an untied host) / wall (tied on a tied host without it); host side wall (rms-weighted host vs a no-emb-norm target is a candidate eps-lever limit -- E_h = E_t / S, weight S sqrt(eps) -- unmeasured, so wall).
- cost: none.
- free by the config rules: eps free; the norm itself is a code switch (no config key) -- not free.
- theorem: untied fold: needs theorem `emb_fold_untied`: for any map N, a table E' with E'[v] = N(E[v]) gives the same residual at layer 0 (trivial). Host-side wall: needs theorem: a weight-free / affine-norm embedding cannot produce rows of unequal RMS (OPEN; cf. `selfcond_norm_not_exact` for the same always-on embedding-norm shape). Lever: `epsLever_bound`, `lever_model_bound` (unmeasured here).

## 4. Test fixtures
- flag: bigscience/bloom-560m -> "ln" (tied default); projecte-aina/FLOR-760M -> "ln" (tied explicit); LumiOpen/Poro-34B-chat -> "ln"; RWKV/rwkv-4-169m-pile -> "ln"; fla-hub/rwkv7-0.1B-g1 -> "ln"; jkminder/d16_optimized_537m_seed5 -> "rms_free"; l0ulan/chinese-babylm-nanochat-d22-step10000 -> "rms_free"; castorini/gaggle-... -> "rms_free"; kingbear329/sn38-r16-c12-step687-alpha05 -> "rms_free"; hweller1/wayfinder-glimmer-30b, RadixArk/Muse-Glimmer-q4-MLX -> "rms_free"; open-athena/snowball-67b-a2b-base-262k-qk175 -> "rms_gated".
- not flag: anas-awadalla/mpt-7b, replit/replit-code-v1-3b, vinai/PhoGPT-4B (MPT: no emb norm); LiquidAI/LFM2.5-1.2B-JP (`embedding_norm` is the final norm); Interchained/Muse-Spark-Open-816B-A42B-GGUF (Muse-Spark: not the Glimmer normed embedding).
- scan: 41 repos (bloom 18, nanochat_gpt 7, grug_moe 7, muse_glimmer(_text) 3, rwkv/rwkv5/rwkv7 3, gaggle, nanochat, sn38-nanochrono). The carriers' "live" list (28) omits grug_moe and sn38-nanochrono; they carry the code fact but are untied, so only their HOST side changes.

## Verdict
real. Fewer edges: BLOOM (18, tied) targets lose every tied host without an LN emb norm; the ~20 emb-norm hosts (nanochat_gpt / nanochat / gaggle / nanochrono / Glimmer / Grug / RWKV) lose every target without a nesting emb norm. Untied emb-norm targets keep all their edges (exact fold).
