# norm_placement_keys_unread
live 22 / placed 23 (ouro 4, molmo 3, plamo3 3, beam 2, bolmo 2, minimax 2, needle 2, cost_wise_gemma 1, domynedge 1, ethoric 1, iquest_q1 1, kolibri1 1); direction both; batch a4_norms

## 1. What the code computes
Sandwich (pre + post-sublayer norm), fixed in code, no config key -> today no `post_norm`:
- PLaMo-3: `post_mixer_norm` / `post_mlp_norm` (offsets 1/5, 1/5^1.5 fold into the gains) applied to the sublayer outputs (pfnet/plamo-3-translate-2b/modeling_plamo.py:575-577, :598, :608).
- Ethoric: attn_out_norm / mlp_out_norm, `x = x + self.attn_out_norm(a)`, `x + self.mlp_out_norm(self.mlp(h))` (techdotus/Ethoric-10M/modeling_ethoric.py:120-122, :155, :160); config peri_ln true, not read by the code.
- Domyn-Edge: post_attention_norm / post_feedforward_norm (domyn/Domyn-Edge-Base/modeling_domynedge.py:284-285, :303, :306).
- Ouro: input_layernorm_2 on the attention output, post_attention_layernorm_2 on the MLP output (ByteDance/Ouro-1.4B-Thinking/modeling_ouro.py:399, :405, :436, :443).
- bge-reranker-v2.5-gemma2-lightweight: builds transformers' `Gemma2DecoderLayer` (BAAI/.../gemma_model.py:56, :258).
- Beam: post_attention_layernorm on the attention output, pre/post_feedforward_layernorm (tfmain:models/beam/modular_beam.py:307, :335, :341).
- Kolibri-1: `Kolibri1DecoderLayer(Glm4DecoderLayer)` (tfmain:models/kolibri1/modular_kolibri1.py:150) -> post_self_attn_layernorm / post_mlp_layernorm (tfmain:models/glm4/modeling_glm4.py:76-77, :103, :109); glm4 itself is already in POST_NORM_TYPES.
Post-only (no pre-norm), key not read:
- Molmo `norm_after` (Molmo-7B-O-0924 true; 7B-D / 72B false): `if not self.config.norm_after:` pre-norm, `if self.config.norm_after:` norm the output (allenai/Molmo-7B-D-0924/modeling_molmo.py:478, :500, :514, :527; passed through at :2029).
- Bolmo `block_type` (default and both snapshots "reordered_norm"): only post norms, `pre_norm = self.block_type != "reordered_norm"` (allenai/Bolmo-1B-Stage1/modeling_bolmo.py:310-311, :333).
Normed residual stream (post-LN), read today as a sandwich:
- MiniMax (tf5): `hidden_states = self.input_layernorm(hidden_states); residual = hidden_states` ... `residual * alpha + attn * beta`, then `post_attention_layernorm`, `residual = hidden_states` (tf5:models/minimax/modular_minimax.py:398-413): the residual IS the normed stream, unconditionally (no postnorm key in tf5). Remote MiniMax-Text-01 / M1: the same iff `self.postnorm` (modeling_minimax_text_01.py:963, :1025, :1044; modeling_minimax_m1.py same lines). alpha (3.5566) / beta are scales: free.
- IQuest-Q1 (layers without use_sandwich_norm): `x = self.attention_norm(h_in); h = x + self.attn_out_norm(attn(x)) * s_a; output = h + ffn_out * s_f` (ext:iquest_q1/gh__IQuestLab__vllm-iquest-q1/model.py:498-505): the raw stream is discarded (post-LN) and the attention output is post-normed; the sandwich layer (:486-496) has both post norms.
Attention-only post-norm:
- Needle: `post_attn_norm` after attention, no norm after the Hadamard MLP (ext:needle/gh__cactus-compute__cactus/architecture.py:363-370).
- lmstudio-community/MiniMax-M2-MLX-6bit is model_type "minimax" but architectures MiniMaxM2ForCausalLM: mlx-lm `minimax.py` is plain pre-norm `r = x + self.self_attn(self.input_layernorm(x))` (_repos/mlx-lm-main/mlx_lm/models/minimax.py:220). Not live.

## 2. Classifier change
- reads: model_type; molmo `norm_after`; bolmo `block_type`; minimax_text_01 / minimax_m1 `postnorm`; architectures (MiniMaxM2).
- new: POST_NORM_TYPES += {plamo3, ethoric, domynedge, ouro, cost_wise_gemma, beam, kolibri1, needle, iquest_q1}; POST_ONLY by config (molmo norm_after, bolmo reordered_norm); FEATURES `post_ln` for minimax (non-M2 arch) / postnorm-true remote MiniMax / iquest_q1 (and NOT post_norm for MiniMax); new FEATURES key `post_norm_attn` ("limit", "wall") for needle / iquest_q1. All in `_feats` / tables.
- detection:
```python
_G1_POST_NORM_BY_CODE = {"plamo3", "ethoric", "domynedge", "ouro", "cost_wise_gemma", "beam", "kolibri1"}
_G1_POST_LN_REMOTE = {"minimax_text_01", "minimax_m1"}

def detect_norm_placement_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in _G1_POST_NORM_BY_CODE:
        return "post_norm"
    if model_type == "molmo" and lm.get("norm_after") is True:
        return "post_only"
    if model_type == "bolmo" and (lm.get("block_type") or "reordered_norm") == "reordered_norm":
        return "post_only"
    if model_type == "minimax" and not any("MiniMaxM2" in a for a in archs(cfg)):
        return "post_ln"
    if model_type in _G1_POST_LN_REMOTE and truthy_flag(lm.get("postnorm")):
        return "post_ln"
    if model_type == "iquest_q1":
        return "post_ln+post_norm+post_norm_attn"
    if model_type == "needle":
        return "post_norm+post_norm_attn"
    return None
```
- apply note: p5:model_graph.py `POST_NORM_TYPES` (~1968) and `POST_ONLY_TYPES` (~1975); in `_feats` (~1820) the `post_only` / `post_norm` branch: add the molmo / bolmo config tests to the post_only test, and make MiniMax-family `postnorm` give `post_ln` INSTEAD of `post_norm` (today `d.get("postnorm") is True` adds post_norm); extend the post_ln test (~1824). New FEATURES entry `post_norm_attn`; in emul_matrix a target's `post_norm_attn` the host lacks is allowed only when the host's eps_lever dim is 1 (its MLP post-norm switched off, as `host_off`).

## 3. Relation rule
- target side: post_norm / post_only / post_ln targets need the same feature on the host (all wall/wall, existing rows). post_norm_attn target on a full-sandwich host: the host's FFN post-norm switched off by the eps lever (limit).
- host side: as the existing rows; a post_norm_attn host has no FFN post-norm to play a full-sandwich target: wall.
- kind: wall both ways for the existing features; post_norm_attn: limit (target side, eps lever) / wall (host side).
- cost: none.
- free by the config rules: norm on/off flags are free only where the code reads one; here placement is fixed in code (or a key that changes the block: norm_after, block_type, postnorm) -> feature switches, not free.
- theorem: post_norm OPEN (PROOFS "Walls of the edge-case audit": post_norm both directions, measured 14 % top-1); lever `postNorm_layer_bound`, `postNorm_model_bound`, `postNorm_greedy` ("The eps lever" table); post_ln / post_only: needs theorem (no Lean row; FEATURES text only).

## 4. Test fixtures
- flag: Molmo-7B-O-0924 -> post_only; Bolmo-1B, Bolmo-1B-Stage1 -> post_only; plamo-3-translate-2b, Ethoric-10M, Domyn-Edge-Base, Ouro-1.4B, bge-reranker-v2.5-gemma2-lightweight, Beam-1.0-501B-A23B, Kolibri-1-mlx -> post_norm; MiniMax-Text-01-hf, MiniMax-M1-40k -> post_ln; IQuest-Q1 -> post_ln+post_norm+post_norm_attn; needle2 -> post_norm+post_norm_attn.
- not flag: Molmo-7B-D-0924, Molmo-72B-0924 (norm_after false); lmstudio-community/MiniMax-M2-MLX-6bit (M2 code, pre-norm); AdarshSingh7647/Eklav-9B-Math-CotGen (glm4: already post_norm in p5, not this finding).
- scan: 23 repos = the 22 live + MiniMax-Text-01 / M1-40k / M1-80k (remote, postnorm true; not in the sweep list, same code fact) - lmstudio M2.

## Verdict
real, not fixed in round2-p5 (none of these types is in POST_NORM_TYPES / POST_ONLY_TYPES; MiniMax still maps postnorm -> post_norm). Fewer edges: ~22 targets / hosts move from plain pre-norm to walled post_norm / post_only / post_ln (they lose every pre-norm partner, gain sandwich partners: Ouro / PLaMo-3 / Beam / Kolibri / Domyn / Ethoric / bge pair with Gemma-2/3 / GLM-4 hosts). MiniMax (4) moves from post_norm (sandwich partners) to post_ln (OPT-350m / GPT-1 / CENO only): both.
