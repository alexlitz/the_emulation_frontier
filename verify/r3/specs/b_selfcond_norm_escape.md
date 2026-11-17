# b_selfcond_norm_escape
live 0 / placed 3 (diffusion_gemma_text 2, sdar 1 (QwenDLM-4B-ORL)); direction over-wall (claimed); batch bd

## 1. What the code computes
- DiffusionGemma self-conditioning: `post_norm = DiffusionGemmaRMSNorm(hidden, eps=config.rms_norm_eps, with_scale=False)`; `combined = inputs_embeds + sc_signal; return self.post_norm(combined)` (tf5:models/diffusion_gemma/modeling_diffusion_gemma.py:804-805, :820-823). The RMSNorm is `x * (mean(x^2) + eps)^-0.5` (:148-159), weight-free; eps is the SAME config `rms_norm_eps` as every stream norm (decoder `self.norm`, :1218).
- With no previous logits the signal is zeros and the norm still runs: `soft_embeddings = torch.zeros_like(inputs_embeds)`; `inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)` (:1268-1272); `inputs_embeds = embed_tokens(ids)` scaled by `hidden_size**0.5` (:1207-1212, :758, :1259).
- The head is tied to the decoder table: `_tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}` (:1597), `tie_word_embeddings: bool = True` (configuration_diffusion_gemma.py:93, :212). Both snapshot DiffusionGemma configs are tied; QwenDLM-4B-ORL (sdar, self_conditioning_post_norm true) is tied too (its modeling_sdar.py is not public; p5 takes the same norm).
- p5 relation: `FEATURES["selfcond_norm"] = ("wall", "wall", ...)` (p5:model_graph.py:1724-1727), set for diffusion_gemma* or `self_conditioning_post_norm` (:1829-1833).
- Construction (`selfcond_norm_escape`): zero the path (down_proj = 0), host row e' = a e with a = sqrt(eps / (1 - ms(e))) gives RMSNorm_eps(e') = e for a target row with ms(e) < 1. Extension (not in PROOFS.md): rescale the host stream by s (sublayer outputs x s folded into o_proj / down_proj, every stream norm's eps = eps_t s^2: RMSNorm_{eps s^2}(s x) = RMSNorm_eps(x)); the selfcond norm shares that eps, so a^2 = eps_t s^4 / (1 - s^2 ms(e)) works for EVERY row once s^2 max ms < 1. The ms < 1 condition disappears; no per-model embedding table is needed.

## 2. Classifier change
- reads: diffusion_gemma* / `self_conditioning_post_norm`, and the head tie (`tie_word_embeddings`, class default True).
- new: none applied today (keep the wall). Ready rule: a host-off lever for "selfcond_norm" when the host is UNTIED and the target's stream-norm eps > 0 (dims `selfcond_untied`, in `_pass5_dims`; merged into `host_off` at emul_matrix :4229).
- detection:
```python
def detect_b_selfcond_norm_escape(cfg, model_type):
    lm = lm_of(cfg)
    norm = model_type.startswith("diffusion_gemma") or lm.get("self_conditioning_post_norm") is True
    if not norm:
        return None
    tie = lm.get("tie_word_embeddings", (cfg or {}).get("tie_word_embeddings"))
    if tie is None:
        tie = True
    return "untied" if tie is False else None
```
- apply note: only if a host ever flags: p5:model_graph.py emul_matrix :4229 `offs` gains `{"selfcond_norm"}` for `selfcond_untied` hosts whose target has eps > 0 (an untied host also needs an untied target: tied_emb host side is already a wall). FEATURES["selfcond_norm"] stays ("wall", "wall").

## 3. Relation rule
- target side: unchanged wall (a selfcond_norm target's self-conditioning input is read by no host without it; FEATURES "selfcond" target side is already a wall).
- host side: an untied selfcond_norm host plays a plain target exactly (stream rescale s, eps_t s^2, a per row); a tied host makes logits a_i (h . e_i) with a per-row a_i: with the final-norm gain folded, a_i -> constant as s -> 0, i.e. a LIMIT (relative logit error ~ s^2 ms_i / 2), not measured.
- kind: exact (untied host, eps_t > 0) | limit, unmeasured -> wall (tied host) | wall (eps_t = 0: ms of every normed row is 1).
- cost: none (weights and the free eps only).
- free by the config rules: the eps (norm eps is free) and the weights; no feature switch.
- theorem: `selfcond_norm_escape` (PROOFS.md Round-2 rules, ms < 1 at s = 1); needs theorem: "for eps > 0 and any rows e_i, s with s^2 max_i ms(e_i) < 1, eps_h = eps s^2, a_i^2 = eps s^4 / (1 - s^2 ms(e_i)): RMSNorm_{eps_h}(a_i e_i) = s e_i, and RMSNorm_{eps_h}(s x) = RMSNorm_eps(x)" (the rescale composes with `rmsNorm_scale`-type facts already used by the eps lever).
- interaction: on DiffusionGemma the sandwich post_norm / vnorm are switched off by the eps lever, which wants a LARGE eps (stream scaled up); the escape wants s^2 ms < 1. The lever can instead shrink the post-norm inputs (o_proj scaled down) at a small eps, but that composition is unproved.

## 4. Test fixtures
- flag: none in the snapshot (every selfcond_norm host is tied).
- not flag: yifanyu/DiffusionGemma-26B-A4B-it-ORL, nvidia/diffusiongemma-26B-A4B-it-NVFP4 (tied: escape not exact); yifanyu/QwenDLM-4B-ORL (tied); yifanyu/LLaDA2.1-mini-ORL (self-conditioning without the post norm: selfcond only, host side already free); JetLM/SDAR-8B-Chat (sdar without self-conditioning); Continuous-Rivals-Discrete/langflow-owt (self_conditioning key, no post norm).

## Verdict
real in principle (the wall is not tight for an untied host, and the stream rescale removes the ms < 1 condition, so
no measured embedding table is needed), but keep the wall: 0 live hosts -- the 3 selfcond_norm hosts are all tied,
where the escape is only an unmeasured limit. For scale: flipping the host side outright would add 26 (exact) / 129
(payable) shape pairs, 415 target repos, all on the 3 tied hosts -- edges the construction does not justify.
