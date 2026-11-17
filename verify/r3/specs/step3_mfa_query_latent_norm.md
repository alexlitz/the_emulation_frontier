# step3_mfa_query_latent_norm
live 1 / placed 1 (step3_text 1, under the step3_vl wrapper); direction over-credit; batch a8_mla_dsv4

## 1. What the code computes
- stepfun-ai/step3/modeling_step3.py:403 `self.num_key_value_heads = 1` (hard-coded MQA; num_attention_groups is not read here)
- :407 `self.q_size = getattr(config, "share_q_dim", self.head_dim)`; configuration_step3.py:57,83 `share_q_dim: int = 2048` (config default; the snapshot sets 2048)
- :412 `q_proj = Linear(hidden, q_size)`; :417 `inter_norm = Step3vRMSNorm(q_size, eps=config.rms_norm_eps)` (weighted: :265 `self.weight`, :272 `output * self.weight`; eps default 1e-5, configuration_step3.py:49); :419 `wq = Linear(q_size, head_dim * num_attention_heads)`
- forward :433-438: `q = wq(inter_norm(q_proj(x)))`, then RoPE on q and the single K head (:441). So the query is a rank-share_q_dim (2048) bottleneck with a weighted RMSNorm on the latent: MLA's q_lora path (q_a_proj -> q_a_layernorm -> q_b_proj) on an MQA head.
- p5 today: kv = "MQA" (nkv = num_attention_groups 1, p5:model_graph.py:2187-2188, :2261), dims_of q_latent = 0 (it is set only when kv_lora_rank, :3315), q_cap = 64 x 256 = 16384: a plain full-rank MQA query. A plain MQA host with the shapes is credited with Step-3; Step-3 as a host is credited with full-rank queries it cannot form (rank <= 2048 < hidden 7168) and with q-norm-free targets.

## 2. Classifier change
- reads: model_type step3_text / step3_vl (code fact), share_q_dim (default 2048)
- new: dims field `"q_lat_norm"` (the normed query-latent rank; 0 = none), in `dims_of`; compared in emul_matrix
- detection:
```python
def detect_step3_mfa_query_latent_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in ("step3_text", "step3_vl"):
        return num(lm.get("share_q_dim")) or 2048
    return None
```
- apply note: p5:model_graph.py `dims_of` return dict (beside "q_latent", :3315): `"q_lat_norm": (_num(d.get("share_q_dim")) or 2048) if mt in ("step3_text", "step3_vl") else 0`. MLA keeps its own q_latent (do NOT reuse q_latent: the MLA<->plain kv wall and the MLA latent rule key on it). Add "q_lat_norm" to the compared dims list (:4616-4619). In emul_matrix, next to the "kv MLA latents" rule (:4366-4374), for non-MLA pairs: `ql_t = dim("q_lat_norm")`; ok &= `(ql_t[:, None] == 0) | ((ql_h[None, :] > 0) & (ql_h[None, :] >= ql_t[:, None]) & (mult(ql_h, ql_t) | lever_h))` (the MLA latent rule's multiple / eps-lever logic) and, for a host with ql_h > 0 and a target with ql_t == 0: `eps_h & (ql_h >= min(hidden_t, q_cap_t))` (switch the latent norm off by the lever; the bottleneck must hold the target's query rank).

## 3. Relation rule
- target side: a q_lat_norm target needs a host with a normed query latent at least as wide (an integer multiple exact; any wider one on an eps-lever host, payable), with its other MQA dims
- host side: a q_lat_norm host plays a plain-query target only with the latent norm switched off (eps lever, limit) and q_lat_norm_h >= min(hidden_t, q_cap_t) (the bottleneck rank)
- kind: wall (target vs plain hosts); exact (multiple) / limit (lever) between latent hosts; limit (host -> plain target)
- cost: none (dims compare)
- free by the config rules: share_q_dim is shape-changing (not free); eps is free (the lever)
- theorem: target wall: `mla_latent_norm_not_affine` argument on the query side (a plain query is linear in the normed input, the latent-normed one is not; needs theorem: the q-side instance, "W2 N(W1 x) != A x for rank W1 < d on a pre-norm sphere"); host lever: `leverAttn_layer_bound` (q-latent norm instance named in the PROOFS.md row); rank: `fullrank_plays_lowrank` / `lowrank_pad` (a wider bottleneck plays a narrower one, zero padding) with `qknorm_pad` / `rmsNorm_eps_inj` for the padded norm's eps (multiple: `rmsNorm_rep`)

## 4. Test fixtures
- flag: stepfun-ai/step3 -> 2048 (the only Step-3 MFA config; step3_vl wrapper, step3_text LM)
- not flag: stepfun-ai/Step-3.5-Flash, stepfun-ai/Step-3.5-Flash-Base-Midtrain (step3p5: GQA, plain q_proj); deepseek-ai/DeepSeek-V3 (MLA q_lora: q_latent, not this field); skt/A.X-K2-NVFP4

## Verdict
real (over-credit): Step-3 as a target loses the plain MQA hosts with q_cap >= 16384 and head_dim >= 256 (few; its MoE / dims already wall most); as a host it loses full-rank-query targets wider than rank 2048 and keeps the rest only under the eps lever. Fewer edges, roughly 0-10 pairs.
