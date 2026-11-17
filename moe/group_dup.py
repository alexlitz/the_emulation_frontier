"""Expert duplication under group-limited routing (Lean EP.moeR_group_copies), on stock DeepSeek-V3 MoE blocks.

DeepSeek-V3 routes in two levels: it keeps the top `topk_group` of `n_group` groups (group score: the sum of
the group's top-2 selection scores), then the top `k` experts inside the kept groups. serving/embed duplicates
experts only for Qwen-MoE routers (no groups). Here copy r of target group g becomes its OWN host group (g, r):

    host experts  E m,  n_group  G m,  topk_group  tg m,  top-k  k m,
    host expert  (g m + r) (E/G) + q   =  copy r of target expert  g (E/G) + q.

Each copy group holds the same experts as g, so it has g's group score. If both levels of the target's
selection are strict, every host selection is the copies of the target's (EP.moeR_group_copies). Then the
renormalised gates (norm_topk_prob) give each copy 1/m of the target's weight, so whole copies reproduce the
target (EP.moeR_dup_renorm); without renormalisation the copies' down-projections are scaled by 1/m.

* `dup_index(e, r, E, G, m)`: the host index of copy r of expert e.
* `group_dup_moe(target_moe, m)`: the host DeepseekV3MoE (a stock module of the host config).
"""
from __future__ import annotations

import torch


def dup_index(e, r, E, G, m):
    per = E // G
    g, q = divmod(e, per)
    return (g * m + r) * per + q


def host_config(cfg, m):
    h = type(cfg).from_dict(cfg.to_dict())
    h.n_routed_experts = cfg.n_routed_experts * m
    h.n_group = cfg.n_group * m
    h.topk_group = cfg.topk_group * m
    h.num_experts_per_tok = cfg.num_experts_per_tok * m
    h._attn_implementation = getattr(cfg, "_attn_implementation", "eager")
    return h


@torch.no_grad()
def group_dup_moe(target, m):
    """The duplication host of a stock DeepseekV3MoE `target` with m copies per expert."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
    cfg = target.config
    E, G = cfg.n_routed_experts, cfg.n_group
    hc = host_config(cfg, m)
    dt = target.gate.weight.dtype
    host = DeepseekV3MoE(hc).to(dt).eval()
    host.shared_experts.load_state_dict(target.shared_experts.state_dict())
    scale = 1.0 if cfg.norm_topk_prob else 1.0 / m
    for e in range(E):
        for r in range(m):
            j = dup_index(e, r, E, G, m)
            host.experts[j].load_state_dict(target.experts[e].state_dict())
            host.experts[j].down_proj.weight.mul_(scale)
            host.gate.weight[j] = target.gate.weight[e]
            host.gate.e_score_correction_bias[j] = target.gate.e_score_correction_bias[e]
    return host
