# router_group_balanced
live 1 / placed 1 (interns1_pro_text 1); direction over-credit; batch a9_router

## 1. What the code computes
- internlm/Intern-S1-Pro/modeling_interns1_pro.py:70-73: the MoE block uses InternS1ProMoeTextGroupedRouter iff
  `router_n_groups > 1` (snapshot 8), else the plain Qwen3-VL-MoE top-k router.
- :350-359 grouped router reads num_experts_per_tok, num_experts, norm_topk_prob, router_n_groups.
- :361-388 forward: softmax over all E = 512 experts; view as (G, E/G); `topk(k / G)` INSIDE every group (k = 8,
  G = 8: exactly one expert from each 64-expert group); gather; renormalise iff norm_topk_prob (`/ (sum + 1e-20)`).
- No group score and no group drop: every group contributes k/G experts. A plain top-k router cannot force one
  expert per group; a DeepSeek-V3 group limit (top2sum, topk_group < n_group) drops groups instead.
- p5 reads only n_group / num_expert_group / topk_group (`_moe_sig` ~939-942): Intern-S1-Pro gets n_group 0, i.e. a
  plain softmax top-8 R router.

## 2. Classifier change
- reads: router_n_groups, model_type interns1_pro / interns1_pro_text
- new: group fields in _moe_sig: n_group = router_n_groups, topk_group = router_n_groups, group_cap "per_group";
  router_group feature in _feats
- detection:
```python
def detect_router_group_balanced(cfg, model_type):
    lm = lm_of(cfg)
    if not _n_experts(lm):
        return None
    if num(lm.get("router_n_groups")) > 1 and model_type.startswith("interns1_pro"):
        return "per_group"
    return None
```
- apply note: p5:model_graph.py `_moe_sig` (~939): `R = _num(d.get("router_n_groups"))`; if R > 1 (interns1_pro*):
  G = g = R, active = True, gcap = "per_group" (bypass `0 < g < G`). `_feats` (~1789): add router_group when
  router_n_groups > 1. moe_why needs no new branch: the group_cap equality, the top2sum m != 1 rule (not taken) and
  `Eh % (m * n_group)` already express "host per_group, k_h = m k_t, copies inside the target's groups".

## 3. Relation rule
- target side: a per_group target needs a per_group host with router_n_groups = G_t (free), k_h = m k_t, E_h = m E_t
  (copies of group g's experts inside host group g): each host group picks top-(m k_t / G) = the m copies of the
  target's top-(k_t / G) (strict order)
- host side: a per_group host plays ungrouped targets by router_n_groups = 1 (code :70-73 falls back to the plain
  router); it cannot play a top2sum group-limited target (different code)
- kind: exact (per_group <- per_group; plain <- per_group host at G = 1); wall (per_group <- plain / top2sum host)
- cost: none
- free by the config rules: router_n_groups is a routing number (free on the host); the per-group code is not
- theorem: `moeR_topk_unique`, `moeR_copies_topk` applied per group | needs theorem: moeR_per_group_copies (per-group
  top-(k/G) with in-group copies selects exactly the copies) and per_group_not_plain (no plain top-k router forces one
  expert per group; OPEN, kept as a wall)

## 4. Test fixtures
- flag: internlm/Intern-S1-Pro -> "per_group"
- not flag: deepseek-ai/DeepSeek-V3 (top2sum group limit), Qwen/Qwen3-30B-A3B (plain), inclusionAI/Ling-1T

## Verdict
real, small. Fewer edges: Intern-S1-Pro stops being played by plain softmax-R hosts with >= 512 x m experts (a
handful of 512-expert hosts: Qwen3.5 / 3.6-MoE-class, Ling-2.6); as a host it is unchanged for plain targets.
