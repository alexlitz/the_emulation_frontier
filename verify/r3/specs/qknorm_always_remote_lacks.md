# qknorm_always_remote_lacks
live 1 / placed 1 (doge 1); direction both; batch a1_typetables

## 1. What the code computes
- SmallDoge/Doge-320M/config.json:7-9 `auto_map: {"AutoModelForCausalLM": "modeling_doge.DogeForCausalLM", ...}`: the repo runs its own remote modeling_doge.py, not transformers' Doge.
- Remote SmallDoge/Doge-320M/modeling_doge.py:243 DogeAttention: :284-285 `query_states = q_proj(x).view(...)`, `key_states = k_proj(x).view(...)` then :289 RoPE -- no q_norm / k_norm anywhere in the class; :264 `A = Parameter(zeros(num_attention_heads))`, :265-266 dt_proj -> num_attention_heads (dynamic-mask parameters, not this finding).
- transformers Doge has the norms: tf5:models/doge/modeling_doge.py:280-281 `q_norm / k_norm = DogeRMSNorm(head_dim)`, :294-295; also tf4.57 models/doge/modeling_doge.py:262-263, :278-279.
- SmallDoge/Doge-20M has no auto_map (config) and runs transformers' Doge: qknorm=head is right for it.
- p5 lists doge in QKNORM_ALWAYS and QKNORM_HEAD_CHECKED (p5:model_graph.py:1852-1862, :1893-1901), so Doge-320M gets qknorm=head.

## 2. Classifier change
- reads: model_type doge + the auto_map pointing at the remote modeling_doge
- new: qknorm slot "no" for those configs, in classify
- detection:
```python
def detect_qknorm_always_remote_lacks(cfg, model_type):
    if model_type == "doge" and "modeling_doge" in auto_map(cfg):
        return "no"
    return None
```
- apply note: in p5:model_graph.py classify, right after the QKNORM_KEY_NOT_SOFTMAX override (:2337-2338) add `if mt == "doge" and "modeling_doge" in _auto_map_str(d): qknorm = "no"` (a small helper joining the auto_map values of d and outer); keep doge in QKNORM_ALWAYS for transformers-run configs. A more general form: `QKNORM_REMOTE_LACKS = {"doge": "modeling_doge"}`.

## 3. Relation rule
- target side: Doge-320M as a target no longer needs a qk-norm host (any plain host of the right shape plays it, given its other slots); qk-norm hosts play it only by the eps lever (limit).
- host side: Doge-320M as a host can no longer play head-qknorm targets (it was over-credited: it has no norm to set).
- kind: exact for the plain hosts now allowed; the removed edges were wrong (no construction)
- cost: none
- free by the config rules: nothing (the norm is absent from the code, no switch)
- theorem: `knorm_not_plain` / `qknorm_out_of_regime` (PROOFS.md "qk-norm target <- plain host": why the old host-side edges were wrong); `epsLever_bound` (head host -> Doge-320M target, limit)

## 4. Test fixtures
- flag: SmallDoge/Doge-320M -> "no" (remote modeling_doge, no q / k norm)
- not flag: SmallDoge/Doge-20M (same type, no auto_map: transformers Doge with q / k norm); Qwen/Qwen3-0.6B (unrelated head type)

## Verdict
real. One model: Doge-320M loses its head-qknorm targets as a host (over-credit removed, dozens of edges since every Qwen3 / Gemma-3 sized target was reachable on shape grounds) and gains plain-attention hosts as a target (over-wall removed). Note Doge-320M's dynamic-mask A / dt_proj per query head is a separate issue (not this batch).
