# multimax_logits
live 1 / placed 1 (ernie5_5_moe 1); direction over-credit; batch a4_norms

## 1. What the code computes
- `Ernie5_5ParallelLMHead` registers buffers `multimax_ranges` r (4) and `multimax_ts` t (4), fp32, loaded from the checkpoint (ext:ernie5_5_moe/gh__baidu__vLLM-ERNIE/model.py:84-91).
- `multimax(l) = l + t0 relu(r0 - l) + t1 relu(l - r1) + t2 relu(r2 - l)^2 + t3 relu(l - r3)^2`, elementwise on the logits (:93-101); applied in `compute_logits` / `compute_logits_local` (:750-758). sglang applies the same kernel to the logits and to the MTP draft (ext:ernie5_5_moe/sglang/ernie55.py:936-943; ernie55_nextn.py:243).
- Config: `"multimax": "lm_head"` (ext:ernie5_5_moe/hub__baidu__ERNIE-5.5-Flash-Base/config.json:97); sglang lists it as a config switch (sglang/config_ernie55.py:63, :72).
- Effect on the relation's metric (top-1 / greedy): f'(l) = 1 - t0 [l < r0] + t1 [l > r1] - 2 t2 (r2 - l)[l < r2] + 2 t3 (l - r3)[l > r3]. Monotone (so argmax-preserving, like the final softcap) iff t0 <= 1 and t2 = 0, or the logits never reach the region where f' < 0; with t2 > 0, f(l) -> +inf as l -> -inf, so a very low logit can overtake the top one. The checkpoint's r / t are not local (no weights in ~/.cache/huggingface/hub/models--baidu--ERNIE-5.5-Flash-Base): whether the map is monotone on the reachable range is unknown here.

## 2. Classifier change
- reads: config `multimax` (truthy = applied), model_type ernie5_5*.
- new: FEATURES key `multimax` ("wall", "free"), in `_feats`.
- detection:
```python
def detect_multimax_logits(cfg, model_type):
    v = get(cfg, "multimax")
    if model_type.startswith("ernie5_5") and v:
        return str(v)
    return None
```
- apply note: p5:model_graph.py FEATURES (near "selfcond", ~1720); `_feats` end (~1836): `if mt.startswith("ernie5_5") and d.get("multimax"): f.add("multimax")`. If the checkpoint's t is later read and f is strictly increasing on the logit range, relax to ("free", "free") with the softcap "final <- no" justification (argmax).

## 3. Relation rule
- target side: a multimax target on a plain host: the host's logits are l; greedy / top-1 agree iff f preserves the argmax on the reached logits. Unknown -> wall (conservative). With f strictly increasing it would be exact for greedy (the `final` softcap argument).
- host side: t = 0 is the identity: exact (the buffers are weights, set by the emulator).
- kind: target side wall (argmax-exact only if f is monotone, unverified); host side exact.
- cost: none.
- free by the config rules: the config key only switches the map on; t / r are weights (t = 0 is the switch-off). Not a listed free value.
- theorem: host side: trivial identity (needs theorem `multimax_zero`: t = 0 -> f = id). Target side if monotone: `softcap_strictMono`-style generalisation, needs theorem `strictMono_logit_greedy` (any strictly increasing elementwise logit map preserves every argmax; `final_softcap_greedy` is the tanh instance).

## 4. Test fixtures
- flag: baidu/ERNIE-5.5-Flash-Base -> "lm_head".
- not flag: baidu/ERNIE-4.5-21B-A3B-PT (ernie4_5_moe: plain lm_head).

## Verdict
real, tiny. Fewer edges: the one ERNIE-5.5 target loses plain hosts (its code also runs mhc hyper-connections, sglang/ernie55.py:40, and a 512-expert MoE, so few plain hosts remain: ~0-few visible edges; not checked against the p5 matrix); as a host unchanged. Revisit with the checkpoint's buffers. Region: classification (output head).
