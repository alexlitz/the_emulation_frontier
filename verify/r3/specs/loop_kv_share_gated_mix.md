# loop_kv_share_gated_mix
live 1 / placed 1 (iquestloopcoder 1); direction over-credit; batch a11_recurrent_loops

## 1. What the code computes
(IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct/modeling_iquestloopcoder.py)
- `IQuestLoopCoderModel.__init__`: `loop_num = getattr(config, "loop_num", 2)`, `loop_window_size =
  getattr(config, "loop_window_size", 64)`, one `LoopGateProjection` per layer (:875-881).
- forward: `for loop_idx in range(loop_num)` over all 80 layers, gate_proj passed for loop_idx > 0, final norm only
  after the last loop (:953-975).
- Attention dispatch: loop 0 -> `forward_loop1` (plain GQA, fills the shared KV cache); loop >= 1 -> `forward_loop2`
  (:492-499, :554).
- `forward_loop2`: global attention of q over loop 1's K/V of the SAME layer (all positions), local attention over the
  current loop's K/V kept to the last loop_window_size tokens, `mixed = local * (1 - gate) + global * gate` (:700).
- `LoopGateProjection`: weight [heads, head_dim] and bias [heads] (:433-434), `gate = sigmoid(einsum('bhsd,hd->bhs', q,
  w) + b)` (:449-451): a per-head, per-position data-dependent mix of two softmaxes.
- Cache: a shared cache for loop 0 and a local cache of (loop_num - 1) x layers windowed entries (:145-157, :214-235).
- p5: loop=looped, eff 160 / param 80, feats '' (p5:model_graph.py `_looped` via loop_num, :682-690); the loop note
  (:3828-3833) already names "IQuest-Loop's loop-1 KV + window" a wall for an untied target on it as a HOST.

## 2. Classifier change
- reads: model_type iquestloopcoder; loop_num, loop_window_size.
- new: FEATURES key `loop_kv_mix` ("wall", "wall"), in `_feats`; dims `loop_window` = loop_window_size (for the record).
- detection:
```python
def detect_loop_kv_share_gated_mix(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "iquestloopcoder" and (num(lm.get("loop_num")) or 2) > 1:
        return num(lm.get("loop_window_size")) or 64
    return None
```
- apply note: p5:model_graph.py FEATURES (:1597): `"loop_kv_mix": ("wall", "wall", "loops >= 2 mix attention over loop
  1's K/V of the same layer with a local-window attention over the current loop, by a per-head sigmoid gate of q
  (IQuest-Loop)")`; `_feats` (:1741): `if mt == "iquestloopcoder" and (_num(d.get("loop_num")) or 2) > 1`.

## 3. Relation rule
- target side: unrolled on an untied host, layer 80 + i must read layer i's K/V (cross-layer KV, crosslayer_kv) AND its
  own windowed K/V, and form (1 - g) A_local + g A_global with g = sigmoid(q . w): two softmaxes per head plus a
  q-dependent convex mix. No plain (or CLA) host layer does that. Wall.
- host side: an IQuest host plays a plain looped / untied target only if loop 2's attention equals fresh full
  attention: gate 0 gives the local window (64 tokens) only, gate 1 the loop-1 keys; neither is fresh full attention
  beyond 64 tokens. Wall (as p5 already says for untied targets).
- kind: wall both ways.
- cost: none.
- free by the config rules: loop_window_size is a window size (free) -- a window >= context makes the local branch full
  attention over the current loop, which removes the window but not the gated mix with loop-1 K/V; still a wall.
- theorem: needs theorem `gated_two_softmax_mix_not_single`: (1 - g(q)) softmax(q K1) V1 + g(q) softmax(q K2) V2 is not a
  single softmax head over any one K / V set (except g constant 0 / 1).

## 4. Test fixtures
- flag: IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct -> 64 (the only loop-coder config).
- not flag: IQuestLab/IQuest-Q1 (same org, iquest_q1, no loop keys); Nanbeige/Nanbeige4.2-3B (loop with fresh
  attention; its loop_share_kv option is null); ByteDance/Ouro-1.4B; ProCreations/Dory-Beam-4B.

## Verdict
real. Fewer edges for 1 point: IQuest-Loop as a TARGET loses its exact looped <- untied edges (untied hosts with
eff_layers >= 160 at hidden >= 5120: a few tens of pairs); as a host p5 already walls it for untied targets. Region:
attention (loop).
