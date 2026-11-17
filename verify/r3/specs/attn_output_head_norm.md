# attn_output_head_norm
live 2 / placed 2 (muse_spark_text 2); direction both; batch a4_norms

## 1. What the code computes
- `self.o_norm = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)` in MuseSparkTextAttention (tfmain:models/muse_spark/modular_muse_spark.py:397-400): weight-free.
- `attn_output = self.o_norm(attn_output).reshape(*input_shape, -1)` on the (…, heads, head_dim) attention output, i.e. per head over head_dim, then `attn_output * sigmoid(self.gate_proj(hidden_states))` (out_gate, already FEATURES out_gate) and `o_proj` (:445-447).
- Per token and head the softmax-mixed value o = sum_j a_j v_j is replaced by o / rms_eps(o). With a single visible key, o = v_j and the head emits N(v_j): exactly a weight-free v-norm (FEATURES vnorm's case, `vnorm_not_plain`).
- Related code not in the snapshot: BitNet `attn_sub_norm = BitNetRMSNorm(hidden_size)` on the concatenated attention output before o_proj, weighted, full width (tf5:models/bitnet/modeling_bitnet.py:177, :216).

## 2. Classifier change
- reads: model_type (code fact, no config key).
- new: FEATURES key `onorm` ("wall", "wall"), set in `_feats` for ONORM_TYPES = {muse_spark, muse_spark_text} (value "head_free"; BitNet "full_w" would be a distinct key `onorm_full` -- not needed for the snapshot); add `onorm` to EPS_LEVER_FEATS (host side, see 3).
- detection:
```python
def detect_attn_output_head_norm(cfg, model_type):
    if model_type in ("muse_spark", "muse_spark_text"):
        return "head_free"
    if model_type == "bitnet":
        return "full_w"
    return None
```
- apply note: p5:model_graph.py: new table `ONORM_TYPES` beside VNORM_TYPES (~1941); FEATURES entry after "vnorm" (~1625); in `_feats` next to the vnorm test (~1766): `if mt in ONORM_TYPES: f.add("onorm")`; `EPS_LEVER_FEATS = frozenset({"post_norm", "vnorm", "onorm"})` (~3787) only once the lever is measured for it (see 3).

## 3. Relation rule
- target side: an o-normed target needs an o-normed host: a plain head's output scales with |v|, the normed one does not (one-key context: o-norm = v-norm).
- host side: a weight-free o-norm cannot be switched off by a weight, but the eps lever works through o_proj: write v / S (v_proj rows / S), then o_norm(o / S) = o / (S sqrt(eps) sqrt(1 + u)), u = |o|^2 / (head_dim S^2 eps) -> o / (S sqrt(eps)); o_proj x S sqrt(eps) (the sigmoid gate is elementwise and commutes with the scalar). Limit, host eps > 0 kept (rms_norm_eps).
- kind: target side wall; host side limit (eps lever) -- unmeasured, so wall until a measurement passes the standard (top-1 >= 99 %, >= 90 % identical 32-token greedy, fp32).
- cost: none.
- free by the config rules: rms_norm_eps is free (the lever keeps eps > 0, needs no eps change); the o-norm itself is code, not free.
- theorem: target side `vnorm_not_plain` (one-key context: the o-norm head equals a v-norm head); host side `epsLever_eq`, `epsLever_bound`, `leverAttn_layer_bound`, `lever_model_bound` / `leverAttn_greedy` ("The eps lever" table; the normed vector here is the softmax output, linear in v, so the per-vector lever applies with o_proj absorbing S sqrt(eps)). Needs theorem for the exact instance: `oNorm_lever` (weight-free norm, factor moved into the next linear map).

## 4. Test fixtures
- flag: Interchained/Muse-Spark-Open-816B-A42B-GGUF, staccs/Muse-Spark-Open-816B-A42B-lecore-mlx -> "head_free".
- not flag: hweller1/wayfinder-glimmer-30b, RadixArk/Muse-Glimmer-q4-MLX (Muse-Glimmer: same AfmoeAttention lineage, no o_norm).

## Verdict
real, small. Fewer edges: the 2 Muse-Spark points lose any host without an o-norm (as targets; they are already walled by post_norm + residual gate + MoE specifics, so the visible change is ~0 edges), and as hosts they keep plain targets only once the lever is measured. Region: attention.
