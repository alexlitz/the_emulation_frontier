# fla_num_heads_unplaced
live 1 / placed 1 (gla 1); direction over-wall; batch a11_recurrent_loops

## 1. What the code computes
(flash-linear-attention, ext:delta_net/fla/* and ext:gla/fla/*)
- fla configs name heads `num_heads` / `num_kv_heads`, never `num_attention_heads`: fla-hub/transformer-1.3B-100B
  (num_heads 32, num_kv_heads null), delta_net-1.3B-100B (16), gla-1.3B-100B (4).
- `FLAModel.forward` (ext:delta_net/fla/modeling_utils.py:153-233): embeddings -> pre-norm blocks -> final RMSNorm;
  attnres off in these configs. The transformer's attention comes from `get_attention_spec(config.attn ...)` with
  `num_heads=attn_spec['num_heads'], num_kv_heads=...` (:32-41); MLP width `256 * ceil(int(hidden * hidden_ratio *
  2/3) / 256)` when intermediate_size is null (ext:delta_net/fla/mlp.py:46-50).
- DeltaNet layer (ext:delta_net/fla/delta_net.py:75-168): key/value widths hidden * expand_k / expand_v (:116-121), head
  dims key_dim / num_heads; q, k, v short convs with silu when use_short_conv (:137-156); beta = sigmoid(b_proj x) (:239);
  q, k silu then l2-normed in the kernel (:221, :256); NO decay gate (delta rule, not gated); o_norm a plain per-head
  RMSNorm since use_gate=false (:162-166). modeling_delta_net.py:44-55 passes num_heads etc. from the config.
- p5: `classify` returns None for a softmax config without num_attention_heads / n_head (p5:model_graph.py:2182-2184), so
  transformer is dropped; delta_net is not in LINEAR_TYPES (:442) / RECURRENT_FAMILY, falls to "softmax" and is dropped
  too. `_normalize` (:2543) maps num_heads only for gpt_neo (:2568-2569).

## 2. Classifier change
- reads: architectures TransformerForCausalLM / DeltaNetForCausalLM / GLAForCausalLM; num_heads, num_kv_heads,
  hidden_ratio.
- new: a `_normalize` branch (FLA_ARCHS): num_attention_heads = num_heads, num_key_value_heads = num_kv_heads or
  num_heads, intermediate_size = fla MLP width; `LINEAR_TYPES += {"delta_net"}`, `RECURRENT_FAMILY["delta_net"] =
  "deltanet"` (new family, RC_EQ dims: see gla_rc_dims_unread).
- detection:
```python
FLA_ARCHS = ("TransformerForCausalLM", "DeltaNetForCausalLM", "GLAForCausalLM")

def detect_fla_num_heads_unplaced(cfg, model_type):
    lm = lm_of(cfg)
    if not any(a in FLA_ARCHS for a in archs(cfg)):
        return None
    if lm.get("num_attention_heads") or not num(lm.get("num_heads")):
        return None
    return num(lm.get("num_heads"))
```
- apply note: `_normalize` (p5:model_graph.py:2543), a new `elif any(a in FLA_ARCHS for a in g("architectures") or [])
  and not g("num_attention_heads"):` branch: `u.update(num_attention_heads=g("num_heads"),
  num_key_value_heads=g("num_kv_heads") or g("num_heads"))`, and if intermediate_size is null `u["intermediate_size"] =
  256 * -(-int(hidden * (g("hidden_ratio") or 4) * 2 / 3) // 256)` (gla too: GLAMLP is the same rule). Gate on the
  architecture, not model_type "transformer" (too generic). Add "delta_net" to LINEAR_TYPES (:442) and
  `"delta_net": "deltanet"` to RECURRENT_FAMILY (:2034, next to gla).

## 3. Relation rule
- target side: fla transformer = an ordinary Llama-like point (RMSNorm, RoPE 1e4, MHA, SwiGLU): the existing softmax
  rules. DeltaNet: a rf_deltanet target needs rf_deltanet layers (itself, or the universal host's gated-delta block with
  decay fixed at 1).
- host side: transformer hosts by the existing rules; DeltaNet's recurrent layers are zeroed for targets without them.
- kind: exact (existing rules); rf_deltanet vs rf_gdn a wall both ways: GDN's output norm is gated (norm(o) * silu(z),
  no weight makes silu(W_z x) == 1) and GDN's decay is exp(-A softplus(.)) per token; DeltaNet has neither.
- cost: none.
- free by the config rules: the head count is a shape read, not a rule.
- theorem: none new for the transformer; DeltaNet padding by `deltaRow_inert` / `delta_out_pad` (RC_EQ kept equal here).

## 4. Test fixtures
- flag: fla-hub/gla-1.3B-100B -> 4, fla-hub/delta_net-1.3B-100B -> 16, fla-hub/transformer-1.3B-100B -> 32.
- not flag: EleutherAI/gpt-neo-125m (num_heads, handled by gpt_neo); fla-hub/rwkv7-0.1B-g1 (fla, RWKV7 arch: read by
  the rwkv7 dims); ridger/MMfreeLM-370M (fla-style hgrn_bit, not this arch set); JetLM/SDAR-8B-Chat (fuse_* keys, has
  num_attention_heads); google-t5/t5-3b (num_heads, encoder-decoder); 01-ai/Yi-1.5-34B-32K (plain llama).

## Verdict
real. More edges: two new points. fla-hub/transformer-1.3B-100B joins the plain pre-norm softmax cluster (hosts /
is hosted by ~1000+ Llama-like points on the existing rules); delta_net-1.3B-100B is a pure linear point (rf_deltanet
only pairs with itself / the universal host; as a host it plays no softmax target). gla is already placed (its heads
reach q_cap dims only). Region: classification.
