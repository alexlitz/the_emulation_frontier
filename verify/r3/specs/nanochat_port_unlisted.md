# nanochat_port_unlisted
live 3 / placed 3 (gaggle 1, pit 1, sn38-nanochrono 1); direction both; batch a1_typetables

## 1. What the code computes
Three renamed ports of nanochat (karpathy) with no hidden_act / mlp key; p5 knows only nanochat_gpt (RAW_ACT_BY_CODE :484, UNGATED_BY_CODE :494, NORM_BY_CODE :469, QKNORM_VARIANT_BY_TYPE head_free :1881):
- MLP up -> relu^2 -> down, no gate:
  - gaggle: castorini/gaggle-.../modeling_gaggle.py:116-125 `c_fc`, `c_proj`, `c_proj(F.relu(c_fc(x)).square())`
  - pit: Diamegs/PIT-1B-201912/modeling_pit.py:144-152 `c_fc (n_embd -> 4 n_embd)`, `c_proj(F.relu(c_fc(x)).square())`
  - sn38-nanochrono: ext sn38-nanochrono/github__chronollm__sn38/modeling_nanochrono.py:86-93 `down_proj(F.relu(up_proj(x)).square())`
  p5 gives all three ffn=glu, act=SiLU (the unknown-type defaults, classify :2277-2283, :2287-2304).
- weight-free per-head q / k RMSNorm:
  - pit: modeling_pit.py:114-115 `rope(F.rms_norm(q, eps=rms_norm_eps))` (before RoPE; weight-free, so order is irrelevant): p5 qknorm=no
  - sn38-nanochrono: :50-51 RoPE then `_norm` (weight-free F.rms_norm, :13), :52-53 `q, k *= attention_multiplier` (1.2 in the snapshot config :6 -> gain 1.44: the qknorm_fixed_gain variant): p5 qknorm=no
  - gaggle: already head_free in p5 (QKNORM_BY_CODE :1875); its 1.2 qk_scale is qknorm_fixed_gain
- stream norms weight-free F.rms_norm (pit :173, :177, :371; gaggle :36): RMSNorm, already p5's default.

## 2. Classifier change
- reads: model_type
- new: RAW_ACT_BY_CODE["gaggle" | "pit" | "sn38-nanochrono"] = "relu2"; UNGATED_BY_CODE += the three (ffn=vanilla); QKNORM_VARIANT_BY_TYPE head_free += pit, sn38-nanochrono (and QKNORM_BY_CODE += them, as gaggle); NORM_BY_CODE += the three ("RMSNorm", for the record)
- detection:
```python
NANOCHAT_PORTS = {"gaggle", "pit", "sn38-nanochrono"}

def detect_nanochat_port_unlisted(cfg, model_type):
    return "ffn=vanilla,act=relu2,qknorm=head_free" if model_type in NANOCHAT_PORTS else None
```
- apply note: in p5:model_graph.py extend `RAW_ACT_BY_CODE` (:484), `UNGATED_BY_CODE` (:488), `QKNORM_BY_CODE` (:1875) and the head_free tuple of `QKNORM_VARIANT_BY_TYPE` (:1881). The gain (gaggle 1.44, sn38 1.44) then upgrades head_free to head_free_gain via qknorm_fixed_gain; pit has no gain and stays head_free.

## 3. Relation rule
- target side: vanilla relu2 targets need a host whose FFN plays up -> relu^2 -> down (a relu2 vanilla host exactly; a GLU host via `glu_ungated` with an up bias or constant channel; SiLU hosts only by the act floors / act-width fits, which price relu2 separately); a head_free target needs a qk-normed host (head_free / head / head_post / head_distinct, or head_free_gain set to 1)
- host side: as hosts they play vanilla relu2 targets (nanochat_gpt, arcee-like) and head_free / plain-by-lever targets: head_free cannot be levered off (weight-free), so they no longer play plain attention targets
- kind: exact (same-function edges; head_free <- head by gain 1); wall (qk-normed target <- plain host; SiLU-GLU edges they were credited with)
- cost: none on the exact edges
- free by the config rules: none (activation and FFN shape are not free)
- theorem: `glu_ungated` (PROOFS.md "ffn: vanilla -> GLU", needs an up bias / constant channel); `qkn_gain_one`, `weightFree_comm` (head_free <- head); `knorm_not_plain` (qk-norm target <- plain host impossible)

## 4. Test fixtures
- flag (every live repo): castorini/gaggle-nanochat-pretrained-climbmix-20260924, Diamegs/PIT-1B-201912, kingbear329/sn38-r16-c12-step687-alpha05 -> "ffn=vanilla,act=relu2,qknorm=head_free"
- not flag: jkminder/d16_optimized_537m_seed5 (nanochat_gpt: already listed); l0ulan/chinese-babylm-nanochat-d22-step10000 (transformers nanochat type: listed); Qwen/Qwen3-0.6B

## Verdict
real, both directions, 3 models: each loses its SiLU-GLU plain-attention edges (it was classified as a Llama-like model) and gains the nanochat_gpt / relu2-vanilla qk-normed neighbourhood. Tens of edges move per model; mostly fewer (the Llama-like pool is large, the nanochat pool small).
