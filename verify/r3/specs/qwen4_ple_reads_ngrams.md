# qwen4_ple_reads_ngrams
live 25 / placed 25 (qwen4_exp_text 24, qwen4_exp 1); direction over-credit; batch a12_extra_inputs

## 1. What the code computes
- Config: `ple_layer_ids` one-indexed PLE layers, `[]` if None (tf5:models/qwen4_exp/modular_qwen4_exp.py:175); `ple_embed_dim` defaults to hidden (:176); `number_of_conv_states = 3 if ple_layer_ids else 1` (:180); PLE only on linear_attention layers (:256-262), needs eos (:265).
- A PLE module is built only for layers in `ple_layer_ids` (:796-797); the decoder layer adds it to the hc-stream residual before the hyper-connection: `hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, ...)` (:811-814).
- `Qwen4ExpTextNGramEmbedding.forward` (:659-708): token history = cached `context_len = ngram_size - 1` previous ids + input (:660-684); `shifted_tokens[s]` = the token s positions back within the EOS-delimited segment, EOS-filled (:643-657, :685); for each order n in 2..ngram_size: `mixed = XOR_{p<n} shifted[p] * layer_multipliers[p]` (:688-696), `ids = mixed mod prime_head_size + offset` (:697-700); one embedding row per (order, head) concatenated (:702-708). Multipliers: splitmix64 seeded per PLE layer (:576-586); per-head prime vocab sizes (:627-641).
- `Qwen4ExpTextPLELayer.forward` (:763-783): `key = RMSNorm_group(key_proj(e))`, `query = RMSNorm_group(h)` per hc stream; `gate = <key, query>/sqrt(hidden)`, signed sqrt, `sigmoid(gate) * value_proj(e)` per stream (:770-776); output `gated + silu(dilated depthwise conv(norm(gated)))`, kernel `ple_conv_kernel_size`, dilation `ngram_size` (:727-734, :743-761, :777-783).
- Contrast, Gemma-4 PLE: `embed_tokens_per_layer(input_ids)` -- one row per CURRENT token (tf5:models/gemma4/modeling_gemma4.py:1749). DeepSeek-V4.1 Engram: XOR-of-multiplied tokens mod primes + the same signed-sqrt gate per hc copy, NO short conv, compressed-tokenizer multipliers (ext deepseek_v41 .../inference/model.py:328-366, engram.py:64-80, :178-182).
- Snapshot: 24 repos `ple_layer_ids [2]`, PeetPedro `[1, 6, ..., 36]`; Jab1718/qwen3.8-flash-coder-85gb-bf16 has `ple_layer_ids []` = no PLE layer (yet p5 flags `ple` there via `ple_embed_dim > 0`).
- So the input reads the previous ngram_size - 1 tokens (and the conv reaches (k-1)*ngram_size positions further): an Engram, not a per-token PLE.

## 2. Classifier change
- reads: model_type in (qwen4_exp, qwen4_exp_text), `ple_layer_ids`
- new: FEATURES key `engram_qwen4` (prefix family `engram_*`), in `_feats`; drop `ple` for these types
- detection:
```python
QWEN4_TYPES = ("qwen4_exp", "qwen4_exp_text")

def detect_qwen4_ple_reads_ngrams(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in QWEN4_TYPES:
        return None
    ids = lm.get("ple_layer_ids")
    return "engram_qwen4" if isinstance(ids, list) and ids else None
```
- apply note: in `_feats` (p5:model_graph.py:1763-1764) replace `or d.get("ple_layer_ids") or _num(d.get("ple_embed_dim")) > 0` (Qwen4-only keys) by `if mt in QWEN4_TYPES and d.get("ple_layer_ids"): f.add("engram_qwen4")`; keep `hidden_size_per_layer_input` -> `ple` (Gemma). In `feats_ok`'s `rule` lambda (:2116) add `x.startswith("engram_")` -> ("wall", "free") beside `rf_`. Update the FEATURES `ple` text (drop "Qwen4-exp"). Same family as ngram_memory_unread (shared prefix rule). Optional dims `engram_sig` = (ple_layer_ids, ngram_size, heads_per_ngram, ple_embed_dim, ple_conv_kernel_size) compared for equality on `engram_qwen4` pairs.

## 3. Relation rule
- target side: needs a host carrying `engram_qwen4` (same hash code, multipliers seeded by PLE-layer index) at the aligned layers with equal sig; any other host (Gemma PLE, DS-V4.1 engram, plain) is a wall.
- host side: free -- zero value_proj (output 0, conv of 0 = 0): plays targets without it.
- kind: target wall; host exact.
- cost: none.
- free by the config rules: none (ngram_size / heads / table sizes change the hash and the table shapes; the PLE itself is a feature switch).
- theorem: host side `extra_input_off` (PROOFS.md "engram / ple / x0 lambdas / smear host -> plain"); target side OPEN ("switch_lora / engram / ple / value_emb target <- plain host").

## 4. Test fixtures
- flag: all 24 live repos with `ple_layer_ids` non-empty -> "engram_qwen4" (23 qwen4_exp_text, PeetPedro/qwen4_exp-8b-... qwen4_exp flat config).
- not flag: Jab1718/qwen3.8-flash-coder-85gb-bf16 (`ple_layer_ids []`: no PLE layer -- a live carrier the code does NOT take; p5's `ple` there is itself wrong); farbodtavakkoli/OTel-2.0-Decision-26B-A4B-4bit (gemma4_text PLE), jialinyyzz/humanizer (gemma4_unified_text), mlx-community/gemma-3n-E2B-it-lm-4bit (gemma3n); Qwen/Qwen3-Next-80B-A3B-Instruct, nvidia/Qwen3.6-35B-A3B-NVFP4 (parent families); deepseek-ai/DeepSeek-V4.1-Flash (existing `engram`).
- scan: 24 repos, exactly the live set minus Jab1718.

## Verdict
real (mislabel). Frontier change ~0 today: Qwen4-exp also carries hyper_conn (wall both ways), so Gemma-PLE hosts never reached it. The relabel prevents a future over-credit (Qwen4 <-> DS-V4.1 engram, Qwen4 <-> Gemma PLE once hyper_conn variants are split, see hyper_conn_variant) and removes the spurious `ple` on Jab1718.
