# ngram_memory_unread
live 5 / placed 5 (needle 2, klm 1, crowd 1, allgdn 1); direction over-credit; batch a12_extra_inputs

## 1. What the code computes
- KLM-1.5: `NgramMemory` (MinimaLabs/KLM-1.5/modeling_klm.py:94-120): sorted key banks `bigram_keys` (n_bigrams) / `trigram_keys` (n_trigrams), embeddings of `ngram_dim` (+1 unknown row), projected to hidden (:101-106). forward: `prev1, prev2` = the two previous tokens, zeroed after `<eos>` (:112-116); `bigram = prev1*V + t`, `trigram = (prev2*V + prev1)*V + t` (:117-118); exact lookup `searchsorted`, unknown -> row 0 (:87-91); output `bigram_proj(E_bi) + trigram_proj(E_tri)` (:119-120). Injected as the second AttnRes source after the embedding (:338-339).
- Crowd-v1: `CrowdEngram` = `Embedding(engram_buckets, hidden)` (Vxtzq/Crowd-v1/modeling_crowd.py:86-89); `h = (prev * 1000003 + ids) % buckets`, prev = previous token (0 at position 0) (:91-97); input `x = wte(ids) + engram(ids)` (:183). Config `engram_buckets: 227865`.
- AllGDN (summerAIO/Cyber-9B-Qwen-AllGDN-Engram/modeling_allgdn.py): `Engram` (:76-126): per (order in `orders`, head) a rolling hash `x = x*P_h + g + 1` over the last n ids, `(x ^ (x >> 17)) mod slots` (:89-103); `gate = sigmoid(<rms(h) w_h, rms(key e) w_k>/sqrt(d))`, `v = gate * val(e)`, `out = v + causal depthwise conv(v)` (:110-126); a pre-hook adds it before each layer in `config.engram["layers"]` (:136-148, :162-177). Snapshot `engram: {layers [1, 15], orders [2, 3], heads 4, slots 65536, dim 128}`.
- needle (ext:needle/gh__cactus-compute__cactus/architecture.py): `engram_indices` seeded XOR-multiply hash per (order, head), `acc ^ (acc >> 15) mod slots` (:95-108, constants :73-75); `Engram` (:133-158): tables -> key / value projections, value passed through 4 LINEAR dilated taps (dilation max(orders), no silu); in `Block.__call__` `x += sigmoid(<rms(x), rms(k)>/sqrt d) * v` at the engram sites (:348-354); sites `config.engram_layers` (:673-679, :686-696). Snapshot `extras.engram.sites` [2, 15] (needle2) / [3, 7, 11, 15, 19] (needle3).
- p5 `_feats` fires `engram` only on `engram_layer_ids` / `engram_vocab_size` (p5:model_graph.py:1761-1762): none of these four.
- Each reads the previous token(s); the four hashes / addressings / injection points differ from each other and from DeepSeek-V4.1's (prime-bucket XOR, compressed tokenizer, no conv).

## 2. Classifier change
- reads: model_type; KLM `n_bigrams` / `n_trigrams`, Crowd `engram_buckets`, AllGDN `engram.layers`, needle `extras.engram.sites`
- new: FEATURES keys `engram_klm`, `engram_crowd`, `engram_allgdn`, `engram_needle` (prefix family `engram_*`, rule ("wall", "free")), in `_feats`
- detection:
```python
def detect_ngram_memory_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "klm":
        return "engram_klm" if num(lm.get("n_bigrams")) + num(lm.get("n_trigrams")) > 0 else None
    if model_type == "crowd":
        return "engram_crowd" if num(lm.get("engram_buckets")) > 0 else None
    if model_type == "allgdn":
        ec = lm.get("engram")
        return "engram_allgdn" if isinstance(ec, dict) and ec.get("layers") else None
    if model_type == "needle":
        ex = lm.get("extras") if isinstance(lm.get("extras"), dict) else {}
        ec = ex.get("engram") if isinstance(ex.get("engram"), dict) else {}
        return "engram_needle" if ec.get("sites") else None
    return None
```
- apply note: in `_feats` after the `engram` line (p5:model_graph.py:1762) add the four cases; in `feats_ok`'s `rule` (:2116) map `x.startswith("engram_")` -> ("wall", "free") (shared with qwen4_ple_reads_ngrams). Keep `engram` for the DeepSeek-V4.1 key path. Universal host: its UEngram (elephantine/universal.py:874-904) IS the AllGDN hash, so its feature string should list `engram_allgdn`, not `engram` (today it claims DS-V4.1's engram with a different hash).

## 3. Relation rule
- target side: a host carrying the same `engram_<kind>` at aligned layers with equal table geometry; every other host is a wall (a token embedding / another hash cannot read the same n-gram rows).
- host side: free -- zero the output (Crowd: the table; AllGDN / needle: val / value_proj). KLM: a zeroed memory is still an AttnRes source (score q . rms(0) = 0 keeps softmax mass), so it is exact only against KLM-shaped targets -- moot, KLM is attn_res-walled both ways.
- kind: target wall; host exact (Crowd / AllGDN / needle).
- cost: none.
- free by the config rules: none (orders / heads / slots / buckets change the hash and shapes).
- theorem: host `extra_input_off` (PROOFS.md "engram / ple / x0 lambdas / smear host -> plain"); target OPEN ("switch_lora / engram / ple / value_emb target <- plain host").

## 4. Test fixtures
- flag: MinimaLabs/KLM-1.5 -> "engram_klm"; Vxtzq/Crowd-v1 -> "engram_crowd"; summerAIO/Cyber-9B-Qwen-AllGDN-Engram -> "engram_allgdn"; Cactus-Compute/needle2, Cactus-Compute/needle3 -> "engram_needle".
- not flag: deepseek-ai/DeepSeek-V4.1-Flash, RuVI-AI/Pragya-EXP2 (already `engram` by keys); kenpeter123/x-small (san: engram_* keys, engram_heads 0, code not read -- unverified, left out); openai/circuit-sparsity (bigram table, not this finding); autotrust/JEV-9B (qwen3_5_text: AllGDN's parent); hstolte/Qwen3.8-Flash-Next-HTR (engram_qwen4, the other finding); a gemma4_text repo.
- scan: exactly the 5 live repos.

## Verdict
real. Fewer edges: the 5 targets (KLM, Crowd, AllGDN, needle2/3) lose every host without their own kind -- in practice all hosts but themselves (thousands of target-side edges, 5 models). Host side unchanged (free).
