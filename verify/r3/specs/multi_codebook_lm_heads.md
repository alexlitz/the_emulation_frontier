# multi_codebook_lm_heads
live 4 / placed 4 (parler_tts_decoder 4); direction over-credit; batch a3_placement

## 1. What the code computes
- ext:parler_tts/gh__huggingface__parler-tts/modeling_parler_tts.py:1834-1838: `use_fused_lm_heads` (config default False,
  configuration_parler_tts.py:133, :161; v1.1 sets True) -> one `Linear(hidden, vocab_size * num_codebooks)`, else
  `num_codebooks` heads `Linear(hidden, vocab_size)`.
- modeling_parler_tts.py:1917-1920: logits = the fused head reshaped to (B, num_codebooks, T, vocab), or the stack of
  the per-codebook heads: identical maps (the stacked heads ARE the row blocks of one Linear(h, 9 x 1088)).
- modeling_parler_tts.py:1433: the input at a position is `sum_c embed_tokens[c](input[:, c])`, a sum of 9 codebook
  rows: a function of that position's 9-tuple only (a per-position lookup).
- Snapshot: num_codebooks 9, vocab_size 1088 (dims_of reads vocab 1088); USE_VOCAB = False (p5:model_graph.py :2483).

## 2. Classifier change
- reads: model_type parler_tts_decoder, num_codebooks
- new: dims "codebooks" = num_codebooks (record only), in dims_of
- detection:
```python
def detect_multi_codebook_lm_heads(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "parler_tts_decoder" and num(lm.get("num_codebooks")) > 1:
        return num(lm.get("num_codebooks"))
    return None
```
- apply note: optional: p5:model_graph.py `dims_of` return dict, `"codebooks": g("num_codebooks") or 1` for
  parler_tts_decoder. Not a relation key.

## 3. Relation rule
- target side: logits = one linear head of 9 x 1088 rows on the last hidden state, reshaped; a host with a
  9 x 1088-row head reproduces them exactly (vocab ignored); the frame input is a per-position table lookup (a
  table indexed by the 9-tuple, the sum of the rows, vocab ignored)
- host side: a Parler host plays a one-codebook target with codebooks 1..8's embedding rows zero and head 0
- kind: exact both ways at the logit level
- cost: none (vocab not priced)
- free by the config rules: use_fused_lm_heads is a layout switch with the same map; vocab ignored
- theorem: none needed (linear head rows; per-position lookup)

## 4. Test fixtures
- flag: parler-tts/parler-tts-large-v1, parler-tts-mini-v1, parler-tts-mini-v1.1, parler-tts-mini-multilingual-v1.1
  (live) and parler-tts/parler_tts_mini_v0.1 (same code, not placed) -> 9
- not flag: BreezeBlue/Breeze-TTS-2 (16 codebooks, but its backbone head predicts codebook 0 only),
  kmhf/hf-moshika (moshi: text head + depth decoder), openai-community/gpt2

## Verdict
non-issue: the 9 heads are one Linear(h, 9V) reshaped (modeling_parler_tts.py:1918) and the codebook-sum input is a
per-position lookup, so with vocab ignored (USE_VOCAB off, a design rule) logits match exactly; the per-codebook
argmax is decoding, not the forward. Moot anyway: Parler decoders are target-walled by
cross_attention_decoder_unflagged. Region: classification.
