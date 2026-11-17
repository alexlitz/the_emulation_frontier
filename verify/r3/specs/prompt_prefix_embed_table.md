# prompt_prefix_embed_table
live 4 / placed 4 (parler_tts_decoder 4); direction over-credit; batch a3_placement

## 1. What the code computes
- ext:parler_tts/gh__huggingface__parler-tts/modeling_parler_tts.py:2395: `self.embed_prompts = nn.Embedding(config.vocab_size,
  decoder.hidden_size)` (outer vocab_size 32128 / 90714); :2397-2398 prompt_cross_attention (config default False,
  configuration_parler_tts.py:240, :259; every snapshot False or absent).
- modeling_parler_tts.py:3033-3043 (and :1437-1439 in the decoder): with prompt_cross_attention False the prompt
  (transcript) embeddings `embed_prompts(prompt_input_ids)` (:3100) are PREPENDED to the codebook-sum input stream:
  one causal sequence, positions 0..P-1 for the prompt, then the audio frames.
- So the decoder input at each position is a per-position lookup from one of two tables (prompt ids vs frame
  tuples): a lookup in their disjoint union.
- USE_VOCAB = False (p5:model_graph.py :2483): input / output table sizes are not priced.

## 2. Classifier change
- reads: outer model_type parler_tts, prompt_cross_attention
- new: none (record only); if ever wanted, dims "in_vocab" = outer vocab_size, in dims_of
- detection:
```python
def detect_prompt_prefix_embed_table(cfg, model_type):
    if outer_type(cfg) == "parler_tts" and not truthy_flag((cfg or {}).get("prompt_cross_attention")):
        return True
    return None
```
- apply note: no change to p5:model_graph.py; the real Parler wall is the cross-attention (cross_attention_decoder_unflagged).

## 3. Relation rule
- target side: an input table on a disjoint id range = one larger input table (an input-only table distinct from the
  output head = untied, which the decoder already is: separate lm_heads)
- host side: unchanged
- kind: exact (vocab ignored)
- cost: none
- free by the config rules: vocab ignored
- theorem: none needed (a per-position lookup)

## 4. Test fixtures
- flag: the 4 live parler-tts decoders and parler-tts/parler_tts_mini_v0.1 (no prompt_cross_attention key = class
  default False) -> True
- not flag: BreezeBlue/Breeze-TTS-2 (its text input is an encoder output: text_encoder_input_embeds),
  kmhf/hf-moshika, openai-community/gpt2

## Verdict
non-issue: a prepended prompt embedded by a second table is still a per-position lookup in one causal stream; with
vocab ignored it is the same as a larger untied input table. Moot anyway (cross-attention wall). Region:
classification.
