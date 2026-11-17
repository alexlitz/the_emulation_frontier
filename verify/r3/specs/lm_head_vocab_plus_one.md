# lm_head_vocab_plus_one
live 2 / placed 2 (breeze 2); direction over-credit; batch a3_placement

## 1. What the code computes
- ext:breeze/gh__breezeblue-ai__breeze-tts/models/breeze.py:921-923: `self.backbone_eos_token_id = config.vocab_size`;
  `self.lm_head = nn.Linear(config.hidden_size, config.vocab_size + 1, bias=False)`: 2052 output rows (codebook-0 ids
  0..2050 plus a backbone-EOS class) while dims_of vocab reads vocab_size 2051 (p5:model_graph.py :3295).
- breeze.py:1818-1828: backbone_logits = lm_head(backbone hidden states), out_features = vocab_size + 1.
- The relation ignores vocab: USE_VOCAB = False (p5:model_graph.py :2483; the vocab test at :4519 runs only when on),
  and the design rules say vocab size is ignored.

## 2. Classifier change
- reads: model_type breeze, vocab_size
- new: dims "vocab" = vocab_size + 1 for breeze (only matters under USE_VOCAB), in dims_of
- detection:
```python
def detect_lm_head_vocab_plus_one(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "breeze" and num(lm.get("vocab_size")):
        return num(lm.get("vocab_size")) + 1
    return None
```
- apply note: optional. p5:model_graph.py `dims_of` (:3295) `"vocab": ... + (1 if mt == "breeze" else 0)`. No headline
  effect (USE_VOCAB off).

## 3. Relation rule
- target side: an output table one row larger (vocab capacity)
- host side: same
- kind: exact (vocab is ignored; a larger head is zero-padded rows)
- cost: none in the headline relation
- free by the config rules: vocab size ignored by rule
- theorem: none needed (vocab not priced)

## 4. Test fixtures
- flag: BreezeBlue/Breeze-TTS-2 -> 2052; drbaph/Breeze-TTS-2-comfyui -> 2052
- not flag: Qwen/Qwen3-0.6B (vocab = lm_head rows), kmhf/hf-moshika

## Verdict
non-issue: vocab size is ignored by the relation (USE_VOCAB = False, a design rule); record the +1 only if
USE_VOCAB is ever turned on. Region: classification.
