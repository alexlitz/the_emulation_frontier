# factorized_embedding_proj
live 1 / placed 15 (opt 15; live opt-350m only); direction both; batch a3_placement

## 1. What the code computes
- tf5:models/opt/modeling_opt.py:289: `embed_tokens = Embedding(vocab, word_embed_proj_dim)`; :292-300: when
  word_embed_proj_dim != hidden_size, bias-free `project_out = Linear(hidden, wpd)` and `project_in = Linear(wpd,
  hidden)`; :368-369 inputs = project_in(E[ids]); :393 the final stream goes through project_out; :454
  `lm_head = Linear(wpd, vocab)`, tied to embed_tokens (:447 `_tied_weights_keys`).
- So the effective input embedding at width h is E P_in^T and the effective unembedding is E P_out: NOT transposes of
  each other (P_in != P_out^T in general), and both have rank <= wpd.
- Snapshot: only facebook/opt-350m has wpd (512) != hidden (1024); the other 14 OPT points have wpd = hidden (no
  projections). opt-350m is also do_layer_norm_before = false, so feats post_ln (wall both ways) already isolate it.
- p5 reads tie_word_embeddings true -> feats tied_emb (p5:model_graph.py `_tied` :1839) and no factorization.
- (OLMo / LLaDA / Molmo embedding_size != hidden is a PADDED VOCAB, not a factorization: not this finding.)

## 2. Classifier change
- reads: model_type opt, word_embed_proj_dim, hidden_size
- new: dims "emb_rank" = wpd; `_tied` false when wpd != hidden; in dims_of / _tied
- detection:
```python
def detect_factorized_embedding_proj(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "opt":
        return None
    wpd, h = num(lm.get("word_embed_proj_dim")), num(lm.get("hidden_size"))
    if wpd and h and wpd != h:
        return wpd
    return None
```
- apply note: p5:model_graph.py `_tied` (:1839): return False for opt with wpd != hidden (the width-h tables are not
  transposes); `dims_of` return dict (:3295 area): `"emb_rank": wpd if mt == "opt" and wpd != hidden else hidden`;
  emul_matrix: host emb_rank >= target hidden (rank of the target's embedding / head, generically its hidden) --
  add to the capacity checks next to hidden.

## 3. Relation rule
- target side: an untied host plays it exactly: host embedding := E P_in^T, host lm_head := E P_out (linear folds);
  a tied host cannot (needs E P_in^T = (E P_out)^T)
- host side: a factorized host's embedding and head have rank <= wpd: it plays a target only if the target's
  hidden (embedding rank) <= wpd, with the target's tables written through P_in / P_out
- kind: exact (target, untied host) / capacity (host)
- cost: host emb_rank >= target hidden
- free by the config rules: word_embed_proj_dim is shape-changing (not free)
- theorem: `tied_sub_untied` (untied host plays tied-like target), `scalar_fold`-style linear fold (needs theorem:
  "an untied host realises any factorized embedding / head pair E P_in^T, E P_out" -- trivial linear algebra);
  host-side rank bound needs theorem (rank argument)

## 4. Test fixtures
- flag: facebook/opt-350m -> 512
- not flag: facebook/opt-125m, facebook/opt-1.3b, facebook/galactica-125m (OPT with wpd = hidden),
  allenai/OLMo-1B (embedding_size is a padded vocab)

## Verdict
real but no current edge changes: opt-350m is walled both ways by post_ln. Becomes live only if post_ln gets a
construction. Region: classification.
