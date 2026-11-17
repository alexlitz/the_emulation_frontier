# eos_document_mask
live 2 / placed 2 (ethoric 1, klm 1); direction over-credit; batch a6_masks

## 1. What the code computes
- Ethoric-10M: `document_ids = cumsum(input_ids == eos) - is_eos` (Ethoric-10M/modeling_ethoric.py:435-443). Every eos
  (eos_token_id 1) closes a document. The model computes doc ids whenever `config.doc_masking` (default True,
  configuration_ethoric.py:46, :66; true in the snapshot) and passes them to every layer (modeling_ethoric.py:228-235).
  The dense mask ANDs causal, same document (`q_doc == k_doc`) and the window (attention.py:80-98; flex variant
  :102-127). The Canon short convs also stop at document starts (modeling_ethoric.py:130-135).
- KLM-1.5: `segments`: a new document starts after every <eos> (eos_token_id 2) and at each row's first real token
  (modeling_klm.py:66-84). Attention allows only same-document keys (:327-332). The GDN-2 state, the short-conv taps and
  the n-gram history reset at each start (:108-116, :171-189). This is code-fixed (no config switch).
- On any input containing the eos id, both compute something a plain causal LM does not. On eos-free inputs both are
  plain causal (one document).

## 2. Classifier change
- reads: model_type ethoric (doc_masking, default True), model_type klm
- new: FEATURES key `doc_mask` = ("wall", "free"), in `_feats` + FEATURES
- detection:
```python
def detect_eos_document_mask(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "ethoric" and lm.get("doc_masking", True) is not False:
        return "doc_mask"
    if model_type == "klm":
        return "doc_mask"
    return None
```
- apply note: FEATURES (p5:model_graph.py ~L1597): `"doc_mask": ("wall", "free", "in-sequence documents split at the
  eos token: same-document attention and conv / recurrent / n-gram resets (Ethoric doc_masking, KLM segments): a
  token-id-dependent mask no plain layer forms; a doc-mask host never sees its eos id (the emulator maps no target
  token to it; vocab is free)")`. In `_feats` (~L1741): `if (mt == "ethoric" and d.get("doc_masking", True) is not
  False) or mt == "klm": f.add("doc_mask")`.

## 3. Relation rule
- target side: wall. The mask depends on token identity through a running count of eos tokens: a hard 0 / -inf
  same-document mask. A plain softmax host forms masks only as finite logits (`finite_mask_limit`: a limit), and the
  document id is a running count that uniform attention gives only as a fraction. Not exact, not measured.
- host side: free. The emulator chooses the host token ids (vocab ignored, a spare id is assumed), so the host's eos id
  never appears in the input and the host is plain causal. For Ethoric, doc_masking is also a config switch. That is a
  feature switch, not free, but it is not needed.
- kind: wall (target); exact (host)
- cost: none
- free by the config rules: doc_masking is a feature switch (not free); eos_token_id is config, but the host side needs
  neither
- theorem: host side, by restriction (on eos-free inputs the document mask is the causal mask: no lemma needed).
  Target side: needs theorem "no plain causal softmax stack equals an eos-segmented one on inputs containing eos"
  (a two-input witness: same suffix, with / without an eos in the prefix).

## 4. Test fixtures
- flag: techdotus/Ethoric-10M -> "doc_mask"; MinimaLabs/KLM-1.5 -> "doc_mask"
- not flag: AxiomicLabs/GPT-X3-150M (GDN2 hybrid, no doc reset), AxiomicLabs/GPT-S3-3M, Qwen/Qwen3-0.6B

## Verdict
Real, fewer edges for 2 models as targets: every host lacking doc_mask loses Ethoric-10M and KLM-1.5. As hosts nothing
changes. They play each other only if every other slot agrees, which is unlikely (softmax vs GDN-2 hybrid). If the
evaluation is restricted to eos-free inputs instead, the feature is free both ways and nothing moves. Region: attention
(and recurrent for KLM).
