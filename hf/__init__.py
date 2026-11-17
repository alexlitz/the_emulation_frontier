"""Emulation constructions against REAL, UNMODIFIED HuggingFace modules.

Each module here takes a stock `transformers` class (LlamaModel, MixtralSparseMoeBlock,
LlamaDecoderLayer, ...) with NO code edits, and shows that a more-general
architecture reproduces a special case forward-for-forward using only:
  - a config change, and/or
  - a state_dict remap (tile / tie / mask weights).

This is the strong form of the subset claim: the generality lives in the shipped
module, and the special case is just a parameterization of it -- not a bespoke
re-implementation. Everything runs in float64 on CPU for fp-exact tolerances.
"""
