"""ElephantineConfig: the transformers config of the ONE universal elephantine architecture.

Every subfolder of the Hub repo uses this same config class and the same model class
(`modeling_elephantine.ElephantineForCausalLM`). What differs between settings is

  host      the host's sizes (the `UConfig` fields of elephantine/universal.py: vocab, d, L, heads,
            head_dim, the SSM / FFN widths, and the optional structural branches);
  causal    the default mask (True: autoregressive; False: the full mask a diffusion LM runs under);
  weights   model.safetensors: the weights AND the branch coefficients / buffers (an activation
            switch, a parallel-block flag, a norm's eps, a RoPE table, ...). Tensors still at the
            architecture's default value are not stored; the model class recreates them at load.

`branches` and `stored_tensors` are a readable summary of the weights file (which coefficients
are switched away from their defaults, which tensors are stored); they are informational, the
safetensors file is authoritative. `source` records the checkpoint the setting was loaded from.
"""
from transformers import PretrainedConfig


class ElephantineConfig(PretrainedConfig):
    model_type = "elephantine"

    def __init__(self, host=None, causal=True, source=None, branches=None, stored_tensors=None, **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)     # the host keeps its own lm_head
        kwargs.setdefault("use_cache", False)               # no KV cache: every step re-runs the prefix
        super().__init__(**kwargs)
        self.host = dict(host or {})
        self.causal = bool(causal)
        self.source = dict(source or {})
        self.branches = dict(branches or {})
        self.stored_tensors = list(stored_tensors or [])
        h = self.host                                       # the usual names, for tools that read them
        self.vocab_size = h.get("vocab", 0)
        self.hidden_size = h.get("d", 0)
        self.num_hidden_layers = h.get("L", 0)
        self.num_attention_heads = h.get("heads", 1)
