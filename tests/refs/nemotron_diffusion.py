"""Nemotron-Labs-Diffusion's bidirectional encoder: a labelled reference transcription, never the remote code.

Source:   nvidia/Nemotron-Labs-Diffusion-3B, snapshot 0d51902da1f8869f83413ce642fab402fa5641e0
Files (read as text):
  modeling_nemotron_labs_diffusion.py  sha256 29d73c5709e90e3be3c7e537edb61a84fbb3dc1c286b1eaed42d899a3a4e4760
      lines 157-185: NemotronLabsDiffusionModel.__init__ -- the encoder is Ministral3Model(config) with
      diffusion_lm = True and attn_class = Ministral3Attention for dlm_paradigm "bidirectional" (the snapshot's);
      lines 288-299: the encoder's last hidden state feeds diffusion_head (a bias-free Linear).
  modeling_ministral.py                sha256 021afd8a72e23c5bf3a309f9dc5f0a7771db2424c148be5ce2a2737bf4cb7d65
      lines 104-196: Ministral3Attention; with diffusion_lm the attention interface gets mask None and
      is_causal=False (lines 168-179): every position attends to every position;
      lines 199-461: MLP, RMSNorm, decoder layer, rotary embedding (yarn via ROPE_INIT_FUNCTIONS), Ministral3Model
      (no mask is built unless use_causal_mask is passed, lines 421-441).
  configuration_nemotron_labs_diffusion.py  sha256 2b18216e1b4e0d89b728c1c871744088a28004564f99009809294b39ec677b57
      lines 110-186 (the fields, read straight off the snapshot's config.json here).

Lines 199-461 of modeling_ministral.py are, line for line, transformers 5.18's stock models/ministral3 (MLP, RMSNorm,
decoder layer, rotary embedding, model); the attention differs only in the diffusion branch and in reading the
Llama-4 query scale off cache_position (== position_ids without a cache). So the reference is the stock 5.18
Ministral3Model (library code) with its attention forward replaced by the transcription of lines 139-196 below.
Needs transformers 5.x; import it only inside the TF5 subprocess. Weights are the real checkpoint's safetensors.
"""
import json
import os

import torch
from transformers.models.ministral3 import modeling_ministral3 as mm
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Nemotron-Labs-Diffusion-3B/snapshots/"
                          "0d51902da1f8869f83413ce642fab402fa5641e0")
_CFG_KEYS = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings", "rms_norm_eps",
             "rope_parameters", "sliding_window", "attention_dropout", "tie_word_embeddings",
             "bos_token_id", "eos_token_id")


class DiffusionAttention(mm.Ministral3Attention):
    """modeling_ministral.py lines 129-196 (no cache): diffusion_lm True -> mask None, is_causal=False."""
    diffusion_lm = True

    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = mm.apply_rotary_pos_emb(q, k, cos, sin)
        cache_position = torch.arange(q.shape[2], device=q.device)              # no cache: 0..T-1
        rp = self.config.rope_parameters
        scale = 1 + rp.get("llama_4_scaling_beta") * torch.log(1 + torch.floor(cache_position / rp.get("original_max_position_embeddings")))
        q = q * scale.unsqueeze(-1).to(q.dtype)
        attn = mm.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, mm.eager_attention_forward)
        if self.diffusion_lm:
            out, w = attn(self, q, k, v, None, dropout=0.0, scaling=self.scaling, is_causal=False)
        else:                                                                   # the AR branch (lines 181-192)
            out, w = attn(self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling,
                          sliding_window=getattr(self.config, "sliding_window", None))
        return self.o_proj(out.reshape(*input_shape, -1).contiguous()), w


def config(attn_implementation="sdpa"):
    raw = json.load(open(os.path.join(SNAP, "config.json")))
    cfg = Ministral3Config(**{k: raw[k] for k in _CFG_KEYS if k in raw})
    cfg._attn_implementation = attn_implementation
    return cfg


def encoder(dtype=torch.float32, diffusion_lm=True):
    """The real checkpoint's encoder (encoder.* tensors of model.safetensors), as the reference above."""
    from safetensors import safe_open
    cfg = config()
    with torch.device("meta"):
        enc = mm.Ministral3Model(cfg)
    enc = enc.to_empty(device="cpu")
    sd = {}
    with safe_open(os.path.join(SNAP, "model.safetensors"), "pt") as f:
        for key in f.keys():
            if key.startswith("encoder."):
                sd[key[len("encoder."):]] = f.get_tensor(key).to(dtype)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    enc.rotary_emb = mm.Ministral3RotaryEmbedding(cfg)                         # non-persistent inv_freq, rebuilt
    enc = enc.to(dtype).eval()
    for layer in enc.layers:
        layer.self_attn.__class__ = DiffusionAttention
        layer.self_attn.diffusion_lm = diffusion_lm
    return cfg, enc
