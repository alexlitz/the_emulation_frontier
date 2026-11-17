"""dQwen3.5 (diffusion Qwen3.5): a labelled reference transcription, never the remote code.

Source:   UT-IFML/dQwen3.5-0.8B-Base, snapshot 1b6ae9426071d5e372d58e309bd4ed85de39c762
File:     modeling_dqwen3_5.py, sha256 e1f58b23aa2fee44a1b1dea1e2d251de65e8ddda0664a158f77d2ab9296960df
Lines:    12-15 (BOS_TOKEN_ID), 31-75 (DiffuQwen35Config, DiffuQwen35.__init__ / forward), read as text.
          77-313 (masked-diffusion generate, from_qwen3_5 seeding, auto-class registration) are sampling /
          loading code, not the forward pass, and are not transcribed.

The remote class subclasses the stock transformers 5.x Qwen3_5ForCausalLM; what it changes is transcribed
here on top of the same stock class (library code):
  - __init__: every full_attention layer's self_attn.is_causal = False (the gated-delta layers stay causal);
  - forward: prepend BOS_TOKEN_ID, run the body with the per-layer-type mask dict {"full_attention": None,
    "linear_attention": None} (no padding: the dict stops the body building a causal mask), and read the
    logits off hidden[:, :-1] (the BOS / logit shift). The padded-mask branch (lines 66-70) is transcribed too.
Needs transformers 5.x (Qwen3_5); import it only inside the TF5 subprocess.
"""
import torch
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

BOS_TOKEN_ID = 248073          # <tts_text_bos> (line 15); tests rebind it to fit a tiny vocabulary


class DiffuQwen35Config(Qwen3_5TextConfig):
    model_type = "diffuqwen35"


class DiffuQwen35(Qwen3_5ForCausalLM):
    config_class = DiffuQwen35Config

    def __init__(self, config, bidirectional=True, bos=True):
        super().__init__(config)
        self.bos = bos                                   # switches for the branch checks (the remote has both on)
        for layer in self.model.layers:
            if layer.block_type == "full_attention" and bidirectional:
                layer.self_attn.is_causal = False

    def forward(self, input_ids, attention_mask=None):
        B = input_ids.shape[0]
        if self.bos:
            bos = torch.full((B, 1), BOS_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([bos, input_ids], dim=1)
            if attention_mask is not None:
                ones = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([ones, attention_mask], dim=1)
        inputs_embeds = self.model.embed_tokens(input_ids)
        if attention_mask is None or attention_mask.all():
            attn_mask = {"full_attention": None, "linear_attention": None}
        else:
            _, L1, _ = inputs_embeds.shape
            mask_4d = attention_mask.reshape(B, 1, 1, L1).expand(B, 1, L1, L1)
            mask_4d = (1.0 - mask_4d.to(inputs_embeds.dtype)) * -10000
            attn_mask = {"full_attention": mask_4d, "linear_attention": attention_mask}
        hidden = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False).last_hidden_state
        return self.lm_head(hidden[:, :-1, :] if self.bos else hidden)
