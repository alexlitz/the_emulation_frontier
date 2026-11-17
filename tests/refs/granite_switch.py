"""REFERENCE, not the remote code: Granite-Switch's Hugging Face forward pass (the single-switch, single-stream
LoRA / aLoRA layout), transcribed as plain torch from its source READ AS TEXT, never imported or executed.

Source: github.com/generative-computing/granite-switch at commit 24aa420f71b6703f96e40533d83041ce311c30ae (the
single-switch layout the ibm-granite/granite-switch-4.1-* checkpoints are composed with; fetched as a tarball by
elephantine/fetch_upstream_refs.py into ~/.cache/upstream_refs/granite_switch-24aa420). Files and the line ranges
transcribed:
  granite_switch/hf/modeling_granite_switch.py  sha256 a4b8b5530827a932b6640f0d0872542f4ebac7eb1ca19d741301b97d974deda0
      48-231   GraniteSwitchAttentionDecoderLayer (dense shared_mlp path, no experts; _mlp_block's context passing)
      364-673  GraniteSwitchModel (switch -> token exchange -> embedding * multiplier -> layers -> norm; the switch
               takes one cache slot, so num_hidden_layers - 1 decoder layers)
      676-789  GraniteSwitchForCausalLM (lm_head, logits / logits_scaling)
  granite_switch/hf/core/lora.py                sha256 127e72c1b528428dcc1d1a236c4b4242cee9e187db3d046611f370aad7dcd601
      23-169   SwitchedLoRALinear (per-token adapter, index 0 = base, B pre-scaled)
      172-348  MergedSwitchedLoRALinear (fused qkv / gate-up, one (A, B) pair per slice)
      351-555  GraniteLoRAEmbeddedAttention.forward (fused qkv, RoPE, eager attention at attention_multiplier)
      662-703  replace_shared_mlp_projections_with_lora
  granite_switch/hf/switch/single.py            sha256 88891d871993634e0b6fc8294833061ef5355e309f9f9d26cc8d461b36b9800c
      27-246   SingleSwitch (one head, dim 0 live: q 1, k +-gain, v adapter id; causal softmax average, round, clamp)
  granite_switch/hf/switch/__init__.py          sha256 fe7b3b8534aab51242dd6305447bc9608573811b91490db3a2346061faba0136
      20-41    create_switch ("single")
  granite_switch/token_exchange.py              sha256 72c1f7c2b39ede5bff4dd30511c8089913548b458ed5eb85e7be25011231644d
      32-156   build_control_to_substitute_lut / apply_token_exchange
  granite_switch/config.py                      sha256 c3413d21e16e04a08382ed1d9018ce7852817fb4373478d2dfd20820509c6d34
      101-320  the fields read here (control_token_gain 15, shared_intermediate_size defaults to intermediate_size,
               projection_head_dim = hidden_size // num_attention_heads)

The stock pieces it imports are used as they are: transformers' GraniteMoeHybridConfig / MLP / RMSNorm / RotaryEmbedding
/ apply_rotary_pos_emb / eager_attention_forward. Not transcribed (not used by the single-switch LoRA checkpoints):
the Shadow-Residual dual stream, MultiSwitch, experts, qk-norm, KV caching.
"""
import torch
import torch.nn as nn
from transformers import GraniteMoeHybridConfig
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridMLP, GraniteMoeHybridRMSNorm, GraniteMoeHybridRotaryEmbedding, apply_rotary_pos_emb,
    eager_attention_forward)


def granite_switch_config(num_adapters, adapter_token_ids, adapter_substitute_token_ids, adapter_ranks,
                          lora_target_modules, control_token_gain=15.0, **kw):
    """GraniteSwitchConfig (config.py 101-320): the stock GraniteMoeHybridConfig plus the switch fields."""
    n = kw.get("num_hidden_layers", 32)
    c = GraniteMoeHybridConfig(num_local_experts=0, position_embedding_type="rope", layer_types=["attention"] * n,
                               attn_implementation="eager", **kw)
    if c.shared_intermediate_size is None:
        c.shared_intermediate_size = c.intermediate_size
    c.num_adapters, c.adapter_token_ids = num_adapters, list(adapter_token_ids)
    c.adapter_substitute_token_ids = list(adapter_substitute_token_ids)
    c.adapter_ranks, c.max_lora_rank, c.lora_target_modules = list(adapter_ranks), max(adapter_ranks), list(lora_target_modules)
    c.control_token_gain = control_token_gain
    c.projection_head_dim = kw.get("head_dim") or c.hidden_size // c.num_attention_heads
    return c


class SwitchedLoRALinear(nn.Module):                                 # lora.py 23-169
    def __init__(self, in_features, out_features, num_adapters, max_lora_rank, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.zeros(num_adapters, 1, max_lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_adapters, 1, out_features, max_lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self._adapter_indices = None

    def forward(self, x, adapter_indices=None):
        if adapter_indices is None:
            adapter_indices = self._adapter_indices
        out = self.base_layer(x)
        if adapter_indices is None or not torch.any(adapter_indices > 0):
            return out
        shape = out.shape
        xf, af, of = x.reshape(-1, self.in_features), adapter_indices.reshape(-1), out.reshape(-1, self.out_features).clone()
        for a in af[af > 0].unique():
            ix = torch.where(af == a)[0]
            of[ix] += (xf[ix] @ self.lora_A[a - 1, 0].t()) @ self.lora_B[a - 1, 0].t()
        return of.view(shape)


class MergedSwitchedLoRALinear(nn.Module):                           # lora.py 172-348
    def __init__(self, in_features, output_slices, num_adapters, max_lora_rank, bias=False):
        super().__init__()
        self.in_features, self.output_slices = in_features, tuple(output_slices)
        self.base_layer = nn.Linear(in_features, sum(output_slices), bias=bias)
        self.lora_A_slices = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_adapters, 1, max_lora_rank, in_features)) for _ in output_slices])
        self.lora_B_slices = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_adapters, 1, o, max_lora_rank)) for o in output_slices])
        for A in self.lora_A_slices:
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
        self._adapter_indices = None

    def forward(self, x, adapter_indices=None):
        if adapter_indices is None:
            adapter_indices = self._adapter_indices
        out = self.base_layer(x)
        if adapter_indices is None or not torch.any(adapter_indices > 0):
            return out
        shape = out.shape
        xf, af, of = x.reshape(-1, self.in_features), adapter_indices.reshape(-1), out.reshape(-1, shape[-1]).clone()
        off = 0
        for A, B, size in zip(self.lora_A_slices, self.lora_B_slices, self.output_slices):
            for a in af[af > 0].unique():
                ix = torch.where(af == a)[0]
                of[ix, off:off + size] += (xf[ix] @ A[a - 1, 0].t()) @ B[a - 1, 0].t()
            off += size
        return of.view(shape)


class GraniteLoRAEmbeddedAttention(nn.Module):                       # lora.py 351-555 (no cache, no qk-norm)
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.hidden_size, self.num_heads = config.hidden_size, config.num_attention_heads
        self.head_dim = config.projection_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.attention_multiplier
        self.is_causal, self.training = True, False
        self.q_size, self.kv_size = self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim
        E, r = config.num_adapters, config.max_lora_rank
        assert {"qkv_proj", "o_proj"} <= set(config.lora_target_modules)
        self.qkv_proj = MergedSwitchedLoRALinear(self.hidden_size, (self.q_size, self.kv_size, self.kv_size), E, r,
                                                 bias=config.attention_bias)
        self.o_proj = SwitchedLoRALinear(self.q_size, self.hidden_size, E, r, bias=config.attention_bias)

    def forward(self, h, adapter_indices, position_embeddings, attention_mask):
        b, T, _ = h.shape
        q, k, v = self.qkv_proj(h, adapter_indices).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        o, _ = eager_attention_forward(self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling)
        return self.o_proj(o.reshape(b, T, self.hidden_size), adapter_indices)


class GraniteSwitchAttentionDecoderLayer(nn.Module):                 # modeling 48-231 (shared_mlp only)
    def __init__(self, config, layer_idx):
        super().__init__()
        self.residual_multiplier = config.residual_multiplier
        self.self_attn = GraniteLoRAEmbeddedAttention(config, layer_idx)
        self.shared_mlp = GraniteMoeHybridMLP(config)
        E, r, s = config.num_adapters, config.max_lora_rank, config.shared_intermediate_size
        assert {"shared_input_linear", "shared_output_linear"} <= set(config.lora_target_modules)
        self.shared_mlp.input_linear = MergedSwitchedLoRALinear(config.hidden_size, (s, s), E, r, bias=False)
        self.shared_mlp.output_linear = SwitchedLoRALinear(s, config.hidden_size, E, r, bias=False)
        self.input_layernorm = GraniteMoeHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GraniteMoeHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h, mask, pe, adapter_indices):
        h = h + self.self_attn(self.input_layernorm(h), adapter_indices, pe, mask) * self.residual_multiplier
        x = self.post_attention_layernorm(h)
        self.shared_mlp.input_linear._adapter_indices = self.shared_mlp.output_linear._adapter_indices = adapter_indices
        y = self.shared_mlp(x)
        self.shared_mlp.input_linear._adapter_indices = self.shared_mlp.output_linear._adapter_indices = None
        return h + y * self.residual_multiplier


class SingleSwitch(nn.Module):                                       # single.py 27-246, token_exchange.py 32-156
    def __init__(self, config):
        super().__init__()
        self.num_adapters, self.control_token_gain = config.num_adapters, config.control_token_gain
        self.head_dim = config.projection_head_dim
        self.num_key_value_groups, self.training = 1, False
        lut = torch.full((max(config.vocab_size, max(config.adapter_token_ids) + 1),), -1, dtype=torch.long)
        for c, s in zip(config.adapter_token_ids, config.adapter_substitute_token_ids):
            lut[c] = s
        self.register_buffer("control_to_substitute_lut", lut)

    num_cache_layers = 1

    def forward(self, input_ids, adapter_token_ids, attention_mask):
        b, T = input_ids.shape
        # the source allocates q / k / v in the default dtype (fp32); they hold exact small integers and the
        # output is rounded to an adapter index, so the mask's dtype is used here (consistent under an fp32 lift)
        dt = attention_mask.dtype
        q = torch.zeros(b, 1, T, self.head_dim, dtype=dt); q[..., 0] = 1.0
        k = torch.zeros(b, 1, T, self.head_dim, dtype=dt); k[..., 0] = -self.control_token_gain
        v = torch.zeros(b, 1, T, self.head_dim, dtype=dt)
        for i in range(self.num_adapters):
            m = input_ids == adapter_token_ids[i]
            k[:, 0, :, 0][m] = self.control_token_gain
            v[:, 0, :, 0][m] = float(i + 1)
        o, _ = eager_attention_forward(self, q, k, v, attention_mask, dropout=0.0, scaling=1.0)
        idx = torch.clamp(torch.round(o[:, :, 0, 0]).long(), 0, self.num_adapters)
        sub = self.control_to_substitute_lut[input_ids]
        return idx, torch.where(sub >= 0, sub, input_ids)


class GraniteSwitchModel(nn.Module):                                 # modeling 364-673
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embedding_multiplier = config.embedding_multiplier
        self.switch = SingleSwitch(config)
        self.layers = nn.ModuleList(GraniteSwitchAttentionDecoderLayer(config, i + self.switch.num_cache_layers)
                                    for i in range(config.num_hidden_layers - self.switch.num_cache_layers))
        self.norm = GraniteMoeHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GraniteMoeHybridRotaryEmbedding(config=config)

    def forward(self, input_ids):
        b, T = input_ids.shape
        dt = self.embed_tokens.weight.dtype
        mask = torch.full((T, T), torch.finfo(dt).min, dtype=dt).triu(1)[None, None].expand(b, 1, T, T)
        ctrl = torch.as_tensor(self.config.adapter_token_ids, dtype=torch.long)
        adapter_indices, ids = self.switch(input_ids, ctrl, mask)
        h = self.embed_tokens(ids) * self.embedding_multiplier
        pe = self.rotary_emb(h, position_ids=torch.arange(T)[None])
        for layer in self.layers:
            h = layer(h, mask, pe, adapter_indices)
        self._last_adapter_indices = adapter_indices
        return self.norm(h)


class GraniteSwitchForCausalLM(nn.Module):                           # modeling 676-789
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GraniteSwitchModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids)) / self.config.logits_scaling
