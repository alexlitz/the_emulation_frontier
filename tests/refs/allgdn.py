"""REFERENCE transcription of summerMC/Cyber-9B-Qwen-AllGDN-Engram, read as text, never run.

Source repo: summerMC/Cyber-9B-Qwen-AllGDN-Engram, snapshot 4aa32fa9b0fd644909f0f2da7b1d56432454361d
  modeling_allgdn.py  sha256 c955beee54187ac087daec54a59af68a4ff43d6c7456434e1deadc8100b03f3f
    AllGDNConfig (l. 23-24), EngramRMS (l. 65-73), Engram (l. 76-126), AllGDNForCausalLM.__init__ /
    _init_weights / _make_inject (l. 132-178), _set_engram_ids / _forward_one (l. 183-207),
    forward (l. 209-229: the single-pass path; the chunked-prefill loop l. 231-246 is not transcribed).
  (engram.py, sha256 1b67d2d9..., is the repo's standalone attach-to-stock-Qwen3.5 variant; the
   AutoModel path the old test ran is modeling_allgdn.py, transcribed here.)

The remote file subclasses the stock transformers 5.x Qwen3_5ForCausalLM / Qwen3_5TextConfig (its l. 19-20);
so does this reference (library code). Not transcribed: `_patch_cache_classes` (l. 30-59), which only wraps
cache `get_seq_length` / `get_mask_sizes` for generation with zero attention layers (unused with
use_cache=False), and the `_allgdn_len` bookkeeping on a cache. EngramRMS's fp32 casts (l. 72-73) are lifted
to fp64. The stock Qwen3.5 body keeps its own fp32 casts (RMSNorm, gated norm, delta rule).
Needs transformers 5.x (run it inside the TF5 subprocess, tests/test_universal_elephantine._tf5_run).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

F64 = torch.float64


class AllGDNConfig(Qwen3_5TextConfig):
    model_type = "allgdn"


class EngramRMS(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x.to(F64)                                         # remote: x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w.to(F64)


class Engram(nn.Module):
    PRIMES = [1000003, 999983, 1000033, 999979, 1000037, 999961, 1000039, 999953]

    def __init__(self, hidden, vocab, orders=(2, 3), heads=4, slots=2**16, dim=128, conv=4):
        super().__init__()
        self.orders, self.heads, self.slots, self.pad = tuple(orders), heads, slots, vocab
        self.table = nn.Embedding(len(self.orders) * heads * slots, dim)
        mem = len(self.orders) * heads * dim
        self.key = nn.Linear(mem, hidden, bias=False)
        self.val = nn.Linear(mem, hidden, bias=False)
        self.nh, self.nk = EngramRMS(hidden), EngramRMS(hidden)
        self.conv = nn.Conv1d(hidden, hidden, conv, groups=hidden, padding=conv - 1, bias=False)

    def hash_ids(self, ids):
        B, T = ids.shape
        maxn = max(self.orders)
        p = F.pad(ids, (maxn - 1, 0), value=self.pad)
        out, off = [], 0
        for n in self.orders:
            grams = [p[:, maxn - 1 - j: maxn - 1 - j + T] for j in range(n)]
            for h in range(self.heads):
                P = self.PRIMES[h % len(self.PRIMES)]
                x = torch.zeros_like(ids)
                for g in grams:
                    x = x * P + g + 1
                out.append(torch.remainder(x ^ (x >> 17), self.slots) + off)
                off += self.slots
        return torch.stack(out, -1)

    def forward(self, h, ids, conv_state=None, return_state=False):
        T = h.shape[1]
        e = self.table(self.hash_ids(ids)).flatten(-2)[:, -T:].to(self.key.weight.dtype)
        k, v = self.key(e), self.val(e)
        gate = torch.sigmoid((self.nh(h) * self.nk(k)).sum(-1, keepdim=True) / math.sqrt(h.shape[-1]))
        v = gate.to(v.dtype) * v
        K = self.conv.kernel_size[0]
        vt = v.transpose(1, 2)
        if conv_state is None:
            conv_state = vt.new_zeros(vt.shape[0], vt.shape[1], K - 1)
        x = torch.cat([conv_state.to(device=vt.device, dtype=vt.dtype), vt], dim=-1)
        y = F.conv1d(x, self.conv.weight, groups=self.conv.groups)
        out = v + y.transpose(1, 2)
        if return_state:
            return out, x[..., -(K - 1):].detach()
        return out


class AllGDNForCausalLM(Qwen3_5ForCausalLM):
    config_class = AllGDNConfig

    def __init__(self, config):
        super().__init__(config)
        ec = getattr(config, "engram", None) or {}
        self.engram_layers = [int(i) for i in ec.get("layers", [])]
        self.engrams = nn.ModuleDict({
            str(i): Engram(config.hidden_size, config.vocab_size, ec.get("orders", (2, 3)), ec.get("heads", 4),
                           ec.get("slots", 2**16), ec.get("dim", 128))
            for i in self.engram_layers})
        self._eg_ids = None
        self._eg_buf = None
        self._eg_conv = {}
        for i in self.engram_layers:
            self.model.layers[i].register_forward_pre_hook(self._make_inject(i), with_kwargs=True)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, Engram):
            nn.init.normal_(module.table.weight, std=0.02)
            nn.init.zeros_(module.val.weight)
            nn.init.zeros_(module.conv.weight)
            return
        if isinstance(module, EngramRMS):
            nn.init.ones_(module.w)
            return
        super()._init_weights(module)

    def _make_inject(self, i):
        def inject(mod, args, kwargs):
            if self._eg_ids is None:
                return
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            state = self._eg_conv.get(i)
            if state is not None and state.shape[0] != h.shape[0]:
                state = None
            add, new_state = self.engrams[str(i)](h, self._eg_ids, state, return_state=True)
            self._eg_conv[i] = new_state
            add = add[:, -h.shape[1]:].to(h.dtype)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = h + add
            else:
                args = (h + add,) + tuple(args[1:])
            return args, kwargs
        return inject

    def _set_engram_ids(self, input_ids, continue_seq):
        if input_ids is None:
            self._eg_ids = None
            return
        buf = self._eg_buf
        if continue_seq and buf is not None and buf.shape[0] == input_ids.shape[0]:
            full = torch.cat([buf.to(input_ids.device), input_ids], 1)
        else:
            full = input_ids
            self._eg_conv = {}
        self._eg_ids, self._eg_buf = full, full[:, -2:]

    def _forward_one(self, input_ids, continue_seq, **kwargs):
        self._set_engram_ids(input_ids, continue_seq)
        return super().forward(input_ids=input_ids, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        chunk = int(getattr(self.config, "prefill_chunk", 2048) or 0)
        T = input_ids.shape[1] if input_ids is not None else (inputs_embeds.shape[1] if inputs_embeds is not None else 0)
        assert not (chunk > 0 and T > chunk), "chunked prefill is not transcribed"
        assert past_key_values is None
        return self._forward_one(input_ids, continue_seq=False, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                 logits_to_keep=logits_to_keep, **kwargs)
