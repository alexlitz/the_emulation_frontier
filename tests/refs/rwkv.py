"""REFERENCE transcriptions (read as text, never run) of the RWKV-5 / -6 / -7 time-mixing blocks.

No transformers release ships these: 4.57 and 5.18 have only `rwkv` (RWKV-4, RwkvSelfAttention with
time_first / time_decay scalars per channel). Each class below mirrors the remote module's parameter
names and shapes so the host's block loaders (rwkv5_block / rwkv6_block / rwkv7_block) read it exactly
as they read the remote module. Plain torch, fp64: the remote CPU paths' `.float()` / float32 compute
casts are lifted to the input dtype (that is the only change).

RWKV-5  RWKV/v5-Eagle-7B-HF, snapshot bb01ae9434eb9f4934c1ebe486eb7d3e25883d72, modeling_rwkv5.py
        sha256 d0df9e09371901242d4c9d17c212169d0b6e65a6e90a5b5af0c2014428a5fdcf
        l. 166-206 rwkv_linear_attention_v5_cpu (the CPU path rwkv_linear_attention takes off-CUDA, l. 209-240),
        l. 259-353 RwkvSelfAttention (__init__, extract_key_value, forward).
RWKV-6  RWKV/rwkv-6-world-3b, snapshot d398e6d9dddbc95185a40de4ea437b2193bbe3be, modeling_rwkv6.py
        sha256 cabf408d8452e4feb228f9a66dcd2ebd874a56fba7c4e7474f08563246eaa76a
        l. 54-76 rwkv6_linear_attention_cpu (taken in eval: `not training`, l. 78-92),
        l. 107-206 Rwkv6SelfAttention (data-dependent ddlerp token shift, LoRA decay, GroupNorm eps
        1e-5 * head_size_divisor ** 2 from configuration_rwkv6.py l. 90-107).
RWKV-7  RWKV/RWKV7-G1j-13.3B-20260831, snapshot 1ead51ecbe85f5ac1d2f3f0201ecbbd5ff7ce301, modeling_rwkv7.py
        sha256 28fc373673b2c98805275cc55381db0dd1dd9325ec0c1b34264d49ca1bc16958
        l. 41-63 Rwkv7TokenShift (no cu_seq_lens / keep), l. 107-156 rwkv7_recurrent (the 'eager' WKV,
        l. 292-337, compute_dtype lifted from float32 to the input dtype), l. 375-571 Rwkv7Attention
        (__init__, forward without padding / packing, lora_gates). `tokenwise` (l. 94-104) only changes
        the evaluation order of the projections (per token vs batched), not their values; batched here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- RWKV-5 ------------------------------------------------------------------------------------------
class Rwkv5Config:
    def __init__(self, hidden_size=32, head_size=8, attention_hidden_size=None, **kw):
        self.hidden_size, self.head_size = hidden_size, head_size
        self.attention_hidden_size = attention_hidden_size
        self.__dict__.update(kw)


def rwkv5_linear_attention_cpu(B, H, S, T, n_head, hidden, time_decay, time_first, receptance, key, value,
                               gate, lxw, lxb, ow, state):
    dt = hidden.dtype                                                     # remote: torch.float32
    key = key.to(dt).view(B, T, H, S).transpose(1, 2).transpose(-2, -1)
    value = value.to(dt).view(B, T, H, S).transpose(1, 2)
    receptance = receptance.to(dt).view(B, T, H, S).transpose(1, 2)
    time_decay = torch.exp(-torch.exp(time_decay.to(dt))).reshape(-1, 1, 1).reshape(n_head, -1, 1)
    time_first = time_first.to(dt).reshape(-1, 1, 1).reshape(n_head, -1, 1)
    lxw, lxb = lxw.to(dt), lxb.to(dt)
    out = torch.zeros_like(key).reshape(B, T, H, S)
    for t in range(T):
        rt = receptance[:, :, t:t + 1, :]
        kt = key[:, :, :, t:t + 1]
        vt = value[:, :, t:t + 1, :]
        at = kt @ vt
        out[:, t] = (rt @ (time_first * at + state)).squeeze(2)
        state = at + time_decay * state
    out = out.reshape(B * T, H * S)
    out = F.group_norm(out, num_groups=H, weight=lxw, bias=lxb).reshape(B, T, H * S)
    out = out.to(dtype=hidden.dtype) * gate
    return out @ ow, state


class Rwkv5SelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config, self.layer_id = config, layer_id
        h = config.hidden_size
        H = h // config.head_size
        self.num_attention_heads = H
        A = config.attention_hidden_size if config.attention_hidden_size is not None else h
        self.attention_hidden_size = A
        self.time_decay = nn.Parameter(torch.empty(H, config.head_size))
        self.time_faaaa = nn.Parameter(torch.empty(H, config.head_size))
        self.time_mix_gate = nn.Parameter(torch.empty(1, 1, h))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, h))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, h))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, h))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(h, A, bias=False)
        self.value = nn.Linear(h, A, bias=False)
        self.receptance = nn.Linear(h, A, bias=False)
        self.gate = nn.Linear(h, A, bias=False)
        self.output = nn.Linear(A, h, bias=False)
        self.ln_x = nn.GroupNorm(h // config.head_size, h)

    def extract_key_value(self, hidden):
        shifted = self.time_shift(hidden)
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        gate = hidden * self.time_mix_gate + shifted * (1 - self.time_mix_gate)
        return self.receptance(receptance), self.key(key), self.value(value), F.silu(self.gate(gate))

    def forward(self, hidden, state=None):
        """`state`: the (B, H, S, S) wkv state (zeros = a sequence start, as Rwkv5Model allocates)."""
        B, T = hidden.shape[0], hidden.shape[1]
        H = self.time_decay.shape[0]
        S = hidden.shape[-1] // H
        r, k, v, g = self.extract_key_value(hidden)
        if state is None:
            state = torch.zeros(B, H, S, S, dtype=hidden.dtype)
        return rwkv5_linear_attention_cpu(B, H, S, T, self.num_attention_heads, hidden, self.time_decay,
                                          self.time_faaaa, r, k, v, g, self.ln_x.weight, self.ln_x.bias,
                                          self.output.weight.t(), state)


# --- RWKV-6 ------------------------------------------------------------------------------------------
class Rwkv6Config:
    def __init__(self, hidden_size=32, attention_hidden_size=None, head_size=64, head_size_divisor=8, **kw):
        self.hidden_size = hidden_size
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        self.head_size, self.head_size_divisor = head_size, head_size_divisor
        self.__dict__.update(kw)


def rwkv6_linear_attention_cpu(receptance, key, value, time_decay, time_first, state):
    batch, seq_length, _ = receptance.shape
    num_heads, head_size = time_first.shape
    dt = receptance.dtype                                                 # remote: .float()
    key = key.to(dt).view(batch, seq_length, num_heads, head_size).transpose(1, 2).transpose(-2, -1)
    value = value.to(dt).view(batch, seq_length, num_heads, head_size).transpose(1, 2)
    receptance = receptance.to(dt).view(batch, seq_length, num_heads, head_size).transpose(1, 2)
    time_decay = torch.exp(-torch.exp(time_decay.to(dt))).view(batch, seq_length, num_heads, head_size).permute(0, 2, 3, 1)
    time_first = time_first.to(dt).reshape(-1, 1, 1).reshape(num_heads, -1, 1)
    out = torch.zeros_like(key).reshape(batch, seq_length, num_heads, head_size)
    for i in range(seq_length):
        r_i = receptance[:, :, i:i + 1, :]
        k_i = key[:, :, :, i:i + 1]
        v_i = value[:, :, i:i + 1, :]
        w_i = time_decay[:, :, :, i:i + 1]
        kv = k_i @ v_i
        out[:, i] = (r_i @ (time_first * kv + state)).squeeze(2)
        state = kv + w_i * state
    return out, state


class Rwkv6SelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config, self.layer_id = config, layer_id
        h, A = config.hidden_size, config.attention_hidden_size
        self.attention_hidden_size = A
        H = A // config.head_size
        self.time_maa_x = nn.Parameter(torch.empty(1, 1, h))
        self.time_maa_w = nn.Parameter(torch.empty(1, 1, h))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, h))
        self.time_maa_v = nn.Parameter(torch.empty(1, 1, h))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, h))
        self.time_maa_g = nn.Parameter(torch.empty(1, 1, h))
        E = 32                                                            # TIME_MIX_EXTRA_DIM
        self.time_maa_w1 = nn.Parameter(torch.empty(h, E * 5))
        self.time_maa_w2 = nn.Parameter(torch.empty(5, E, h))
        self.time_decay = nn.Parameter(torch.empty(1, 1, A))
        Dd = 64                                                           # TIME_DECAY_EXTRA_DIM
        self.time_decay_w1 = nn.Parameter(torch.empty(h, Dd))
        self.time_decay_w2 = nn.Parameter(torch.empty(Dd, A))
        self.time_faaaa = nn.Parameter(torch.empty(H, config.head_size))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(h, A, bias=False)
        self.key = nn.Linear(h, A, bias=False)
        self.value = nn.Linear(h, A, bias=False)
        self.gate = nn.Linear(h, A, bias=False)
        self.output = nn.Linear(A, h, bias=False)
        self.ln_x = nn.GroupNorm(H, h, eps=(1e-5) * (config.head_size_divisor ** 2))

    def extract_key_value(self, hidden):
        shifted = self.time_shift(hidden)
        x = hidden
        B, T, C = hidden.shape
        xx = shifted - x
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        time_decay = x + xx * (self.time_maa_w + mw)
        key = x + xx * (self.time_maa_k + mk)
        value = x + xx * (self.time_maa_v + mv)
        receptance = x + xx * (self.time_maa_r + mr)
        gate = x + xx * (self.time_maa_g + mg)
        receptance, key, value = self.receptance(receptance), self.key(key), self.value(value)
        gate = F.silu(self.gate(gate))
        time_decay = torch.tanh(time_decay @ self.time_decay_w1) @ self.time_decay_w2
        return receptance, key, value, gate, self.time_decay + time_decay

    def forward(self, hidden, state=None):
        r, k, v, g, w = self.extract_key_value(hidden)
        B, T, C = r.shape
        H, S = self.time_faaaa.shape
        if state is None:
            state = torch.zeros(B, H, S, S, dtype=hidden.dtype)
        out, state = rwkv6_linear_attention_cpu(r, k, v, w, self.time_faaaa, state)
        out = out.reshape(B * T, H * S)
        out = F.group_norm(out, num_groups=H, weight=self.ln_x.weight.to(out.dtype),
                           bias=self.ln_x.bias.to(out.dtype), eps=self.ln_x.eps).reshape(B, T, H * S)
        out = out.to(dtype=hidden.dtype) * g
        return self.output(out), state


# --- RWKV-7 ------------------------------------------------------------------------------------------
_INV_SQRT_E = 0.6065306597126334


def rwkv7_recurrent(r, w_log, k, v, kk, a, state):
    """l. 107-156 with compute_dtype = the input dtype (remote default float32)."""
    batch, seq_len, num_heads, head_dim = r.shape
    dtype = r.dtype
    out = torch.empty(batch, seq_len, num_heads, head_dim, dtype=dtype)
    state = state.to(dtype)
    for t in range(seq_len):
        decay = torch.exp(w_log[:, t])[:, :, None, :]
        kk_t = kk[:, t]
        value_key = v[:, t][..., None] @ k[:, t][:, :, None, :]
        anti_value_key = (-kk_t)[..., None] @ (kk_t * a[:, t])[:, :, None, :]
        matrix = state
        state = matrix * decay
        state = state + matrix @ anti_value_key
        state = state + value_key
        out[:, t] = (state @ r[:, t][..., None]).squeeze(-1)
    return out, state


class Rwkv7Attention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config, self.layer_id = config, layer_id
        C = config.hidden_size
        self.hidden_size, self.num_heads, self.head_dim = C, config.num_heads, config.head_dim
        for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, n, nn.Parameter(torch.zeros(1, 1, C)))
        self.w1 = nn.Parameter(torch.zeros(C, config.decay_low_rank_dim))
        self.w2 = nn.Parameter(torch.zeros(config.decay_low_rank_dim, C))
        self.w0 = nn.Parameter(torch.zeros(1, 1, C))
        self.a1 = nn.Parameter(torch.zeros(C, config.a_low_rank_dim))
        self.a2 = nn.Parameter(torch.zeros(config.a_low_rank_dim, C))
        self.a0 = nn.Parameter(torch.zeros(1, 1, C))
        self.g1 = nn.Parameter(torch.zeros(C, config.gate_low_rank_dim))
        self.g2 = nn.Parameter(torch.zeros(config.gate_low_rank_dim, C))
        if layer_id == 0:                                                 # layer 0 produces v_first: never read
            self.register_buffer("v1", torch.zeros(C, config.v_low_rank_dim))
            self.register_buffer("v2", torch.zeros(config.v_low_rank_dim, C))
            self.register_buffer("v0", torch.zeros(1, 1, C))
        else:
            self.v1 = nn.Parameter(torch.zeros(C, config.v_low_rank_dim))
            self.v2 = nn.Parameter(torch.zeros(config.v_low_rank_dim, C))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C))
        self.k_k = nn.Parameter(torch.zeros(1, 1, C))
        self.k_a = nn.Parameter(torch.zeros(1, 1, C))
        self.r_k = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(config.num_heads, C, eps=config.norm_eps * config.head_dim)

    def lora_gates(self, xw, xa, xg, xv):
        w_log = -_INV_SQRT_E * torch.sigmoid(torch.tanh(xw @ self.w1) @ self.w2 + self.w0)
        a = torch.sigmoid(xa @ self.a1 @ self.a2 + self.a0)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        v_gate = None if xv is None else torch.sigmoid(xv @ self.v1 @ self.v2 + self.v0)
        return w_log, a, g, v_gate

    def forward(self, hidden_states, v_first=None):
        """-> (y, v_first). Sequence start (zero shift state, zero wkv state), no padding."""
        batch, seq_len, C = hidden_states.shape
        H, N = self.num_heads, self.head_dim
        shifted = torch.cat([torch.zeros_like(hidden_states[:, :1]), hidden_states[:, :-1]], dim=1)
        delta = shifted - hidden_states
        xr, xw, xk, xv, xa, xg = (hidden_states + getattr(self, n) * delta
                                  for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))
        r, k, v = self.receptance(xr), self.key(xk), self.value(xv)
        w_log, a, g, v_gate = self.lora_gates(xw, xa, xg, None if self.layer_id == 0 else xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * v_gate
        kk = k * self.k_k
        kk = F.normalize(kk.view(batch, seq_len, H, N), dim=-1, p=2.0).view(batch, seq_len, C)
        k = k * (1 + (a - 1) * self.k_a)
        heads = lambda t: t.view(batch, seq_len, H, N)
        state = torch.zeros(batch, H, N, N, dtype=hidden_states.dtype)
        y, _ = rwkv7_recurrent(heads(r), heads(w_log), heads(k), heads(v), heads(kk), heads(a), state)
        y = self.ln_x(y.reshape(batch * seq_len, C)).view(batch, seq_len, C)
        bonus = ((heads(r) * heads(k) * self.r_k).sum(dim=-1, keepdim=True) * heads(v)).reshape(batch, seq_len, C)
        return self.output((y + bonus) * g), v_first
