"""DeepSeek-V4.1 (deepseek_v41 / deepseek_v41_text) on the universal host: from_deepseek_v4 -> from_deepseek_v41.

No transformers class exists for V4.1. Its code is the official inference package in deepseek-ai/DeepSeek-V4.1-Flash
(snapshot 314113f8a7dadb388215396ba5fbee2c610a04d6), READ AS TEXT, NEVER IMPORTED OR RUN (it needs tilelang
kernels and fp8 / fp4 GPU GEMMs). The model below is a REFERENCE: a transcription of its prefill forward
(start_pos = 0, no cache), dense and in fp64, with the official module / parameter names:
    inference/model.py   md5 0f88edd28a7d3d07944426b003c69a85
        ModelArgs l. 44-149; RMSNorm l. 281-293; Engram l. 328-365; precompute_freqs_cis l. 368-389;
        apply_rotary_emb l. 392-406; get_window_topk_idxs l. 409-426; Compressor l. 429-485; Indexer l. 488-581;
        select_candidate_blocks l. 584-611; Attention l. 614-791; Gate l. 794-829; Expert l. 832-853;
        MoE l. 856-906; Block l. 909-996; make_identity_pre_mix l. 1162-1166; SharedAttentionRuntime l. 1169-1183;
        Transformer.__init__ / forward l. 1186-1275 (text only)
    inference/kernel.py  md5 f6a45ea638bd37cea9161a52368948db
        sparse_attn_kernel l. 311-389 (a softmax over the selected slots, -1 slots masked, the sink adding
        exp(attn_sink) to the denominator only) -> dense masked attention here; hc_split_sinkhorn_kernel l. 407-474
    inference/engram.py  md5 827186d041fa816be8e206102635d79a
        find_next_prime / compute_hash_multipliers / EngramLayout / NgramHashState l. 9-184
Dropped, as numerics not architecture: act_quant / fp4_act_quant (in-place quantise-dequantise) and the fp8 / fp4
weight GEMMs, the fp8 Engram table (here a dequantised float table), the bf16 / fp32 casts. Replaced: the tokenizer-
derived compressed token map (build_compressed_token_map) by a random map of the same shape; ParallelEmbedding /
ParallelHead / Column- / RowParallelLinear by their world_size = 1 forms. Not transcribed: decode (start_pos > 0),
vision, the DSpark draft head. The reference keeps its ModelArgs as `.args` for the loader.

The test config makes each V4.1 difference from V4 bind: (a) no per-head q norm, (b) the lagged hyper-connection
pre-mix (layer 0 from stream 0 alone, the final collapse by the last FFN's pre), (c) ratio-2 non-overlapping pools,
ratio-1 per-token keys, kv_source_layers / index_source_layers sharing, candidate pre-filtering; plus the hc-stream
Engram and a SwiGLU limit that clamps.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

F64 = torch.float64


@dataclass
class Args:                                     # ModelArgs (l. 44-149), the fields the text forward reads
    vocab_size: int = 61
    dim: int = 32
    moe_inter_dim: int = 24
    n_layers: int = 6
    n_heads: int = 4
    n_routed_experts: int = 6
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.3
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 0.4
    q_lora_rank: int = 12
    head_dim: int = 16
    rope_head_dim: int = 8
    norm_eps: float = 1e-6
    o_groups: int = 2
    o_lora_rank: int = 6
    window_size: int = 4
    compress_ratios: tuple = (0, 2, 2, 1, 1, 1)
    kv_source_layers: tuple = (1, 3)
    index_source_layers: tuple = (1, 3, 4)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 4
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 2
    index_head_dim: int = 12
    index_topk: int = 3
    candidate_source_layer: int = 3
    candidate_topk_blocks: int = 2
    candidate_block_size: int = 3
    hc_mult: int = 3
    hc_sinkhorn_iters: int = 5
    hc_eps: float = 1e-6
    engram_layer_ids: tuple = (1, 4)
    engram_max_ngram_size: int = 3
    engram_vocab_size: int = 50
    engram_n_heads: int = 2
    engram_head_dim: int = 4
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 40
    max_seq_len: int = 64


class RMSNorm(nn.Module):                                       # l. 281-293
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim, self.eps, self.weight = dim, eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps))


def lin(i, o):
    return nn.Linear(i, o, bias=False)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):   # l. 368-389
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=F64) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=F64) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, dtype=F64), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x, freqs_cis, inverse=False):              # l. 392-406 (out of place)
    xc = torch.view_as_complex(x.contiguous().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    fc = freqs_cis.view(1, xc.size(1), xc.size(-1)) if xc.ndim == 3 else freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * fc).flatten(-2)


def rope_tail(x, fc, rd, inverse=False):
    return torch.cat([x[..., :-rd], apply_rotary_emb(x[..., -rd:], fc, inverse)], -1)


def get_window_topk_idxs(window_size, bsz, seqlen):              # l. 409-426, start_pos 0
    end = torch.arange(seqlen).unsqueeze(1)
    idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
    idxs = torch.where(idxs > end, -1, idxs)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


class Shared:                                                    # SharedAttentionRuntime, l. 1169-1183
    compress_kv = index_k = topk_idxs = candidates = None


class Compressor(nn.Module):                                     # l. 429-485, start_pos 0
    def __init__(self, a, layer_id):
        super().__init__()
        self.compress_ratio = r = a.compress_ratios[layer_id]
        self.norm = RMSNorm(a.head_dim, a.norm_eps)
        self.wkv = lin(a.dim, a.head_dim)
        if r > 1:
            self.wgate = lin(a.dim, a.head_dim)

    def forward(self, x):
        r = self.compress_ratio
        if r == 1:
            return self.norm(self.wkv(x))
        T = x.shape[1]
        if T < r:
            return None
        cut = T - T % r
        kv, score = self.wkv(x)[:, :cut].unflatten(1, (-1, r)), self.wgate(x)[:, :cut].unflatten(1, (-1, r))
        return self.norm((kv * score.softmax(dim=2)).sum(dim=2))


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):   # l. 584-611
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(nn.Module):                                        # l. 488-581, start_pos 0
    def __init__(self, a, layer_id, shared):
        super().__init__()
        self.owns_k = layer_id in a.kv_source_layers
        self.compress_ratio = a.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == a.candidate_source_layer
        self.uses_candidates = 0 <= a.candidate_source_layer < layer_id
        self.candidate_topk_blocks, self.candidate_block_size = a.candidate_topk_blocks, a.candidate_block_size
        self.n_heads, self.index_head_dim, self.rope_head_dim = a.index_n_heads, a.index_head_dim, a.rope_head_dim
        self.index_topk = a.index_topk
        self.softmax_scale = self.index_head_dim ** -0.5
        self.wq_b = lin(a.q_lora_rank, self.n_heads * self.index_head_dim)
        self.weights_proj = lin(a.dim, self.n_heads)
        self.freqs_cis = None
        self.shared = shared
        if self.owns_k:
            self.wk = lin(a.head_dim, self.index_head_dim)
            self.k_norm = RMSNorm(self.index_head_dim, a.norm_eps)

    def forward(self, x, qr, latent):
        sh, T = self.shared, x.shape[1]
        ratio, rd = self.compress_ratio, self.rope_head_dim
        if self.owns_k and latent is not None:
            k = self.k_norm(self.wk(latent))
            sh.index_k = rope_tail(k, self.freqs_cis[: T - T % ratio: ratio], rd)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        q = rope_tail(q, self.freqs_cis[:T], rd)
        index_k = sh.index_k[:, : T // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        compress_lens = (torch.arange(1, T + 1) // ratio).unsqueeze(-1)
        index_score = index_score.masked_fill(torch.arange(T // ratio) >= compress_lens, -torch.inf)
        if self.is_candidate_source:
            sh.candidates = select_candidate_blocks(index_score, compress_lens, self.candidate_topk_blocks,
                                                    self.candidate_block_size)
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~sh.candidates, -torch.inf)
        topk = min(self.index_topk, T // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + T, -1)        # offset = the window kv length (T)


class Attention(nn.Module):                                      # l. 614-791, start_pos 0
    def __init__(self, layer_id, a, shared):
        super().__init__()
        self.n_heads, self.head_dim, self.rope_head_dim = a.n_heads, a.head_dim, a.rope_head_dim
        self.n_groups, self.o_lora_rank, self.window_size = a.o_groups, a.o_lora_rank, a.window_size
        self.compress_ratio = a.compress_ratios[layer_id]
        self.shared = shared
        self.attn_sink = nn.Parameter(torch.zeros(a.n_heads))
        self.wq_a = lin(a.dim, a.q_lora_rank)
        self.q_norm = RMSNorm(a.q_lora_rank, a.norm_eps)
        self.wq_b = lin(a.q_lora_rank, a.n_heads * a.head_dim)
        self.wkv = lin(a.dim, a.head_dim)
        self.kv_norm = RMSNorm(a.head_dim, a.norm_eps)
        self.wo_a = lin(a.n_heads * a.head_dim // a.o_groups, a.o_groups * a.o_lora_rank)
        self.wo_b = lin(a.o_groups * a.o_lora_rank, a.dim)
        self.softmax_scale = a.head_dim ** -0.5
        self.is_kv_source = layer_id in a.kv_source_layers
        self.is_index_source = layer_id in a.index_source_layers
        self.compressor = Compressor(a, layer_id) if self.is_kv_source else None
        self.indexer = Indexer(a, layer_id, shared) if self.is_index_source else None
        if self.compress_ratio:
            osl, theta = a.original_seq_len, a.compress_rope_theta
        else:
            osl, theta = 0, a.rope_theta
        self.register_buffer("freqs_cis", precompute_freqs_cis(a.rope_head_dim, a.max_seq_len, osl, theta,
                                                               a.rope_factor, a.beta_fast, a.beta_slow))

    def forward(self, x):
        sh, (B, T, _) = self.shared, x.shape
        fc, rd = self.freqs_cis[:T], self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = rope_tail(self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)), fc, rd)   # (a): no head norm
        kv = rope_tail(self.kv_norm(self.wkv(x)), fc, rd)
        idx = get_window_topk_idxs(self.window_size, B, T)
        if self.compress_ratio:
            ratio = self.compress_ratio
            latent = None
            if self.is_kv_source:
                latent = self.compressor(x)
            if self.is_index_source:                               # _compress_topk_idxs
                if self.indexer.freqs_cis is None:
                    self.indexer.freqs_cis = self.freqs_cis
                sh.topk_idxs = self.indexer(x, qr, latent)
            cidx = sh.topk_idxs
            if latent is not None:
                sh.compress_kv = rope_tail(latent, self.freqs_cis[: T - T % ratio: ratio], rd)
            kv = torch.cat([kv, sh.compress_kv[:, : T // ratio]], 1)
            idx = torch.cat([idx, cidx], -1)
        # sparse_attn (kernel.py l. 311-403) as a dense mask over the gathered slots, the sink in the denominator
        sel = torch.zeros(B, T, kv.shape[1] + 1, dtype=torch.bool).scatter_(-1, idx.where(idx >= 0, kv.shape[1]), True)
        s = torch.einsum("bthd,bsd->bhts", q, kv) * self.softmax_scale
        s = s.masked_fill(~sel[:, None, :, : kv.shape[1]], -torch.inf)
        m = s.amax(-1, keepdim=True).clamp(min=-1e30)
        e = (s - m).exp()
        den = e.sum(-1, keepdim=True) + (self.attn_sink[None, :, None, None] - m).exp()
        o = torch.einsum("bhts,bsd->bthd", e / den, kv)
        o = rope_tail(o, fc, rd, inverse=True)
        o = o.view(B, T, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        return self.wo_b(torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2))


class Gate(nn.Module):                                           # l. 794-829
    def __init__(self, a):
        super().__init__()
        self.topk, self.score_func, self.gate_temp = a.n_activated_experts, a.score_func, a.gate_temp
        self.norm_topk_prob, self.route_scale = a.norm_topk_prob, a.route_scale
        self.weight = nn.Parameter(torch.zeros(a.n_routed_experts, a.dim))
        self.bias = nn.Parameter(torch.zeros(a.n_routed_experts))

    def forward(self, x):
        scores = F.linear(x, self.weight) / self.gate_temp
        scores = (scores.softmax(-1) if self.score_func == "softmax" else
                  scores.sigmoid() if self.score_func == "sigmoid" else F.softplus(scores).sqrt())
        indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


class Expert(nn.Module):                                         # l. 832-853
    def __init__(self, dim, inter, swiglu_limit=0.0):
        super().__init__()
        self.w1, self.w2, self.w3, self.swiglu_limit = lin(dim, inter), lin(inter, dim), lin(dim, inter), swiglu_limit

    def forward(self, x, weights=None):
        gate, up = self.w1(x), self.w3(x)
        if self.swiglu_limit > 0:
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            gate = gate.clamp(max=self.swiglu_limit)
        x = F.silu(gate) * up
        return self.w2(x if weights is None else weights * x)


class MoE(nn.Module):                                            # l. 856-906
    def __init__(self, a):
        super().__init__()
        self.dim = a.dim
        self.gate = Gate(a)
        self.experts = nn.ModuleList(Expert(a.dim, a.moe_inter_dim, a.swiglu_limit) for _ in range(a.n_routed_experts))
        self.shared_experts = Expert(a.dim, a.moe_inter_dim, a.swiglu_limit)

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        for i, ex in enumerate(self.experts):
            idx, top = torch.where(indices == i)
            if idx.numel():
                y[idx] += ex(x[idx], weights[idx, top, None])
        return (y + self.shared_experts(x)).view(shape)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):   # kernel.py l. 407-474
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def find_next_prime(start, seen):                                # engram.py l. 9-14
    from sympy import isprime
    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


def compute_hash_multipliers(layer_ids, max_ngram_size, vocab):  # engram.py l. 64-83
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    rows = [torch.tensor(np.random.default_rng(10007 * l).integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
                         * 2 + 1) for l in layer_ids]
    return torch.stack(rows)


def engram_primes(a):                                            # EngramLayout.from_args, engram.py l. 102-126
    primes, seen = [], set()
    for _ in a.engram_layer_ids:
        per = []
        for _ in range(a.engram_max_ngram_size - 1):
            sizes, cur = [], a.engram_vocab_size - 1
            for _ in range(a.engram_n_heads):
                cur = find_next_prime(cur, seen); seen.add(cur); sizes.append(cur)
            per.append(sizes)
        primes.append(per)
    return primes


class NgramHashState(nn.Module):                                 # engram.py l. 129-184, start_pos 0, no image mask
    def __init__(self, a, token_map):
        super().__init__()
        self.max_n = a.engram_max_ngram_size
        primes = engram_primes(a)
        flat = [[p for per in layer for p in per] for layer in primes]
        self.register_buffer("primes", torch.tensor(primes))
        self.register_buffer("offsets", torch.tensor(np.array([np.cumsum([0, *s[:-1]]) for s in flat])))
        self.register_buffer("multipliers", compute_hash_multipliers(a.engram_layer_ids, self.max_n,
                                                                     a.engram_compressed_vocab_size))
        self.register_buffer("token_map", token_map)
        self.pad_id = int(token_map[a.engram_pad_id])
        self.num_embeddings = [sum(s) for s in flat]

    def forward(self, input_ids):
        B, T = input_ids.shape
        cache = self.token_map[input_ids]
        positions = torch.arange(T).expand(B, T)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_n):
            source = cache.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, -1)
        products = tokens.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_n):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class Engram(nn.Module):                                         # model.py l. 328-365 (float table)
    def __init__(self, a, layer_id, num_embeddings):
        super().__init__()
        self.layer_hash_index = a.engram_layer_ids.index(layer_id)
        self.dim, self.hc_mult, self.clamp_value, self.eps = a.dim, a.hc_mult, 1e-6, a.norm_eps
        self.embed = nn.Embedding(num_embeddings, a.engram_head_dim)
        cols = (a.engram_max_ngram_size - 1) * a.engram_n_heads
        self.wkv = lin(cols * a.engram_head_dim, a.dim * (a.hc_mult + 1))
        self.q_weight = nn.Parameter(torch.ones(a.hc_mult, a.dim))
        self.k_weight = nn.Parameter(torch.ones(a.hc_mult, a.dim))

    def forward(self, x, hash_ids):
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight * self.k_weight
        rstd = torch.rsqrt(x.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (x * weight * key).sum(-1) * rstd * self.dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        return x + gate.unsqueeze(-1) * value.unsqueeze(-2)


class Block(nn.Module):                                          # l. 909-996
    def __init__(self, layer_id, a, shared, hash_state):
        super().__init__()
        self.norm_eps = a.norm_eps
        self.attn, self.ffn = Attention(layer_id, a, shared), MoE(a)
        self.engram = (Engram(a, layer_id, hash_state.num_embeddings[a.engram_layer_ids.index(layer_id)])
                       if layer_id in a.engram_layer_ids else None)
        self.attn_norm, self.ffn_norm = RMSNorm(a.dim, a.norm_eps), RMSNorm(a.dim, a.norm_eps)
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps = N, it, e = a.hc_mult, a.hc_sinkhorn_iters, a.hc_eps
        mix = (2 + N) * N
        self.hc_attn_fn, self.hc_ffn_fn = nn.Parameter(torch.zeros(mix, N * a.dim)), nn.Parameter(torch.zeros(mix, N * a.dim))
        self.hc_attn_base, self.hc_ffn_base = nn.Parameter(torch.zeros(mix)), nn.Parameter(torch.zeros(mix))
        self.hc_attn_scale, self.hc_ffn_scale = nn.Parameter(torch.ones(3)), nn.Parameter(torch.ones(3))

    def hc_mixes(self, x, fn, scale, base):
        x = x.flatten(2)
        mixes = F.linear(x, fn) * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        return hc_split_sinkhorn(mixes, scale, base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    @staticmethod
    def hc_pre(x, pre_mix):
        return torch.sum(pre_mix.unsqueeze(-1) * x, dim=2)

    @staticmethod
    def hc_post(x, residual, post, comb):
        return post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

    def forward(self, x, pre_mix):
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)))
        x = self.hc_post(x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)))
        return self.hc_post(x, residual, ffn_post, ffn_comb), ffn_pre


class Transformer(nn.Module):                                    # l. 1186-1275, text forward, logits at every position
    def __init__(self, a, token_map):
        super().__init__()
        self.args, self.hc_mult = a, a.hc_mult
        self.shared = Shared()
        self.engram_hash = NgramHashState(a, token_map) if a.engram_layer_ids else None
        self.embed = nn.Embedding(a.vocab_size, a.dim)
        self.layers = nn.ModuleList(Block(i, a, self.shared, self.engram_hash) for i in range(a.n_layers))
        self.norm = RMSNorm(a.dim, a.norm_eps)
        self.head = lin(a.dim, a.vocab_size)

    def forward(self, input_ids):
        hashes = self.engram_hash(input_ids) if self.engram_hash is not None else None
        h = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = torch.zeros(*h.shape[:3], dtype=h.dtype)       # make_identity_pre_mix, l. 1162-1166
        pre_mix[:, :, 0] = 1.0
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
            h, pre_mix = layer(h, pre_mix)
        return self.head(self.norm(Block.hc_pre(h, pre_mix)))


def _build(seed=0, **kw):
    a = Args(**kw)
    torch.manual_seed(seed)
    token_map = torch.randint(0, a.engram_compressed_vocab_size, (a.vocab_size,))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(F64)                                  # fp64 params; the rotary tables stay complex128
    try:
        t = Transformer(a, token_map).eval()
    finally:
        torch.set_default_dtype(prev)
    with torch.no_grad():
        for n, p in t.named_parameters():
            if n.endswith("norm.weight") or "_weight" in n:
                p.copy_(1 + 0.2 * torch.randn_like(p))
            elif "hc_" in n and "scale" in n:
                p.copy_(0.5 + torch.rand_like(p))
            elif n.endswith("gate.bias") or n.endswith("attn_sink"):
                p.copy_(0.5 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 if "embed" in n else 0.3) / max(p.shape[-1], 1) ** 0.25)
    return t


@pytest.mark.parametrize("yarn", [False, True])
@torch.no_grad()
def test_same_host_carries_deepseek_v41(yarn):
    """from_deepseek_v4 (dispatching to from_deepseek_v41) vs the REFERENCE transcription above, T = 20: every
    layer slides (window 4), layers 1-2 read ratio-2 pools built by layer 1, layers 3-5 per-token keys built by
    layer 3, index sources 1 / 3 / 4 (4 reading layer 3's index keys and its candidate blocks), engrams on layers
    1 and 4; with yarn the compressed layers' rotary is YaRN-scaled."""
    from elephantine.universal import from_deepseek_v4
    t = _build(original_seq_len=8 if yarn else 0)
    x = torch.randint(0, t.args.vocab_size, (2, 20), generator=torch.Generator().manual_seed(1))
    ref = t(x)
    u = from_deepseek_v4(t)
    rel = lambda o: (ref - o).abs().max().item() / ref.abs().max().item()
    assert rel(u(x)) < 1e-12, rel(u(x))
    if yarn:
        return
    # each V4.1 difference binds: turning its host branch to the V4 / off setting moves the logits
    def no_lag(v):
        for l in v.layers:
            l.hc_a.lag.zero_(); l.hc_f.lag.zero_()
        v.hc_lag.zero_()
    flips = {
        "(a) q head norm": lambda v: [l.attn.q_coef.copy_(torch.tensor([0.0, 1.0])) for l in v.layers],
        "(b) lagged pre": no_lag,
        "(c) pool picks": lambda v: v.layers[1].attn.cmp.cfg[3].fill_(float("inf")),
        "(c) candidates": lambda v: v.layers[4].attn.cmp.share[3].zero_(),
        "(c) shared picks": lambda v: v.layers[5].attn.cmp.share[2].zero_() and None or
                                      v.layers[5].attn.cmp.cfg[3].fill_(float("inf")),
        "engram": lambda v: v.layers[4].engram_hc.wkv.weight.zero_(),
        "swiglu limit": lambda v: [l.ffn.c_clip.zero_() for l in v.layers],
    }
    for name, flip in flips.items():
        v = from_deepseek_v4(t)
        flip(v)
        assert rel(v(x)) > 1e-6, name
