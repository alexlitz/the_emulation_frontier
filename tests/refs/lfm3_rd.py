"""LFM3-RD (recurrent-depth LFM3): a labelled reference transcription of the remote modelling code, read as text,
never run.

Source:   cybertruck32489/LFM3-1.7B-RD, snapshot fd41791dc768883f1bace7aa44bcc8d309699c04
  modeling_lfm3_rd.py       sha256 ecfe5ec35615a841381a59b2a426521d50b9dd96344af97b7cd92fd797bd0dab
  configuration_lfm3_rd.py  sha256 30d97b6fc6007b98b03d42e746a42f5eebb8dd58e3a505d0e095b6bd078008bc
Transcribed (modeling_lfm3_rd.py); the LFM3 pieces it shares with LiquidAI's modeling_lfm3.py are taken from
tests/refs/lfm3.py (the RD file's copies, 300-347 rotary, 349-357 MLP, 366-399 attention helpers, 418-471 attention,
515-616 short conv, 619-665 decoder layer, 745-760 masks, read side by side: the same forward bar the LoRA
projections below and RMSNorm as F.rms_norm, 63-74):
   77-90    Lfm3SharedCoreBank: 12 shared matrices, [q k v o | conv in out] for two attention and two conv motif slots
   93-138   Lfm3SharedLoRALinear, eval path: W = shared + B @ A (`_get_eval_weight`, 122-128)
  141-192   Lfm3SharedGroupedLoRALinear, eval path: W = shared + concat_g(B_g @ A_g) (172-179)
  410-415   _core_pos; 430-436 / 528-538: core layers take bank slots 0 / 6 (attention, motif position 0 / else) and
            4 / 10 (conv, motif position 1 / else), prelude and coda layers plain nn.Linear
  713-742   Lfm3RDPhase / Lfm3RDCore: num_motifs phases of motif_size layers, all sharing one bank
  899-918   Lfm3RDModel layout: prelude, core, coda, rd_controller, rotary_emb, embedding_norm
  962-1202  Lfm3RDModel.forward, eval, no cache, reasoning_mode "manual" with recurrent_loops T == num_motifs
            (1101-1180): prelude, each phase once in order (s < n_motifs: h_stream advances, the last one is the exit
            state), then x <- p + step_scales[T - 1] * (x - p) with p the prelude output, coda, embedding_norm.
            With T == num_motifs the controller's halting / injection / refinement paths are not reached (they run
            for s >= n_motifs or in training), so of Lfm3RDController (195-224) only step_scales is transcribed.
 1205-1263  Lfm3RDForCausalLM: lm_head (tied to embed_tokens, 1206)
configuration_lfm3_rd.py 19-77: the RD fields and defaults (on top of LFM3's).

Deviations, all numerical: the remote code merges the LoRA weights in fp32 (`.float()` in _get_eval_weight) and runs
RoPE and the softmax in fp32; here they run at the input dtype (fp64 merge, the fp32 casts lifted), so an fp64 run is
fp64 throughout.
"""
import torch
from torch import nn

from tests.refs.lfm3 import (Lfm3Attention, Lfm3Config, Lfm3DecoderLayer, Lfm3RMSNorm, Lfm3RotaryEmbedding,
                             Lfm3ShortConv)


class Lfm3RDConfig(Lfm3Config):
    model_type = "lfm3_rd"

    def __init__(self, num_prelude_layers=2, num_motifs=5, motif_size=4, num_coda_layers=2, rank_attn=640,
                 rank_conv=640, rank_kv=512, conv_lora_groups=3, reasoning_mode="manual", recurrent_loops=5,
                 max_position_embeddings=262_144, **kw):
        super().__init__(max_position_embeddings=max_position_embeddings, **kw)
        self.num_prelude_layers, self.num_motifs, self.motif_size = num_prelude_layers, num_motifs, motif_size
        self.num_coda_layers, self.rank_attn, self.rank_conv, self.rank_kv = num_coda_layers, rank_attn, rank_conv, rank_kv
        self.conv_lora_groups, self.reasoning_mode, self.recurrent_loops = conv_lora_groups, reasoning_mode, recurrent_loops
        assert self.num_hidden_layers == num_prelude_layers + num_motifs * motif_size + num_coda_layers
        assert motif_size == 4


class Lfm3SharedCoreBank(nn.Module):                                # 77-90
    def __init__(self, config):
        super().__init__()
        D, hd = config.hidden_size, config.head_dim
        q, kv = config.num_attention_heads * hd, config.num_key_value_heads * hd
        shapes = [(q, D), (kv, D), (kv, D), (D, q), (3 * D, D), (D, D)] * 2
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(s)) for s in shapes])


class Lfm3SharedLoRALinear(nn.Module):                              # 93-138 (eval path)
    def __init__(self, bank, core_idx, in_features, out_features, rank):
        super().__init__()
        object.__setattr__(self, "_core_bank", bank)
        self.core_idx, self.in_features, self.out_features = core_idx, in_features, out_features
        self.B = nn.Parameter(torch.empty(out_features, rank))
        self.A = nn.Parameter(torch.empty(rank, in_features))

    @property
    def shared_weight(self):
        return self._core_bank.weights[self.core_idx]

    def _get_eval_weight(self, dt, dev):
        return (self.shared_weight + self.B @ self.A).to(dtype=dt, device=dev)

    def forward(self, x):
        return x @ self._get_eval_weight(x.dtype, x.device).T


class Lfm3SharedGroupedLoRALinear(Lfm3SharedLoRALinear):           # 141-192 (eval path)
    def __init__(self, bank, core_idx, in_features, group_out_features, rank, num_groups=3):
        nn.Module.__init__(self)
        object.__setattr__(self, "_core_bank", bank)
        self.core_idx, self.in_features = core_idx, in_features
        self.B = nn.Parameter(torch.empty(num_groups, group_out_features, rank))
        self.A = nn.Parameter(torch.empty(num_groups, rank, in_features))

    def _get_eval_weight(self, dt, dev):
        lora = torch.bmm(self.B, self.A).reshape(-1, self.in_features)
        return (self.shared_weight + lora).to(dtype=dt, device=dev)


def _core_pos(config, layer_idx, bank):                             # 410-415
    start = config.num_prelude_layers
    end = start + config.num_motifs * config.motif_size
    return (layer_idx - start) % config.motif_size if bank is not None and start <= layer_idx < end else None


def _layer(config, i, bank=None):
    pos = _core_pos(config, i, bank)
    if pos is None:
        return Lfm3DecoderLayer(config, i)
    D = config.hidden_size

    def attn(cfg, idx):                                             # 430-436
        b = 0 if pos == 0 else 6
        r = {"q": (0, config.rank_attn), "k": (1, config.rank_kv), "v": (2, config.rank_kv), "o": (3, config.rank_attn)}
        return Lfm3Attention(cfg, idx, make_linear=lambda i_, o_, w: Lfm3SharedLoRALinear(bank, b + r[w][0], i_, o_, r[w][1]))

    def conv(cfg, idx):                                             # 528-535
        b = 4 if pos == 1 else 10
        return Lfm3ShortConv(cfg, idx,
                             make_in=lambda: Lfm3SharedGroupedLoRALinear(bank, b, D, D, config.rank_conv, config.conv_lora_groups),
                             make_out=lambda: Lfm3SharedLoRALinear(bank, b + 1, D, D, config.rank_conv))
    return Lfm3DecoderLayer(config, i, attn_factory=attn, conv_factory=conv)


class Lfm3RDPhase(nn.Module):                                       # 713-723
    def __init__(self, config, phase_idx, bank):
        super().__init__()
        base = config.num_prelude_layers + phase_idx * config.motif_size
        self.blocks = nn.ModuleList(_layer(config, base + p, bank) for p in range(config.motif_size))


class Lfm3RDCore(nn.Module):                                        # 738-742
    def __init__(self, config):
        super().__init__()
        self.shared = Lfm3SharedCoreBank(config)
        self.phases = nn.ModuleList(Lfm3RDPhase(config, m, self.shared) for m in range(config.num_motifs))


class Lfm3RDController(nn.Module):                                  # 195-224: only what T == num_motifs reads
    def __init__(self, config):
        super().__init__()
        self.step_scales = nn.Parameter(torch.ones(32))


class Lfm3RDModel(nn.Module):                                       # 899-918, 962-1202
    def __init__(self, config):
        super().__init__()
        self.config = config
        coda_start = config.num_prelude_layers + config.num_motifs * config.motif_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.prelude = nn.ModuleList(_layer(config, i) for i in range(config.num_prelude_layers))
        self.core = Lfm3RDCore(config)
        self.coda = nn.ModuleList(_layer(config, i) for i in range(coda_start, config.num_hidden_layers))
        self.rd_controller = Lfm3RDController(config)
        self.rotary_emb = Lfm3RotaryEmbedding(config)
        self.embedding_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, input_ids):
        c = self.config
        T_loops = max(1, min(32, int(c.recurrent_loops)))
        assert c.reasoning_mode == "manual" and T_loops == c.num_motifs, "only the T == num_motifs manual path is transcribed"
        h = self.embed_tokens(input_ids)
        pos = torch.arange(h.shape[1])[None]
        pe = {lt: self.rotary_emb(h, pos, lt) for lt in self.rotary_emb.layer_types}
        for blk in self.prelude:                                    # 1017
            h = blk(h, pe)
        p = h
        for phase in self.core.phases:                              # 1113-1137: s < n_motifs, each phase once
            for blk in phase.blocks:
                h = blk(h, pe)
        scale = self.rd_controller.step_scales[T_loops - 1].to(h.dtype)
        h = p + scale * (h - p)                                     # 1179-1180
        for blk in self.coda:                                       # 1191
            h = blk(h, pe)
        return self.embedding_norm(h)


class Lfm3RDForCausalLM(nn.Module):                                 # 1205-1263
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Lfm3RDModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))
