"""fr-mixer rules (graph/model_graph.py; dist/fr_mixer.md; Lean FrMixer.lean): a pure Mamba-2 stack on a Bamba hybrid
(attention layers zeroed), the norm-free Falcon-H1 Mamba-2 on a gated-norm Falcon-H1 (residual-scaled eps lever), a
plain Mamba-1 on Falcon-Mamba / Jamba (the B / C / dt norms levered). Stock transformers classes, fp64, random weights;
the host builders are hf/mixer_rules_agreement.py's (the same ones measured on the real checkpoints)."""
from __future__ import annotations

import math

import torch

from verify.spot.catalog import entry
from verify.spot.catalog.ffn_moe import _cmp
from verify.spot.catalog.rec import _assert_edge, _stock

_M2 = dict(hidden_size=32, state_size=8, expand=2, num_heads=8, head_dim=8, n_groups=1, conv_kernel=4,
           num_hidden_layers=6, tie_word_embeddings=False, residual_in_fp32=False, chunk_size=16)


def _bamba(kv):
    from hf.mixer_rules_agreement import host_bamba
    t = _stock("mamba2", **_M2)
    h = host_bamba(t, period=3, heads=4, kv=kv).to(torch.float64)
    _assert_edge(t, h)
    return _cmp(t, h, note=f"6 Mamba-2 layers on 8 Bamba layers, 2 attention (kv {kv}) zeroed")


@entry("slot:mixer:ssm->hybrid", "slot:kv:none->GQA",
       instance="mamba2 (6 layers) -> bamba (6 mamba + 2 GQA attention layers, attention / FFN outputs zero)")
def frm_mamba2_on_bamba():
    return _bamba(kv=2)


@entry("slot:kv:none->MQA", instance="mamba2 (6 layers) -> bamba (6 mamba + 2 MQA attention layers, zeroed)")
def frm_mamba2_on_bamba_mqa():
    return _bamba(kv=1)


_FH = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
           head_dim=8, mamba_d_ssm=32, mamba_n_heads=4, mamba_d_head=8, mamba_d_state=8, mamba_n_groups=1,
           mamba_d_conv=4, mamba_chunk_size=16, mamba_rms_norm=False, tie_word_embeddings=True, use_mamba_kernels=False)


@entry("feature:rf_mamba2_nonorm:nest:rf_mamba2", standard="limit",
       instance="falcon_h1 (mamba_rms_norm false) -> falcon_h1 (gated norm on): residual x S = 2^24, eps e S^2, "
                "gain S sqrt(e)")
def frm_nonorm_lever():
    from hf.mixer_rules_agreement import host_nonorm
    t = _stock("falcon_h1", **_FH)
    h = host_nonorm(t, 2.0 ** 24).to(torch.float64)
    _assert_edge(t, h)
    return _cmp(t, h, standard="limit", tol=1e-6, note="the gated norm linear: rho ~ ms / (e S^2)")


_M1 = dict(hidden_size=32, state_size=8, expand=2, time_step_rank=4, conv_kernel=4, num_hidden_layers=3,
           tie_word_embeddings=True)


@entry("feature:rf_mamba1:nest:rf_mamba1_bcdt", standard="limit",
       instance="mamba (3 layers) -> falcon_mamba: x_proj rows x 2^-a (a = 20), dt_proj / D / out_proj fold the "
                "weight-free norms' constant")
def frm_mamba1_on_falcon_mamba():
    from hf.mixer_rules_agreement import host_falcon_mamba
    t = _stock("mamba", **_M1)
    h = host_falcon_mamba(t, 20).to(torch.float64)
    _assert_edge(t, h)                 # (an edge since round 3's mamba_ffn_none: both blocks have no FFN)
    return _cmp(t, h, standard="limit", tol=1e-6, note="the dt / B / C norms linear: rho ~ ms 2^-2a / e")


@entry("feature:rf_mamba1:nest:rf_mamba1_bcdtw", standard="limit",
       instance="mamba (3 layers) -> jamba (3 mamba + 1 zeroed attention layer): x_proj rows x 2^-a (a = 20), "
                "dt / B / C norm gains 2^a sqrt(e)")
def frm_mamba1_on_jamba():
    from hf.mixer_rules_agreement import host_jamba
    t = _stock("mamba", **_M1)
    h = host_jamba(t, 20, period=3, heads=4, kv=2).to(torch.float64)
    # (the nest is the lever's, checked here on its own; the pair also takes pure_on_hybrid's ssm <- hybrid edge)
    return _cmp(t, h, standard="limit", tol=1e-6, note="the gained dt / B / C norms linear")
