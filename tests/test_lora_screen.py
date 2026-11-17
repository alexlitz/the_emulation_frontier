"""The LoRA-exactness screen on synthetic local checkpoints: a merged low-rank adapter stored in
bf16 reads as its true rank, a dtype round-trip as identical, and any edit to a non-adaptable
tensor (norm, decay parameter, router) as not-a-LoRA.

Run:  python -m pytest tests/test_lora_screen.py -q
"""
import pytest
import torch
from safetensors.torch import save_file

import serving.lora_screen as ls
from serving.st_reader import Repo, delta_rank


def _base(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.02
    return {
        "model.embed_tokens.weight": r(5000, 64),
        "model.layers.0.self_attn.q_proj.weight": r(128, 64),
        "model.layers.0.mlp.gate.weight": r(8, 64),
        "model.layers.0.mlp.experts.gate_up_proj": r(8, 96, 64),
        "model.layers.0.input_layernorm.weight": 1 + r(64),
        "model.layers.0.linear_attn.A_log": r(4),
        "model.visual.blocks.0.attn.qkv.weight": r(16, 16),   # ignored: vision tower
    }


def _save(sd, path, dtype=torch.bfloat16):
    path.mkdir()
    save_file({k: v.to(dtype).contiguous() for k, v in sd.items()}, str(path / "model.safetensors"))
    return str(path)


def _lowrank(rows, cols, r, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, r, generator=g) @ torch.randn(r, cols, generator=g) * 2e-3


@pytest.fixture
def ckpts(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "CACHE", str(tmp_path / "cache"))
    base = {k: v.to(torch.bfloat16).float() for k, v in _base().items()}   # what's on disk
    out = {"base": _save(base, tmp_path / "base")}
    lora = dict(base)
    lora["model.layers.0.self_attn.q_proj.weight"] = base["model.layers.0.self_attn.q_proj.weight"] + _lowrank(128, 64, 3, 1)
    ex = base["model.layers.0.mlp.experts.gate_up_proj"].clone()
    ex[4] += _lowrank(96, 64, 2, 2)                    # the middle expert: sampled by the screen
    lora["model.layers.0.mlp.experts.gate_up_proj"] = ex
    lora["model.visual.blocks.0.attn.qkv.weight"] = torch.randn(16, 16)    # vision edits don't count
    out["lora"] = _save(lora, tmp_path / "lora")
    out["fp32"] = _save(base, tmp_path / "fp32", torch.float32)
    norm = dict(lora)
    norm["model.layers.0.input_layernorm.weight"] = base["model.layers.0.input_layernorm.weight"] * 1.1
    out["norm"] = _save(norm, tmp_path / "norm")
    alog = dict(base)
    alog["model.layers.0.linear_attn.A_log"] = base["model.layers.0.linear_attn.A_log"] + 0.5
    out["alog"] = _save(alog, tmp_path / "alog")
    full = dict(base)
    full["model.layers.0.self_attn.q_proj.weight"] = base["model.layers.0.self_attn.q_proj.weight"] + torch.randn(128, 64) * 2e-3
    out["full"] = _save(full, tmp_path / "full")
    return out


def test_merged_low_rank_adapter_reads_as_its_rank(ckpts):
    r = ls.screen_full(ckpts["base"], ckpts["lora"], rows=96, workers=2)
    assert r["verdict"] == "lora<=16" and r["max_linear_rank"] == 4    # rank 3 -> first listed cap 4
    assert r["per_module"]["layers.N.mlp.experts.gate_up_proj"] == 2


def test_dtype_round_trip_is_identical(ckpts):
    assert ls.screen_full(ckpts["base"], ckpts["fp32"], workers=2)["verdict"] == "identical"


@pytest.mark.parametrize("which", ["norm", "alog"])
def test_non_adaptable_edit_is_not_a_lora(ckpts, which):
    r = ls.screen_full(ckpts["base"], ckpts[which], workers=2)
    assert r["verdict"] == "not-lora" and r["n_changed_fixed"] == 1


def test_full_rank_delta_is_not_a_lora(ckpts):
    r = ls.screen_full(ckpts["base"], ckpts["full"], workers=2)
    assert r["verdict"] == "not-lora" and r["max_linear_rank"] is None


def test_row_slices_read_correctly(ckpts):
    R = Repo(ckpts["lora"])
    full = R.read("layers.0.mlp.experts.gate_up_proj")
    assert torch.equal(R.read("layers.0.mlp.experts.gate_up_proj", rows=10, lead=4), full[4, :10])
    assert torch.equal(R.read("embed_tokens.weight", rows=7), R.read("embed_tokens.weight")[:7])


def test_delta_rank_floor():
    w = torch.randn(96, 64) * 0.02
    stored = w.to(torch.bfloat16).double()
    assert delta_rank(stored, stored.float().to(torch.bfloat16).double())[0] == 0
    assert delta_rank(stored, (w + _lowrank(96, 64, 1, 3)).to(torch.bfloat16).double())[0] == 1


def test_sub_ulp_update_keeps_its_structured_rank():
    # an update smaller than one bf16 step per weight: storage flips some weights by 1 ulp, yet the
    # rank-2 structure stands out of the rounding noise, and the delta is flagged sub-ulp
    g = torch.Generator().manual_seed(4)
    w = torch.randn(512, 256, generator=g) * 0.02
    u = torch.randn(512, 2, generator=g) @ torch.randn(2, 256, generator=g)
    u *= 0.3 * 2.0 ** -7 * 0.02 / u.std()                 # ~0.3 ulp of a typical weight
    r, rel, sub = delta_rank(w.to(torch.bfloat16).double(), (w + u).to(torch.bfloat16).double())
    assert sub and r == 2, (r, rel, sub)
    noise = torch.randn(512, 256, generator=g) * 0.3 * 2.0 ** -7 * 0.02   # full-rank sub-ulp update
    r, _, sub = delta_rank(w.to(torch.bfloat16).double(), (w + noise).to(torch.bfloat16).double())
    assert sub and r == 0                                 # indistinguishable from rounding
