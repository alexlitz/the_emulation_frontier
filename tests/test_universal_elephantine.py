"""ONE elephantine architecture (`elephantine/universal.py`) carries every bucket of the zoo.

Each test builds a small REAL `transformers` target, loads its weights into the same
`UniversalElephantine` class, and compares outputs. Real modules round some internals to
fp32 (RMSNorm, rotary tables, the Mamba / gated-delta scans, router softmax), so whole-model
agreement is at that fp32 level; tests/test_mamba1_gated_delta.py shows the gap closes to
~1e-17 once those casts are mirrored.

Run:  pytest tests/test_universal_elephantine.py -q
"""
import pytest
import torch

from elephantine.universal import from_llama, from_mamba, from_qwen3next

F64 = torch.float64
TOL = 1e-6


def _ids(vocab, T=10, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (2, T))


def _rand_(model, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() >= 1:
                p.add_(torch.randn_like(p) * 0.05)
    return model


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3,
                      num_attention_heads=6, num_key_value_heads=2, rms_norm_eps=1e-5,
                      attn_implementation="eager")
    return _rand_(LlamaForCausalLM(cfg).to(F64).eval())


@torch.no_grad()
def test_one_host_carries_a_softmax_ar_model():
    t = _llama()
    ids = _ids(97)
    d = (t(ids).logits - from_llama(t)(ids, causal=True)).abs().max().item()
    assert d < TOL, d


@torch.no_grad()
def test_same_host_carries_a_diffusion_forward_with_the_full_mask():
    t = _llama()
    ids = _ids(97)
    T = ids.shape[1]
    full = torch.zeros(ids.shape[0], 1, T, T, dtype=F64)                 # bidirectional
    ref = t(ids, attention_mask=full).logits
    host = from_llama(t)
    assert (ref - host(ids, causal=False)).abs().max().item() < TOL
    assert (ref - host(ids, causal=True)).abs().max().item() > 1e-3      # the mask really matters


@torch.no_grad()
def test_same_host_carries_a_parallel_block_cohere():
    # the host's parallel-block branch (ULayer.par = 1: the FFN reads the layer input): a stock
    # CohereForCausalLM (x + attn(ln x) + mlp(ln x), one LayerNorm, interleaved RoPE, logit_scale)
    # runs in the host at its OWN width -- no 2x carry. This is why the elephantine join does not
    # charge parallel-block targets (Command-R+) the 2x width a sequential host pays.
    from transformers import CohereConfig, CohereForCausalLM
    from elephantine.universal import from_cohere
    cfg = CohereConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3,
                       num_attention_heads=6, num_key_value_heads=2, logit_scale=0.0625,
                       layer_norm_eps=1e-5, attn_implementation="eager")
    t = _rand_(CohereForCausalLM(cfg).to(F64).eval())
    ids = _ids(97)
    host = from_cohere(t)
    d = (t(ids).logits - host(ids, causal=True)).abs().max().item()
    assert d < TOL, d                                   # fp32 LayerNorm inside the real module
    for ul in host.layers:
        ul.par.fill_(0.0)                               # the sequential path is NOT the same model
    assert (t(ids).logits - host(ids, causal=True)).abs().max().item() > 1e-3


@torch.no_grad()
def test_same_host_carries_a_mamba1_model():
    from transformers import MambaConfig, MambaForCausalLM
    cfg = MambaConfig(vocab_size=97, hidden_size=32, state_size=8, num_hidden_layers=3, expand=2,
                      conv_kernel=4, time_step_rank=4, residual_in_fp32=False, use_mambapy=False)
    t = _rand_(MambaForCausalLM(cfg).to(F64).eval())
    ids = _ids(97)
    d = (t(ids).logits - from_mamba(t)(ids)).abs().max().item()
    assert d < TOL, d


def _qwen3next():
    q3 = pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")
    cfg = q3.Qwen3NextConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.25,
                             linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                             linear_value_head_dim=8, linear_conv_kernel_dim=4, num_experts=4,
                             num_experts_per_tok=2, moe_intermediate_size=24,
                             shared_expert_intermediate_size=32, intermediate_size=48,
                             full_attention_interval=2, attn_implementation="eager")
    cfg.dtype = F64
    t = q3.Qwen3NextForCausalLM(cfg).to(F64).eval()
    for layer in t.model.layers:                                         # CPU torch paths (fla is GPU-only)
        g = getattr(layer, "linear_attn", None)
        if g is not None:
            g.chunk_gated_delta_rule = q3.torch_chunk_gated_delta_rule
            old = g.norm
            norm = q3.Qwen3NextRMSNormGated(old.weight.shape[0], eps=float(getattr(old, "eps", 1e-6))).to(F64)
            norm.weight.data.copy_(old.weight.data)
            g.norm = norm
    return _rand_(t), cfg


@torch.no_grad()
def test_same_host_carries_a_qwen3next_hybrid():
    t, cfg = _qwen3next()
    assert {l.layer_type for l in t.model.layers} == {"linear_attention", "full_attention"}
    ids = _ids(97)
    d = (t(ids).logits - from_qwen3next(t)(ids)).abs().max().item()
    assert d < TOL, d


# --- the same SSM block carries every other recurrent family (block level) ------------------

def _init_(module, seed=0):
    """Standalone blocks skip transformers' _init_weights (params start as torch.empty)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return module


def _x(d, T=10, seed=3):
    torch.manual_seed(seed)
    return torch.randn(2, T, d, dtype=F64)


@torch.no_grad()
def test_ssm_block_carries_mamba2():
    from transformers import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer
    from elephantine.universal import mamba2_block
    cfg = Mamba2Config(hidden_size=32, num_heads=8, head_dim=8, expand=2, n_groups=2, state_size=8,
                       conv_kernel=4, use_conv_bias=True, use_bias=False, chunk_size=4)
    m = _init_(Mamba2Mixer(cfg, layer_idx=0).to(F64).eval())
    x = _x(32)
    assert (m.torch_forward(x) - mamba2_block(m)(x)).abs().max().item() < TOL


@torch.no_grad()
def test_ssm_block_carries_lfm2_gated_short_conv():
    from transformers import Lfm2Config
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv
    from elephantine.universal import lfm2_block
    m = _init_(Lfm2ShortConv(Lfm2Config(hidden_size=16, conv_L_cache=3, conv_bias=True), 0).to(F64).eval())
    x = _x(16)
    assert (m.slow_forward(x) - lfm2_block(m)(x)).abs().max().item() < TOL


@torch.no_grad()
def test_ssm_block_carries_minimax_lightning():
    from transformers import MiniMaxConfig
    from transformers.models.minimax.modeling_minimax import MiniMaxLightningAttention
    from elephantine.universal import minimax_block
    cfg = MiniMaxConfig(hidden_size=32, num_attention_heads=4, head_dim=8, num_hidden_layers=3, block_size=4)
    m = _init_(MiniMaxLightningAttention(cfg, layer_idx=1).to(F64).eval())
    x = _x(32)
    ref = m(x, position_embeddings=None, attention_mask=None)[0]
    assert (ref - minimax_block(m)(x)).abs().max().item() < TOL


@torch.no_grad()
def test_ssm_block_carries_rwkv4_ratio():
    from transformers import RwkvConfig
    from transformers.models.rwkv.modeling_rwkv import RwkvSelfAttention
    from elephantine.universal import rwkv4_block
    m = _init_(RwkvSelfAttention(RwkvConfig(hidden_size=16, attention_hidden_size=16, num_hidden_layers=1), 0).to(F64).eval())
    x = _x(16)
    assert (m(x)[0] - rwkv4_block(m)(x)).abs().max().item() < TOL


@torch.no_grad()
def test_ssm_block_carries_rg_lru():
    from transformers import RecurrentGemmaConfig
    from transformers.models.recurrent_gemma.modeling_recurrent_gemma import RecurrentGemmaRecurrentBlock
    from elephantine.universal import rglru_block
    cfg = RecurrentGemmaConfig(hidden_size=16, lru_width=16, num_attention_heads=2, conv1d_width=4,
                               hidden_activation="gelu_pytorch_tanh")
    m = _init_(RecurrentGemmaRecurrentBlock(cfg).to(F64).eval())
    x = _x(16)
    pos = torch.arange(x.shape[1])[None].expand(2, -1)
    ref = m(x, position_ids=pos, attention_mask=None, cache_position=pos[0], use_cache=False)
    assert (ref - rglru_block(m)(x)).abs().max().item() < TOL


def _remote(repo_glob, fname):
    import glob, importlib.util, sys
    hits = glob.glob(f"{__import__('os').path.expanduser('~')}/.cache/huggingface/hub/{repo_glob}/snapshots/*/{fname}")
    if not hits:
        pytest.skip(f"{fname} not cached")
    name = "remote_" + fname[:-3]
    spec = importlib.util.spec_from_file_location(name, hits[0])
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def test_ssm_block_carries_mamba3_mimo():
    from elephantine.universal import mamba3_block
    q3m = _remote("models--arianraje--qwen3-4b-mamba3-hybrid-stage2b-kd-bias0", "modeling_qwen3_mamba3.py")
    cfg = q3m.Qwen3Mamba3Config(hidden_size=32, mamba_n_heads=4, mamba_d_head=8, mamba_d_state=16,
                                mamba_n_groups=2, mamba_mimo_rank=2, mamba_expand=1, mamba_chunk_size=16,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                num_hidden_layers=2, layer_types=["linear_attention", "full_attention"])
    m = _init_(q3m.Qwen3Mamba3Mixer(cfg, 0).to(F64).eval())
    assert m.num_rope_angles > 0 and m.mimo_rank > 1
    x = _x(32)
    ref = m(x)
    assert (ref - mamba3_block(m)(x)).abs().max().item() < TOL


def _upstream_kernels():
    """Make the REAL upstream pure-torch reference functions importable on CPU: mamba_ssm's
    ssd_combined / selective_state_update `*_ref` and causal_conv1d's `causal_conv1d_ref`,
    loaded from their source tarballs (python -m elephantine.fetch_upstream_refs) without the
    packages' compiled CUDA extensions. Only the CUDA entry points are stubbed; never called."""
    import glob, importlib, os, sys, types
    root = os.path.expanduser("~/.cache/upstream_refs")
    ms, cc = glob.glob(f"{root}/mamba_ssm-*/mamba_ssm"), glob.glob(f"{root}/causal_conv1d-*/causal_conv1d")
    if not ms or not cc:
        pytest.skip("upstream mamba_ssm / causal_conv1d sources not fetched")
    import importlib.machinery as im

    # (each stub carries a ModuleSpec: a spec-less module in sys.modules makes importlib.util.find_spec -- which
    # transformers' is_*_available probes call -- raise ValueError in every later test of the same xdist worker)
    def pkg(name, path):
        mod = types.ModuleType(name); mod.__path__ = [path]
        mod.__spec__ = im.ModuleSpec(name, None, is_package=True); mod.__spec__.submodule_search_locations = [path]
        sys.modules.setdefault(name, mod)
    for n, p in (("mamba_ssm", ms[0]), ("mamba_ssm.ops", ms[0] + "/ops"), ("mamba_ssm.ops.triton", ms[0] + "/ops/triton"),
                 ("causal_conv1d", cc[0])):
        pkg(n, p)
    cpp = types.ModuleType("causal_conv1d.cpp_functions")
    cpp.__spec__ = im.ModuleSpec("causal_conv1d.cpp_functions", None)
    for f in ("causal_conv1d_fwd_function", "causal_conv1d_bwd_function", "causal_conv1d_update_function"):
        setattr(cpp, f, None)                                            # CUDA-only, unused on CPU
    sys.modules.setdefault("causal_conv1d.cpp_functions", cpp)
    import mamba_ssm.ops.triton.ssd_combined, mamba_ssm.ops.triton.selective_state_update  # noqa: F401
    sys.modules["mamba_ssm"].ops = sys.modules["mamba_ssm.ops"]
    sys.modules["mamba_ssm.ops"].triton = sys.modules["mamba_ssm.ops.triton"]
    importlib.import_module("causal_conv1d.causal_conv1d_interface")


@torch.no_grad()
def test_same_host_carries_plamo2():
    from elephantine.universal import from_plamo2
    _upstream_kernels()
    pm = _remote("models--pfnet--plamo-2-1b", "modeling_plamo.py")
    cfg = pm.Plamo2Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
                          hidden_size_per_head=16, attention_window_size=32, mamba_d_state=16, mamba_d_conv=4,
                          mamba_num_heads=4, mamba_step=2, mamba_chunk_size=8, intermediate_size=96,
                          vocab_size=97, tie_word_embeddings=False)
    t = _rand_(pm.Plamo2ForCausalLM(cfg).to(F64).eval())
    assert {l.is_mamba for l in t.model.layers.layers} == {True, False}
    ids = _ids(97)
    ref = t(ids).logits
    d = (ref - from_plamo2(t)(ids)[..., : ref.shape[-1]]).abs().max().item()   # embed is vocab-padded
    assert d < TOL, d


def _hymba_module():
    """Import Hymba's REAL modeling code on CPU. Its fast path calls mamba_ssm/causal_conv1d CUDA
    wrappers; route those to the upstream pure-torch references with the same semantics
    (selective_scan_ref, causal_conv1d_ref), stub flash_attn (unused on the SDPA path), and lift
    its two CUDA-only guards (the rotary built on device='cuda', mixer_forward's device check).
    Everything else -- the block, norms, KV sharing, meta tokens, masks -- is Hymba's own code."""
    import glob, importlib.util, os, sys, types
    _upstream_kernels()
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Hymba-1.5B-Instruct/snapshots/*/"))
    if not snap or not os.path.exists(snap[0] + "configuration_hymba.py"):
        pytest.skip("Hymba remote code not cached")
    import causal_conv1d.causal_conv1d_interface as ci
    sys.modules["causal_conv1d"].causal_conv1d_fn = ci.causal_conv1d_ref
    sys.modules["causal_conv1d"].causal_conv1d_update = ci.causal_conv1d_update_ref
    import importlib.machinery as im
    scc = types.ModuleType("selective_scan_cuda"); scc.__spec__ = im.ModuleSpec("selective_scan_cuda", None)
    sys.modules.setdefault("selective_scan_cuda", scc)
    import mamba_ssm.ops.selective_scan_interface as ssi
    fa = types.ModuleType("flash_attn"); fa.__path__ = []
    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1)):
        raise RuntimeError("flash_attn is GPU-only")
    fa.flash_attn_func = fa.flash_attn_varlen_func = flash_attn_func
    bp = types.ModuleType("flash_attn.bert_padding")
    bp.index_first_axis = bp.pad_input = bp.unpad_input = None
    import importlib.machinery as im
    for stub in (fa, bp):
        stub.__spec__ = im.ModuleSpec(stub.__name__, None)
    sys.modules.setdefault("flash_attn", fa); sys.modules.setdefault("flash_attn.bert_padding", bp)
    import torch.nn.attention.flex_attention as fx                         # Hymba builds its block mask
    if not getattr(fx.create_block_mask, "_cpu", False):                     # without a device (-> cuda)
        orig = fx.create_block_mask
        fx.create_block_mask = lambda *a, **k: orig(*a, **{**k, "device": "cpu"})
        fx.create_block_mask._cpu = True
    pkg = types.ModuleType("hymba_remote"); pkg.__path__ = [snap[0]]; sys.modules["hymba_remote"] = pkg
    mod = sys.modules.get("hymba_remote.modeling_hymba")
    if mod is None:
        spec = importlib.util.spec_from_file_location("hymba_remote.modeling_hymba", snap[0] + "modeling_hymba.py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[spec.name]
            raise
        mod.selective_scan_fn = ssi.selective_scan_ref
        mod.causal_conv1d_fn = ci.causal_conv1d_ref
        init = mod.LlamaRotaryEmbedding.__init__
        mod.LlamaRotaryEmbedding.__init__ = lambda self, *a, device=None, **k: init(self, *a, device=None, **k)
        mod.HymbaBlock.mixer_forward = lambda self, h, cache_params=None, **k: self.cuda_kernels_forward(h, cache_params, **k)
    return mod


def _hymba(**over):
    hm = _hymba_module()
    from hymba_remote.configuration_hymba import HymbaConfig
    kw = dict(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=5, num_attention_heads=4,
              num_key_value_heads=2, num_experts=1, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2,
              mamba_dt_rank=8, mamba_inner_layernorms=True, kv_reuse_group=[[1, 2], [3, 4]],
              global_attn_idx=[0], sliding_window=64, num_memory_tokens=6, memory_tokens_interspersed_every=0,
              attn_implementation_new="sdpa", rope=True, rope_type="default", kq_norm="none",
              attn_hidden_size=-1, kq_head_dim=-1, v_head_dim=32, mlp_hidden_act="silu", seq_length=64,
              tie_word_embeddings=False)
    kw.update(over)
    dtype = kw.pop("dtype", F64)
    cfg = HymbaConfig(**kw)
    cfg._pre_quantization_dtype = dtype          # its fp32 branch otherwise reads a q_proj it doesn't have
    t = _rand_(hm.HymbaForCausalLM(cfg).to(dtype).eval())
    from torch.nn.attention.flex_attention import flex_attention
    for layer in t.model.layers:                 # compiled CPU flex needs AVX-512; eager is the same op
        if hasattr(layer.mamba.self_attn, "flex_attention"):
            layer.mamba.self_attn.flex_attention = flex_attention
    return t


# Hymba's own RMSNorms and the upstream selective_scan_ref run in fp32; layer by layer the gap
# stays at ~2e-7 relative with no jump at the KV-sharing layers, accumulating to ~4e-6 at the logits.
HYMBA_TOL = 1e-5


@torch.no_grad()
def test_same_host_carries_hymba_parallel_kv_sharing_meta_tokens():
    from elephantine.universal import from_hymba
    t = _hymba()
    assert any(l.mamba.reuse_kv for l in t.model.layers)
    ids = _ids(97)
    ref = t(ids, use_cache=False).logits                                   # Hymba casts logits to fp32
    d = (ref - from_hymba(t)(ids)).abs().max().item()
    assert d < HYMBA_TOL * ref.abs().max().item(), d


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")   # eager on purpose
@torch.no_grad()
def test_same_host_carries_hymba_window_with_meta_token_sinks():
    """Hymba's authoritative mask is its flex-attention block mask: causal AND (t - s <= window OR
    s is a meta token) on non-global layers. (Its SDPA path can't run a window: the block treats
    the 4-D mask as a padding mask.) Short window, so both the window and the sinks bind."""
    from elephantine.universal import from_hymba
    t = _hymba(attn_implementation_new="flex", sliding_window=3, global_attn_idx=[0], seq_length=12)
    ids = _ids(97, T=12)
    ref = t(ids, use_cache=False).logits.double()
    host = from_hymba(t)
    assert any(l.attn.window < 12 for l in host.layers) and host.layers[1].attn.n_sink == 6
    d = (ref - host(ids)).abs().max().item()
    assert d < HYMBA_TOL * ref.abs().max().item(), d


@torch.no_grad()
def test_ssm_block_carries_rwkv5():
    from elephantine.universal import rwkv5_block
    import glob, importlib.util, os, sys, types
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--RWKV--v5-Eagle-7B-HF/snapshots/*/"))
    if not snap:
        pytest.skip("RWKV-5 remote code not cached")
    pkg = types.ModuleType("rwkv5_remote"); pkg.__path__ = [snap[0]]; sys.modules["rwkv5_remote"] = pkg
    spec = importlib.util.spec_from_file_location("rwkv5_remote.modeling_rwkv5", snap[0] + "modeling_rwkv5.py")
    rm = importlib.util.module_from_spec(spec); sys.modules[spec.name] = rm; spec.loader.exec_module(rm)
    from rwkv5_remote.configuration_rwkv5 import Rwkv5Config
    rm.rwkv5_cuda_kernel = None
    rm.is_ninja_available = lambda: False                            # CPU reference path only
    cfg = Rwkv5Config(hidden_size=32, head_size=8, num_hidden_layers=1, attention_hidden_size=32)
    m = _init_(rm.RwkvSelfAttention(cfg, 0).to(F64).eval())
    x = _x(32)
    state = [torch.zeros(2, 32, 1, dtype=F64), torch.zeros(2, 4, 8, 8, 1, dtype=torch.float32)]   # as Rwkv5Model does (fp32 wkv state)
    ref = m(x, state=state)[0]
    assert (ref - rwkv5_block(m)(x)).abs().max().item() < TOL * ref.abs().max().item()


def _fla_layers():
    """fla's real layers: the installed fla core provides the Triton kernels (fla.ops,
    fla.modules); the layer definitions come from the matching flash-linear-attention 0.4.0
    source tarball in ~/.cache/upstream_refs, added to the fla package path (no install)."""
    import glob, os
    fla = pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("fla kernels are Triton/GPU")
    src = glob.glob(os.path.expanduser("~/.cache/upstream_refs/flash_linear_attention-*/fla"))
    if not src:
        pytest.skip("flash-linear-attention source not fetched")
    if src[0] not in fla.__path__:
        fla.__path__.append(src[0])
    import fla.layers
    return fla.layers


def _fla_check(layer, block_fn, d):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    x = _x(d).float().cuda()
    with torch.no_grad():
        ref = layer(x)[0].double().cpu()                               # real Triton kernels, fp32
        ours = block_fn(layer)(x.double().cpu())
    d = (ref - ours).abs().max().item()
    assert d < 1e-5 * ref.abs().max().item(), d                         # fp32 kernel accumulation


def test_ssm_block_carries_gla():
    from elephantine.universal import gla_block
    L = _fla_layers()
    _fla_check(L.GatedLinearAttention(hidden_size=32, num_heads=2, expand_k=0.5, expand_v=1.0,
                                      use_short_conv=True, layer_idx=0).cuda().eval(), gla_block, 32)


def test_ssm_block_carries_retnet():
    from elephantine.universal import retnet_block
    L = _fla_layers()
    _fla_check(L.MultiScaleRetention(hidden_size=32, num_heads=2, expand_k=1.0, expand_v=1.0,
                                     layer_idx=0).cuda().eval(), retnet_block, 32)


def test_ssm_block_carries_kimi_kda():
    from elephantine.universal import kda_block
    L = _fla_layers()
    _fla_check(L.KimiDeltaAttention(hidden_size=32, num_heads=2, head_dim=16, use_short_conv=True,
                                    layer_idx=0).cuda().eval(), kda_block, 32)


def _rwkv7_module():
    """The real RWKV-7 remote code (written for transformers v5) on CPU in fp64: stub the v5-only
    imports it never uses in a forward (init helpers, a cache base class, the strict config class),
    use its 'eager' recurrence, and run that recurrence in fp64 (it hard-codes an fp32 compute dtype)."""
    import glob, importlib.util, os, sys, types
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--RWKV--RWKV7-G1j-13.3B-20260831/snapshots/*/"))
    if not snap:
        pytest.skip("RWKV-7 remote code not cached")
    if "rwkv7_remote.modeling_rwkv7" in sys.modules:
        return sys.modules["rwkv7_remote.modeling_rwkv7"]
    init = types.ModuleType("transformers.initialization")
    def _noop(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None
    init.__getattr__ = _noop
    sys.modules.setdefault("transformers.initialization", init)
    import transformers, transformers.cache_utils as cu, transformers.utils as tu
    transformers.initialization = sys.modules["transformers.initialization"]
    if not hasattr(cu, "LinearAttentionLayer"):
        cu.LinearAttentionLayer = type("LinearAttentionLayer", (), {})
    ident = lambda f=None, **k: f if f is not None else (lambda g: g)
    saved = {n: getattr(tu, n, None) for n in ("auto_docstring", "can_return_tuple")}
    for name in saved:                     # 4.57's docstring builder chokes on v5 `X | None` hints
        setattr(tu, name, ident)
    pkg = types.ModuleType("rwkv7_remote"); pkg.__path__ = [snap[0]]; sys.modules["rwkv7_remote"] = pkg
    cfgmod = types.ModuleType("rwkv7_remote.configuration_rwkv7")
    cfgmod.Rwkv7Config = type("Rwkv7Config", (), {})
    sys.modules["rwkv7_remote.configuration_rwkv7"] = cfgmod
    spec = importlib.util.spec_from_file_location("rwkv7_remote.modeling_rwkv7", snap[0] + "modeling_rwkv7.py")
    rm = importlib.util.module_from_spec(spec); sys.modules[spec.name] = rm
    try:
        spec.loader.exec_module(rm)
    except Exception:
        del sys.modules[spec.name]
        raise
    finally:
        for name, f in saved.items():
            if f is not None:
                setattr(tu, name, f)
    rm.rwkv7_recurrent.__defaults__ = (F64,)
    return rm


@torch.no_grad()
def test_ssm_block_carries_rwkv7_with_v_first():
    from types import SimpleNamespace
    from elephantine.universal import rwkv7_block
    rm = _rwkv7_module()
    cfg = SimpleNamespace(hidden_size=32, head_dim=8, num_heads=4, decay_low_rank_dim=8, a_low_rank_dim=8,
                          gate_low_rank_dim=8, v_low_rank_dim=8, intermediate_size=64, norm_eps=1e-5,
                          norm_bias=True, wkv_implementation="eager", wkv_state_dtype="float64")
    a0, a1 = (_init_(rm.Rwkv7Attention(cfg, i).to(F64).eval(), seed=i) for i in (0, 1))
    x0, x1 = _x(32, seed=3), _x(32, seed=4)
    y0, v_first, _, _ = a0(x0, None, None, None)
    y1, _, _, _ = a1(x1, v_first, None, None)
    reg = {}
    h0, h1 = rwkv7_block(a0)(x0, reg), rwkv7_block(a1)(x1, reg)
    assert (y0 - h0).abs().max().item() < TOL * y0.abs().max().item()
    assert (y1 - h1).abs().max().item() < TOL * y1.abs().max().item()       # v_first via the register


TF5 = __import__("os").path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")
TF5_PRELUDE = '''
import sys, types, torch
m = types.ModuleType("transformers.dependency_versions_check"); m.dep_version_check = lambda *a, **k: None
sys.modules["transformers.dependency_versions_check"] = m
sys.modules["fla"] = None              # hide fla: tf5 would route its torch fallbacks to Triton
F64 = torch.float64
def rand_(model, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model
def ids(vocab, T=10, seed=0):
    torch.manual_seed(seed); return torch.randint(0, vocab, (2, T))
def report(ref, ours):
    print("GAP", (ref - ours).abs().max().item(), "SCALE", ref.abs().max().item())
'''


def _tf5_run(body):
    """Run a reference that needs transformers 5.x in a subprocess (4.57 is installed here; the
    two can't share a process): transformers 5.18 + huggingface_hub 1.33 unpacked into
    ~/.cache/upstream_refs, nothing installed. Returns (gap, scale) printed by `report`."""
    import os, re, subprocess, sys
    if not os.path.isdir(TF5):
        pytest.skip("transformers 5.18 not unpacked (python -m elephantine.fetch_upstream_refs)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, root]))
    r = subprocess.run([sys.executable, "-c", TF5_PRELUDE + body], env=env, capture_output=True, text=True, timeout=900)
    m = re.search(r"GAP (\S+) SCALE (\S+)", r.stdout)
    assert m, r.stdout[-2000:] + r.stderr[-4000:]
    return float(m.group(1)), float(m.group(2))


def test_same_host_carries_qwen35_and_diba():
    gap, scale = _tf5_run('''
from transformers.models.qwen3_5 import modeling_qwen3_5 as q, configuration_qwen3_5 as qc
from elephantine.universal import from_qwen35
cfg = qc.Qwen3_5TextConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48)
t = q.Qwen3_5ForCausalLM(cfg).to(F64).eval()
for l in t.model.layers:                          # CPU torch paths (fla kernels are GPU Triton)
    g = getattr(l, "linear_attn", None)
    if g is not None:
        g.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
        g.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
        g.causal_conv1d_fn = None
rand_(t)
x = ids(97)
with torch.no_grad():
    report(t(x).logits, from_qwen35(t)(x))
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


TF5_REMOTE = '''
import glob, importlib.util, os
def remote(repo_dir, pkg, files):
    hit = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/" + repo_dir + "/snapshots/*/" + files[0] + ".py"))
    snap = os.path.dirname(hit[0]) + "/"           # the snapshot that has the code (not a config-only one)
    p = types.ModuleType(pkg); p.__path__ = [snap]; sys.modules[pkg] = p
    mods = []
    for f in files:
        spec = importlib.util.spec_from_file_location(pkg + "." + f, snap + f + ".py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        mods.append(mod)
    return mods
'''


def test_same_host_carries_agnes_parallel_ffn():
    gap, scale = _tf5_run(TF5_REMOTE + '''
import transformers.cache_utils as cu
if not hasattr(cu, "LAYER_TYPE_CACHE_MAPPING"):
    cu.LAYER_TYPE_CACHE_MAPPING = {}
cfgm, am = remote("models--Agnes-AI--Agnes-3.0-Flash", "agnes_remote", ["configuration_agnes", "modeling_agnes"])
from elephantine.universal import from_qwen35
cfg = cfgm.AgnesTextConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=6,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48,
    parallel_ffn_intermediate_size=16,
    layer_types=[am.LAYER_DELTA, am.LAYER_DELTA, am.LAYER_DELTA, am.LAYER_GLOBAL])
t = rand_(am.AgnesForCausalLM(cfg).to(F64).eval())
assert t.model.layers[0].mlp.parallel_ffn is not None
x = ids(97)
with torch.no_grad():
    report(t(x, use_cache=False).logits, from_qwen35(t)(x))
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


def test_same_host_carries_dqwen35_diffusion():
    """dQwen3.5: Qwen3.5 with its attention layers made bidirectional (the gated-delta layers stay
    causal) and a BOS/shift. On the host: the same weights run with causal=False, BOS prepended,
    last position dropped."""
    gap, scale = _tf5_run(TF5_REMOTE + '''
(dq,) = remote("models--UT-IFML--dQwen3.5-0.8B-Base", "dqwen_remote", ["modeling_dqwen3_5"])
dq.BOS_TOKEN_ID = 5                                   # fit the tiny test vocab
from elephantine.universal import from_qwen35
cfg = dq.DiffuQwen35Config(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48)
t = rand_(dq.DiffuQwen35(cfg).to(F64).eval())
x = ids(97)
with torch.no_grad():
    ref = t(x).logits
    u = from_qwen35(t)
    bos = torch.full((x.shape[0], 1), 5)
    ours = u(torch.cat([bos, x], 1), causal=False)[:, :-1]
    causal = u(torch.cat([bos, x], 1), causal=True)[:, :-1]
    assert (ref - causal).abs().max().item() > 1e-3            # bidirectional attention matters
    report(ref, ours)
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


def test_same_host_carries_aliceai_attnres_sigmoid_router():
    """AliceAI: KDA + gated attention, sigmoid top-k router with bias-corrected selection, and
    Block AttnRes (no running residual: depth-softmax mixes of block sums feed every sublayer)."""
    gap, scale = _tf5_run(TF5_REMOTE + '''
cm, am = remote("models--AMAImedia--AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF", "alice_remote",
                ["configuration_alice_ai", "modeling_alice_ai"])
from elephantine.universal import from_alice
cfg = cm.AliceAIConfig(vocab_size=97, hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=16, num_experts=6, num_experts_per_tok=2,
    moe_intermediate_size=24, shared_expert_intermediate_size=32, block_attn_res_block_size=2,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention",
                 "linear_attention", "full_attention"])
t = rand_(am.AliceAIForCausalLM(cfg).to(F64).eval())
with torch.no_grad():                                   # make the selection bias matter
    for l in t.model.layers:
        l.mlp.gate.e_score_correction_bias.copy_(torch.randn(6, dtype=F64) * 0.3)
x = ids(97)
with torch.no_grad():
    report(t(x, use_cache=False).logits, from_alice(t)(x))
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


@torch.no_grad()
def test_same_host_carries_mini_k3():
    """Mini-K3 (Mamba-2 variants): Mamba-2 with grouped gated norm, NoPE MLA with latent RMSNorms,
    SiTU-GLU dense + latent MoE (sigmoid router, post-routing norm), Block AttnRes. Pure-torch
    reference; it computes in fp32 internally, so compare at fp32 level."""
    import glob, importlib.util, os, sys, types
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nkkbr--Mini-K3-1H-mamba2-n128-g8-v1_B/snapshots/*/"))
    if not snap:
        pytest.skip("Mini-K3 remote code not cached")
    pkg = types.ModuleType("mk3_remote"); pkg.__path__ = [snap[0]]; sys.modules["mk3_remote"] = pkg
    mods = {}
    for f in ("configuration_mini_k3", "mamba2_mixer", "modeling_mini_k3"):
        spec = importlib.util.spec_from_file_location("mk3_remote." + f, snap[0] + f + ".py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        mods[f] = mod
    from elephantine.universal import from_mini_k3
    cfg = mods["configuration_mini_k3"].MiniK3Config.tiny_for_test()
    torch.manual_seed(0)
    t = mods["modeling_mini_k3"].MiniK3ForCausalLM(cfg).eval()
    _rand_(t)
    with torch.no_grad():
        for l in t.model.layers:
            if l.is_moe:
                l.feed_forward.router.e_score_correction_bias.copy_(torch.randn_like(l.feed_forward.router.e_score_correction_bias) * 0.3)
    assert any(l.is_mamba for l in t.model.layers) and any(l.is_moe for l in t.model.layers)
    x = _ids(128, T=10)
    out = t(x)
    ref = (out.logits if hasattr(out, "logits") else out[0] if isinstance(out, tuple) else out).double()
    d = (ref - from_mini_k3(t)(x)).abs().max().item()
    assert d < 1e-5 * max(ref.abs().max().item(), 1), d


def _nemotron_flash_file(name):
    import glob, importlib.util, os, sys, types
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Nemotron-Flash-1B/snapshots/*/"))
    if not snap or not os.path.exists(snap[0] + name + ".py"):
        pytest.skip("Nemotron-Flash remote code not cached")
    pkg = types.ModuleType("nflash_remote"); pkg.__path__ = [snap[0]]; sys.modules.setdefault("nflash_remote", pkg)
    full = "nflash_remote." + name
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, snap[0] + name + ".py")
        mod = importlib.util.module_from_spec(spec); sys.modules[full] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[full]
            raise
    return sys.modules[full]


def test_ssm_block_carries_nemotron_flash_deltanet():
    from elephantine.universal import deltanet_block
    _fla_layers()                                                   # skips without CUDA / fla
    dn = _nemotron_flash_file("delta_net")
    _fla_check(dn.DeltaNet(hidden_size=32, num_heads=2, use_short_conv=True, conv_size=4, layer_idx=0).cuda().eval(),
               deltanet_block, 32)


def test_ssm_block_carries_nemotron_flash_mamba2():
    """Nemotron-Flash's Mamba-2 is upstream mamba_ssm's module: real Triton SSD scan and gated
    RMSNorm on GPU (the conv's CUDA extension routed to the upstream causal_conv1d_ref)."""
    from types import SimpleNamespace
    from elephantine.universal import mamba2_block, mamba_ssm_mamba2_shim
    if not torch.cuda.is_available():
        pytest.skip("Triton kernels need a GPU")
    import os, sys
    os.environ["TRITON_F32_DEFAULT"] = "ieee"   # mamba_ssm's tl.dot defaults to TF32 (~1e-3) on Ampere
    _upstream_kernels()
    import causal_conv1d.causal_conv1d_interface as ci
    sys.modules["causal_conv1d"].causal_conv1d_fn = ci.causal_conv1d_ref
    sys.modules["causal_conv1d"].causal_conv1d_update = ci.causal_conv1d_update_ref
    m2 = _nemotron_flash_file("mamba2")
    m2.causal_conv1d_fn = ci.causal_conv1d_ref
    cfg = SimpleNamespace(hidden_size=32, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba2_headdim=8)
    mx = m2.Mamba2(cfg, ngroups=1, rmsnorm=True, norm_before_gate=False, use_mem_eff_path=False,
                   chunk_size=16, layer_idx=0).cuda().eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in mx.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    x = _x(32).float().cuda()
    with torch.no_grad():
        ref = mx(x)
        ref = (ref[0] if isinstance(ref, tuple) else ref).double().cpu()
        ours = mamba2_block(mamba_ssm_mamba2_shim(mx))(x.double().cpu())
    d = (ref - ours).abs().max().item()
    assert d < 1e-5 * ref.abs().max().item(), d


def _phi4flash_module():
    """Phi-4-mini-flash's REAL modeling code on CPU. Stubs: flash_attn (a reference
    flash_attn_func: causal + left window + GQA), causal_conv1d / *_cuda (unused on the slow path),
    selective_scan_fn -> upstream selective_scan_ref, swiglu (a CUDA jiterator) -> silu(a) * b.
    Its differential-attention forward runs as written except the bf16 cast and is_cuda asserts."""
    import glob, importlib.util, importlib.machinery as im, inspect, os, sys, types
    import torch.nn.functional as F
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--Phi-4-mini-flash-reasoning/snapshots/*/"))
    if not snap:
        pytest.skip("Phi-4-flash remote code not cached")
    if "phi4f_remote.modeling_phi4flash" in sys.modules:
        return sys.modules["phi4f_remote.modeling_phi4flash"]
    _upstream_kernels()
    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kw):
        B, S, H, D = q.shape
        rep = H // k.shape[2]
        k, v = k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)
        sc = torch.einsum("bihd,bjhd->bhij", q, k) * (softmax_scale or D ** -0.5)
        i, j = torch.arange(S)[:, None], torch.arange(S)[None]
        ok = torch.ones(S, S, dtype=torch.bool)
        if causal:
            ok &= j <= i
        if window_size[0] >= 0:
            ok &= (i - j) <= window_size[0]
        sc = sc.masked_fill(~ok, float("-inf"))
        return torch.einsum("bhij,bjhd->bihd", sc.softmax(-1), v)
    fa = types.ModuleType("flash_attn"); fa.__path__ = []
    fa.flash_attn_func, fa.flash_attn_varlen_func = flash_attn_func, flash_attn_func
    bp = types.ModuleType("flash_attn.bert_padding"); bp.index_first_axis = bp.pad_input = bp.unpad_input = None
    for stub in (fa, bp):
        stub.__spec__ = im.ModuleSpec(stub.__name__, None)
    saved = {n: sys.modules.get(n) for n in ("flash_attn", "flash_attn.bert_padding")}
    sys.modules["flash_attn"], sys.modules["flash_attn.bert_padding"] = fa, bp
    for n in ("causal_conv1d_cuda", "selective_scan_cuda"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules["causal_conv1d"].causal_conv1d_fn = None
    sys.modules["causal_conv1d"].causal_conv1d_update = None
    import mamba_ssm.ops.selective_scan_interface as ssi
    pkg = types.ModuleType("phi4f_remote"); pkg.__path__ = [snap[0]]; sys.modules["phi4f_remote"] = pkg
    spec = importlib.util.spec_from_file_location("phi4f_remote.modeling_phi4flash", snap[0] + "modeling_phi4flash.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[spec.name]
        raise
    finally:
        for n, v in saved.items():                      # don't leak our flash_attn stub to other tests
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v
    mod.selective_scan_fn = ssi.selective_scan_ref
    mod.swiglu = lambda a, b: F.silu(a) * b
    mod.causal_conv1d_fn = mod.causal_conv1d_update = None
    src = inspect.getsource(mod.FlashDiffCustomAttention.forward)
    src = "\n".join(l[4:] for l in src.splitlines())   # dedent the method
    for line in ("q = q.to(torch.bfloat16)", "k = k.to(torch.bfloat16)", "v = v.to(torch.bfloat16)",
                 "assert q.dtype in [torch.float16, torch.bfloat16]", "assert q.is_cuda and k.is_cuda and v.is_cuda"):
        assert line in src, line
        src = src.replace(line, "pass")
    ns = {}
    exec(src, mod.__dict__, ns)
    mod.FlashDiffCustomAttention.forward = ns["forward"]
    return mod


@torch.no_grad()
def test_same_host_carries_phi4flash_sambay():
    from elephantine.universal import from_phi4flash
    mod = _phi4flash_module()
    import sys
    Phi4FlashConfig = sys.modules["phi4f_remote.configuration_phi4flash"].Phi4FlashConfig
    cfg = Phi4FlashConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, sliding_window=4, mb_per_layer=2,
                          mamba_d_state=8, mamba_d_conv=4, mamba_expand=2)
    t = mod.Phi4FlashForCausalLM(cfg).to(F64).eval()
    for l in t.model.layers:
        if l.use_mamba:
            l.attn.use_fast_path = False
    _rand_(t)
    kinds = [("gmu" if l.attn.yoco_cross else "mamba") if l.use_mamba else ("xattn" if l.attn.yoco_cross else "attn") for l in t.model.layers]
    assert {"gmu", "mamba", "xattn", "attn"} <= set(kinds), kinds
    x = _ids(97, T=10)
    ref = t(x, use_cache=False).logits.double()
    d = (ref - from_phi4flash(t)(x)).abs().max().item()
    assert d < 1e-5 * max(ref.abs().max().item(), 1), d


@torch.no_grad()
def test_ssm_block_carries_rwkv6_ddlerp():
    import glob, importlib.util, inspect, os, sys, types
    from elephantine.universal import rwkv6_block
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--RWKV--rwkv-6-world-3b/snapshots/*/"))
    if not snap:
        pytest.skip("RWKV-6 remote code not cached")
    pkg = types.ModuleType("rwkv6_remote"); pkg.__path__ = [snap[0]]; sys.modules["rwkv6_remote"] = pkg
    mods = {}
    for f in ("configuration_rwkv6", "modeling_rwkv6"):
        spec = importlib.util.spec_from_file_location("rwkv6_remote." + f, snap[0] + f + ".py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        mods[f] = mod
    rm = mods["modeling_rwkv6"]
    # its CPU recurrence hard-casts to fp32; run the same code in fp64
    exec(inspect.getsource(rm.rwkv6_linear_attention_cpu).replace(".float()", ".double()"), rm.__dict__)
    cfg = mods["configuration_rwkv6"].Rwkv6Config(hidden_size=32, attention_hidden_size=32, head_size=8,
                                                 num_hidden_layers=1)
    m = _init_(rm.Rwkv6SelfAttention(cfg, 0).to(F64).eval())
    x = _x(32)
    state = [torch.zeros(2, 32, 1, dtype=F64), torch.zeros(2, 4, 8, 8, 1, dtype=F64)]
    ref = m(x, state=state)[0]
    d = (ref - rwkv6_block(m)(x)).abs().max().item()
    assert d < TOL * max(ref.abs().max().item(), 1), d


@torch.no_grad()
def test_same_host_carries_granite_switch_alora():
    """Granite-Switch (pinned to the single-switch commit 24aa420 of
    github.com/generative-computing/granite-switch, which this checkpoint was composed with):
    LoRA adapters chosen per token by a switch head, so an aLoRA adapter turns on mid-sequence."""
    import os, sys
    src = os.path.expanduser("~/.cache/upstream_refs/granite_switch-24aa420")
    if not os.path.isdir(src):
        pytest.skip("granite-switch source not fetched")
    # earlier tests may have registered spec-less mamba_ssm / causal_conv1d stubs, which transformers'
    # granitemoehybrid import probes with find_spec; set them aside while importing
    stubs = {n: sys.modules.pop(n) for n in list(sys.modules)
             if n.split(".")[0] in ("mamba_ssm", "causal_conv1d") and getattr(sys.modules[n], "__spec__", None) is None}
    sys.path.insert(0, src)
    try:
        from granite_switch.config import GraniteSwitchConfig
        from granite_switch.hf import modeling_granite_switch as gm
    finally:
        sys.path.remove(src)
        sys.modules.update(stubs)
    from elephantine.universal import from_granite_switch
    cfg = GraniteSwitchConfig(vocab_size=97, hidden_size=64, intermediate_size=96, shared_intermediate_size=96,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, num_adapters=2,
        adapter_token_ids=[90, 91], adapter_substitute_token_ids=[5, 6], max_lora_rank=4, adapter_ranks=[4, 4],
        lora_target_modules=["qkv_proj", "o_proj", "shared_input_linear", "shared_output_linear"],
        num_local_experts=0, embedding_multiplier=12.0, residual_multiplier=0.22, logits_scaling=10.0,
        attention_multiplier=1 / 16, rope_theta=1e7)
    t = _rand_(gm.GraniteSwitchForCausalLM(cfg).to(F64).eval())
    with torch.no_grad():                                           # non-zero adapters (B inits to 0)
        for n, p in t.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.2)
    x = _ids(88, T=10)
    x[0, 4], x[1, 6] = 90, 91                                       # adapter 1 from t=4, adapter 2 from t=6
    ref = t(x, use_cache=False).logits
    host = from_granite_switch(t)
    d = (ref - host(x)).abs().max().item()
    assert d < TOL * max(ref.abs().max().item(), 1), d
    with torch.no_grad():                                           # the switch really changes the output
        for l in host.layers:
            for P in list(l.attn.lb.values()) + list(l.ffn.lb.values()):
                P.zero_()
    assert (ref - host(x)).abs().max().item() > 1e-3


def _gigachat35_module():
    """GigaChat-3.5 remote code (transformers 4.57), fla / causal_conv1d hidden so its pure-torch
    fallbacks (gated-delta rule, gated RMSNorm) run on CPU."""
    import glob, importlib.util, os, sys, types
    hit = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ai-sage--GigaChat3.5-432B-A28B-Reasoning/"
                                       "snapshots/*/modeling_gigachat3_5.py"))
    if not hit:
        pytest.skip("GigaChat-3.5 remote code not cached")
    if "gc35_remote.modeling_gigachat3_5" in sys.modules:
        return sys.modules["gc35_remote.configuration_gigachat3_5"], sys.modules["gc35_remote.modeling_gigachat3_5"]
    snap = os.path.dirname(hit[0]) + "/"
    pkg = types.ModuleType("gc35_remote"); pkg.__path__ = [snap]; sys.modules["gc35_remote"] = pkg
    mods = []
    # earlier tests may have registered spec-less mamba_ssm / causal_conv1d stubs, which the import's
    # availability probes (find_spec) reject; set them aside while importing
    stubs = {n: sys.modules.pop(n) for n in list(sys.modules)
             if n.split(".")[0] in ("mamba_ssm", "causal_conv1d") and getattr(sys.modules[n], "__spec__", None) is None}
    try:
        for f in ("configuration_gigachat3_5", "modeling_gigachat3_5"):
            spec = importlib.util.spec_from_file_location("gc35_remote." + f, snap + f + ".py")
            mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
            mods.append(mod)
    finally:
        sys.modules.update(stubs)
    for n in ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
              "causal_conv1d_fn", "causal_conv1d_update"):
        setattr(mods[1], n, None)
    return mods


def _gigachat35(cm, gm):
    cfg = cm.GigaChat35Config(vocab_size=97, hidden_size=64, intermediate_size=48, moe_intermediate_size=16,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=6,
        routed_scaling_factor=2.5, kv_lora_rank=24, q_lora_rank=32, qk_rope_head_dim=8, v_head_dim=12,
        qk_nope_head_dim=16, n_group=1, topk_group=1, num_experts_per_tok=2, first_k_dense_replace=1,
        norm_topk_prob=True, rms_norm_eps=1e-6, rope_theta=10000.0, rope_interleave=True, max_position_embeddings=512,
        rope_scaling={"beta_fast": 32, "beta_slow": 1, "factor": 8.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                      "original_max_position_embeddings": 64, "type": "yarn"},
        norm_type="ZeroCenteredGatedNorm", layernorm_type="pre_post", layernorm_gating_weight=2.0,
        gated_attention=True, use_mla_scaling_factor=True, linear_attention_type="GigaChat35GatedDeltaNet",
        full_attention_layers=[1, 3], linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_gating_type="gated_rmsnorm_sigmoid_zero_centered",
        linear_sigmoid_gate_scale=2.0, linear_attn_o_norm_eps=1e-6, swiglu_limit=0.08, attn_implementation="eager")
    torch.manual_seed(0)
    t = _rand_(gm.GigaChat35ForCausalLM(cfg).eval())            # fp32: its MoE accumulates in fp32
    with torch.no_grad():                                        # make the selection bias matter
        for l in t.model.layers:
            if hasattr(l.mlp, "gate"):
                l.mlp.gate.e_score_correction_bias.copy_(torch.randn(6) * 0.3)
    return t


@torch.no_grad()
def test_same_host_carries_gigachat35():
    """GigaChat-3.5: Qwen3-Next-layout gated-DeltaNet with a zero-centered sigmoid gated norm (x2),
    gated MLA (latent norms, interleaved RoPE + YaRN, alpha scaling), sandwich zero-centered GATED
    norms (rank-16 sigmoid self-gate), sigmoid router + selection bias, hard SwiGLU clipping. Against
    the fp32 reference at fp32 level; then the same reference code with its fp32 casts lifted to fp64
    agrees to ~1e-14."""
    import inspect
    from elephantine.universal import from_gigachat35
    cm, gm = _gigachat35_module()
    t = _gigachat35(cm, gm)
    x = _ids(97)
    ref = t(x, use_cache=False).logits.double()
    host = from_gigachat35(t)
    d = (ref - host(x)).abs().max().item()
    assert d < 1e-5 * max(ref.abs().max().item(), 1), d
    for l in host.layers:                                        # the SwiGLU clip really bites
        l.ffn.c_clip.zero_()
    assert (ref - host(x)).abs().max().item() > 1e-3
    # fp64 floor: rerun the reference with its internal fp32 casts made fp64 (restored afterwards)
    lift = lambda s: s.replace("torch.float32", "torch.float64").replace(".float()", ".double()")
    saved = {}

    def patch(owner, name, src, key):
        saved[(owner, name)] = getattr(owner, name)
        ns = dict(gm.__dict__); exec(src, ns); setattr(owner, name, ns[key])
    try:
        for fn in ("torch_chunk_gated_delta_rule", "eager_attention_forward"):
            patch(gm, fn, lift(inspect.getsource(getattr(gm, fn))), fn)
        saved[(gm, "_apply_sigmoid_gate_fp32")] = gm._apply_sigmoid_gate_fp32
        gm._apply_sigmoid_gate_fp32 = lambda h, g: h * torch.sigmoid(g)
        for cls in (gm.GigaChat35TopkRouter, gm.GigaChat35GatedDeltaRMSNorm, gm.GigaChat35RMSNorm,
                    gm.GigaChat35ZeroCenteredRMSNorm, gm.GigaChat35ZeroCenteredGatedNorm,
                    gm.GigaChat35RotaryEmbedding, gm.GigaChat35GatedDeltaNetWrapper):
            src = lift(inspect.getsource(cls.forward))
            src = "\n".join(l[4:] for l in src.splitlines() if not l.strip().startswith("@"))
            patch(cls, "forward", src.replace("super().forward(", "GigaChat35ZeroCenteredRMSNorm.forward(self, "),
                  "forward")
        t = t.to(F64)
        for l in t.model.layers:
            if l.layer_type == "linear_attention":
                l.self_attn.chunk_gated_delta_rule = gm.torch_chunk_gated_delta_rule
        ref64 = t(x, use_cache=False).logits
    finally:
        for (owner, name), v in saved.items():
            setattr(owner, name, v)
    d64 = (ref64 - from_gigachat35(t)(x)).abs().max().item()
    assert d64 < 1e-12 * max(ref64.abs().max().item(), 1), d64


def test_same_host_carries_allgdn_engram():
    """AllGDN-Engram: an all-gated-DeltaNet Qwen3.5 (no attention layers) with Engram hashed n-gram
    memories injected before chosen layers. The n-gram hash is a deterministic function of the input
    ids, so the host carries it as a hashed n-gram embedding branch (UEngram, the token embedding
    generalised to n-grams) feeding a normed sigmoid gate and a causal depthwise conv."""
    gap, scale = _tf5_run(TF5_REMOTE + '''
(ag,) = remote("models--summerMC--Cyber-9B-Qwen-AllGDN-Engram", "allgdn_remote", ["modeling_allgdn"])
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
from elephantine.universal import from_qwen35
cfg = ag.AllGDNConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48,
    layer_types=["linear_attention"] * 4)
cfg.engram = {"layers": [1, 3], "orders": [2, 3], "heads": 2, "slots": 16, "dim": 8}
torch.manual_seed(0)
t = ag.AllGDNForCausalLM(cfg).to(F64).eval()
for l in t.model.layers:                          # CPU torch paths (fla kernels are GPU Triton)
    g = l.linear_attn
    g.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
    g.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
    g.causal_conv1d_fn = None
rand_(t)
assert set(t.engrams) == {"1", "3"}
x = ids(97, T=12)
x[0, 3:6] = x[0, 7:10] = torch.tensor([5, 6, 7])                 # a repeated trigram hits the same slots
with torch.no_grad():
    ref = t(x, use_cache=False).logits
    u = from_qwen35(t)
    ours = u(x)
    for l in u.layers:                            # the n-gram memory really matters
        if l.engram is not None:
            l.engram.c.zero_()
    assert (ref - u(x)).abs().max().item() > 1e-3
    report(ref, ours)
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


def test_same_host_carries_edgerock_fixed_memory():
    """EdgeRock: Qwen3.5 gated-DeltaNet + gated attention whose attention layers are sliding-window
    and carry an independent fixed-capacity memory: keys / values written through a SwiGLU writer,
    admitted by a learned score, readable one chunk later while they stay in the top `capacity`
    scores, read with a null key and a per-head tanh-sigmoid gate. On the host: the window branch
    plus a parallel memory attention (score-selected mask, null logit, head gate, kv writer)."""
    gap, scale = _tf5_run(TF5_REMOTE + '''
cm, em = remote("models--rockmatrix-ai--EdgeRock", "edgerock_remote", ["configuration_edgerock", "modeling_edgerock"])
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
from elephantine.universal import from_qwen35
cfg = cm.EdgeRockConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48,
    layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"])
cfg.fixed_memory_revision = em.INDEPENDENT_MEMORY_REVISION
cfg.fixed_memory_swa_layers, cfg.fixed_memory_swa_window = [1, 3], 4
cfg.independent_memory = {"admission_rank": 8, "capacity": 3, "max_commit_chunk": 2, "writer_intermediate_size": 24}
torch.manual_seed(0)
t = em.EdgeRockForCausalLM(cfg).to(F64).eval()
for l in t.model.layers:                          # CPU torch paths (fla kernels are GPU Triton)
    g = getattr(l, "linear_attn", None)
    if g is not None:
        g.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
        g.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
        g.causal_conv1d_fn = None
rand_(t)
with torch.no_grad():                             # open the read gates (they init at tanh(0) = 0)
    for l in t.model.layers:
        if hasattr(l, "memory"):
            l.memory.read_scale.fill_(1.0)
assert sum(hasattr(l, "memory") for l in t.model.layers) == 2
x = ids(97, T=14)
with torch.no_grad():
    ref = t(x, use_cache=False).logits
    u = from_qwen35(t)
    ours = u(x)
    for l in u.layers:                            # selection really bites: capacity 3 < keys before the chunk
        if l.mem is not None:
            l.mem.sel_cap.fill_(float("inf"))
    assert (ref - u(x)).abs().max().item() > 1e-4
    for l in u.layers:                            # the memory really matters
        if l.mem is not None:
            l.mem.hg_c.zero_()
    assert (ref - u(x)).abs().max().item() > 1e-3
    u = from_qwen35(t)
    for l in u.layers:                            # so does the window
        l.attn.window.fill_(float("inf"))
    assert (ref - u(x)).abs().max().item() > 1e-3
    report(ref, ours)
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


def test_edgerock_memory_and_engram_blocks_fp64():
    """Block level, with the references' fp32 casts (Qwen3.5 / Engram RMSNorm, the rotary table)
    lifted to fp64: EdgeRock's memory read and AllGDN's Engram agree with the host to ~1e-16."""
    gap, scale = _tf5_run(TF5_REMOTE + '''
cm, em = remote("models--rockmatrix-ai--EdgeRock", "edgerock_remote", ["configuration_edgerock", "modeling_edgerock"])
(ag,) = remote("models--summerMC--Cyber-9B-Qwen-AllGDN-Engram", "allgdn_remote", ["modeling_allgdn"])
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
from elephantine.universal import from_qwen35
rms64 = lambda w1: lambda self, x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (w1 + getattr(self, "weight", getattr(self, "w", None)))
q.Qwen3_5RMSNorm.forward = rms64(1)
ag.EngramRMS.forward = rms64(0)
kw = dict(vocab_size=97, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=8,
    linear_conv_kernel_dim=4, intermediate_size=48)
cfg = cm.EdgeRockConfig(layer_types=["linear_attention", "full_attention"], **kw)
cfg.fixed_memory_revision = em.INDEPENDENT_MEMORY_REVISION
cfg.fixed_memory_swa_layers, cfg.fixed_memory_swa_window = [1], 4
cfg.independent_memory = {"admission_rank": 8, "capacity": 3, "max_commit_chunk": 2, "writer_intermediate_size": 24}
torch.manual_seed(0)
t = rand_(em.EdgeRockForCausalLM(cfg).to(F64).eval())
mem = t.model.layers[1].memory
with torch.no_grad():
    mem.read_scale.fill_(1.0)
torch.manual_seed(5)
h = torch.randn(2, 14, 64, dtype=F64)
with torch.no_grad():
    fr = torch.arange(14, dtype=F64)[:, None] * t.model.rotary_emb.inv_freq.double()[None]
    emb = torch.cat((fr, fr), -1)[None].expand(2, -1, -1)
    ref = mem(h, (emb.cos(), emb.sin()))
    ul = from_qwen35(t).layers[1]
    gaps = [(ref - ul.mem(ul.norm_m(h), True, {})).abs().max().item()]
    scales = [ref.abs().max().item()]
cfg = ag.AllGDNConfig(layer_types=["linear_attention"] * 2, **kw)
cfg.engram = {"layers": [1], "orders": [2, 3], "heads": 2, "slots": 16, "dim": 8}
torch.manual_seed(0)
t = rand_(ag.AllGDNForCausalLM(cfg).to(F64).eval())
x = ids(97, T=14)
with torch.no_grad():
    ref = t.engrams["1"](h, x)
    gaps.append((ref - from_qwen35(t).layers[1].engram(h, x)).abs().max().item())
    scales.append(ref.abs().max().item())
print("GAP", max(g / max(s, 1) for g, s in zip(gaps, scales)), "SCALE", 1.0)
''')
    assert gap < 1e-13, gap


@torch.no_grad()
def test_same_host_carries_lfm2_whole_model():
    """LFM2 as a whole model (it was block-tested): gated short-conv layers + qk-normed GQA."""
    from transformers import Lfm2Config, Lfm2ForCausalLM
    from elephantine.universal import from_lfm
    cfg = Lfm2Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, conv_L_cache=3, conv_bias=True, block_auto_adjust_ff_dim=False,
                     layer_types=["conv", "full_attention", "conv", "full_attention"], attn_implementation="eager")
    torch.manual_seed(0)
    t = _rand_(Lfm2ForCausalLM(cfg).to(F64).eval())
    x = _ids(97)
    ref = t(x, use_cache=False).logits
    d = (ref - from_lfm(t)(x)).abs().max().item()
    assert d < TOL * max(ref.abs().max().item(), 1), d


TF5_LFM3_CFG = '''
kw = dict(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=6, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, conv_L_cache=7, conv_bias=False, sliding_window=4, use_sandwich_norm=True,
    pad_token_id=0, bos_token_id=1, eos_token_id=2,
    layer_types=["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"],
    rope_parameters={"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1e6, "rope_type": "default"},
                     "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})
'''


def test_same_host_carries_lfm2_moe_and_lfm3():
    """LFM2-MoE (transformers 5.18): LFM2 + a sigmoid top-k MoE with expert-bias selection and
    (sum + 1e-6) renormalisation. LFM3 (remote code): width-7 gated short conv, per-head sigmoid
    attention gate, full + sliding layers with their own (zero-tail partial) RoPE, sandwich norms."""
    gap, scale = _tf5_run(TF5_REMOTE + TF5_LFM3_CFG + '''
from transformers.models.lfm2_moe import modeling_lfm2_moe as lm, configuration_lfm2_moe as lc
from elephantine.universal import from_lfm
cfg = lc.Lfm2MoeConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, conv_L_cache=3, num_dense_layers=1,
    num_experts=6, num_experts_per_tok=2, use_expert_bias=True, norm_topk_prob=True, routed_scaling_factor=1.5,
    layer_types=["conv", "full_attention", "conv", "full_attention"], experts_implementation="eager")
torch.manual_seed(0)
t = rand_(lm.Lfm2MoeForCausalLM(cfg).to(F64).eval())
with torch.no_grad():
    for l in t.model.layers:
        if hasattr(l.feed_forward, "expert_bias"):
            l.feed_forward.expert_bias.copy_(torch.randn(6, dtype=F64) * 0.3)
x = ids(97, T=12)
gaps, scales = [], []
with torch.no_grad():
    ref = t(x, use_cache=False).logits
    gaps.append((ref - from_lfm(t)(x)).abs().max().item()); scales.append(ref.abs().max().item())
(cm, mm) = remote("models--LiquidAI--LFM3-1.7B-Base", "lfm3_remote", ["configuration_lfm3", "modeling_lfm3"])
cfg = cm.Lfm3Config(**kw)
torch.manual_seed(0)
t = rand_(mm.Lfm3ForCausalLM(cfg).to(F64).eval())
with torch.no_grad():
    ref = t(x, use_cache=False).logits
    u = from_lfm(t)
    gaps.append((ref - u(x)).abs().max().item()); scales.append(ref.abs().max().item())
    for l in u.layers:                                    # the sliding window really bites
        l.attn.window.fill_(float("inf"))
    assert (ref - u(x)).abs().max().item() > 1e-4
print("GAPS", gaps, scales)
print("GAP", max(g / max(s, 1) for g, s in zip(gaps, scales)), "SCALE", 1.0)
''')
    assert gap < 1e-5, gap


def test_same_host_carries_lfm3_rd_unrolled():
    """LFM3-RD (looped, recurrent-depth LFM3) at its shipped manual setting: prelude, 5 motifs whose
    attention / conv projections are a shared core + per-layer LoRA (merged as at inference), the
    rescale p + step_scale (x - p) around the core (the residual-snapshot branch), coda."""
    gap, scale = _tf5_run(TF5_REMOTE + TF5_LFM3_CFG + '''
(cm, mm) = remote("models--cybertruck32489--LFM3-1.7B-RD", "lfm3rd_remote", ["configuration_lfm3_rd", "modeling_lfm3_rd"])
from elephantine.universal import from_lfm
kw.update(num_hidden_layers=12, num_prelude_layers=2, num_motifs=2, motif_size=4, num_coda_layers=2,
          recurrent_loops=2, rank_attn=4, rank_conv=4, rank_kv=4, conv_lora_groups=3, reasoning_mode="manual",
          layer_types=["conv", "conv"] + ["full_attention", "conv", "sliding_attention", "conv"] * 2
                      + ["full_attention", "conv"])
cfg = cm.Lfm3RDConfig(**kw)
torch.manual_seed(0)
t = mm.Lfm3RDForCausalLM(cfg).to(F64).eval()
with torch.no_grad():                                     # the shared core / LoRA factors start as torch.empty
    for n, p in t.named_parameters():
        if n.startswith("model.core."):
            p.copy_(torch.randn_like(p) * 0.2)
rand_(t)
with torch.no_grad():
    t.model.rd_controller.step_scales.fill_(0.7)          # make the rescale matter
assert t.model.core.phases[0].blocks[0].self_attn.q_proj.shared_weight is t.model.core.phases[1].blocks[0].self_attn.q_proj.shared_weight
x = ids(97, T=12)
with torch.no_grad():
    ref = t(x, use_cache=False).logits
    u = from_lfm(t)
    ours = u(x)
    for l in u.layers:
        l.res_mix.zero_()
    assert (ref - u(x)).abs().max().item() > 1e-4
    report(ref, ours)
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


@torch.no_grad()
def test_same_host_carries_ceno():
    """CENO: Nemotron-H-style Mamba-2 / MoE / attention / MLP hybrid whose non-Mamba blocks replace the
    residual stream by a norm of it (stream-norm branches), double norms in series (the norm pre
    stage), ungated squared-ReLU experts with post-top-k softmax routing, NoPE attention."""
    import glob, importlib.util, os, sys, types
    hit = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--CladeTeam--CENO-1.3B-A450M-base/snapshots/*/modeling_ceno.py"))
    if not hit:
        pytest.skip("CENO remote code not cached")
    snap = os.path.dirname(hit[0]) + "/"
    pkg = types.ModuleType("ceno_remote"); pkg.__path__ = [snap]; sys.modules["ceno_remote"] = pkg
    stubs = {n: sys.modules.pop(n) for n in list(sys.modules)
             if n.split(".")[0] in ("mamba_ssm", "causal_conv1d") and getattr(sys.modules[n], "__spec__", None) is None}
    mods = {}
    try:
        for f in ("configuration_ceno", "modeling_ceno"):
            spec = importlib.util.spec_from_file_location("ceno_remote." + f, snap + f + ".py")
            mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
            mods[f] = mod
    finally:
        sys.modules.update(stubs)
    # earlier tests may have imported the real (Triton) mamba_ssm; keep CENO on its torch gated norm
    mods["modeling_ceno"].rmsnorm_fn = None
    from elephantine.universal import from_ceno
    cfg = mods["configuration_ceno"].CENOConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
        num_hidden_layers=7, hybrid_override_pattern="ME*E-M*", num_attention_heads=4, num_key_value_heads=2,
        attention_head_dim=8, mamba_num_heads=4, mamba_head_dim=8, ssm_state_size=8, n_groups=2, conv_kernel=4,
        chunk_size=4, num_experts=4, moe_top_k=2, mlp_hidden_act="relu2", mlp_use_swiglu=False,
        qkv_layernorm=True, mamba_in_proj_layernorm=True, use_mamba_kernels=False, use_conv_bias=True,
        layer_norm_epsilon=1e-5, attn_implementation="eager", pad_token_id=1, bos_token_id=0, eos_token_id=0)
    cfg.ceno_mamba_gate_in_ssm = True
    torch.manual_seed(0)
    t = _rand_(mods["modeling_ceno"].CENOForCausalLM(cfg).to(F64).eval())
    assert [b.block_type for b in t.backbone.layers] == ["mamba", "moe", "attention", "moe", "mlp", "mamba", "attention"]
    x = _ids(97)
    out = t(x, use_cache=False)
    ref = out.logits if hasattr(out, "logits") else out[0]
    host = from_ceno(t)
    d = (ref - host(x)).abs().max().item()
    assert d < 1e-5 * max(ref.abs().max().item(), 1), d
    for l in host.layers:                                        # the stream norms really matter
        l.sn_c.zero_()
    assert (ref - host(x)).abs().max().item() > 1e-3


_APERTUS_ROPE = {"default": None, "llama3": {"rope_type": "llama3", "factor": 8.0, "original_max_position_embeddings": 16,
                                             "low_freq_factor": 1.0, "high_freq_factor": 4.0},
                 "linear": {"rope_type": "linear", "factor": 2.0}}


@pytest.mark.parametrize("rope", list(_APERTUS_ROPE))
@torch.no_grad()
def test_same_host_carries_apertus_xielu(rope):
    """Apertus (transformers' ApertusForCausalLM): Llama body + per-head qk-norm + an ungated up / down
    MLP under xIELU, whose learned alpha_p / alpha_n (and bf16-rounded beta / eps buffers) differ per
    layer -> the c_xielu branch. RoPE: default (v1.1), llama3 (8B / 70B) and linear (v1.1-0.5B)."""
    from transformers import ApertusConfig, ApertusForCausalLM
    from elephantine.universal import from_apertus
    cfg = ApertusConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3,
                        num_attention_heads=6, num_key_value_heads=2, max_position_embeddings=64,
                        rope_theta=10000.0, rope_scaling=_APERTUS_ROPE[rope], attn_implementation="eager")
    t = ApertusForCausalLM(cfg).to(F64).eval()
    torch.manual_seed(0)
    for n, p in t.named_parameters():                            # alphas spread so layers differ
        p.add_(torch.randn_like(p) * (0.5 if "alpha" in n else 0.05))
    x = _ids(97)
    ref, host = t(x).logits, from_apertus(t)
    d = (ref - host(x)).abs().max().item()
    assert d < 1e-5 * max(ref.abs().max().item(), 1), d          # Apertus' RMSNorms round to fp32
    with _lift_fp32():                                           # ... and only that: ~1e-15 in fp64
        ref64 = t(x).logits
    assert (ref64 - host(x)).abs().max().item() < 1e-12 * ref64.abs().max().item()
    for tl, ul in zip(t.model.layers, host.layers):              # the MLP alone has no fp32 casts: exact
        h = _x(48)
        a = tl.mlp(h)
        assert (a - ul.ffn(h)).abs().max().item() < 1e-13 * a.abs().max().item()
        assert (tl.mlp.up_proj(h) < 0).any() and (tl.mlp.up_proj(h) > 0).any()   # both xIELU branches
    for l in host.layers:                                        # the branch really matters
        l.ffn.c_xielu.zero_()
    assert (ref - host(x)).abs().max().item() > 1e-2


# --- exact GeLU / mish branches (UFFN c_gelu / c_geglu / gelu_tanh / c_mish) ------------------------

def _host_vs(t, host, x):
    """(max |ref - host|, the same with the target's fp32 casts lifted, max |ref|)."""
    ref, out = t(x).logits, host(x)
    with _lift_fp32():
        ref64 = t(x).logits
    return (ref - out).abs().max().item(), (ref64 - out).abs().max().item(), ref64.abs().max().item()


def _flip_changes(host, x, attr, val, tol=1e-6):
    out = host(x)
    for ul in host.layers:
        getattr(ul.ffn, attr).fill_(val)
    return (out - host(x)).abs().max().item() > tol


@pytest.mark.parametrize("act", ["gelu_new", "gelu", "gelu_pytorch_tanh"])
@torch.no_grad()
def test_same_host_carries_gpt2_gelu(act):
    """GPT-2 (stock GPT2LMHeadModel): learned-abs positions + NoPE, biased fused Conv1D attention,
    biased LayerNorms and an ungated biased MLP -> the c_gelu branch, tanh-approx for gelu_new /
    gelu_pytorch_tanh (gelu_tanh = 1), erf for gelu. No fp32 casts inside: fp64-exact."""
    from transformers import GPT2Config, GPT2LMHeadModel
    from elephantine.universal import from_gpt2
    cfg = GPT2Config(vocab_size=97, n_embd=48, n_layer=3, n_head=6, n_positions=32, activation_function=act,
                     attn_implementation="eager")
    t = _rand_(GPT2LMHeadModel(cfg).to(F64).eval())
    x = _ids(97, T=12)
    host = from_gpt2(t)
    assert all(float(ul.ffn.gelu_tanh) == (act != "gelu") for ul in host.layers)
    d, d64, sc = _host_vs(t, host, x)
    assert d < 1e-12 * sc, d
    assert _flip_changes(host, x, "gelu_tanh", float(act == "gelu"))       # erf vs tanh really differ
    assert _flip_changes(host, x, "c_gelu", 0.0, 1e-2)                     # ... and the branch matters


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "sequential"])
@torch.no_grad()
def test_same_host_carries_gpt_neox_gelu(parallel):
    """GPT-NeoX / Pythia (stock GPTNeoXForCausalLM): per-head fused qkv, partial rotary (rotary_pct
    0.25), the parallel residual (ULayer.par) or sequential, biased ungated MLP -> the erf c_gelu
    branch. Its attention softmax rounds to fp32; lifted, fp64-exact."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from elephantine.universal import from_gpt_neox
    cfg = GPTNeoXConfig(vocab_size=97, hidden_size=48, num_hidden_layers=3, num_attention_heads=6,
                        intermediate_size=80, rotary_pct=0.25, use_parallel_residual=parallel,
                        hidden_act="gelu", attn_implementation="eager")
    t = _rand_(GPTNeoXForCausalLM(cfg).to(F64).eval())
    x = _ids(97, T=12)
    host = from_gpt_neox(t)
    d, d64, sc = _host_vs(t, host, x)
    assert d < 1e-6 * sc and d64 < 1e-12 * sc, (d, d64)
    assert _flip_changes(host, x, "gelu_tanh", 1.0)


def _gemma3_tiny():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=4,
                           layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
                           query_pre_attn_scalar=16, rope_local_base_freq=10000.0, rope_theta=1e6,
                           attn_implementation="eager")
    return _rand_(Gemma3ForCausalLM(cfg).to(F64).eval())


@torch.no_grad()
def test_same_host_carries_gemma3_geglu_tanh():
    """Gemma-3 text (stock Gemma3ForCausalLM): sqrt(d)-scaled embeddings, (1 + w) RMSNorms and qk-norms,
    sandwich norms, query_pre_attn_scalar, sliding-window layers (window 4 < T, local RoPE base) and a
    full layer -> the gated GeGLU branch c_geglu with the tanh approximation (gelu_pytorch_tanh)."""
    from elephantine.universal import from_gemma
    t = _gemma3_tiny()
    x = _ids(97, T=12)
    host = from_gemma(t)
    assert all(ul.ffn.c_geglu == 1 and ul.ffn.gelu_tanh == 1 for ul in host.layers)
    d, d64, sc = _host_vs(t, host, x)
    assert d < 1e-5 * sc and d64 < 1e-12 * sc, (d, d64)                  # Gemma's norms round to fp32
    out = host(x)
    for ul in host.layers:                                               # the window really bites
        ul.attn.window.fill_(float("inf"))
    assert (out - host(x)).abs().max().item() > 1e-4
    assert _flip_changes(from_gemma(t), x, "gelu_tanh", 0.0)


@torch.no_grad()
def test_same_host_carries_gemma1_geglu():
    """Gemma (1) (stock GemmaForCausalLM; its MLP reads hidden_act): no sandwich / qk norms, MQA."""
    from transformers import GemmaConfig, GemmaForCausalLM
    from elephantine.universal import from_gemma
    for act in ("gelu_pytorch_tanh", "gelu"):
        cfg = GemmaConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=1, head_dim=12, hidden_act=act,
                          attn_implementation="eager")
        t = _rand_(GemmaForCausalLM(cfg).to(F64).eval())
        x = _ids(97, T=12)
        host = from_gemma(t)
        assert all(float(ul.ffn.gelu_tanh) == (act != "gelu") for ul in host.layers)
        d, d64, sc = _host_vs(t, host, x)
        assert d < 1e-5 * sc and d64 < 1e-12 * sc, (act, d, d64)


@torch.no_grad()
def test_same_host_carries_llama_mish():
    """A mish GLU: stock LlamaForCausalLM with hidden_act='mish' (the DanTagGen architecture, the zoo's
    4 mish models) -> from_llama selects the c_mish branch from hidden_act."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3,
                      num_attention_heads=6, num_key_value_heads=2, rms_norm_eps=1e-5, hidden_act="mish",
                      attn_implementation="eager")
    t = _rand_(LlamaForCausalLM(cfg).to(F64).eval())
    x = _ids(97)
    host = from_llama(t)
    assert all(ul.ffn.c_mish == 1 for ul in host.layers)
    d, d64, sc = _host_vs(t, host, x)
    assert d < TOL and d64 < 1e-12 * sc, (d, d64)
    assert _flip_changes(host, x, "c_mish", 0.0, 1e-3)                  # SiLU is not mish


_REAL_GELU = {"gpt2": "from_gpt2", "EleutherAI/pythia-70m": "from_gpt_neox", "google/gemma-3-270m": "from_gemma"}


@pytest.mark.parametrize("repo", list(_REAL_GELU))
@torch.no_grad()
def test_same_host_carries_real_gelu_checkpoints(repo):
    """The real checkpoints (CPU): GPT-2 124M (gelu_new), Pythia-70m (erf GeLU, partial RoPE, parallel
    block), Gemma-3-270m (GeGLU-tanh, 5:1 sliding / full). Loaded in fp64 with their fp32 casts lifted the
    host agrees to ~1e-13 rel; against the checkpoint run in its own fp32, to fp32 noise (same argmax)."""
    import elephantine.universal as U
    from transformers import AutoModelForCausalLM
    try:
        t = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True, dtype=F64,
                                                 attn_implementation="eager").eval()
    except OSError:
        pytest.skip(f"{repo} not cached")
    x = _ids(min(t.config.vocab_size, 50000), T=16)
    host = getattr(U, _REAL_GELU[repo])(t)
    d, d64, sc = _host_vs(t, host, x)
    assert d64 < 1e-12 * sc, d64
    out = host(x)
    ref32 = t.float()(x).logits.double()
    assert (ref32 - out).abs().max().item() < 1e-4 * sc
    assert torch.equal(ref32.argmax(-1), out.argmax(-1))


def test_every_loader_returns_the_host():
    """Every whole-model loader ends in `return u` (the one UniversalElephantine it filled)."""
    import ast, inspect
    import elephantine.universal as U
    tree = ast.parse(inspect.getsource(U))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("from_")]
    assert {"from_gpt2", "from_gpt_neox", "from_gemma", "from_llama"} <= {f.name for f in fns}
    for f in fns:
        last = f.body[-1]
        assert isinstance(last, ast.Return) and isinstance(last.value, ast.Name) and last.value.id == "u", f.name


def _motif_modules():
    """Motif-3's own HF remote code (configuration_motif.py, modeling_motif.py; no weights). The
    `kernels` hub-kernel import is blocked so PolyNorm is the file's PolyNormTorch."""
    import glob, importlib.util, os, sys, types
    hit = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Motif-Technologies--Motif-3/snapshots/*/modeling_motif.py"))
    if not hit:
        pytest.skip("Motif-3 remote code not cached")
    snap = os.path.dirname(hit[0]) + "/"
    pkg = types.ModuleType("motif_remote"); pkg.__path__ = [snap]; sys.modules["motif_remote"] = pkg
    saved, sys.modules["kernels"] = sys.modules.get("kernels"), None
    mods = {}
    try:
        for f in ("configuration_motif", "modeling_motif"):
            spec = importlib.util.spec_from_file_location("motif_remote." + f, snap + f + ".py")
            mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
            mods[f] = mod
    finally:
        if saved is None:
            sys.modules.pop("kernels", None)
        else:
            sys.modules["kernels"] = saved
    assert mods["modeling_motif"].PolyNorm is mods["modeling_motif"].PolyNormTorch
    return mods["configuration_motif"], mods["modeling_motif"]


def _lift_fp32():
    """Context: run a reference with its fp32 casts kept in fp64 (elephantine.emulate.lift_fp32), to show the
    remaining gap is only that rounding."""
    from elephantine.emulate import lift_fp32
    return lift_fp32()


@pytest.mark.parametrize("hc,score", [(1e6, "sigmoid"), (0.7, "softmax")], ids=["motif3", "clamped-softmax"])
@pytest.mark.parametrize("kind", ["dense", "moe"])
@torch.no_grad()
def test_ffn_block_carries_motif3_polynorm(kind, hc, score):
    """Motif-3 FFNs (its own remote code) -> UFFN's PolyNorm-GLU branch: the dense MotifMLP (PolyNormTorch,
    sigmoid weights, hidden clamp, output scale) and the MoE (sigmoid / softmax router, top-k on scores +
    expert_bias, route_norm, route_scale 2; per-expert GroupedPolyNorm with the bias clamp and result
    clamp; a PolyNormTorch shared expert). Motif-3's own config: hidden_clamp 1e6, scale 0.5, bias clamp
    0.5; the second case clamps hard (0.7) so the clamps are exercised. Block level: Motif-3's GDLA
    (differential latent) attention is not a host branch."""
    from elephantine.universal import motif_ffn_block
    mc, mm = _motif_modules()
    cfg = mc.MotifConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
                         num_experts=6, experts_top_k=2, num_shared_experts=1, score_func=score, route_norm=True,
                         route_scale=2.0, hidden_act="poly_norm", polynorm_output_scale=0.5,
                         polynorm_bias_clamp=0.5, hidden_clamp=hc, load_balance_coeff=1e-4)
    torch.manual_seed(0)
    m = (mm.MotifMLP(cfg) if kind == "dense" else mm.MoE(cfg)).to(F64).eval()
    for p in m.parameters():                                     # (experts start as torch.empty)
        p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 1.0))
    if kind == "moe":
        m.expert_bias.copy_(torch.randn_like(m.expert_bias) * 0.1)
    x = _x(32)
    run = lambda: (lambda o: o[0] if isinstance(o, tuple) else o)(m(x))
    ref, f = run(), motif_ffn_block(m, cfg)
    ours = f(x)
    assert (ref - ours).abs().max().item() < 1e-5 * max(ref.abs().max().item(), 1)   # Motif computes in fp32
    with _lift_fp32():
        ref64 = run()
    assert ref64.dtype == F64 and (ref64 - ours).abs().max().item() < 1e-12 * ref64.abs().max().item()
    f.c_poly.zero_()                                             # the branch really matters
    assert (ref - f(x)).abs().max().item() > 1e-1


def test_same_host_carries_motif3_whole_model():
    """Motif-3 (MotifForCausalLM, its own remote code, transformers 5.18; config + code only), tiny:
    GDLA attention (q / kv LoRA latents with RMSNorms, NoPE + shared-RoPE key split, GQA, v_head_dim <
    head_dim, groups of 2 signal + 1 noise heads: o_i - sigmoid(lambda_proj x)_i o_noise, elementwise
    sigmoid output gate), YaRN rotary + mscale^2 on full layers, interleaved sliding-window layers on
    their own plain RoPE (swa_rope_theta), a dense PolyNorm-GLU layer then sigmoid-routed PolyNorm MoE
    layers with a shared expert (narrower than the dense bank), mHC (3 streams, 20 Sinkhorn steps).
    Motif refuses any attn_implementation but flash_attention_2 and the flash_attn wheel here is
    broken, so its "flash_attention_2" entry is swapped for a torch kernel with the same semantics
    (causal, GQA, window_size = (W - 1, W - 1) when T > W). Motif's code cannot run in fp64 as is
    (fp32 casts meet fp64 weights), so: the stock fp32 run vs the host at the fp32 level, and the same
    weights in fp64 with the fp32 casts lifted at ~1e-15."""
    import os
    if not __import__("glob").glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--Motif-Technologies--Motif-3/snapshots/*/modeling_motif.py")):
        pytest.skip("Motif-3 remote code not cached")
    gap, scale = _tf5_run(TF5_REMOTE + '''
import contextlib
import torch.nn.functional as tF
sys.modules["kernels"] = None                     # Motif's hub-kernel import -> its torch PolyNorm / RMSNorm
mc, mm = remote("models--Motif-Technologies--Motif-3", "motif_remote", ["configuration_motif", "modeling_motif"])
import transformers.modeling_utils as mu
mu.lazy_import_flash_attention = lambda *a, **k: None
def fa2(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None,
        softcap=None, is_causal=None, **kw):
    """flash_attn_func semantics in torch: (B,H,T,D) q, (B,Hk,T,D) k / v -> (B,T,H,Dv)."""
    assert attention_mask is None and dropout == 0.0 and softcap is None
    B, H, T, D = query.shape
    rep = H // key.shape[1]
    k, v = key.repeat_interleave(rep, 1), value.repeat_interleave(rep, 1)
    s = query @ k.transpose(-1, -2) * scaling
    tt, ss = torch.arange(T)[:, None], torch.arange(T)[None]
    ok = ss <= tt
    if sliding_window is not None and T > sliding_window:
        ok = ok & (tt - ss <= sliding_window - 1)
    return (s.masked_fill(~ok, float("-inf")).softmax(-1) @ v).transpose(1, 2), None
mu.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = fa2
@contextlib.contextmanager
def lift():
    T, zl, sm = torch.Tensor, torch.zeros_like, tF.softmax
    of, ot = T.float, T.to
    def to(self, *a, **k):
        a = tuple(F64 if v is torch.float32 else v for v in a)
        if k.get("dtype") is torch.float32:
            k["dtype"] = F64
        return ot(self, *a, **k)
    T.float, T.to = (lambda self: ot(self, F64)), to
    torch.zeros_like = lambda v, **k: zl(v, **({**k, "dtype": F64} if k.get("dtype") is torch.float32 else k))
    tF.softmax = lambda v, dim=None, dtype=None, **k: sm(v, dim=dim, dtype=F64 if dtype is torch.float32 else dtype, **k)
    try:
        yield
    finally:
        T.float, T.to, torch.zeros_like, tF.softmax = of, ot, zl, sm
from elephantine.universal import from_motif
cfg = mc.MotifConfig(vocab_size=97, hidden_size=32, intermediate_size=40, num_hidden_layers=4, num_attention_heads=6,
    num_key_value_heads=2, head_dim=12, num_noise_heads=2, q_lora_rank=16, kv_lora_rank=12, qk_rope_head_dim=4,
    v_head_dim=8, attention_cls="gdla", diff_v2=True, elementwise_attn_output_gate=True,
    max_position_embeddings=64, rope_theta=10000.0, swa_rope_theta=500.0, original_seq_len=16, rope_factor=4.0,
    mscale=1.0, rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16,
                              "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "rope_theta": 10000.0},
    use_sliding_window=True, sliding_window=4, sliding_window_pattern="interleave", sliding_window_period=2,
    num_experts=6, experts_top_k=2, num_shared_experts=1, moe_intermediate_size=16, interleave_moe_layer_step=1,
    n_dense_first_layers=1, score_func="sigmoid", route_norm=True, route_scale=2.0, load_balance_coeff=1e-4,
    hidden_act="poly_norm", polynorm_output_scale=0.5, polynorm_bias_clamp=0.5, hidden_clamp=1e6,
    mhc_enabled=True, mhc_expansion_rate=3, mhc_sinkhorn_iters=20, rms_norm_eps=1e-5,
    attn_implementation="flash_attention_2")
t = mm.MotifForCausalLM(cfg).float().eval()
torch.manual_seed(0)
with torch.no_grad():
    for n, p in t.named_parameters():
        p.add_(torch.randn_like(p) * (0.3 if ("alpha" in n or "bias" in n or "act_fn" in n) else 0.05))
    for l in t.model.layers:
        if l.moe_enabled:
            l.moe.expert_bias.copy_(torch.randn_like(l.moe.expert_bias) * 0.05)
L = t.model.layers
assert [l.self_attn.sliding_window for l in L] == [None, 5, None, 5] and [l.moe_enabled for l in L] == [False, True, True, True]
assert L[0].self_attn.scaling > L[1].self_attn.scaling            # mscale^2 on the full (YaRN) layers
x = ids(97, T=12)
with torch.no_grad():
    ref = t(x, use_cache=False).logits                            # stock fp32 path
    host = from_motif(t)
    ours = host(x)
    t.to(F64)
    with lift():
        r64 = t(x, use_cache=False).logits
    assert r64.dtype == F64 and (r64 - ours).abs().max().item() < 1e-12 * r64.abs().max().item(), (r64 - ours).abs().max()
    for name, off in (("diff", lambda a: a.dl_c.zero_()), ("gate", lambda a: a.gate_coef.copy_(torch.tensor([1.0, 0.0]))),
                      ("window", lambda a: a.window.fill_(float("inf")))):
        h2 = from_motif(t)
        for l in h2.layers:
            off(l.attn)
        assert (r64 - h2(x)).abs().max().item() > 1e-2, name      # each branch really matters
    report(ref.double(), ours)
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


def test_same_host_carries_glm5_next():
    """GLM-5.3-Flash (glm5_next, transformers 5.18): mHC (3 Sinkhorn-mixed residual streams), KDA with
    the safe lower-bound decay, NoPE MLA under DeepSeek sparse attention (k-pool indexer: pools of 2,
    top-4 keys, plus the tail), sigmoid router + selection bias + shared expert, SwiGLU clip. The text
    model with an lm_head; T = 12 so the indexer drops keys."""
    gap, scale = _tf5_run('''
from transformers.models.glm5_next import modeling_glm5_next as gm, configuration_glm5_next as gc
from types import SimpleNamespace as NS
from elephantine.universal import from_glm5_next
cfg = gc.Glm5NextTextConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=6,
    num_experts_per_tok=2, routed_scaling_factor=2.5, kv_lora_rank=16, q_lora_rank=24, qk_nope_head_dim=8,
    qk_rope_head_dim=0, v_head_dim=8, n_group=1, topk_group=1, norm_topk_prob=True,
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    hc_mult=3, hc_sinkhorn_iters=5, hc_eps=1e-6, index_n_heads=2, index_head_dim=8, index_topk=4, index_kpool=2,
    index_kpool_always_select_tail=True, swiglu_limit=0.1, rms_norm_eps=1e-5,
    layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    linear_attn_config={"num_heads": 4, "head_dim": 8, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
                        "kda_layers": [0, 2], "full_attn_layers": [1, 3]},
    attn_implementation="eager", experts_implementation="eager", pad_token_id=0)
torch.manual_seed(0)
m = rand_(gm.Glm5NextTextModel(cfg).to(F64).eval())
head = torch.nn.Linear(32, 97, bias=False).to(F64)
with torch.no_grad():
    head.weight.normal_(0, 0.2)
    for l in m.layers:
        if hasattr(l.mlp, "gate"):
            l.mlp.gate.e_score_correction_bias.copy_(torch.randn(6, dtype=F64) * 0.3)
        ix = getattr(l.self_attn, "indexer", None)
        if ix is not None:                                   # learned pooling, not the uniform init
            ix.index_kpool_compress_ape.normal_(0, 0.5); ix.index_kpool_compress_gate.normal_(0, 0.3)
target = NS(config=cfg, model=m, lm_head=head)
x = ids(97, T=12)
with torch.no_grad():
    ref = head(m(x).last_hidden_state)
    u = from_glm5_next(target)
    ours = u(x)
    for l in u.layers:                                       # the indexer really drops keys
        l.attn.dsa_topk.fill_(float("inf"))
    assert (ref - u(x)).abs().max().item() > 1e-4
    report(ref, ours)
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


@torch.no_grad()
def test_same_host_carries_glm_superexpert_own_code():
    """GLM-5.3-SuperExpert-V2 ships its own modeling code: a dense NoPE MHA decoder WITHOUT a causal
    mask, 4-bit packed weights and a dense softmax mixture of all experts. The host runs it with the
    full mask (the diffusion setting) and the dequantised weights."""
    import glob, importlib.util, os, sys, types
    hit = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--shashanth-27xA9--GLM-5.3-SuperExpert-v2-9.2GB/"
                                       "snapshots/*/modeling_glm_superexpert.py"))
    if not hit:
        pytest.skip("GLM-SuperExpert remote code not cached")
    snap = os.path.dirname(hit[0]) + "/"
    pkg = types.ModuleType("gse_remote"); pkg.__path__ = [snap]; sys.modules["gse_remote"] = pkg
    mods = {}
    for f in ("configuration_glm_superexpert", "modeling_glm_superexpert"):
        spec = importlib.util.spec_from_file_location("gse_remote." + f, snap + f + ".py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        mods[f] = mod
    from elephantine.universal import from_glm_superexpert
    gm = mods["modeling_glm_superexpert"]
    cfg = mods["configuration_glm_superexpert"].GLMSuperExpertConfig(vocab_size=97, hidden_size=32,
        intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=3, num_attention_heads=4,
        n_routed_experts=4, n_shared_experts=1, first_k_dense_replace=1, pad_token_id=0, eos_token_id=[1])
    torch.manual_seed(0)
    t = gm.GLMSuperExpertForCausalLM(cfg).eval()                # fp32: its embedding dequantises to fp32
    for mod in t.modules():                                      # random packed nibbles and scales
        if isinstance(mod, (gm.DequantizedPackedLinear, gm.DequantizedPackedEmbedding)):
            mod.weight.copy_(torch.randint(0, 256, mod.weight.shape, dtype=torch.uint8))
            mod.scale.fill_(0.3)
    _rand_(t)
    x = _ids(97)
    ref = t(x).logits.double()
    host = from_glm_superexpert(t)
    d = (ref - host(x, causal=False)).abs().max().item()
    assert d < 1e-5 * max(ref.abs().max().item(), 1), d
    assert (ref - host(x, causal=True)).abs().max().item() > 1e-3   # it really is bidirectional


TF5_Q35MOE = '''
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qm, configuration_qwen3_5_moe as qc
qkw = dict(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=8,
    linear_conv_kernel_dim=4, moe_intermediate_size=24, shared_expert_intermediate_size=32, num_experts=6,
    num_experts_per_tok=2, layer_types=["linear_attention"] * 3 + ["full_attention"])
def cpu_paths(t):
    for l in t.model.language_model.layers if hasattr(t.model, "language_model") else t.model.layers:
        g = getattr(l, "linear_attn", None)
        if g is not None:
            g.chunk_gated_delta_rule = qm.torch_chunk_gated_delta_rule
            g.recurrent_gated_delta_rule = qm.torch_recurrent_gated_delta_rule
            g.causal_conv1d_fn = None
    return t
'''


def test_same_host_carries_qwen35_moe_and_memt():
    """Qwen3.5-MoE (transformers 5.18, 3-D expert tensors) and MemT-1.0-Nano, a Qwen3.5-MoE whose
    full-attention layers carry a session memory (a stable-GDN2 state written by commit events and
    read from frozen snapshots). A plain LM forward has no memory state, so MemT is Qwen3.5-MoE; the
    session memory API is not carried."""
    gap, scale = _tf5_run(TF5_REMOTE + TF5_Q35MOE + '''
from elephantine.universal import from_qwen35
gaps, scales = [], []
cfg = qc.Qwen3_5MoeTextConfig(**qkw, experts_implementation="eager")
torch.manual_seed(0)
t = rand_(cpu_paths(qm.Qwen3_5MoeForCausalLM(cfg).to(F64).eval()))
x = ids(97)
with torch.no_grad():
    ref = t(x, use_cache=False).logits
    gaps.append((ref - from_qwen35(t)(x)).abs().max().item()); scales.append(ref.abs().max().item())
qm.torch_causal_conv1d_update = getattr(qm, "torch_causal_conv1d_update", None) or qm.causal_conv1d_update
qm.chunk_gated_delta_rule = qm.fused_recurrent_gated_delta_rule = None
cm, mm = remote("models--MemTensor--MemT-1.0-Nano", "memt_remote", ["configuration_memt", "modeling_memt"])
mm.MemTDecoderLayer.block_type = property(lambda s: s.layer_type)        # tf 5.18 renamed the attribute
import json
snap = os.path.dirname(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--MemTensor--MemT-1.0-Nano/snapshots/*/config.json"))[0])
memt = dict(json.load(open(snap + "/config.json"))["memt"], num_memory_heads=4, num_kv_heads=4)
text = dict(qkw, rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                  "mrope_interleaved": True, "mrope_section": [1, 1, 0]}, attn_output_gate=True,
            model_type="qwen3_5_moe_text")
cfg = cm.MemTQwen3_5MoeConfig(backbone_configs={"model_type": "qwen3_5_moe", "text_config": text}, memt=memt,
                              experts_implementation="eager")
torch.manual_seed(0)
t = rand_(cpu_paths(mm.MemTQwen3_5MoeForCausalLM(cfg).to(F64).eval()))
assert any(hasattr(getattr(l, "self_attn", None), "memt") for l in t.model.language_model.layers)
with torch.no_grad():
    ref = t(x, use_cache=False).logits
    gaps.append((ref - from_qwen35(t)(x)).abs().max().item()); scales.append(ref.abs().max().item())
print("GAP", max(g / max(s, 1) for g, s in zip(gaps, scales)), "SCALE", 1.0)
''')
    assert gap < 1e-5, gap


def _fla052_kda_reference():
    """chunk_kda with fla 0.5.2's in-kernel gating, composed from fla 0.5.2's own torch references
    (naive_kda_gate / naive_kda_lowerbound_gate, naive_recurrent_kda); the l2 norm and sigmoid are
    the kernel's documented prologue (x / sqrt(sum x^2 + 1e-6), sigmoid(beta))."""
    import ast, os
    root = os.path.expanduser("~/.cache/upstream_refs/fla_core-0.5.2/fla/ops/kda/")
    if not os.path.isdir(root):
        pytest.skip("fla-core 0.5.2 not fetched (python -m elephantine.fetch_upstream_refs)")
    ns = {}
    exec(compile(open(root + "naive.py").read(), root + "naive.py", "exec"), ns)
    src = open(root + "gate.py").read()
    fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("naive_kda_gate", "naive_kda_lowerbound_gate")]
    exec(compile(ast.Module(body=fns, type_ignores=[]), root + "gate.py", "exec"), {"torch": torch, "F": torch.nn.functional}, ns)

    def chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                  safe_gate=False, lower_bound=None, scale=None, **kw):
        if use_qk_l2norm_in_kernel:
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        if use_beta_sigmoid_in_kernel:
            beta = beta.sigmoid()
        if use_gate_in_kernel:
            g = (ns["naive_kda_lowerbound_gate"](g, A_log, dt_bias, lower_bound, output_dtype=q.dtype) if lower_bound is not None
                 else ns["naive_kda_gate"](g, A_log, dt_bias, output_dtype=q.dtype))
        return ns["naive_recurrent_kda"](q, k, v, g, beta.to(q.dtype), scale=scale, initial_state=initial_state,
                                         output_final_state=output_final_state)
    return chunk_kda


@torch.no_grad()
def test_same_host_carries_youtu_kda_mla():
    """Youtu-LLM text model: KDA (fused qkv, one short conv, low-rank forget gate with the lower-bound
    sigmoid decay, full-rank output gate) + MLA (latent norms, Megatron-interleaved RoPE, head-wise
    gate) + a quantile-balancing sigmoid MoE (selection on logits - bias) with a shared expert.
    fp32 reference (its router is hard fp32); KDA through fla 0.5.2's torch references."""
    import glob, importlib.util, os, sys, types
    hit = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--tencent--Youtu-LLM-A7B-Base/snapshots/*/modeling_youtu_vl_moe.py"))
    if not hit:
        pytest.skip("Youtu-LLM remote code not cached")
    kda = _fla052_kda_reference()
    snap = os.path.dirname(hit[0]) + "/"
    pkg = types.ModuleType("youtu_remote"); pkg.__path__ = [snap]; sys.modules["youtu_remote"] = pkg
    mods = {}
    for f in ("configuration_youtu_vl_moe", "modeling_youtu_vl_moe"):
        spec = importlib.util.spec_from_file_location("youtu_remote." + f, snap + f + ".py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        mods[f] = mod
    M = mods["modeling_youtu_vl_moe"]
    M.chunk_kda = kda
    from elephantine.universal import from_youtu
    cfg = mods["configuration_youtu_vl_moe"].YoutuVLMoEConfig(vocab_size=97, hidden_size=64, intermediate_size=96,
        moe_intermediate_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, head_dim=16,
        q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        linear_attention_pattern=[1, 1, 1, 0], n_routed_experts=8, num_experts_per_tok=2,
        moe_shared_experts_intermediate_size=32, mrope_section=[2, 1, 1],
        rope_parameters={"rope_type": "default", "rope_theta": 1600000.0, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        kda_use_fla_short_conv=False, kda_use_fused_rmsnorm_gated=False, mla_attn_implementation="eager",
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16,
                           num_position_embeddings=16, model_type="qwen3_vl_moe"))
    torch.manual_seed(0)
    t = M.YoutuVLForConditionalGeneration(cfg).eval()
    for n, p in t.named_parameters():                            # some parameters start as torch.empty
        if "language_model" in n or "lm_head" in n:
            p.copy_(torch.randn_like(p) * 0.1 + (1.0 if n.endswith(("norm.weight", "layernorm.weight")) else 0.0))
    for mod in t.modules():
        if isinstance(mod, M.YoutuVLTopkRouter) and mod.qb_beta is not None:
            mod.qb_beta.normal_(0, 0.3)
    x = _ids(97)
    ref = t(input_ids=x, use_cache=False).logits.double()
    host = from_youtu(t)
    d = (ref - host(x)).abs().max().item()
    assert d < 1e-5 * max(ref.abs().max().item(), 1), d
    for l in host.layers:                                        # selection on logits - bias really differs
        l.ffn.sel_logit.zero_()
    assert (ref - host(x)).abs().max().item() > 1e-4


@torch.no_grad()
def test_same_host_carries_jet_nemotron():
    """Jet-Nemotron (its own remote code, dconv_implementation="naive" = its torch dynamic conv): full
    attention, SWA and JetBlocks (dynamic per-token conv on v -> the SSM block's dconv branch). The
    fla Triton pieces it imports are swapped for torch references: transformers' gated-delta rule
    (split per sequence for its unpadded varlen call), a gated RMSNorm, flash-attn-style unpad helpers.
    Always called with a 2-D attention mask (its mask-less path drops the last key)."""
    import glob, importlib.util, os, sys, types, typing
    import torch.nn.functional as F
    hit = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--jet-ai--Jet-Nemotron-2B/snapshots/*/jet_block.py"))
    if not hit:
        pytest.skip("Jet-Nemotron remote code not cached")
    import transformers.utils as tu
    from transformers.models.qwen3_next import modeling_qwen3_next as q3
    if not hasattr(tu, "LossKwargs"):
        tu.LossKwargs = typing.TypedDict("LossKwargs", {}, total=False)
    lu = types.ModuleType("fla.layers.utils")

    def get_unpad_data(mask):
        lens = mask.sum(-1, dtype=torch.int32)
        idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        return idx, F.pad(torch.cumsum(lens, 0, dtype=torch.int32), (1, 0)), int(lens.max())
    lu.get_unpad_data = get_unpad_data
    lu.index_first_axis = lambda x, idx: x[idx]

    def pad_input(x, idx, b, t):
        out = x.new_zeros(b * t, *x.shape[1:]); out[idx] = x
        return out.view(b, t, *x.shape[1:])
    lu.pad_input = pad_input
    saved = {n: sys.modules.get(n) for n in ("fla.layers.utils",)}
    sys.modules["fla.layers.utils"] = lu
    snap = os.path.dirname(hit[0]) + "/"
    pkg = types.ModuleType("jet_remote"); pkg.__path__ = [snap]; sys.modules["jet_remote"] = pkg
    mods = {}
    try:
        for f in ("configuration_jet_nemotron", "kv_cache", "dynamic_conv", "jet_block", "modeling_jet_nemotron"):
            spec = importlib.util.spec_from_file_location("jet_remote." + f, snap + f + ".py")
            mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
            mods[f] = mod
    finally:
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v
    jb = mods["jet_block"]

    class GatedNorm(torch.nn.Module):                            # FusedRMSNormGated(x, g) = rms(x) w silu(g)
        def __init__(self, hidden_size, eps=1e-5, **kw):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(hidden_size)); self.eps = eps

        def forward(self, x, g):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight * F.silu(g)

    def delta_rule(q, k, v, g, beta, initial_state=None, output_final_state=False, cu_seqlens=None,
                   use_qk_l2norm_in_kernel=False, **kw):
        cuts = cu_seqlens.tolist() if cu_seqlens is not None else [0, q.shape[1]]
        outs = []
        for b in range(q.shape[0]):
            for s0, s1 in zip(cuts[:-1], cuts[1:]):
                sl = lambda t: t[b:b + 1, s0:s1]
                o, _ = q3.torch_chunk_gated_delta_rule(sl(q), sl(k), sl(v), sl(g), sl(beta),
                                                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                outs.append((b, s0, s1, o))
        out = torch.zeros(*q.shape[:3], v.shape[-1], dtype=v.dtype)
        for b, s0, s1, o in outs:
            out[b:b + 1, s0:s1] = o
        return out, None
    jb.FusedRMSNormGated = GatedNorm
    jb.chunk_gated_delta_rule = jb.fused_recurrent_gated_delta_rule = delta_rule
    from elephantine.universal import from_jet_nemotron
    cfg = mods["configuration_jet_nemotron"].JetNemotronConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, layer_types=["jet", "attn", "swa", "jet"],
        max_position_embeddings=128, rope_theta=10000.0, tie_word_embeddings=True,
        efficient_attention_config={"jet": {"mode": "chunk", "expand_v": 2, "num_heads": 2, "head_dim": 8, "norm_eps": 1e-5,
                                            "conv_size": 4, "dconv_generator_reduction": 8, "dconv_implementation": "naive"},
                                    "swa": {"window_size": 4}})
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    t = _rand_(mods["modeling_jet_nemotron"].JetNemotronForCausalLM(cfg).to(F64).eval())
    x = _ids(97, T=12)
    ref = t(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
    host = from_jet_nemotron(t)
    d = (ref - host(x)).abs().max().item()
    assert d < TOL * max(ref.abs().max().item(), 1), d
    for l in host.layers:                                        # the dynamic kernel really matters
        l.ssm.dc_w2.weight.zero_()
    assert (ref - host(x)).abs().max().item() > 1e-4


@torch.no_grad()
def test_same_host_carries_gpts3_gdn2_full_attnres():
    """GPT-S3 (its own pure-torch remote code): GDN-2 (erase gate on the read key, sigmoid write
    gate on v, per-channel decay, separate q / k / v short convs) + gated GQA (offset qk-norm,
    interleaved partial RoPE) under Full Attention Residuals (every sub-block output a source)."""
    import glob, importlib.util, os, sys, types
    hit = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--AxiomicLabs--GPT-S3-3M/snapshots/*/modeling_gpts3.py"))
    if not hit:
        pytest.skip("GPT-S3 remote code not cached")
    snap = os.path.dirname(hit[0]) + "/"
    pkg = types.ModuleType("gpts3_remote"); pkg.__path__ = [snap]; sys.modules["gpts3_remote"] = pkg
    mods = {}
    for f in ("configuration_gpts3", "modeling_gpts3"):
        spec = importlib.util.spec_from_file_location("gpts3_remote." + f, snap + f + ".py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        mods[f] = mod
    from elephantine.universal import from_gpts3
    cfg = mods["configuration_gpts3"].GPTS3Config(vocab_size=97, hidden_size=32, num_hidden_layers=4,
        gdn_layers_per_attention=1, gdn_head_dim=16, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        rotary_dim=8, intermediate_size=48, max_position_embeddings=128)
    torch.manual_seed(0)
    t = _rand_(mods["modeling_gpts3"].GPTS3ForCausalLM(cfg).to(F64).eval())
    with torch.no_grad():                                        # AttnRes queries start at 0 (uniform mix)
        for mod in t.modules():
            if isinstance(mod, mods["modeling_gpts3"].AttnResAggregator):
                mod.query.normal_(0, 0.5)
    assert [b.is_attention for b in t.transformer["h"]] == [False, True, False, True]
    x = _ids(97, T=12)
    ref = t(x).logits
    host = from_gpts3(t)
    d = (ref - host(x)).abs().max().item()
    assert d < 1e-5 * max(ref.abs().max().item(), 1), d
    for l in host.layers:                                        # the erase gate really matters
        l.ssm.c_er.zero_()
    assert (ref - host(x)).abs().max().item() > 1e-4
    # fp64 floor: the same source with its fp32 casts made fp64 (a separate module), fp64 RoPE table
    src = open(hit[0]).read().replace(".float()", ".double()").replace("torch.float32", "torch.float64")
    m64 = types.ModuleType("gpts3_remote.modeling_gpts3_fp64"); m64.__package__ = "gpts3_remote"
    sys.modules[m64.__name__] = m64
    exec(compile(src, hit[0], "exec"), m64.__dict__)
    t64 = m64.GPTS3ForCausalLM(cfg).to(F64).eval()
    t64.load_state_dict(t.state_dict())
    ref64 = t64(x).logits
    host = from_gpts3(t64)
    for l in host.layers:
        l.attn.inv_freq[: cfg.rotary_dim // 2] = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2, dtype=F64) / cfg.rotary_dim))
    d64 = (ref64 - host(x)).abs().max().item()
    assert d64 < 1e-12 * max(ref64.abs().max().item(), 1), d64


def test_same_host_carries_qwen35_gdn24_all_linear():
    """Qwen3.5-GDN24 (summerAIO/Qwen3.8-27B-Longbook27-v2): Qwen3.8-27B with its 16 full-attention layers
    converted, so every layer is the STOCK transformers Qwen3_5GatedDeltaNet with Qwen3.5's offset
    RMSNorm and SwiGLU MLP; its fused_ops are bit-for-bit those formulas (Triton fast paths aside).
    Its remote code is read here, not executed: the static check pins that it builds nothing but
    those pieces, and the numerical check runs the same composition through transformers' own
    Qwen3.5 classes with every layer linear_attention."""
    import glob, os, re
    src = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--summerAIO--Qwen3.8-27B-Longbook27-v2"
                                       "/snapshots/*/modeling_qwen35_gdn24.py"))
    if src:
        code = open(src[0]).read()
        layer = code[code.index("class Qwen35GDN24DecoderLayer"):code.index("class Qwen35GDN24PreTrainedModel")]
        assert "self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)" in layer
        assert set(re.findall(r"self\.(\w+) = ", layer)) == {"layer_idx", "linear_attn", "mlp", "input_layernorm",
                                                              "post_attention_layernorm"}
        assert "SuperSwiGLUMLP(config)" in layer and layer.count("SuperRMSNorm(") == 2
    gap, scale = _tf5_run('''
from transformers.models.qwen3_5 import modeling_qwen3_5 as q, configuration_qwen3_5 as qc
from elephantine.universal import from_qwen35
cfg = qc.Qwen3_5TextConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48,
    layer_types=["linear_attention"] * 4)
t = q.Qwen3_5ForCausalLM(cfg).to(F64).eval()
assert all(getattr(l, "linear_attn", None) is not None for l in t.model.layers)
for l in t.model.layers:                          # CPU torch paths (fla kernels are GPU Triton)
    g = l.linear_attn
    g.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
    g.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
    g.causal_conv1d_fn = None
rand_(t)
x = ids(97, T=12)
with torch.no_grad():
    report(t(x, use_cache=False).logits, from_qwen35(t)(x))   # stock DynamicCache needs an attention layer;
                                                                # GDN24 ships its own recurrent cache instead
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


HMF = __import__("os").path.expanduser("~/.cache/upstream_refs/hybrid-model-factory/training/src/hmf")
HMF_PRELUDE = '''
import os, sys, types, importlib, torch
os.environ["TRITON_F32_DEFAULT"] = "ieee"         # IEEE fp32 tl.dot (Ampere's default is TF32)
R = %r
for name, sub in (("hmf", ""), ("hmf.model", "/model"), ("hmf.model.hybrid_zoo", "/model/hybrid_zoo"),
                  ("hmf.model.hybrid_zoo.layers", "/model/hybrid_zoo/layers"), ("hmf.model.hybrid_zoo.models", "/model/hybrid_zoo/models")):
    m = types.ModuleType(name); m.__path__ = [R + sub]; sys.modules[name] = m    # skip hmf's LlamaFactory __init__s
def gap(name, ref, ours):
    print(name, (ref - ours).abs().max().item(), ref.abs().max().item())
''' % HMF


def _hmf_run(body, tf5=False):
    """Run hybrid-model-factory's GKA (Triton) on cuda:0 in a subprocess (TRITON_F32_DEFAULT must be
    set before its kernels compile); returns {name: (gap, scale)} printed by `gap`."""
    import os, re, subprocess, sys
    if not torch.cuda.is_available():
        pytest.skip("GKA's kernels are Triton/GPU")
    pytest.importorskip("fla")
    if not os.path.isdir(HMF):
        pytest.skip("hybrid-model-factory not cloned (python -m elephantine.fetch_upstream_refs)")
    if tf5 and not os.path.isdir(TF5):
        pytest.skip("transformers 5.18 not unpacked (python -m elephantine.fetch_upstream_refs)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(([TF5] if tf5 else []) + [root]), CUDA_VISIBLE_DEVICES="0")
    pre = (TF5_PRELUDE + 'del sys.modules["fla"]\n' if tf5 else "") + HMF_PRELUDE
    r = subprocess.run([sys.executable, "-c", pre + body], env=env, capture_output=True, text=True, timeout=900)
    out = {m[0]: (float(m[1]), float(m[2])) for m in re.findall(r"^(\w+) (\S+) (\S+)$", r.stdout, re.M)}
    assert out, r.stdout[-2000:] + r.stderr[-4000:]
    return out


def test_ssm_block_carries_gated_kalmanet():
    """Gated KalmaNet (amazon/GKA-primed-HQwen3-8B-Instruct's mixer): the real GatedKalmaNet layer
    from awslabs hybrid-model-factory, its chunked Triton prefill (two 64-token chunks) and its
    token-by-token decode kernels, against the host's ridge-readout branch: a key-covariance state
    H_t and a fixed 30-step Chebyshev solve of (H_t + 0.02 ||H_t||_F I) x = q_t feeding the GLA query."""
    out = _hmf_run('''
G = importlib.import_module("hmf.model.hybrid_zoo.layers.gated_kalmanet.gka")
C = importlib.import_module("hmf.model.hybrid_zoo.models.cache")
from elephantine.universal import gka_block
torch.manual_seed(0)
layer = G.GatedKalmaNet(hidden_size=32, head_dim=16, num_q_heads=4, num_k_heads=2, num_v_heads=2, layer_idx=0).cuda().eval()
with torch.no_grad():
    for n, p in layer.named_parameters():
        if n not in ("A_log", "dt_bias"):
            p.copy_(torch.randn_like(p) * 0.3 + (1.0 if n.endswith("norm.weight") else 0.0))
    torch.manual_seed(3)
    x = torch.randn(2, 80, 32)
    host = gka_block(layer)
    ours = host(x.double())
    gap("prefill", layer(x.float().cuda())[0].double().cpu(), ours)
    cache = C.GatedKalmaNetCache(types.SimpleNamespace(_gated_kalmanet_memory_dict={}), 80, 1)
    dec = torch.cat([layer(x[:1, t:t + 1].float().cuda(), past_key_values=cache, use_cache=True)[0].double().cpu()
                     for t in range(80)], 1)
    gap("decode", dec, ours[:1])
    host.ridge_c.zero_()
    gap("no_ridge", dec, host(x[:1].double()))
''')
    for k in ("prefill", "decode"):
        g, scale = out[k]
        assert g < 1e-4 * scale, (k, g, scale)       # fp32 floor, amplified by the solve (|x| ~ 1e3 |q| early on)
    assert out["no_ridge"][0] > 1e-1 * out["no_ridge"][1]        # the readout really matters


def test_same_host_carries_gka_hybrid_qwen3():
    """GKA-primed HQwen3 (model_type hybrid_qwen3): hybrid-model-factory's HybridQwen3ForCausalLM with
    the hybrid_override_pattern '*-GKA-GKA-*' (Qwen3 attention layers and Gated KalmaNet layers, the HF
    checkpoint's gka_config), whole-model logits on cuda:0 (fp32, IEEE dots) vs the host."""
    out = _hmf_run('''
for mod, cls in (("bmojo_f.modules", "BMojoLayer"), ("mamba.modules", "Mamba2Layer"), ("gated_deltanet.modules", "GatedDeltaNetLayer")):
    st = types.ModuleType("hmf.model.hybrid_zoo.layers." + mod); setattr(st, cls, None); sys.modules[st.__name__] = st
M = importlib.import_module("hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3")
Cf = importlib.import_module("hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3")
_ccm = M.create_causal_mask                       # written against tf 5.3; 5.18 dropped cache_position
M.create_causal_mask = lambda cache_position=None, **kw: _ccm(**kw)
from elephantine.universal import from_hybrid_qwen3
gka = dict(bp_lambda=True, chunk_size=64, conv_size=4, gla_rescale=True, norm_eps=1e-6, num_iter=30, ridge_strength=0.02,
           solver_type="chebyshev", use_alpha_connection=True, use_beta_gate=True, use_forgetting_gate=True,
           use_forgetting_gate_kk=True, use_gate=True, use_v_conv=True)
cfg = Cf.HybridQwen3Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, rope_parameters={"rope_type": "default", "rope_theta": 5000000.0},
    gka_config=gka, attn_implementation="eager")
cfg.hybrid_override_pattern = "*-GKA-GKA-*"       # set after init: tf 5.18 validates layer_types names
cfg.layer_types = cfg.hybrid_override_pattern.split("-")
torch.manual_seed(0)
t = M.HybridQwen3ForCausalLM(cfg).cuda().eval()
assert [type(l).__name__ for l in t.model.layers] == ["Qwen3DecoderLayer", "Qwen3GKADecoderLayer", "Qwen3GKADecoderLayer", "Qwen3DecoderLayer"]
with torch.no_grad():
    for n, p in t.named_parameters():
        if not n.endswith(("A_log", "dt_bias")):
            p.copy_(torch.randn_like(p) * 0.1 + (1.0 if n.endswith("norm.weight") else 0.0))
    torch.manual_seed(1)
    x = torch.randint(0, 97, (2, 80))
    ref = t(input_ids=x.cuda(), use_cache=False).logits.double().cpu()
    host = from_hybrid_qwen3(t.cpu())
    gap("logits", ref, host(x))
''', tf5=True)
    g, scale = out["logits"]
    assert g < 1e-5 * max(scale, 1), (g, scale)


VLLM_ENV = __import__("os").path.expanduser("~/.cache/vllm-env")   # python -m elephantine.fetch_upstream_refs vllm
PB27_RUNNER = '''
import json, os, sys
import torch
from vllm import LLM, SamplingParams
from vllm.config import model as mc
import vllm.model_executor.layers.fused_moe.fused_moe as fm
# vLLM 0.31 calls a model non-hybrid when every hf layer_types entry is full_attention (transformers 5
# fills Qwen3Config.layer_types), which kills the GDN state cache; PB27's GDN branch lives outside
# layer_types, so keep the class's IsHybrid flag (the plugin's dense Qwen3 class can't start otherwise).
mc.ModelConfig.is_hybrid = property(lambda self: self._model_info.is_hybrid)
gdc = fm.get_default_config
def small_tiles(*a, **k):            # fp32 fused-MoE tiles: the defaults overflow an A5000's shared memory
    c = dict(gdc(*a, **k)); c["num_stages"], c["BLOCK_SIZE_K"] = min(c.get("num_stages", 2), 2), min(c["BLOCK_SIZE_K"], 64)
    return c
fm.get_default_config = small_tiles
d, out, dtype, ids = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
llm = LLM(model=d, skip_tokenizer_init=True, dtype=dtype, enforce_eager=True, gpu_memory_utilization=0.2,
          max_model_len=256, max_num_seqs=4, max_logprobs=-1, seed=0, enable_prefix_caching=False)
CAP = []
def hook(model):
    kinds = [type(l).__name__ + ("+gdn" if hasattr(l, "gdn") else "") for l in model.model.layers]
    windows = [l.self_attn.attn.sliding_window for l in model.model.layers]
    model.logits_processor.register_forward_hook(lambda m, i, o: CAP.append(o.detach().float().cpu()))
    return kinds, windows
kinds, windows = llm.apply_model(hook)[0]
res = []
for seq in ids:
    CAP.clear()
    o = llm.generate([{"prompt_token_ids": seq}], SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=-1,
                     detokenize=False), use_tqdm=False)[0]
    lg = next(c for c in CAP if c.shape[0] == len(seq))          # the prompt-logprob pass: every position
    lp = torch.tensor([[pl[v].logprob for v in sorted(pl)] for pl in o.prompt_logprobs[1:]])
    res.append(dict(logits=lg, logprobs=lp))
torch.save(dict(res=res, kinds=kinds, windows=windows, vllm=__import__("vllm").__version__), out)
print("DONE")
'''


def _pb27_checkpoint(d, moe, window=8, seed=0):
    """A tiny random PB27Qwen3 / PB27Qwen3Moe checkpoint in the real repos' layout (config.json keys
    and `model.layers.N.gdn.*` names of defnoteatin/PseudoBrain-Qwen3-32B / -Coder-30B-A3B), weights
    rounded to bf16 and stored fp32 so the bf16 and fp32 references read identical values."""
    import json, os
    from safetensors.torch import save_file
    V, D, L, H, KV, hd, Hk, Hv, dk, dv, K = 96, 64, 4, 4, 2, 32, 2, 4, 32, 32, 4
    cfg = dict(architectures=["PB27Qwen3MoeForCausalLM" if moe else "PB27Qwen3ForCausalLM"],
               model_type="qwen3_moe" if moe else "qwen3", vocab_size=V, hidden_size=D, num_hidden_layers=L,
               num_attention_heads=H, num_key_value_heads=KV, head_dim=hd, intermediate_size=96, hidden_act="silu",
               rms_norm_eps=1e-6, rope_theta=1000000, rope_scaling=None, max_position_embeddings=4096,
               tie_word_embeddings=False, attention_bias=False, sliding_window=None, use_sliding_window=False,
               torch_dtype="float32", bos_token_id=0, eos_token_id=1, pb_hybrid_layers=[1, 3],
               pb_attention_window=window, pb_attention_sinks=4, linear_num_key_heads=Hk, linear_key_head_dim=dk,
               linear_num_value_heads=Hv, linear_value_head_dim=dv, linear_conv_kernel_dim=K, mamba_ssm_dtype="float32")
    if moe:
        cfg.update(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=48, norm_topk_prob=True,
                   decoder_sparse_step=1, mlp_only_layers=[])
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, std=0.1: torch.randn(*s, generator=g) * std
    sd = {"model.embed_tokens.weight": r(V, D, std=1.0), "model.norm.weight": 1 + r(D), "lm_head.weight": r(V, D)}
    for i in range(L):
        p = f"model.layers.{i}."
        for n, s in (("input_layernorm", D), ("post_attention_layernorm", D), ("self_attn.q_norm", hd), ("self_attn.k_norm", hd)):
            sd[p + n + ".weight"] = 1 + r(s)
        for n, s in (("q_proj", (H * hd, D)), ("k_proj", (KV * hd, D)), ("v_proj", (KV * hd, D)), ("o_proj", (D, H * hd))):
            sd[p + "self_attn." + n + ".weight"] = r(*s)
        E, f = (8, 48) if moe else (0, 96)
        mlps = [p + f"mlp.experts.{e}." for e in range(E)] if moe else [p + "mlp."]
        if moe:
            sd[p + "mlp.gate.weight"] = r(E, D, std=0.5)
        for m in mlps:
            sd[m + "gate_proj.weight"], sd[m + "up_proj.weight"], sd[m + "down_proj.weight"] = r(f, D), r(f, D), r(D, f)
        if i in cfg["pb_hybrid_layers"]:
            q, kd, vd = p + "gdn.", Hk * dk, Hv * dv
            sd[q + "in_proj_qkv.weight"], sd[q + "in_proj_z.weight"] = r(2 * kd + vd, D), r(vd, D)
            sd[q + "in_proj_b.weight"], sd[q + "in_proj_a.weight"] = r(Hv, D), r(Hv, D)
            sd[q + "conv1d.weight"] = r(2 * kd + vd, 1, K, std=0.3)
            sd[q + "A_log"] = torch.empty(Hv).uniform_(0, 2.5, generator=g).log1p()
            sd[q + "dt_bias"], sd[q + "norm.weight"], sd[q + "out_proj.weight"] = r(Hv, std=0.5), 1 + r(dv), r(D, vd)
    sd = {k: v.to(torch.bfloat16).float().contiguous() for k, v in sd.items()}
    os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    save_file(sd, os.path.join(d, "model.safetensors"))
    return cfg, sd


def _pb27_run(d, dtype, ids):
    """The pb27-vllm plugin under vLLM 0.31 (isolated venv, in-process engine, one GPU, eager). fp32
    needs `python -O`: the only fp32 blocker is the vendored fla chunk_gated_delta_rule's dtype guard
    (an assert); TRITON_F32_DEFAULT=ieee keeps its Triton dots (and TRITON_ATTN's) out of TF32."""
    import json, os, subprocess
    py = os.path.join(VLLM_ENV, "bin", "python")
    if not os.path.exists(py) or not __import__("glob").glob(os.path.join(VLLM_ENV, "lib", "python3*", "site-packages", "pb27_vllm")):
        pytest.skip("vLLM venv with the pb27-vllm plugin missing (python -m elephantine.fetch_upstream_refs vllm)")
    if not torch.cuda.is_available():
        pytest.skip("the pb27-vllm reference runs on a CUDA GPU")
    out = os.path.join(d, f"ref_{dtype}.pt")
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    if dtype == "float32":
        env["TRITON_F32_DEFAULT"] = "ieee"
    cmd = [py] + (["-O"] if dtype == "float32" else []) + ["-c", PB27_RUNNER, d, out, dtype, json.dumps(ids)]
    r = subprocess.run(cmd, env=env, cwd=d, capture_output=True, text=True, timeout=900)
    assert "DONE" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
    return torch.load(out)


@pytest.mark.parametrize("moe", [False, True], ids=["PB27Qwen3", "PB27Qwen3Moe"])
def test_same_host_carries_pseudobrain_vllm_plugin(tmp_path, moe):
    """PseudoBrain-Qwen3-32B (PB27Qwen3ForCausalLM) / -Qwen3-Coder-30B-A3B (PB27Qwen3MoeForCausalLM): their
    only runnable code is the pb27-vllm plugin shipped in the repos, run here under vLLM 0.31 on a tiny
    random checkpoint. Qwen3 (-MoE) whose every attention layer gets a W = pb_attention_window sliding
    window, plus on pb_hybrid_layers a parallel Qwen3.5 gated-DeltaNet on the same normed input, summed
    with attention (host: seq = 0). W = 8 < T so the window is exercised on every layer."""
    from elephantine.universal import from_pseudobrain
    d = str(tmp_path)
    cfg, sd = _pb27_checkpoint(d, moe)
    ids = [[5, 17, 3, 88, 42, 9, 61, 2, 33, 70, 12, 7, 50, 91, 4, 26, 38, 19, 77, 64, 1, 2, 3, 4],
           [9, 9, 1, 40, 23, 80, 6, 55, 13, 2, 71, 44, 30, 8]]
    r32, r16 = _pb27_run(d, "float32", ids), _pb27_run(d, "bfloat16", ids)
    assert r32["windows"] == [8] * 4 and [k.endswith("+gdn") for k in r32["kinds"]] == [False, True, False, True]
    host = from_pseudobrain(cfg, sd)
    with torch.no_grad():
        ours = [host(torch.tensor([s]))[0] for s in ids]
    gap = lambda ref: max((x["logits"].double() - y).abs().max().item() for x, y in zip(ref["res"], ours))
    scale = max(x["logits"].abs().max().item() for x in r32["res"])
    lp_gap = max((x["logprobs"].double() - y.log_softmax(-1)[:-1]).abs().max().item() for x, y in zip(r32["res"], ours))
    noise = max((a["logits"] - b["logits"]).abs().max().item() for a, b in zip(r32["res"], r16["res"]))
    print(f"fp32 gap {gap(r32):.2e} (logprobs {lp_gap:.2e}), bf16 gap {gap(r16):.2e}, reference bf16-vs-fp32 {noise:.2e}, scale {scale:.2f}")
    assert gap(r32) < 2e-5 * max(scale, 1) and lp_gap < 2e-5 * max(scale, 1), (gap(r32), lp_gap, scale)
    assert gap(r16) < 2 * noise, (gap(r16), noise)            # the stock bf16 path: at its own rounding level
    for l in host.layers:                                     # the window (on all layers) is exercised
        l.attn.window.fill_(float("inf"))
    with torch.no_grad():
        assert max((x["logits"].double() - host(torch.tensor([s]))[0]).abs().max().item()
                   for s, x in zip(ids, r32["res"])) > 1e-2


def test_sink_key_branch():
    """UAttention.null_key (UConfig.sink_key; release: IQuest-Q1's learned sink KEY): a zero-value extra key per head
    whose logit is q . s_h x scale, against a hand-written reference; s = 0 is the zero-logit sink (null_logit 0)."""
    from elephantine.universal import UAttention, UConfig
    torch.manual_seed(0)
    a = UAttention(UConfig(vocab=11, d=8, L=1, heads=2, head_dim=4, sink_key=True)).to(F64)
    with torch.no_grad():
        for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            lin.weight.copy_(torch.randn_like(lin.weight) * 0.5)
            if lin.bias is not None:
                lin.bias.zero_()
        a.null_key.copy_(torch.randn_like(a.null_key))
    x = torch.randn(2, 5, 8, dtype=F64)
    B, T, H, hd = 2, 5, 2, 4
    q, k, v = (f(x).view(B, T, H, hd) for f in (a.q_proj, a.k_proj, a.v_proj))
    s = torch.einsum("bthd,bshd->bhts", q, k) * a.scale
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    s = torch.cat((s, torch.einsum("bthd,hd->bht", q, a.null_key)[..., None] * a.scale), -1)
    ref = a.o_proj(torch.einsum("bhts,bshd->bthd", s.softmax(-1)[..., :T], v).reshape(B, T, -1))
    assert torch.allclose(a(x, causal=True), ref, atol=1e-12)
    b = UAttention(UConfig(vocab=11, d=8, L=1, heads=2, head_dim=4)).to(F64)     # no sink key: a scalar sink at 0
    with torch.no_grad():
        for lb, la in ((b.q_proj, a.q_proj), (b.k_proj, a.k_proj), (b.v_proj, a.v_proj), (b.o_proj, a.o_proj)):
            lb.weight.copy_(la.weight)
        a.null_key.zero_()
        b.null_logit.fill_(0.0)
    assert torch.allclose(a(x, causal=True), b(x, causal=True), atol=1e-12)
