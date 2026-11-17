"""GLM-5.3-SuperExpert-V2 on the universal host without running its remote code: the host is compared at fp64
with tests/refs/glm_superexpert.py, a REFERENCE transcription of the repo's configuration / modelling files
(read as text; the header names the snapshot, sha256 and line ranges).

Run:  pytest tests/test_notremote_glm_superexpert.py -q
"""
import torch

from tests.test_universal_elephantine import _ids, _rand_

F64 = torch.float64


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1)


def test_same_host_carries_glm_superexpert_own_code_notremote():
    """GLM-5.3-SuperExpert-V2: a dense NoPE MHA decoder WITHOUT a causal mask, 4-bit packed weights and a dense
    softmax mixture of all experts (plus a shared expert). The host runs it with causal=False and the dequantised
    weights. Same config, seed, random packed nibbles and scales as the remote-code test; the reference is built
    in fp32 (as the remote model was), then lifted to fp64 (its embedding / RMSNorm fp32 casts lifted, see the
    reference header), so the bound is fp64-clean: 1e-12 relative.
    Branch checks: causal=True on the host moves the output, and the reference under an additive causal mask
    (the remote forward's attention_mask) agrees with it; restricting the host's all-expert mixture to the top-2
    experts (no renorm) moves the output."""
    from elephantine.universal import from_glm_superexpert
    from tests.refs import glm_superexpert as gm
    cfg = gm.GLMSuperExpertConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
                                  num_hidden_layers=3, num_attention_heads=4, n_routed_experts=4, n_shared_experts=1,
                                  first_k_dense_replace=1, pad_token_id=0, eos_token_id=[1])
    torch.manual_seed(0)
    t = gm.GLMSuperExpertForCausalLM(cfg).eval()
    with torch.no_grad():
        for mod in t.modules():                                  # random packed nibbles and scales
            if isinstance(mod, (gm.DequantizedPackedLinear, gm.DequantizedPackedEmbedding)):
                mod.weight.copy_(torch.randint(0, 256, mod.weight.shape, dtype=torch.uint8))
                mod.scale.fill_(0.3)
    _rand_(t)
    t = t.to(F64)
    assert t.model.language_model.embed_tokens.weight.dtype == torch.uint8
    kinds = [type(l.mlp).__name__ for l in t.model.language_model.layers]
    assert kinds == ["GLMDenseMLP", "GLMSuperExpertMoE", "GLMSuperExpertMoE"], kinds
    x = _ids(97)
    host = from_glm_superexpert(t)
    with torch.no_grad():
        ref = t(x).logits
        assert ref.dtype == F64
        gap = _rel(ref, host(x, causal=False))
        assert gap < 1e-12, gap
        # it really is bidirectional; and the reference contains the mask path the host switches
        T = x.shape[1]
        mask = torch.full((T, T), float("-inf"), dtype=F64).triu(1)[None, None]
        ref_c = t(x, attention_mask=mask).logits
        hc = host(x, causal=True)
        assert _rel(ref, hc) > 1e-3
        assert _rel(ref_c, hc) < 1e-12
        # the dense all-expert softmax mixture: top-2 only (unrenormalised) is a different model
        moe = [ul.ffn for ul, tl in zip(host.layers, t.model.language_model.layers) if hasattr(tl.mlp, "super_experts")]
        assert moe and all(f.top_k == cfg.n_routed_experts for f in moe)
        for f in moe:
            f.top_k = 2
        assert _rel(ref, host(x, causal=False)) > 1e-3
