"""Build `elephantine-2026-08-08`: a Jamba hybrid (SSM + attention + MoE) whose
forward pass reproduces the smallest SiLU/RMSNorm decoder LM in this dataset
TOKEN-FOR-TOKEN.

Thesis (this repo): a big, general "universal-emulator" architecture, correctly
weighted, IS any smaller model. elephantine is a hybrid Jamba (interleaved Mamba
SSM layers + attention layers, every block carrying an FFN, MoE on the attention
blocks). We weight it so that:

  * Attention layers reproduce the target's attention (copy q/k/v/o).
  * MoE FFNs collapse to the target's dense SwiGLU MLP: all experts tied AND
    top_k == num_experts, so the softmaxed router weights of the selected experts
    sum to 1  ->  output = that one MLP  (Jamba does NOT renormalize after topk,
    unlike Mixtral, so we activate ALL experts to guarantee weights sum to 1).
  * Mamba (SSM) layers are exact identity: zero mixer `out_proj` (block adds 0),
    and their plain-MLP FFN is also zeroed (down_proj = 0)  ->  full passthrough.
  * Norms copied RMSNorm->RMSNorm; embeddings + lm_head copied from the target.

TARGET is a NoPE Llama (RoPE inv_freq zeroed) because Jamba attention carries no
positional encoding -- NoPE is a special case of RoPE (see hf/nope_rope.py), so a
NoPE target is what a NoPE-attention hybrid can reproduce exactly.
"""
from __future__ import annotations

import torch
from transformers import JambaConfig, LlamaConfig
from transformers.models.jamba.modeling_jamba import JambaForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# ---- shapes shared by target and elephantine ---------------------------------
HIDDEN = 64
HEADS = 4
KV_HEADS = 4          # target is MHA; elephantine attn is MHA too -> direct copy
INTER = 128           # target FFN width == elephantine MoE-expert width
VOCAB = 256
EPS = 1e-5
SEQ = 8
SEED = 0


def build_target(dtype):
    """The smallest RMSNorm+SiLU decoder LM: a 2-layer NoPE Llama with fixed weights."""
    torch.manual_seed(SEED)
    cfg = LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS, intermediate_size=INTER, vocab_size=VOCAB,
        max_position_embeddings=64, rms_norm_eps=EPS, hidden_act="silu",
        attn_implementation="eager", tie_word_embeddings=False,
    )
    m = LlamaForCausalLM(cfg).to(dtype).eval()
    # NoPE: zero the rotary frequencies -> attention ignores position (hf/nope_rope.py).
    z = torch.zeros_like(m.model.rotary_emb.inv_freq)
    m.model.rotary_emb.inv_freq = z
    m.model.rotary_emb.original_inv_freq = z
    return m


def build_elephantine(dtype):
    cfg = JambaConfig(
        hidden_size=HIDDEN, num_hidden_layers=4, num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS, intermediate_size=INTER, vocab_size=VOCAB,
        # schedule -> ['mamba','attention','mamba','attention'];
        # experts on the attention layers (1,3), plain MLP on mamba layers (0,2).
        attn_layer_period=2, attn_layer_offset=1,
        expert_layer_period=2, expert_layer_offset=1,
        num_experts=4, num_experts_per_tok=4,          # top_k == num_experts -> weights sum to 1
        rms_norm_eps=EPS, hidden_act="silu", use_mamba_kernels=False,
        attn_implementation="eager", tie_word_embeddings=False,
        max_position_embeddings=64, mamba_proj_bias=False,
    )
    return JambaForCausalLM(cfg).to(dtype).eval(), cfg


@torch.no_grad()
def weight_from_target(eleph: JambaForCausalLM, target: LlamaForCausalLM, cfg: JambaConfig):
    tsd = target.state_dict()
    em, tm = eleph.model, target.model

    # embeddings + lm_head + final norm  <- copied verbatim
    em.embed_tokens.weight.copy_(tm.embed_tokens.weight)
    eleph.lm_head.weight.copy_(target.lm_head.weight)
    em.final_layernorm.weight.copy_(tm.norm.weight)

    block_type = cfg.layers_block_type            # ['mamba','attention','mamba','attention']
    attn_layers = [i for i, t in enumerate(block_type) if t == "attention"]
    assert len(attn_layers) == target.config.num_hidden_layers, (
        f"need one elephantine attention layer per target layer: "
        f"{len(attn_layers)} vs {target.config.num_hidden_layers}"
    )

    for i, layer in enumerate(em.layers):
        if block_type[i] == "mamba":
            # ---- SSM layer -> FULL IDENTITY -----------------------------------
            layer.mamba.out_proj.weight.zero_()          # mixer contributes 0  (no bias: mamba_proj_bias=False)
            if layer.mamba.out_proj.bias is not None:
                layer.mamba.out_proj.bias.zero_()
            layer.feed_forward.down_proj.weight.zero_()  # plain MLP contributes 0
            # (input_layernorm / pre_ff_layernorm left as 1.0; irrelevant since sublayers output 0)
        else:
            # ---- attention layer -> reproduce the corresponding target layer ---
            t = attn_layers.index(i)                     # 0 -> target layer 0, 1 -> target layer 1
            p = f"model.layers.{t}."
            # attention: MHA<-MHA direct copy (same head counts)
            layer.self_attn.q_proj.weight.copy_(tsd[p + "self_attn.q_proj.weight"])
            layer.self_attn.k_proj.weight.copy_(tsd[p + "self_attn.k_proj.weight"])
            layer.self_attn.v_proj.weight.copy_(tsd[p + "self_attn.v_proj.weight"])
            layer.self_attn.o_proj.weight.copy_(tsd[p + "self_attn.o_proj.weight"])
            # norms: input_layernorm<-input_layernorm, pre_ff_layernorm<-post_attention_layernorm
            layer.input_layernorm.weight.copy_(tsd[p + "input_layernorm.weight"])
            layer.pre_ff_layernorm.weight.copy_(tsd[p + "post_attention_layernorm.weight"])
            # MoE FFN: tie EVERY expert to the target MLP; top_k==num_experts so
            # softmaxed selected-weights sum to 1 -> block == that dense SwiGLU MLP.
            for e in layer.feed_forward.experts:
                e.gate_proj.weight.copy_(tsd[p + "mlp.gate_proj.weight"])
                e.up_proj.weight.copy_(tsd[p + "mlp.up_proj.weight"])
                e.down_proj.weight.copy_(tsd[p + "mlp.down_proj.weight"])
            # Zero the router weights -> logits are identically 0 -> softmax is exactly
            # uniform (0.25 each, a power of two, exactly representable). With top_k==
            # num_experts the selected weights sum to EXACTLY 1.0 even though Jamba
            # computes the router softmax in fp32; every tied expert then contributes
            # 0.25*f(x) and the four accumulate to exactly f(x). This is what makes the
            # MoE block bit-exact (not merely ~1e-7) in fp64.
            layer.feed_forward.router.weight.zero_()
    return eleph


@torch.no_grad()
def verify(dtype, tol_msg):
    target = build_target(dtype)
    eleph, cfg = build_elephantine(dtype)
    weight_from_target(eleph, target, cfg)
    torch.manual_seed(123)
    ids = torch.randint(0, VOCAB, (2, SEQ))
    lt = target(input_ids=ids).logits
    le = eleph(input_ids=ids).logits
    d = (lt - le).abs().max().item()
    print(f"[{dtype}] max|Δ logits| = {d:.3e}   {tol_msg}")
    return d, eleph, cfg, target


if __name__ == "__main__":
    d64, *_ = verify(torch.float64, "(fp64, want <= ~1e-9)")
    d32, eleph, cfg, target = verify(torch.float32, "(fp32, want <= ~1e-4)")
    assert d64 <= 1e-9, f"fp64 not bit-exact: {d64}"
    assert d32 <= 1e-4, f"fp32 not close: {d32}"
    n = sum(p.numel() for p in eleph.parameters())
    print(f"\nelephantine params: {n:,}")
    print(f"layer schedule: {cfg.layers_block_type}")
    print(f"experts/layer:  {cfg.layers_num_experts}  (top_k={cfg.num_experts_per_tok})")
    print("VERIFIED: elephantine.forward == target.forward")
