"""make_elephantine — build an elephantine that reproduces an ARBITRARY decoder LM
token-for-token, by embedding its (smaller / "compressed") tensors into the bigger hybrid
architecture.

This generalizes `build_elephantine.py` (which hardcoded one 2-layer target) to any
RMSNorm + SiLU + dense-SwiGLU decoder (Llama / Mistral / Qwen2 / SmolLM / …). The
elephantine is a Jamba hybrid sized to the target and weighted so that:

  * each of the target's L layers is reproduced by one elephantine **attention** layer
    (q/k/v/o copied; GQA carried directly — Jamba supports `num_key_value_heads`);
  * the target's dense **SwiGLU MLP** is reproduced by a tied **MoE**: every expert is a
    copy of the MLP, `top_k == num_experts`, and the router is zeroed, so the softmax
    weights are uniform and sum to exactly 1 → the block computes that one MLP;
  * the interleaved **Mamba (SSM)** layers are the identity (mixer `out_proj` and the
    block's FFN `down_proj` zeroed → pure passthrough);
  * embeddings / `lm_head` / norms are copied straight over.

Jamba attention carries no positional encoding, so the target's **RoPE is zeroed first**
(NoPE is the `inv_freq → 0` special case of RoPE, see `hf/nope_rope.py`). The elephantine
then reproduces the (NoPE) target's whole forward pass exactly — the "big model" running
the small model's compressed tensors.

  python -m elephantine.make_elephantine HuggingFaceTB/SmolLM2-135M --out ./eleph
  python -m elephantine.make_elephantine <hub-id-or-local-path> --experts 4

The target loads through its stock transformers class; --trust-remote-code is needed (and runs the repo's own
code) only for a model whose class is not in transformers.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, JambaConfig
from transformers.models.jamba.modeling_jamba import JambaForCausalLM


def nope_ize(target):
    """Zero the target's rotary frequencies (NoPE ⊆ RoPE) so a positionless hybrid can
    reproduce it exactly. In-place; returns the target."""
    re = getattr(getattr(target, "model", target), "rotary_emb", None)
    if re is not None and hasattr(re, "inv_freq"):
        z = torch.zeros_like(re.inv_freq)
        re.inv_freq = z
        if hasattr(re, "original_inv_freq"):
            re.original_inv_freq = z
    return target


def elephantine_config(target, num_experts=4, ffn_mult=1, depth_mult=2, efficient=False):
    """Size the elephantine to the target, optionally INFLATED to a bigger model:
      num_experts  -- tied MoE experts (a power of 2 keeps it exact); more = bigger FFN.
      ffn_mult     -- widen each expert by this factor (extra units zero-padded, inert).
      depth_mult   -- layers per attention block: 1 attention + (depth_mult-1) identity
                      Mamba layers; more = deeper. depth_mult=2 is the minimal hybrid.
      efficient    -- route to ONE expert (top_k=1) instead of all E, avoiding the redundant
                      tied-expert compute; the kept expert's down_proj is scaled by E so the
                      1/E router weight still lands on the exact MLP. Same outputs, ~E x cheaper FFN."""
    c = target.config
    heads = c.num_attention_heads
    head_dim = getattr(c, "head_dim", None) or (c.hidden_size // heads)
    if head_dim * heads != c.hidden_size:
        raise ValueError(f"this tool needs head_dim*heads == hidden "
                         f"({head_dim}*{heads} != {c.hidden_size})")
    L, P = c.num_hidden_layers, max(2, depth_mult)
    return JambaConfig(
        hidden_size=c.hidden_size, num_hidden_layers=P * L,      # P layers/group: 1 attn + (P-1) mamba
        num_attention_heads=heads, num_key_value_heads=getattr(c, "num_key_value_heads", heads),
        intermediate_size=ffn_mult * c.intermediate_size, vocab_size=c.vocab_size,
        attn_layer_period=P, attn_layer_offset=P - 1,            # attention = last layer of each group
        expert_layer_period=P, expert_layer_offset=P - 1,        # MoE on the attention layers
        num_experts=num_experts,
        num_experts_per_tok=(1 if efficient else num_experts),   # top_k==E → weights sum to 1; efficient → 1
        rms_norm_eps=getattr(c, "rms_norm_eps", 1e-5), hidden_act="silu",
        use_mamba_kernels=False, attn_implementation="eager", tie_word_embeddings=False,
        max_position_embeddings=getattr(c, "max_position_embeddings", 2048),
        mamba_proj_bias=False,
    )


@torch.no_grad()
def make_elephantine(target, num_experts=4, ffn_mult=1, depth_mult=2, efficient=False):
    """Return (elephantine, config) whose forward reproduces the (NoPE) target — optionally
    INFLATED (num_experts/ffn_mult/depth_mult) or run EFFICIENTLY (efficient → top_k=1, ~E x
    cheaper FFN), still token-for-token. Call `nope_ize(target)` first if the target uses RoPE."""
    cfg = elephantine_config(target, num_experts, ffn_mult, depth_mult, efficient)
    dscale = cfg.num_experts if efficient else 1     # top_k=1 router weight is 1/E → scale the MLP by E
    eleph = JambaForCausalLM(cfg).to(next(target.parameters()).dtype).eval()
    tsd, em, tm = target.state_dict(), eleph.model, target.model
    em.embed_tokens.weight.copy_(tm.embed_tokens.weight)
    lm = tm.embed_tokens.weight if target.config.tie_word_embeddings else target.lm_head.weight
    eleph.lm_head.weight.copy_(lm)
    em.final_layernorm.weight.copy_(tm.norm.weight)

    it = target.config.intermediate_size                          # target MLP / expert width
    bt = cfg.layers_block_type
    attn_layers = [i for i, t in enumerate(bt) if t == "attention"]
    if len(attn_layers) != target.config.num_hidden_layers:
        raise RuntimeError("elephantine attention-layer count != target layer count")
    for i, layer in enumerate(em.layers):
        if bt[i] == "mamba":                                      # SSM block → identity
            layer.mamba.out_proj.weight.zero_()
            if layer.mamba.out_proj.bias is not None:
                layer.mamba.out_proj.bias.zero_()
            layer.feed_forward.down_proj.weight.zero_()           # plain-MLP block → 0
        else:                                                     # attention block → target layer t
            t, p = attn_layers.index(i), f"model.layers.{attn_layers.index(i)}."
            for a in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(layer.self_attn, a).weight.copy_(tsd[p + f"self_attn.{a}.weight"])
            layer.input_layernorm.weight.copy_(tsd[p + "input_layernorm.weight"])
            layer.pre_ff_layernorm.weight.copy_(tsd[p + "post_attention_layernorm.weight"])
            for e in layer.feed_forward.experts:                  # tie every expert to the MLP
                # zero-pad to the (possibly wider) expert width: the extra up-proj rows are 0,
                # so SiLU(gate)*up = 0 there and the down-proj adds nothing → inert padding.
                e.gate_proj.weight.zero_(); e.gate_proj.weight[:it].copy_(tsd[p + "mlp.gate_proj.weight"])
                e.up_proj.weight.zero_();   e.up_proj.weight[:it].copy_(tsd[p + "mlp.up_proj.weight"])
                e.down_proj.weight.zero_(); e.down_proj.weight[:, :it].copy_(dscale * tsd[p + "mlp.down_proj.weight"])
            layer.feed_forward.router.weight.zero_()              # uniform router → one dense MLP
    return eleph, cfg


@torch.no_grad()
def make_hybrid_elephantine(target, ffn_mult=1):
    """Emulate a HYBRID (Jamba-family) target token-for-token by CARRYING its Mamba-SSM
    recurrences — not the softmax path. Each target layer maps to the same elephantine
    layer: Mamba mixers are copied whole (the SSM recurrence is carried, NOT set to
    identity), attention layers copy q/k/v/o, and the FFN (MoE or dense) is copied and
    optionally widened (ffn_mult, extra units zero-padded and inert). This is the
    construction behind the '50 Mamba-family hybrids' coverage claim: a bigger hybrid host
    runs a smaller hybrid's compressed tensors. Requires a `JambaForCausalLM` target."""
    tc = target.config
    it = tc.intermediate_size
    eleph = JambaForCausalLM(JambaConfig(**{**tc.to_dict(), "intermediate_size": ffn_mult * it,
                                            "use_mamba_kernels": False,
                                            "attn_implementation": "eager"})).to(
        next(target.parameters()).dtype).eval()
    eleph.model.embed_tokens.weight.copy_(target.model.embed_tokens.weight)
    eleph.lm_head.weight.copy_(target.lm_head.weight)
    eleph.model.final_layernorm.weight.copy_(target.model.final_layernorm.weight)
    for el, tl in zip(eleph.model.layers, target.model.layers):
        el.input_layernorm.weight.copy_(tl.input_layernorm.weight)
        el.pre_ff_layernorm.weight.copy_(tl.pre_ff_layernorm.weight)
        if hasattr(tl, "mamba"):
            el.mamba.load_state_dict(tl.mamba.state_dict())      # CARRY the SSM recurrence
        if hasattr(tl, "self_attn"):
            el.self_attn.load_state_dict(tl.self_attn.state_dict())
        fe, ft = el.feed_forward, tl.feed_forward
        pairs = zip(fe.experts, ft.experts) if hasattr(ft, "experts") else [(fe, ft)]
        for ee, et in pairs:                                     # copy FFN, zero-pad to the wider width
            ee.gate_proj.weight.zero_(); ee.gate_proj.weight[:it].copy_(et.gate_proj.weight)
            ee.up_proj.weight.zero_();   ee.up_proj.weight[:it].copy_(et.up_proj.weight)
            ee.down_proj.weight.zero_(); ee.down_proj.weight[:, :it].copy_(et.down_proj.weight)
        if hasattr(ft, "router"):
            fe.router.weight.copy_(ft.router.weight)
    return eleph, eleph.config


@torch.no_grad()
def emulate_hybrid(target, ffn_mult=1, seq=8, seed=123):
    """Build a hybrid target's elephantine (Mamba carried) and return (eleph, cfg, max|Δ|)."""
    eleph, cfg = make_hybrid_elephantine(target, ffn_mult)
    torch.manual_seed(seed)
    ids = torch.randint(0, target.config.vocab_size, (2, seq))
    d = (target(input_ids=ids).logits - eleph(input_ids=ids).logits).abs().max().item()
    return eleph, cfg, d


@torch.no_grad()
def emulate_diffusion(target, kv=None, inter=None, layers=None, hidden=None, seq=8, seed=123):
    """Emulate a DIFFUSION LM's forward pass token-for-token. Diffusion LMs (LLaDA2,
    Nemotron-Labs-Diffusion, diffusion-Gemma) are standard transformers + MoE trained with a
    masked objective, so their forward uses FULL (bidirectional) attention -- each position
    predicts from BOTH sides. The exact same capacity recipe as the AR elephantine reproduces
    them (KV carry / MoE tie / FFN width / depth / hidden width are all mask-agnostic); the
    only change is you run it with a full attention mask instead of a causal one.

    This is a SEPARATE per-paradigm join (the `gen` hard wall): the causal Jamba elephantine
    cannot cross to diffusion (its Mamba layers are inherently causal), and the diffusion
    *decode* -- iterative denoise / mask-predict over steps -- is the inference loop you run
    the identical weights with, not part of the weight construction. Returns (host, max|Δ|)
    over the bidirectional forward."""
    from elephantine.hidden_width import convert
    host = convert(target, kv=kv, inter=inter, layers=layers, hidden=hidden)   # capacity-widened, dense
    torch.manual_seed(seed)
    dt = next(target.parameters()).dtype
    ids = torch.randint(0, target.config.vocab_size, (1, seq))
    full = torch.zeros(1, 1, seq, seq, dtype=dt)                               # bidirectional (no causal)
    d = (target(ids, attention_mask=full).logits
         - host(ids, attention_mask=full).logits).abs().max().item()
    return host, d


@torch.no_grad()
def emulate_universal(target, causal=True, kv=None, inter=None, layers=None, hidden=None, seq=8, seed=123):
    """ONE full-attention host, universal across BOTH generation paradigms. The gen 'wall' is a
    DECODE CONTRACT, not an architecture wall: run the same host with a CAUSAL mask to reproduce an
    autoregressive target (causal attention is just the full-attention host masked down --
    local <= full, a validated reduction), or with a FULL mask to reproduce a diffusion target's
    bidirectional forward. Same weights-recipe, same architecture; only the mask (and the outer
    decode loop -- left-to-right vs denoise) changes. Add gated-delta SSM layers (idle here, since
    softmax/diffusion targets don't use them) and the one host also carries the AR recurrences.
    So a SINGLE universal model spans AR + diffusion. Returns (host, max|Δ|)."""
    from elephantine.hidden_width import convert
    host = convert(target, kv=kv, inter=inter, layers=layers, hidden=hidden)
    torch.manual_seed(seed)
    dt = next(target.parameters()).dtype
    ids = torch.randint(0, target.config.vocab_size, (1, seq))
    mask = (torch.triu(torch.full((1, 1, seq, seq), float("-inf"), dtype=dt), 1) if causal   # AR
            else torch.zeros(1, 1, seq, seq, dtype=dt))                                       # diffusion
    d = (target(ids, attention_mask=mask).logits
         - host(ids, attention_mask=mask).logits).abs().max().item()
    return host, d


@torch.no_grad()
def emulate(target, num_experts=4, ffn_mult=1, depth_mult=2, efficient=False, seq=8, seed=123):
    """NoPE-reduce the target, build its (optionally inflated / efficient) elephantine, and
    return (elephantine, cfg, max|Δ logits|) on a random batch."""
    nope_ize(target)
    eleph, cfg = make_elephantine(target, num_experts, ffn_mult, depth_mult, efficient)
    torch.manual_seed(seed)
    ids = torch.randint(0, target.config.vocab_size, (2, seq))
    d = (target(input_ids=ids).logits - eleph(input_ids=ids).logits).abs().max().item()
    return eleph, cfg, d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Embed an arbitrary decoder LM into a Jamba elephantine.")
    ap.add_argument("target", help="hub id or local path of a Llama-family decoder LM")
    ap.add_argument("--experts", type=int, default=4, help="tied MoE experts (a power of 2 stays exact)")
    ap.add_argument("--ffn-mult", type=int, default=1, help="INFLATE: widen each expert this many x (zero-padded)")
    ap.add_argument("--depth-mult", type=int, default=2, help="INFLATE: layers per attention block (>=2)")
    ap.add_argument("--efficient", action="store_true", help="route to 1 expert (top_k=1), ~E x cheaper FFN")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64", "bfloat16"])
    ap.add_argument("--out", default=None, help="save the elephantine (save_pretrained) here")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="let transformers import and RUN the target repo's own modeling code (off by default: only stock transformers classes load)")
    args = ap.parse_args(argv)
    dt = getattr(torch, args.dtype)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dt,
                                                  trust_remote_code=args.trust_remote_code).eval()
    eleph, cfg, d = emulate(target, args.experts, args.ffn_mult, args.depth_mult, args.efficient)
    tn = sum(p.numel() for p in target.parameters())
    en = sum(p.numel() for p in eleph.parameters())
    print(f"target {args.target}: {tn:,} params  ->  elephantine (Jamba hybrid): {en:,} params "
          f"({en / max(tn, 1):.1f}x)")
    print(f"inflate: experts={args.experts} ffn_mult={args.ffn_mult} depth_mult={args.depth_mult}")
    print(f"layer schedule: {cfg.layers_block_type}")
    print(f"max|Δ logits| ({args.dtype}, NoPE-reduced) = {d:.3e}")
    if args.out:
        eleph.save_pretrained(args.out)
        print(f"saved elephantine -> {args.out}")


if __name__ == "__main__":
    main()
