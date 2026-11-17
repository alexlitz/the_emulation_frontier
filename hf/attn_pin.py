"""The attention implementation every agreement / measurement harness pins: eager, with one exception.

transformers 4.57 picks `sdpa` by default for every family that supports it, and two families' sdpa paths
silently compute a different model:
  * GPT-2: the sdpa path never applies `scale_attn_by_inverse_layer_idx` (eager divides layer l's logits
    by l + 1) nor `scale_attn_weights = False` (sdpa always scales by 1 / sqrt(d)); Mistral's GPT-2
    checkpoints (stanford-crfm/*) set the first;
  * Gemma-2 (and any Gemma-3 config with attn_logit_softcapping): the eager path applies
    tanh softcapping of the attention logits, sdpa drops the `softcap` argument. Every released Gemma-2
    checkpoint sets attn_logit_softcapping = 50.
so `attn_implementation="eager"` is pinned. The exception:
  * Falcon with ALiBi (falcon-rw-*, `alibi: true`): `_update_causal_mask` already folds alibi / sqrt(d)
    into the mask, and the eager branch adds alibi again before scaling, so eager counts ALiBi twice;
    sdpa (the mask alone) is the correct model.
hf/attn_impl_audit.py measures eager vs sdpa on every model the harnesses recorded (hf/data/attn_impl_audit.jsonl,
2026-11-13): identical models (100% / 100%, max |dlogp| at the fp32 rounding level) on all of them except
gemma-2-2b / -2b-it (sdpa: 99.74% top-1, 92 / 96% identical, max |dlogp| 1.6 / 0.45); and on the two bug
demonstrators, alias-gpt2-small-x21 (66% top-1, 0% identical) and falcon-rw-1b (96%, 33%). Re-running the
recorded Gemma-2 measurements under eager (hf/data/eager_reruns.jsonl) flips one verdict: act_swap's
gemma-2-2b erf-GeLU fit, 87.5% -> 91.7% identical (FAIL -> PASS); act_width's Gemma-2 rows keep theirs.

Note for fp64 harnesses: the eager Llama-style paths compute the softmax in fp32
(softmax(dtype=float32).to(query.dtype)), so an fp64 run under eager is fp32 at the softmax (HF RMS
norms upcast to fp32 only, too). hf/partial_rope_host.py and the bos-token constructions use sdpa for that
reason; that is safe for the families they load (Llama, Qwen2, GLM, OPT, GPT-2 without layer-idx scaling).
"""
from __future__ import annotations

EAGER = "eager"


def attn_impl(config) -> str:
    """The implementation that computes the checkpoint's own model."""
    if getattr(config, "model_type", None) == "falcon" and getattr(config, "alibi", False):
        return "sdpa"
    return EAGER


def attn_impl_for(model_id, trust_remote_code=False) -> str:
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        return EAGER
    return attn_impl(cfg)
