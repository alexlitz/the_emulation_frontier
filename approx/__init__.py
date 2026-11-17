"""approx/ -- APPROXIMATE, performance-preserving architecture conversion.

The rest of this repo builds EXACT emulations (a smaller model's function inside a bigger host's
stock code). When the target architecture is not a superset -- here gpt-oss-20b cannot emulate
Gemma-4-26B-A4B (python -m graph.reduce: kv / q capacity and depth FAIL, GeLU and softcap are
approximate) -- the next best thing is an initialization of the target's weights that starts the
student as close to the teacher as possible, as a basis for distillation.

  plan.py        mismatch analysis (graph.reduce slot/axis gaps + every structural difference) ->
                 per-mismatch EXACT fold / APPROX step
  arch.py        reference forwards of both architectures on weight dicts (streamable layer by
                 layer; checked against the stock HF modules)
  importance.py  teacher passes on a (domain) calibration set: residual stream, per-layer ablation
                 dKL, gradient x activation attribution (heads, experts, expert units, dense units),
                 routing frequency / mass / co-activation
  convert.py     the conversion (sequential, layer by layer):
                   closed form  - exact folds (padding, embed scale, layer_scalar, norm weights,
                                  attention scale, router input scale, per-expert scale, up bias)
                                - attention: rope-pair selection by attention-weighted score
                                  variance + Hungarian assignment to the student's frequencies,
                                  GQA-aware (K shared per group), V by output-weighted PCA, spare
                                  q-heads as a temperature ensemble
                                - layers: 6 dropped by ablation dKL under a type constraint (DP)
                                - FFN: 128 -> 32 experts grouped by (excess) co-activation, units
                                  kept by importance, the dense MLP replicated into every expert
                                  (a unit in all experts is dense since top-k weights sum to 1),
                                  router = LS onto the log-sum-exp of member logits, GeLU ->
                                  swiglu-oai by a per-unit gate scale fitted with the clamp
                   calibration  - LS refits on the student's own stream against the teacher's
                                  residual stream (absorbs dropped layers and upstream drift):
                                  q/k/v (affine, replaces the q/k/v RMSNorms), o_proj, router,
                                  per-expert down_proj, per-channel FFN gain/bias; centered, with
                                  every ridge chosen on held-out calibration sequences (without
                                  this the refits overfit in a cascade: held-out KL 15-18)
                                - optional gradient block reconstruction of the FFN (approximates
                                  Gemma's per-token RMS normalization of its FFN output, which no
                                  linear map can)
  evaluate.py    KL(teacher || student), top-1 agreement, perplexities, streamed; checkpoint writer
  distill.py     logit-KL distillation (LoRA on attention [+ experts], int8-frozen experts,
                 2-GPU pipeline through the CPU)
  data.py        corpora (wikitext-103; Python sources split by package)
  real.py        the Gemma-4-26B-A4B -> gpt-oss-20b driver
  tiny.py        a tiny pair with the same mismatch structure (tests/test_approx.py)

Results: see the "Approximate conversion" section of README.md.
"""
