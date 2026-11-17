# M12: the whole-model limit theorems, with measured constants

The Lean library (`proofs/lean`: `ModelLimit.lean`, `LimitKit.lean`,
`LimitInstances.lean`, `LimitInstances2.lean`, `EpsLever.lean`; proofs/PROOFS.md, "Limits at the whole-model level")
proves two things for a stack of layers:

* the logit error is at most `Lh · Σ_i ε_i Π_{j>i} L_j ≤ Lh · (Π L_i)(Σ ε_i)` (`model_error_bound_head`,
  `errBound_le_prod_sum`);
* greedy text is identical once `2 · Lh · (Π L)(Σ ε) < δ`, where δ is the target's top-1 margin along its own
  greedy trajectory (`greedy_of_bound`, `greedy_of_layer_bounds(_on)`).

Each limit construction makes `ε_i = κ_i · g(param)`, so each one has an explicit threshold. The Lipschitz
constants `L_i` and `Lh`, the value bounds and δ are hypotheses of those theorems. `measure.py` measures them on
real checkpoints and evaluates every threshold. It then measures the host error each construction actually makes,
teacher-forced on the target's own greedy continuation, which is the quantity the theorem bounds.

Run: `python -m verify.lipschitz.measure <model|all> --device cuda:0`, then `--report` to regenerate the tables
below from `out/*.json`. The test is `tests/test_lipschitz.py`: a tiny random Llama on CPU, about 30 s.

## What is measured

* **Text.** The 24 prompts of `act/act_swap_agreement.py`, each with its 64-token greedy continuation
  (fp32, eager attention, TF32 off). Contexts run to T = 77–89 tokens.
* **Norms.** A state is every position's residual. `dist` is the max over positions of the per-position l2 norm:
  Lean's Pi sup norm on `Fin T → V`, with V = R^d in l2. Logits use the sup norm over the vocabulary
  (`pick_eq_of_margin`).
* **Layer constants L_i.** For each block, the attention residual map and the FFN residual map, we take the
  Jacobian at real activations and compute, for every sampled query position t:
  * `spec`: the l2 spectral norm of the row block `J_t = ∂y_t/∂x`, by power iteration with autograd JVP and VJP;
  * `mix`: the (∞,2)→2 norm, which matches the sup-over-positions metric. It comes from the same iteration with
    a per-position normalisation, a monotone ascent that gives a lower bound.

  FFN maps act per position, so for them `spec = mix` and every token is measured. Attention maps and blocks are
  measured on 8 prompts at every 4th position plus the last, on all ten models (see the
  JSON's `jac_prompts` and `stride`). The JSON keeps max, p50, p90 and p99 over tokens, plus the power
  iteration's convergence. The theorem uses `L_i = max over tokens of block mix`. Over 20 + 10 iterations the
  `mix` value is 1.5–3× `spec`.
* **The head's constant Lh.** Final norm plus unembedding, l2 → sup-norm logits: `max_v ‖J_norm(x)ᵀ w_v‖₂`.
  This has a closed form for RMSNorm and LayerNorm, and the test checks it against autograd. Gemma-2's final
  cap is dropped because it is strictly increasing (`final_softcap_greedy`).
* **Value bounds.**
  * `max |pre|`: the largest FFN pre-activation.
  * `R_i`: the largest attention score `|z|` in a layer, causal entries only, before any cap.
  * `M_i = Σ_h max_j ‖W_O,h v_h(x_j)‖₂`. Lean's attention lemmas cover one head, and heads add.
  * Per layer: q, k, v sup norms and the sublayer outputs' sup and l2 norms, for the eps lever.
  * Router margins z(1) − z(2) for the MoE.
* **Margins δ.** The trajectory token's logit minus the best other, at all 64 steps. The theorem needs the
  minimum over a prompt's steps; the construction must pass on every prompt, so the binding value is the worst
  prompt. The median prompt is reported as well.

**These are local constants.** The theorems need Lipschitz constants valid on the whole region the host can reach,
meaning every state within the error radius of the trajectory. A Jacobian norm at a real activation is the local
constant there, so it is a lower bound on the constant the theorem needs. Every threshold below is therefore a
lower bound on what the theorem demands: the true certified threshold is at least this strict.

**Stack granularity.** One decoder layer is one `f_i`. An FFN-side edit changes the block by `ε_ffn`. An
attention-side edit changes it by at most `L_ffn,i · ε_attn`, where `L_ffn,i` is the FFN residual map's constant.
Pythia's parallel blocks use 1 instead of `L_ffn,i`. The "prod" column is the stated product form
`Lh (Π max(L_i, 1)) Σ κ_i`. The "sharp" column is Lean's `errBound` recursion, `Lh Σ κ_i Π_{j>i} L_j`.

**Per-rule constants κ_i** (Lean theorem → what is plugged in):

| rule | Lean | g(param) | κ_i |
|---|---|---|---|
| ReLU ⊆ SiLU (OPT-125m) | `relu_silu_greedy` | 1/β | T · Σ_n ‖W_down[:, n]‖₂ (Lean's units range over the whole state: T positions × neurons) |
| ALiBi ⊆ RoPE (BLOOM-560m) | `alibi_rope_greedy` | θ² | L_ffn · (2/3) m̄ T³ M_i (valid only while m̄T³θ² ≤ 3) |
| softcap host (plain target) | `cap_greedy` | 1/c² | L_ffn · (4/3) R_i³ M_i (valid only while 2R³ ≤ 3c²) |
| hard mask ← −B | `mask_greedy` | e^{−B} | L_ffn · 2 M_i T e^{2R_i} |
| plain ← sink host | `sink_greedy` | e^{b} | L_ffn · 2 M_i e^{R_i} |
| dense ⊇ top-1 MoE, soft router (Huihui-MoE-0.8B-2E) | `soft_router_greedy` | e^{−β δ_r} | 2 M_exp,i (E − 1), with δ_r the minimum router margin |
| eps lever, qk-norm off (Qwen3 host ← Llama-family target) | `leverAttn_greedy` | 1/S², S = 2^a | L_ffn · Σ_h O_h V_b (2 C_B Q K (Q² + K²) + V_b²/2)/ε, with C_B = √(2 d_h) and O_h = ‖W_O,h‖ from l∞ to l2 |
| eps lever, post-norms off (sandwich host ← plain target) | `postNorm_greedy` | 1/S² | √D (L_ffn R_attn³ + R_ffn³)/(2ε). R is the sup norm of the sublayer output; Lean's bound is in the sup norm and √D converts it to l2 |

**The construction settings compared against:**
* β = 3417: OPT's SiLU-host fit, from `act/data/act_swap_agreement.jsonl`.
* θ = 1e-2 … 6.25e-4: the synthetic sweep in `attn/positional_cross.py`. No real-model ALiBi-on-RoPE host exists.
* c = 5·max|z|: `attn/regime_agreement.py`.
* B and b: HF and elephantine mask with finfo.min or −inf, so exp underflows to exactly 0.
* Router β = 1e7: the select sharpness in `moe/dense_host_prenorm.py`, branch `moe-loop-rescue`. That
  construction uses threshold masks `t² σ(βt)`, not the soft router of the Lean theorem, but both have a soft
  zone of width ~1/β around routing ties.
* Lever exponent a = ⌈(log₂(ms_max/ε) + 30)/2⌉: the rule in `graph/eps_lever.py`, branch `measure-batch2`. It
  gives a = 26 on SmolLM2 and Llama-3.2, as measured there.

**The "observed" sweeps simulate the host directly:**
* softcap, mask and sink hosts: a manual eager attention for the Llama-family models;
* the eps lever: its exact identity `S√ε N_ε(x/S) = x/√(1 + ‖x‖²/(dS²ε))`, applied to q and k or to the sublayer
  outputs;
* OPT: `silu(βx)/β`;
* BLOOM: the RoPE pair `(m/θ) sin(θ(j − t))` in place of ALiBi's `m(j − t)`;
* Huihui: `softmax(βz)` over both experts.

## Results

### Measured constants (local: Jacobians at real activations)

| model | n | Lh max | L_block max (mix / spec) | prod L_block | prod sublayers | L_ffn max | L_attn max (mix) | max \|pre\| | max \|z\| | M max | delta min | delta median-prompt min | delta p1 | steps < 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smollm2-135m | 30 | 0.855 | 602 / 602 | 9.84e+39 | 1.67e+62 | 623 | 33.8 | 49.2 | 52.8 | 1.38e+03 | 8.26e-04 | 0.0135 | 0.0209 | 4.6% |
| qwen2.5-0.5b | 24 | 4.11 | 515 / 515 | 1.59e+32 | 1.30e+51 | 491 | 27.6 | 47.6 | 1.73e+03 | 339 | 3.62e-05 | 0.0234 | 0.0264 | 3.9% |
| llama-3.2-1b | 16 | 9.67 | 256 / 256 | 5.00e+23 | 1.58e+38 | 134 | 54.1 | 39.2 | 45.6 | 274 | 0.00111 | 0.0258 | 0.026 | 3.7% |
| pythia-160m | 12 | 9.06 | 229 / 123 | 3.00e+18 | 1.62e+32 | 61.7 | 228 | 109 | 1.50e+05 | 115 | 6.10e-04 | 0.0294 | 0.0296 | 3.2% |
| gpt2 | 12 | 1.46 | 180 / 101 | 5.52e+18 | 1.78e+30 | 428 | 131 | 62.9 | 103 | 4.33e+03 | 0.00188 | 0.0228 | 0.0231 | 4.2% |
| gemma-2-2b | 26 | 1.76 | 42.7 / 16.4 | 6.81e+33 | 4.99e+56 | 23.5 | 26.1 | 11.8 | 23.4 | 199 | 0.00656 | 0.046 | 0.0369 | 2.0% |
| olmo-2-1b | 16 | 191 | 52.6 / 32 | 1.91e+24 | 4.40e+40 | 59 | 33.5 | 53 | 21.8 | 2.46e+03 | 9.55e-04 | 0.036 | 0.0248 | 4.6% |
| opt-125m | 12 | 15.7 | 82.2 / 82.2 | 1.12e+17 | 2.05e+25 | 69.8 | 25.9 | 16 | 21.9 | 55.9 | 4.80e-04 | 0.0471 | 0.0439 | 2.7% |
| bloom-560m | 24 | 10.1 | 53.2 / 53.2 | 3.29e+28 | 4.29e+43 | 61.3 | 38.1 | 81.4 | 5.48e+03 | 493 | 0.00336 | 0.0325 | 0.0304 | 4.4% |
| huihui-moe-0.8b-2e | 28 | 0.633 | 771 / 771 | 2.70e+39 | 2.58e+63 | 450 | 146 | 284 | 47.9 | 3.07e+03 | 0.00323 | 0.0163 | 0.0169 | 4.6% |

### Limit rule -> theorem threshold -> construction setting

| limit rule | model | theorem threshold (prod form, worst prompt) | sharp form (errBound) | median prompt | construction setting | satisfied? | loosest swept setting with identical text | observed max \|dlogit\| there | bound / observed |
|---|---|---|---|---|---|---|---|---|---|
| softcap_host | smollm2-135m | c > 3.38e+26 | c > 1.26e+24 | c > 8.36e+25 | 264 | no | c = 264 | 0.0751 | 10^46 (sharp 10^41) |
| mask | smollm2-135m | B > 218 | B > 154 | B > 215 | 3.40e+38 | yes | B = 20 | 9.43e-04 | 10^86 (sharp 10^58) |
| sink | smollm2-135m | b < -161 | b < -123 | b < -158 | -inf | yes | b = -20 | 6.58e-05 | 10^62 (sharp 10^46) |
| lever_qk | smollm2-135m | a > 101 | a > 93.9 | a > 99.5 | 26 | no | a (S = 2^a) = 16 | 0.0263 | 10^50 (sharp 10^45) |
| lever_post | smollm2-135m | a > 104 | a > 86 | a > 102 | 34 | no | a (S = 2^a) = 22 | 0.00419 | 10^48 (sharp 10^38) |
| softcap_host | qwen2.5-0.5b | c > 1.18e+25 | c > 5.41e+23 | c > 4.64e+23 | 8.64e+03 | no | c = 1.73e+04 | 0.00263 | 10^40 (sharp 10^37) |
| mask | qwen2.5-0.5b | B > 3.55e+03 | B > 3.55e+03 | B > 3.55e+03 | 3.40e+38 | yes | B = 20 | 0.709 | 10^1530 (sharp 10^1527) |
| sink | qwen2.5-0.5b | b < -1.82e+03 | b < -1.82e+03 | b < -1.81e+03 | -inf | yes | none in sweep | - | - |
| lever_qk | qwen2.5-0.5b | a > 95.2 | a > 90.5 | a > 90.5 | 31 | no | a (S = 2^a) = 18 | 0.0952 | 10^43 (sharp 10^40) |
| lever_post | qwen2.5-0.5b | a > 90.6 | a > 78 | a > 85.9 | 31 | no | a (S = 2^a) = 22 | 0.0026 | 10^39 (sharp 10^32) |
| softcap_host | llama-3.2-1b | c > 2.13e+18 | c > 4.79e+16 | c > 4.43e+17 | 228 | no | c = 91.2 | 0.0772 | 10^31 (sharp 10^27) |
| mask | llama-3.2-1b | B > 167 | B > 143 | B > 164 | 3.40e+38 | yes | B = 10 | 0.0595 | 10^66 (sharp 10^56) |
| sink | llama-3.2-1b | b < -117 | b < -93.3 | b < -114 | -inf | yes | b = -5 | 0.0329 | 10^47 (sharp 10^37) |
| lever_qk | llama-3.2-1b | a > 74.5 | a > 69.5 | a > 72.2 | 26 | no | a (S = 2^a) = 16 | 0.00392 | 10^34 (sharp 10^31) |
| lever_post | llama-3.2-1b | a > 68.7 | a > 61.7 | a > 66.4 | 27 | no | a (S = 2^a) = 14 | 0.0995 | 10^31 (sharp 10^26) |
| softcap_host | pythia-160m | c > 1.82e+20 | c > 8.80e+14 | c > 2.62e+19 | 7.50e+05 | no | - | - | - |
| mask | pythia-160m | B > 3.00e+05 | B > 3.00e+05 | B > 3.00e+05 | 3.40e+38 | yes | - | - | - |
| sink | pythia-160m | b < -1.50e+05 | b < -1.50e+05 | b < -1.50e+05 | -inf | yes | - | - | - |
| softcap_host | gpt2 | c > 1.01e+16 | c > 2.19e+14 | c > 2.90e+15 | 514 | no | - | - | - |
| mask | gpt2 | B > 269 | B > 250 | B > 266 | 3.40e+38 | yes | - | - | - |
| sink | gpt2 | b < -161 | b < -143 | b < -159 | -inf | yes | - | - | - |
| softcap_host | olmo-2-1b | c > 3.23e+19 | c > 8.54e+17 | c > 5.26e+18 | 109 | no | c = 109 | 0.195 | 10^32 (sharp 10^29) |
| mask | olmo-2-1b | B > 126 | B > 122 | B > 122 | 3.40e+38 | yes | B = 20 | 0.00442 | 10^45 (sharp 10^43) |
| sink | olmo-2-1b | b < -99.6 | b < -95.5 | b < -95.9 | -inf | yes | b = -10 | 0.0246 | 10^37 (sharp 10^35) |
| relu_silu | opt-125m | beta > 1.90e+28 | beta > 1.81e+25 | beta > 1.94e+26 | 3.42e+03 | no | beta = 1e+03 | 0.00742 | 10^24 (sharp 10^21) |
| softcap_host | opt-125m | c > 3.80e+14 | c > 2.68e+13 | c > 3.83e+13 | 109 | no | - | - | - |
| mask | opt-125m | B > 105 | B > 99.9 | B > 100 | 3.40e+38 | yes | - | - | - |
| sink | opt-125m | b < -79.1 | b < -74.3 | b < -74.5 | -inf | yes | - | - | - |
| alibi_rope | bloom-560m | theta < 9.33e-22 | theta < 4.35e-20 | theta < 2.90e-21 | 0.01 ... 6.25e-04 | no | theta = 0.01 | 0.677 | 10^35 (sharp 10^32) |
| softcap_host | huihui-moe-0.8b-2e | c > 7.88e+25 | c > 1.37e+24 | c > 3.50e+25 | 239 | no | - | - | - |
| mask | huihui-moe-0.8b-2e | B > 205 | B > 200 | B > 204 | 3.40e+38 | yes | - | - | - |
| sink | huihui-moe-0.8b-2e | b < -153 | b < -148 | b < -152 | -inf | yes | - | - | - |
| soft_router | huihui-moe-0.8b-2e | beta > 2.60e+07 | beta > 2.30e+07 | beta > 2.56e+07 | 1.00e+07 | no | beta = 1 | 1.53e-04 | 10^48 (sharp 10^42) |

## Findings

**Measured on ten models:** SmolLM2-135M, Qwen2.5-0.5B, Llama-3.2-1B, Pythia-160m, GPT-2, Gemma-2-2B,
OLMo-2-0425-1B, OPT-125m (the ReLU target), BLOOM-560m (the ALiBi target) and Huihui-MoE-0.8B-2E (the top-1 MoE
target). All ten used 8 Jacobian prompts at stride 4. Llama-3.2, Gemma-2 and OLMo-2 ran with the model in CPU
memory: each layer went to the GPU for its Jacobians, and the host sweeps ran on the CPU.

**Not measured:**
* The host sweeps are built for the Llama family only (Llama, Qwen2, OLMo-2; OLMo-2 without the levers). For GPT-2,
  Pythia, OPT and Huihui the softcap, mask and sink thresholds are evaluated, but no observed error was measured.
* Gemma-2-2B has its own attention softcap, so the script evaluates no rule for it: the softcap host, mask and sink
  rules are gated together on a plain target. Its row in the first table is constants only.

1. **The hypotheses can now be evaluated, and they are large.**
   * Per block, the local constant is `L_i` = 4–770. The largest values come from the layers that write the
     massive activations: Huihui's largest block is 771, and SmolLM2 layer 12's FFN has a per-token Jacobian norm
     of 623.
   * Their product over the stack is 10^17 (OPT, 12 layers) to 10^40 (SmolLM2, 30 layers).
   * Taking the stack at sublayer granularity instead of block granularity costs another 8–24 orders of
     magnitude, because the composition is far smaller than the product of its parts.
   * `Lh` is 0.63–16 on nine models and 191 on OLMo-2.
   * The worst-prompt margin is δ = 3.6e-5 (Qwen2.5) to 6.6e-3 (Gemma-2). Every model has steps with near-ties.
   * The median prompt's minimum margin is 0.013–0.047, and 2–5% of all steps have δ < 0.1.
2. **Satisfied only where the construction is exact by underflow.**
   * Masks and sinks: the theorem asks for B > 105–3,550 and b < −79 to −1,820. Pythia, with attention scores up
     to 1.5e5, asks for B > 3e5. The constructions use finfo.min or −inf, so they satisfy the theorem, but
     trivially.
   * Every rule with a genuinely finite parameter fails the theorem:
     * **ReLU ⊆ SiLU:** β* > 1.9e28 (sharp form 1.8e25), against the fit's β = 3,417.
     * **ALiBi ⊆ RoPE:** θ* < 9e-22, against 1e-2 … 6e-4. θ = 1e-2 is also outside the theorem's own small-angle
       regime, m̄T³θ² ≤ 3, which needs θ ≤ 2.7e-3.
     * **Softcap host:** c* = 1e14 (OPT) to 3e26 (SmolLM2), against 5·max|z| = 109–8,640. The bound's own
       validity condition c² ≥ 2R³/3 already fails at 5·max|z| on SmolLM2, Qwen2.5, Llama-3.2, GPT-2, Pythia and
       Huihui.
     * **Eps lever:** the theorem needs a > 75–101 for qk-norm and a > 69–104 for post-norm. The constructions
       use a = 26–34.
     * **Soft router (Huihui):** β* > 2.6e7 (sharp form 2.3e7), against β = 1e7. This is the closest miss, a
       factor of 2.6, because β enters through e^{−βδ_r}, so the threshold is only logarithmic in the 10^39
       product. The case is degenerate on this model: Huihui's two experts are bit-identical in every layer
       (gate, up and down projections), so the dense mixture equals the top-1 output at any β. The sweep sees
       only fp32 rounding (max |Δlogit| 1.5e-4 at β = 1, 0 at β = 1e7), and the theorem's κ = 2 M_exp (E − 1)
       cannot see that the experts coincide.
3. **What the constructions actually need is modest.** These are the loosest swept settings that already give
   identical text on all 24 prompts:
   * β = 1e3 on OPT;
   * θ = 1e-2 on BLOOM;
   * c = 5·max|z| on SmolLM2 and OLMo-2, and c = 2·max|z| on Llama-3.2. On Qwen2.5, 5·max|z| gives 23 of 24
     prompts identical (top-1 99.93%, which passes regime_agreement's payable standard) and 10·max|z| gives all 24;
   * B = 20 for the mask (B = 10 on Llama-3.2);
   * b = −20 for the sink on SmolLM2, b = −10 on OLMo-2 and b = −5 on Llama-3.2. On Qwen2.5, whose scores reach
     1,728, no swept b ≥ −40 suffices;
   * a = 16–18 for the qk lever and a = 14–22 for the post-norm lever, against the construction rule's a = 26–34.
   So the constructions' settings are safe empirically, with a few binades to spare, while the theorem cannot
   certify them.
4. **How loose the bound is.** The ratio of theorem bound to observed max |Δlogit| at those settings:
   * 10^21–10^24 for ReLU ⊆ SiLU (sharp to product form);
   * 10^32–10^35 for ALiBi ⊆ RoPE;
   * 10^26–10^50 for the softcap host and the levers;
   * 10^35–10^86 for the mask and the sink;
   * 10^42–10^48 for Huihui's soft router, where the observed error is only fp32 rounding (identical experts);
   * 10^1530 for Qwen2.5's mask, through e^{2R} with R = 1,728.

   The looseness has three sources:
   * the product of worst-case per-layer constants (10^17–10^40), when real perturbations do not align with
     every layer's top singular direction;
   * worst-case value bounds, R³ and e^{2R} at the single largest score, when nearly every score is far smaller
     (q999 ≈ 12–18);
   * the worst step's margin, against a typical step's margin of ~1.
5. **Are the limit proofs fully grounded?** No.
   * Every hypothesis now has a measured value, and the thresholds can be computed. With those values, the
     theorems certify identical generation only for the exact-by-underflow settings (masks, sinks).
   * The measured values are local Jacobian norms, so they lower-bound the region constants the theorems need.
     Certified region constants, for example from interval bounds over the error ball, would make the
     thresholds stricter still.
   * For the limit rules the repo pays (activation swaps, ALiBi ⊆ RoPE, the softcap host, the eps lever, the
     MoE router), the proofs establish the limit and its rate, O(1/β), O(θ²), O(1/c²) and O(1/S²), and the
     measured agreement shows that the rate's constant is small in practice. A certificate at the settings
     actually used still rests on measurement, not on the theorem.
   * Closing the gap would take bounds tied to the trajectory rather than worst-case ones: Jacobian-vector
     products along the actual error direction, per-score bounds instead of R_max, and per-step margins.
