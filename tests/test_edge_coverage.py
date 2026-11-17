"""The emulation relation's edge coverage is measured, not asserted.

Pins the construction-verified / atom-verified / approximate-only split over the real
904-shape graph under the HEADLINE (payable) relation (see elephantine/edge_coverage.py), so
the honest coverage number can't silently drift.

Run:  pytest tests/test_edge_coverage.py -q
"""
from elephantine.edge_coverage import coverage, exact_edge_coverage, executable_coverage


def test_edge_coverage_split():
    b = coverage()
    # 124891 (capacity-only headline) -> 77872 (headline-payable, 2026-11-11): the type walls
    # without a currency now block; 77872 -> 66939 (act-floor-calibration): the activation floor
    # edges (GeLU <- SiLU 0.012, ...) are walls at the calibrated ACT_FLOOR_MAX; 66939 -> 60279
    # (mixer-gap-measure): the softmax <- SSM / linear / hybrid near-gap is not payable; 60279 -> 60251
    # (cla-not-payable): the CLA carry (~6e-2 pre-norm floor) is not payable
    # 60251 -> 63618 (partial-rope-host): rope <- partial_rope paid where the host rotates enough dims per
    # head, and the rotary factor nested in rope_parameters read; 63618 -> 29388 (edge-case-audit: per-layer
    # kinds, MLA, measured norm / bias / sink walls, FEATURES, KV heads, the Lean conditions; the rot gate
    # on full-RoPE hosts too, NoPE layers in rope_fits -- 29176 on the audit alone)
    # 29388 -> 28250 (round 2, classification pass: drafters skipped, legacy families read, NoPE / router / sink /
    # bias / GeLU-tanh labels, class-default ties, block-causal and sinusoidal walls)
    # 28250 -> 32355 (round 2, attention / norm pass: KV-head packing adds edges; the qk-norm variants, post-only
    # norms, attention softcap, ALiBi slope schemes, the fixed-rotary gate and the new bias labels remove some)
    # 32355 -> 32263 (round 2, MoE / MLA pass: DSA is dense attention, which adds edges; MoE <- MoE by expert
    # duplication -- top-k a multiple, enough experts, renorm / score / group shape, shared width, dense / MoE layer
    # kinds -- and the MLA latent multiples remove more; 904 -> 909 shapes)
    # 32263 -> 32281 (round 2, recurrent / diffusion pass: the block-causal SDAR hosts play AR targets and the
    # bidirectional ones through their caller mask, which adds edges; the recurrent capacity per family, the split
    # recurrent variants and the self-conditioning embedding norm remove some)
    # 32281 -> 46329 (round 2, rules / rescues pass: the window / rotary-fraction / softcap / norm-flag config levers,
    # the eps lever, the start-token rules, activation width and GeLU erf <-> tanh add edges; the config-only rot gate,
    # the looped-host, LayerNorm-beta, swigluoai-bias and ALiBi-on-learned-abs walls remove some; 909 -> 931 shapes)
    # 46329 -> 43332 (round 3, batch 1, classification: the code facts the type tables / config reads missed -- q / k / v
    # and output biases, ties, post-norm placements, embedding / grouped / gated norms, encoder inputs and cross-attention,
    # head widths, prefix and bidirectional masks -- mostly remove edges; 931 -> 928 shapes)
    # 43332 -> 40903 (round 3, batch 2a: the output gates split by granularity / bias / activation, each walled from the
    # others, and the q / k norms read from code -- LFM3 / Grug / Muse / Dory ... -- remove edges; 928 -> 926 shapes)
    # 40903 -> 40710 (round 3, batch 2b: the positions -- OLMo-3 / PLaMo-3 second RoPE tables, the NoPE layers of
    # Muse / Domyn-Edge / Kolibri-1 / Grug, the q_temp / q_temp_nope / fope / rope_warp walls, Parler's sinusoid, the
    # scaled Falcon slopes -- remove edges; the LFM3 zero-tail tables and the read rotary widths add some)
    # 40710 -> 40269 (round 3, batch 2c: windows and masks -- the sliding layers read from code no longer host full
    # layers, the swa_trunc / attn_out_sat walls (Meditor, Qwen3-D-ROS) and the dyn_mask / doc_mask target walls
    # remove edges; 926 -> 927 shapes)
    # 40269 -> 39286 (round 3, batch 2d: heads and MLA -- each attention shape counted on its own layers with its head
    # dim and V width, and the vha / xsa / canon / cca / moa / mem_attn / mova / mla_q_raw / mla_qmix / kv_compress_v41
    # walls remove edges; clip_qkv's host lever and the read head widths add some; 927 -> 928 shapes)
    # 39286 -> 39188 (round 3, batch 3a: MoE and router -- the router / block code facts (the 0-filled group limit, the
    # code-fixed renorm / score / selection-bias kinds, the identity / latent-norm / branch / Arctic forms) remove
    # edges; K2-Horizon / Motif / Kohaku's settable scores and the denser models freed of router features add some)
    # 39188 -> 37465 (round 3, batch 3b: the spot-check fixes -- no gated / MoE target on a vanilla host, the vanilla <-
    # gated step and up-biased targets on the host's up bias, no act-width fit on a swigluoai host -- remove edges; the
    # act_clip host lever and the FFN widths read from code add some)
    # 37465 -> 37233 (round 3, batch 4a: recurrent mixers and loops -- the hybrid schedules read from code (GPT-X3, KLM,
    # Edge0, Ming-3, Solar-Open2 leave the plain-softmax class), the loop depths (Ouro x4, HRM x32 / 128) and the
    # loop-boundary walls remove edges; the fla transformer, the CFRD cells and LFM3-RD as an untied stack add some;
    # 928 -> 933 shapes)
    # 37233 -> 37205 (round 3, batch 4b: residual streams and extra inputs -- the hyper-connection variants, the n-gram /
    # value-embedding / AttnRes signatures and the residual-stream walls remove edges; 933 -> 935 shapes)
    # 37205 -> 37220 (round 3, the zoo classifier fixes: Phi-3-small-8k placed, Apertus-1.5's ungated MLP, TR-MOE-306's
    # top-2 and Kanana-2-tiny's q / k norm move edges; 935 -> 936 shapes)
    # 37220 -> 37013 (round 3, output heads and inputs as slots, all of it nonlm_heads: the sentence embeddings (mean
    # pooling / L2) leave the LM class both ways, which removes edges -- 136 of them AR targets on the bidirectional
    # embedders' masks --; the score-head rerankers on untied LM hosts add some; 936 -> 946 shapes)
    # 37013 -> 37019 (round 3, kind_order: the order-preserving layer embedding removes reordered hybrid pairs; the
    # PseudoBrain split adds points, 946 -> 948 shapes)
    # 37019 -> 37020 (round 3, mamba_ffn_none / mamba1_lever: mamba-130m / 370m / 790m / 1.4b / 2.8b on falcon-mamba-7b
    # by the lever; the two mamba-130m checkpoints (ArthurZ's, state-spaces') merge, 948 -> 947 shapes)
    # 37020 -> 36997 (round 3, score_not_free: the router score function is a feature switch, no host score lever --
    # the 9 K2-Horizon transitivity triples go, 23 MoE <- MoE edges across score functions with them)
    # 36997 -> 38134 (release-rel: the first-wave rules -- fr-mixer, fr-resnorm, fr-positions, fr-cost, fr-moe -- add
    # more edges than their soundness / reading fixes remove; 947 -> 942 shapes)
    assert b["total"] == 38134                               # all asserted headline edges
    assert b["gen_cross"] == 0                              # no causal host for a diffusion / block target
    # AR targets on diffusion / block hosts and block <-> diffusion (the mask; was 634, 530, 509, 856, 620, 622, 676)
    assert b["gen_mask"] == 683                             # (was 682 before release-rel; 818 before round 3, output heads and inputs as
    #                                                         slots -- the bidirectional embedders gte-Qwen2 / stella /
    #                                                         voyage are embedding heads, walled from the AR LM targets
    #                                                         they hosted through their mask (all of it nonlm_heads);
    #                                                         817 before round 3, batch 4a -- the new /
    #                                                         reclassified recurrent and loop points;
    #                                                         828 before round 3, batch 3b -- the up-bias and
    #                                                         gated-on-vanilla walls between AR and diffusion shapes;
    #                                                         792 before round 3, batch 3a -- LLaDA2.2-mini's block
    #                                                         routing splits it from the LLaDA2.0 / 2.1 class: its 36 AR
    #                                                         targets, by the expert_capacity lever, count on their own;
    #                                                         823 before round 3, batch 2d -- the head / MLA walls,
    #                                                         DiffusionGemma's K = V full layers; 835 before round 3,
    #                                                         batch 2c -- the window / mask
    #                                                         walls; 838 before round 3, batch 2b -- the position walls;
    #                                                         874 before round 3, batch 2a -- the gate / qk-norm
    #                                                         walls; 938 before round 3, batch 1 -- DIVE-Doc prefix, the
    #                                                         bidirectional embedders; 703 before the rules / rescues pass)
    # every headline edge spends no near-gap (every atom exact or real-module verified): the softmax <-
    # SSM / linear mixer gap (9.9% of the edges while it was granted free) is refused
    assert b["exact"] == b["total"]
    assert b["has_gap"] == 0
    assert b["spine_pure"] + b["atom_real"] + b["atom_pending"] + b["has_gap"] == b["total"]


def test_executable_coverage_ladder():
    e = executable_coverage()
    # executable fraction only grows as more type axes become constructible
    assert e["capacity_only"] <= e["live_today"] <= e["clean_exact"] <= e["with_regime"]
    assert e["live_today"] > e["capacity_only"]           # run_spine already builds kv-type
    assert e["clean_exact"] > 0.45 and e["with_regime"] > 0.75
    # the hard-atom remainder is real and bounded; no headline edge is blocked only by the mixer gap
    # (it is refused), and the zeroing mixer edges (hybrid host, softmax target) are clean-exact
    assert e["gap_only"] == 0
    assert e["with_regime"] + e["gap_only"] + e["needs_hard_atom"] > 0.95


def test_every_exact_edge_is_buildable():
    # the EXACT relation = every actually-emulatable edge (all reductions exact). EVERY one
    # now has a token-for-token construction; ~95% are whole-model forwards, the rest are
    # module-level constructions composed with a capacity change.
    x = exact_edge_coverage()
    # 39115 -> 39766 (2026-11-11): Gemma GeLU labels -211 exact edges, Qwen v1 RMSNorm +862
    # 39766 -> 38536 (round 2: the same label / pricing fixes)
    # 38536 -> 43197 (headline-payable): the exact zeroing / mask / mirror-norm edges join the slice
    # 43197 -> 45102 (partial-rope-host): the exact rot edge joins the slice
    # 45102 -> 28108 (edge-case-audit, with the rot gate on full-RoPE hosts; 28132 on the audit alone)
    # 28108 -> 26867 (round 2, classification pass; erf and tanh GeLU are walled from each other for now)
    # 26867 -> 31140 (round 2, attention / norm pass: KV-head packing is exact; the exact qk-norm nestings join)
    # 31140 -> 31728 (round 2, MoE / MLA pass: DSA as dense attention adds edges, the MoE routing rules remove some,
    # and the dense host for a top-1 MoE target is payable only -- measured, a limit in Lean -- not exact)
    # 31728 -> 31746 (round 2, recurrent / diffusion pass: the block-causal mask edges are exact; the recurrent
    # capacity / variant walls and the self-conditioning norm remove some)
    # 31746 -> 32227 (round 2, rules / rescues pass: the exact config levers -- window, rotary fraction, softcap
    # values, q / k norm flags -- and the start-token rules join; the config-only rot gate and the walls remove some)
    # 32227 -> 29820 (round 3, batch 1: the classification fixes, mostly walls)
    # 29820 -> 28421 (round 3, batch 2a: the output-gate keys and the code-read q / k norms, mostly walls)
    # 28421 -> 28255 (round 3, batch 2b: the position code facts, mostly walls)
    # 28255 -> 27955 (round 3, batch 2c: the window / mask code facts, mostly walls)
    # 27955 -> 27346 (round 3, batch 2d: the head / MLA code facts, mostly walls; clip_qkv's host lever adds edges)
    # 27346 -> 27364 (round 3, batch 3a: the router / MoE code facts; the settable scores and renorms read from code
    # (K2-Horizon, Motif, Kohaku; Bailing V2 / CENO / Arctic R) add more exact edges than the new walls remove)
    # 27364 -> 27358 (round 3, batch 3b: the up-bias walls remove exact edges; the act_clip host lever and the read FFN
    # widths add some)
    # 27358 -> 27306 (round 3, batch 4a: the recurrent / loop code facts, mostly walls; the new points add some)
    # 27306 -> 27283 (round 3, batch 4b: the residual-stream and extra-input walls)
    # 27283 -> 27285 (round 3, the zoo classifier fixes)
    # 27285 -> 27147 (round 3, output heads and inputs as slots: the embedding models' readout walls; the rerankers' exact
    # score-head edges on untied LM hosts add some)
    # 27147 -> 27150 (round 3, kind_order: as the headline: reordered pairs removed, the two new points' edges added)
    # 27150 -> 27146 (round 3, mamba_ffn_none / mamba1_lever: the two mamba-130m checkpoints merge into one point -- they
    # differed only in the FFN width read off the mixer --, its pair edges counted once)
    # 27146 -> 27133 (round 3, score_not_free: the exact MoE <- MoE edges across score functions)
    # 27133 -> 27243 (release-rel: the first-wave rules -- fr-mixer, fr-resnorm, fr-positions, fr-cost, fr-moe)
    assert x["total"] == 27243
    assert x["whole_model"] > 0.92                         # whole-model forward (spine + elephantine + tests)
    assert abs(x["composed"] - 1.0) < 1e-12               # every exact axis has a construction
    assert abs(x["with_all_exact_atoms"] - 1.0) < 1e-12   # nothing left over: no gap in the exact relation


def test_spine_pure_edges_are_pure_capacity():
    # spine_pure edges change NO categorical slot -> they are exactly what the capacity
    # spine (verify_edges.run + run_spine) builds and runs end-to-end.
    b = coverage()
    assert b["spine_pure"] > 0
    assert b["atom_real"] > b["spine_pure"]                 # most exact edges also cross a type wall
