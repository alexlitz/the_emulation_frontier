"""Standing soundness guards for the emulation accounting — the honest answer to
"is the relation sound?" made into regression tests.

  1. SUFFICIENCY / NECESSITY (sampled, constructive): draw random Llama pairs, and
     for every edge `emulates(B,A)` asserts, actually BUILD B from A and require
     token-for-token reproduction; for every non-edge, require the build to fail.
     Scope: the buildable capacity axes (KV-sharing, FFN width, depth) on the
     softmax Llama family — not hidden-width / cross-mixer / categorical edges.

  2. ROBUSTNESS / CONTINGENCY: the frontier size is stable to the heuristic
     currency factors. The headline relation is NOT a DAG: its two-way edges (the qk-norm /
     softcap regimes, the activation floors, a hybrid host that is all attention) make
     equal-capacity models mutually emulable; dense->MoE routing depth (MOE_DEPTH_FACTOR >= 2)
     is what keeps the FFN axis acyclic -- drop it to 1 and a dense<->MoE 2-cycle appears.

Run:  python -m pytest tests/test_edge_soundness.py -q
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_asserted_edges_build_nonedges_fail():
    from elephantine.verify_edges import run
    r = run(n=60, seed=0)
    assert r["pos_n"] > 0 and r["neg_n"] > 0, "sample should hit both edges and non-edges"
    assert r["pos_ok"] == r["pos_n"], f"{r['pos_n'] - r['pos_ok']} asserted edges did NOT reproduce"
    assert r["neg_reject"] == r["neg_n"], f"{r['neg_n'] - r['neg_reject']} non-edges spuriously reproduced"
    assert r["worst_pos"] < 1e-9, f"asserted edges not exact: worst |Δ| = {r['worst_pos']}"


def test_headline_relation_is_transitive_on_the_real_shapes():
    """i <- k <- j must imply i <- j over every real checkpoint shape. Before the per-layer kinds
    (model_graph.LAYER_KINDS / kind_match) 2,198 pairs broke it: the ZERO_EDGES depth check ran only for
    plain targets, so a hybrid / sliding host with fewer attention / full layers was credited with a
    hybrid / sliding target that has more. The only non-transitivity the relation means to have is the
    activation slot's one-hop floors (REACH['act']): any violation must be blocked by 'act' alone."""
    import numpy as np
    from graph.currency_curve import points
    from graph.frontier import blockers
    from graph.model_graph import emul_matrix
    P = points()
    ok = emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, True)
    two = (ok.astype(np.int32) @ ok.astype(np.int32)) > 0
    bad = [(i, j) for i, j in zip(*np.where(two & ~ok))
           if not all(b.startswith("act ") for b in blockers(P[i], P[j]))]
    assert not bad, [(P[i]["members"][0], P[j]["members"][0], blockers(P[i], P[j])) for i, j in bad[:5]]


def test_per_layer_kinds_bind_every_target():
    """Every target layer needs a host layer of a kind that reaches it, whatever the target's mixer:
    hybrid <- hybrid needs as many attention AND recurrent layers, sliding <- sliding as many full layers;
    a parallel hybrid layer (attention beside an SSM, Falcon-H1) counts toward both kinds."""
    from graph.model_graph import classify, dims_of, emulates
    from tests.test_graph_rules import mk
    hyb = lambda L, a, **k: mk(mixer="hybrid", layers=L, attn_eff=a, attn_param=a, full_eff=a, full_param=a, **k)
    assert emulates(hyb(48, 12), hyb(24, 12)) and not emulates(hyb(48, 11), hyb(24, 12))   # attention layers
    assert not emulates(hyb(48, 40), hyb(24, 12))                     # 8 recurrent < 12 recurrent
    loc = lambda L, f: mk(window="local", layers=L, full_eff=f, full_param=f)
    assert emulates(loc(30, 6), loc(30, 6)) and not emulates(loc(40, 5), loc(30, 6))
    fh = {"model_type": "falcon_h1", "hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 8,
          "intermediate_size": 2048, "rms_norm_eps": 1e-5, "rope_theta": 1e4, "mamba_d_state": 16}
    d = dims_of(fh)
    assert classify(fh)["mixer"] == "hybrid" and d["kFU_eff"] == 12 and d["kR_eff"] == 12
    # a hybrid model class run with no recurrent layer is a softmax stack (Granite-4.0 micro)
    g4 = {"model_type": "granitemoehybrid", "hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8,
          "intermediate_size": 2048, "rms_norm_eps": 1e-5, "rope_theta": 1e4, "mamba_d_state": 16,
          "layer_types": ["attention"] * 4}
    assert classify(g4)["mixer"] == "softmax" and classify({**g4, "layer_types": ["mamba", "attention"] * 2})["mixer"] == "hybrid"
    # Nemotron-NAS blocks whose attention is a no-op or a linear map host no attention layer
    nas = {"model_type": "nemotron-nas", "hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8,
           "rms_norm_eps": 1e-5, "rope_theta": 1e4,
           "block_configs": [{"attention": {"n_heads_in_group": 4}, "ffn": {"ffn_mult": 2.0}},
                             {"attention": {"no_op": True}, "ffn": {"ffn_mult": 2.0}},
                             {"attention": {"replace_with_linear": True}, "ffn": {"ffn_mult": 2.0}},
                             {"attention": {"no_op": True}, "ffn": {"no_op": True}}]}
    dn = dims_of(nas)
    assert dn["eff_layers"] == 3 and dn["kFU_eff"] == 1


def test_frontier_robust_but_dagness_is_contingent():
    import numpy as np
    import graph.model_graph as mg
    from graph.figures import load
    from graph.model_graph import _dedup_points, emul_matrix
    P = _dedup_points(load())

    def frontier_mutual():
        ok = emul_matrix(P); np.fill_diagonal(ok, False)
        strict = ok & ~ok.T
        return int((~strict.any(1)).sum()), int((ok & ok.T).sum()) // 2

    base = mg.MOE_DEPTH_FACTOR
    try:
        mg.MOE_DEPTH_FACTOR = 2; f2, m2 = frontier_mutual()
        mg.MOE_DEPTH_FACTOR = 3; f3, m3 = frontier_mutual()
        mg.MOE_DEPTH_FACTOR = 1; f1, m1 = frontier_mutual()   # pretend dense->MoE routing is free
    finally:
        mg.MOE_DEPTH_FACTOR = base
    # frontier size moves little across the heuristic factor (97 / 107 / 109 raw not-dominated points;
    # 91 / 101 / 103 while the CLA carry was granted free, 75 / 85 / 88 while the mixer near-gap was
    # too) -> the headline is robust
    assert all(240 <= f <= 310 for f in (f1, f2, f3)), (f1, f2, f3)   # 294 / 297 / 297 (release-rel: the first-wave rules -- fr-mixer, fr-resnorm, fr-positions, fr-cost, fr-moe); 303 / 306 / 306 (round 3, mamba_ffn_none / mamba1_lever); 304 / 307 / 307 (round 3, kind_order); 300 / 303 / 303 (round 3, output heads as slots: the rerankers and sundial kept, the embeddings walled from the LMs; the bound was 300); 293 / 296 / 296 (round 3, zoo classifier fixes); 295 / 298 / 298 (round 3, batch 4b: residual streams and extra inputs; the bound was 290); 287 / 290 / 290 (round 3, batch 4a: recurrent mixers and loops); 273 / 278 / 279 (round 3, batch 3a: MoE and router); 274 / 279 / 280 (round 3, batch 2d: heads and MLA); 266 / 271 / 272 (round 3, batch 2c: windows and masks); 261 / 266 / 267 (round 3, batch 2b: positions); 263 / 268 / 269 (round 3, batch 2a: qk-norm and output gates); 257 / 262 / 263 (round 3, batch 1: classification); 251 / 256 / 257 (round 2, rules / rescues
    #                                                                   pass); 276 / 277 / 278 (round 2, recurrent /
    #                                                                   diffusion pass; 262 / 263 / 264 the MoE / MLA pass;
    #                                                                   223 / 225 / 226 before it, 220 / 222 / 223 before the
    #                                                                   round-2 attention / norm pass, 217 / 221 / 221
    #                                                                   before the classification pass)
    # raw points: 218 classes (the rope <-> partial_rope / rope <-> rope two-way edges at equal rotated dims
    # tie a few points; 219 = 219 on the audit alone, 109 / 104 on partial-rope-host alone)
    assert f2 == 297                                                  # was 306 (round 3 final, before the release-rel: the first-wave rules -- fr-mixer, fr-resnorm, fr-positions, fr-cost, fr-moe), 307 (round 3, mamba_ffn_none / mamba1_lever), 303 (round 3, kind_order), 296 (round 3, output heads as slots: +7), 298 (round 3, zoo classifier fixes: -2), 290 (round 3, batch 4b: +8), 284 (round 3, batch 4a: +6), 278 (round 3, batch 3b: +6), 279 (round 3, batch 3a: -1), 271 (round 3, batch 2d: +8), 266 (round 3, batch 2c: +5), 268 (round 3, batch 2b: -2), 262 (round 3, batch 2a: +6), 256 (round 3, batch 1: +6), 277 (rules / rescues pass: -21), 263 (recurrent / diffusion pass: +14), 225,
    #                                                                   222, 221 (MoE / MLA pass: +38)
    # mutual pairs come only from the two-way edges at equal capacity (25 at the validated factor;
    # 30 while the free mixer near-gap paired hybrid targets with softmax hosts both ways, 34 before
    # the activation calibration dropped the two-way swish-cluster floors);
    # free dense->MoE routing adds a dense<->MoE cycle on top
    # 1 on the audit alone -> 7: rope <-> partial_rope (and rope <-> rope) is two-way where the rotated dims
    # match (34 on partial-rope-host alone, 25 before either). 7 -> 20 (round 2, attention / norm pass): KV-head
    # packing makes a GQA host with wide heads and a narrow-headed MHA / GQA shape of equal capacity mutual.
    # 20 -> 24 (round 2, MoE / MLA pass): DSA as dense attention pairs DSA and dense twins (DeepSeek-V3.2 <-> MAI-DS-R1,
    # Ring-mini-sparse <-> Ling-mini, Naive-N0.5 <-> MiMo-V2.5), and a Qwen3-MoE (renorm a config value) with its
    # code-renormalising Qwen3-VL twin; a few twins of equal shape merge, now split by the routing facts
    # 24 -> 51 (round 2, rules / rescues pass): the new two-way edges at equal capacity (GeLU erf <-> tanh, the
    # config levers between twins of one family, the eps lever between a normed host and its plain twin)
    # 51 -> 49 (round 3, batch 2b, local_rope_theta_unread: OLMo-3's YaRN full layers are a second RoPE table, so
    # Olmo-3-1025-7B <-> BAR-7B and Olmo-3-1125-32B <-> OLMo-2-0325-32B, one-table twins, are one-way)
    # 49 -> 48 (round 3, output heads and inputs as slots: gte-Qwen2-7B-instruct, a last-token + L2 embedding, is walled
    # from its diffusion twin Dream-v0-Base-7B both ways by the readout rule)
    # 48 -> 29 (release-rel, fr-positions' fp_table_length: a learned table hosts only a target with no more rows, so
    # learned-abs shapes of different table lengths are no longer two-way; 49 with that rule off)
    assert m2 == m3 == 29, (m2, m3)
    # (a dense host now plays only top-1 unit-gate MoE targets -- Lean dense_hard_select -- so free dense->MoE
    # depth no longer adds a dense<->MoE cycle among the real shapes)
    assert m1 >= m2


def test_layerwise_exact_per_layer():
    """The strong check: build B>=A and require the RAW residual of EVERY reproduced
    layer (not just the final logits) to match A to fp64 noise. This is the test that
    caught the post-final-norm harness artifact and, once fixed, confirmed the
    constructions are bit-exact layer-by-layer."""
    from elephantine.verify_edges import run_layerwise
    r = run_layerwise(n=120, seed=1)
    assert r["npos"] > 50, "sample should build plenty of positive edges"
    assert r["fails"] == 0, f"{r['fails']} edges diverged at some layer"
    assert r["overall"] < 1e-9, f"worst per-layer |Δ| = {r['overall']}"


def test_context_axis_couples_window_depth():
    """USE_CONTEXT (opt-in, like vocab) prices the max-context DEMAND + the tight
    window<->depth<->context coupling. Checked here on the HEADLINE (payable) relation: short-
    context hosts lose edges (capacity floor), local hosts gain reach edges (depth pays the window
    instead of only their full layers counting) — the tight case a one-way window order can't
    express. Net: frontier 268 -> 296 (raw not-dominated points; 262 -> 291 before round 3, batch 2a; 256 -> 286 before
    round 3, batch 1; 277 -> 304 before
    the round-2 rules / rescues pass; 263 -> 290 before the round-2 recurrent / diffusion
    pass, 225 -> 253 before the round-2 MoE / MLA pass, 222 -> 250 before the round-2 attention / norm
    pass, 221 -> 252 before the classification pass, 107 -> 134 before the edge-case audit,
    101 -> 128 while the CLA carry was granted free, 85 -> 117 with the free mixer gap too).

    With the flag on, the points are deduped on max_context / eff_window too (895 -> 1,098 points):
    a class that mixed contexts kept only its first member's, which under-counted the frontier."""
    import numpy as np
    import graph.model_graph as mg
    from graph.figures import load
    from graph.model_graph import _dedup_points, emul_matrix
    M = load()
    P0 = _dedup_points(M)
    mg.USE_CONTEXT = True
    try:
        P1 = _dedup_points(M)                               # context-aware key
        ok1 = emul_matrix(P1); np.fill_diagonal(ok1, False)
    finally:
        mg.USE_CONTEXT = False
    ok0 = emul_matrix(P0); np.fill_diagonal(ok0, False)
    ok0b = emul_matrix(P1); np.fill_diagonal(ok0b, False)   # flag off, same points as ok1
    f0 = int((~(ok0 & ~ok0.T).any(1)).sum())
    f1 = int((~(ok1 & ~ok1.T).any(1)).sum())
    # 895 / 1098 before the round-2 classification pass (898 / 1103 on the audit alone, 842 / 1058
    # partial-rope-host alone, 841 / 1059 before either)
    # 874 / 1068 before the attention / norm pass (the dedup key now holds the ALiBi scheme and rope_pairs_free)
    # 904 / 1093 before the MoE / MLA pass (the key now holds the routing facts moe_sig and the MLA nope / v widths)
    # (the recurrent / diffusion pass adds the recurrent dims and the Zamba blocks to the key: no softmax point
    # carries them, and the recurrent points they would split were already split, so 909 / 1098 stay)
    # 909 / 1098 before the rules / rescues pass: the key now holds the host levers and start-token facts (fraction-
    # free rotary, a fixed rotary's rope kind, q / FFN bias, a learned-abs table's length) and the per-layer KV shapes
    # 931 / 1108 before round 3, batch 1 (classification: sundial, the two score-head rerankers and two remote-code
    # wrappers leave; the new dims emb_rank / emb_norm / norm_groups / gnorm_* are in the key)
    # 928 / 1105 before round 3, batch 2a (qknorm_by_code_unlisted: Hulu-Med-4B and Isaac-0.1, Qwen3 bodies, now carry
    # their q / k norm and join their Qwen3 twins' classes)
    # 926 / 1103 before round 3, batch 2c (attn_output_saturation: Qwen3-0.6B-D-ROS carries attn_out_sat and leaves the
    # Qwen3-0.6B class)
    # 927 / 1104 before round 3, batch 2d (the per-layer head shapes / V widths / Canon / MoA / latent dims split a
    # point: kv_shape_n, v_head, ... are in the key)
    # 928 / 1105 before round 3, batch 3a (the MoE-block forms in moe_sig split / merge context-aware classes)
    # 928 / 1104 before round 3, batch 4a (the fla transformer / DeltaNet and the three CFRD points are placed -- the
    # CFRD cells give it a depth --, and the loop / recurrent dims split a few)
    # 933 / 1109 before round 3, batch 4b (the hyper-connection variants, n-gram / AttnRes / value-embedding signatures
    # and residual-stream features split a few points: the new dims attn_res_sig / engram_sig / ve_kind are in the key)
    # 935 / 1111 before round 3, max_context_unknown (a context read where the old 4096 default stood -- LLaMA-1's
    # max_sequence_length 2048, a class default -- merges two context-aware classes with their read-context twins)
    # 935 / 1109 before round 3, the zoo classifier fixes (Phi-3-small-8k, unplaced for want of a rope key it calls
    # rope_embedding_base, is placed)
    # 936 / 1110 before round 3, output heads as slots (the readout widths / signatures are in the key: the sentence embeddings, GRM's value
    # head, the two score-head rerankers and sundial split from or join the point set)
    # 946 / 1120 before round 3, kind_order (the layer-kind sequence kseq is in the key: the PseudoBrain parallel-GDN
    # conversions leave their sequential Qwen twins' classes)
    # 948 / 1122 before round 3, mamba_ffn_none / mamba1_lever (the FFN-less Mamba stacks' FFN widths are 0, so the two
    # mamba-130m checkpoints, which differed only in the mixer width read as an FFN width, merge)
    # 947 / 1121 before release-rel (the fr-positions step moved 947 -> 942: its readings and dedup keys)
    assert (len(P0), len(P1)) == (942, 1116)
    assert f0 == 297, f0                                    # default path untouched by the flag (was 306 (round 3 final, before the release-rel: the first-wave rules -- fr-mixer, fr-resnorm, fr-positions, fr-cost, fr-moe), 307 (round 3, mamba_ffn_none / mamba1_lever), 303 (round 3, kind_order), 296 (round 3, output heads as slots), 298 (round 3, zoo classifier fixes), 290 (round 3, batch 4b), 284, 278, 279, 271, 266, 268, 262, 256, 277, 263, 225, 222, 221, 219, 109, 101)
    assert f1 == 322, f1                                   # context axis on (was 328 (round 3 final, before the release-rel: the first-wave rules -- fr-mixer, fr-resnorm, fr-positions, fr-cost, fr-moe), 329 (round 3, mamba_ffn_none / mamba1_lever), 326 (round 3, kind_order), 320 (round 3, output heads as slots), 321 (round 3, zoo classifier fixes), 320 (round 3, max_context_unknown: unknown / unbounded contexts), 310 (round 3, batch 4b), 308 (round 3, batch 4a), 304 (round 3, batch 3b), 306, 300, 296, 291, 286, 304, 290, 253, 250, 252, 252, 136, 128)
    assert (ok1 & ~ok0b).sum() > 0                          # gained: local host reaches a wider target
    assert (ok0b & ~ok1).sum() > (ok1 & ~ok0b).sum()        # removed dominates: the context capacity floor


def test_frontier_definitions_headline_and_slices():
    """The HEADLINE frontier is capacity + every type wall that has a currency paid (payable):
    259 (round 3, batch 2b: positions; 260 after batch 2a, 254 after batch 1, 248 after the round-2 rules / rescues
    pass). Secondary, if every type wall were free (capacity + the gen walls): 46 (45). Stricter slice: 281 strictly
    exact (279, 273); looser: 231 (232, 224, 217) with the
    softmax <- SSM near-gap and the CLA carry granted free (mode="mixgap", the pre-measurement
    headline)."""
    from graph.currency_curve import n_frontier
    from graph.figures import load
    from graph.model_graph import _dedup_points, emul_matrix
    P = _dedup_points(load())

    # count equivalence CLASSES (as FRONTIER.md does): capacity_only creates ties
    # (same-capacity, different-type models are now mutually emulable), so a raw
    # not-dominated count would over-count the tie partners.
    def fsize(**kw):
        return n_frontier(emul_matrix(P, **kw))

    # headline-payable (2026-11-11): the old approx type-slice (70) became the headline, with the new
    # currency edges (hybrid / sliding / KV-sharing zeroing, AR <= diffusion, the LayerNorm mirror,
    # ALiBi <- learned-abs / partial-RoPE, the qk-norm / softcap regimes both ways, the thresholded
    # activation floors) and the relabels (relu2 split from ReLU, sliding layers named in layer_types /
    # sliding_windows / sliding_window_size, MiniMax-M1 hybrid, MiniMax-M2 softmax): 70 -> 66.
    # 66 -> 85 (act-floor-calibration): ACT_FLOOR_MAX 0.15 -> 0.0005, calibrated on real-model swaps
    # (act/act_swap_agreement.py); every activation floor edge is now a wall, folds / limits stay.
    # 85 -> 101 (mixer-gap-measure): the softmax <- SSM / linear / hybrid near-gap is refused -- its
    # softmax constructions pass the standard only at heads / depth / a norm change no host pays
    # (attn/mixer_gap_agreement.py); the old relation is mode="mixgap"
    # 101 -> 106 (cla-not-payable): the CLA (cross-layer KV sharing) width edge is refused too -- its
    # ~6e-2 pre-norm carry floor (block/prenorm_carry.py) is far above ACT_FLOOR_MAX (CLA_CARRY_PAYABLE)
    # 106 -> 104 (partial-rope-host): rope <- partial_rope is paid where the host rotates enough dims
    # per head (hf/partial_rope_host.py), and classify reads the rotary factor nested in rope_parameters
    # (Qwen3.5 / Gemma-4) and an MLA with no RoPE key as NoPE
    # 104 -> 218 (edge-case-audit, 106 -> 219 on its own): per-layer kinds, K = V, MLA latent norm, the
    # measured qk-norm / post-norm / bias / sink walls, the FEATURES and recurrent families, KV heads by
    # count, the Lean conditions; merged with the rot gate, which now gates full-RoPE hosts too (several
    # RoPE tables, RoPE + NoPE layers: rope_fits, replacing the audit's multi_rope feature)
    # 218 -> 219 (round-2 classification pass: drafters / assistants skipped, the dropped legacy families
    # read (GPT-Neo, Falcon-7B / 40B, MPT, XGLM, BioGPT, CTRL, GPT-1, OpenELM, InternLM-1, OLMo-1, LLaDA),
    # config-class defaults, renamed types, NoPE / router / sink / bias / GeLU-tanh labels, block-causal SDAR)
    # 219 -> 221 (round-2 attention / norm pass: qk-norm variants -- per head / whole projection / after RoPE /
    # weight-free / LayerNorm, nested only by gain settings; post-only split from sandwich norms; softcap split into
    # attention / final / none with the measured REGIME_PAYABLE; ALiBi slope schemes and the NoPE <- ALiBi switch;
    # the bias labels; KV-head packing, for RoPE targets only on hosts whose rotary takes a longrope table).
    # Joiners Molmo-7B-O, OLMoE-1B-7B-0125 (whole-projection qk-norm), Hunyuan-A13B (qk-norm after RoPE; its
    # HunyuanImage-3 host is fixed-rotary), MiniMax-Text-01-hf (swaps with MiniMax-M1-40k, now split by the
    # rotary flag); leaver jetmoe-8b (now hosted through packing)
    # 221 -> 258 (round-2 MoE / MLA pass): MoE <- MoE by expert duplication (Lean MoERouting.lean) instead of capacity
    # alone -- top-k fixed so the host's a multiple m, E_h >= m E_t, m e_h >= e_t, renorm / router score / group score
    # as each family's code or config allows (an unrenormalised softmax target only on its own kind with exactly m E_t
    # experts), extra experts sunk by a selection bias or a free dim, groups m = 1 under a top-2-sum score (+34 alone);
    # the shared expert's width on its own (+2); dense / MoE layer kinds (+1, and GLM-4.5-Air's AWQ twin splits off by
    # its padded dense width); MLA latents an integer multiple of the target's (hard-coded latent eps: +2); DSA as dense
    # attention (-4: DeepSeek-V3.2, Naive-N0.5, Ring-mini-sparse, LongCat-Flash-Chat leave); a dense host for a MoE only
    # top-1 unit gate at +E channels / +2E(E-1) units (moe-loop-rescue's measured rule; 0 at the margin).
    # Joiners (42), all MoE: DeepSeek-V2 / V2-Lite, ERNIE-4.5-21B, Ling-lite / Ling-2.6-1T, Trinity-Mini / Nano,
    # Laguna-XS.2, granite-4.0-h-tiny / tiny-preview / swash, Mistral-Small-4, sarvam-30b, Param2, Kolibri-1, Nemotron-3
    # Puzzle-75B / 3.5-Bolt / NemotronH-0.3B, gnani-evon, Qwen3.8-Flash-Next cuts, Gemma-4 REAP-14B, ... (FRONTIER.md);
    # leavers the four DSA models above and GLM-4.5-Air (to its AWQ twin)
    # 258 -> 271 (round-2 recurrent / diffusion pass): recurrent capacity per family (heads, state, head dims, conv
    # width, gate rank, key heads; Mamba-2's grouped gated norm as whole-group copies; GDN / KDA key head dim equal: q's
    # 1 / sqrt(dk) scale is in code), the recurrent families split into the variants that do not nest (Falcon-Mamba's
    # weight-free dt / B / C norms, Jamba's weighted ones, Falcon-H1 without its norm, PLaMo-2, OLMo-Hybrid's beta in
    # (0, 2), GDN-2, GKA, Hymba, SambaY, Zamba-1; KDA's lower bound is config, so lb codes nest the softplus one), the
    # Zamba shared block and DiffusionGemma's always-on self-conditioning embedding norm walled both ways, and
    # block-causal SDAR one class with diffusion under the mask currency (AR below both).
    # Joiners (14): Falcon-H1-1.5B-Base / 1.5B-Deep / 3B-Base / 7B-Base / Tiny-R-0.6B, Nemotron-H-4B-Base-8K,
    # Nemotron-Nano-12B-v2-Base, Zamba2-1.2B-instruct / 2.7B, mamba-2.8b-hf (Falcon-Mamba no longer hosts plain
    # Mamba), Qwen3.8-1.0B-A0.6B (GDN dk 64), GPT-S3-3M, Ling-2.6-flash, Wald-26B-A4B (DiffusionGemma no longer hosts
    # Gemma-4); leaver Qwen3-0.6B-diffusion-mdlm-v0.1 (an SDAR host plays it through an all-ones mask)
    # 271 -> 248 (round-2 rules / rescues / refutations pass): the config rules (a host sets the values its
    # family's code reads: the sliding window -> every sliding layer full (-9 alone), the rotary fraction (-5), softcap
    # values and q / k norm flags (exact slice only)); RoPE frequencies config-only (a fixed rotary sets theta alone:
    # it plays only fixed, default-or-own-type targets; zero-frequency slots only where its config sets them: +3);
    # the eps lever (qk-norm -5, sandwich post-norms / v-norm -3, MLA latents -2, Mamba-2 / GDN / KDA -3, MLA host for
    # an MQA / NoPE target 0); the start-token rules (learned-abs -1, sinks 0); activation width (0 alone) and GeLU erf
    # <-> tanh (-6); walls: untied targets on looped hosts but Nanbeige (0), a LayerNorm-beta target on an RMSNorm host
    # without an lm_head bias (+3), ALiBi on learned-abs (+2), a swigluoai target without a host up bias (0); the
    # compensators credit nothing (transitivity: every measured-family target hosts an unmeasured one); the review's
    # fixes (pure SSM layers recurrent, per-layer KV shapes, mixgap recurrent dims). Leave-one-out in the commit.
    # 248 -> 254 (round 3, batch 1: classification -- type tables, config reading, placement, dims_of; 30 specs,
    # round-3 batch 1; leave-one-out in the commit)
    # 254 -> 260 (round 3, batch 2a: qk-norm and output gates -- the gate split by granularity / bias / activation, the
    # q / k norms read from code, IQuest-Q1's key sink, Muse-Spark's o-norm; 12 specs, round-3 batch 2a;
    # leave-one-out in the commit)
    # 260 -> 259 (round 3, batch 2b: positions -- second RoPE tables (OLMo-3, PLaMo-3), the rotary widths read from code
    # (Step-3.5, K2-Horizon, StableLM-Epoch, ZAYA1-VL, Grug, Ling-1T, DeepSeek-VL2), LFM3's zero-tail tables
    # fraction-free, NoPE layers (Muse, Domyn-Edge, Kolibri-1, Grug, Zamba2-2.7B), the q_temp / q_temp_nope / fope /
    # rope_warp features, Parler's sinusoid, the scaled Falcon slopes, the remote MPT's learned table; 16 specs,
    # round-3 batch 2b; leave-one-out in the commit)
    # 259 -> 264 (round 3, batch 2c: windows and masks -- the window / layer schedules read from each family's code
    # (cost_wise_gemma, needle3, IQuest-Q1, Motif-3, Meditor, gaggle, Jet-Nemotron, Grug, PLaMo-3, EdgeRock, Ethoric,
    # Bluebird), Llama-4's chunk reach, sn38's inclusive band, nanowhale's full-window sink-free code, the swa_trunc /
    # dyn_mask / doc_mask / attn_out_sat features (Meditor -1, Doge -2, Ethoric / KLM -1, Qwen3-D-ROS -1); Phi-3-small
    # dense by a design decision; 8 specs changed code, round-3 batch 2c; leave-one-out in the commit)
    # 264 -> 272 (round 3, batch 2d: heads and MLA -- the per-layer KV shapes counted layer by layer (Laguna, DeciLM,
    # Domyn-Edge, MiMo), the V widths (MiMo, Hymba, needle3), DiffusionGemma's K = V full layers, the vha / diff_attn /
    # xsa / canon / cca / moa / mem_attn / mova / mla_q_raw / mla_qmix / kv_compress_v41 features, the grouped o_proj,
    # Step-3's query latent, ERNIE-5.5's MLA dims, Cyrah's GQA, clip_qkv's host lever; 19 specs changed code,
    # round-3 batch 2d; leave-one-out in the commit)
    # 272 -> 271 (round 3, batch 3a: MoE and router -- renorm / score / selection-bias / group codes read from each
    # family's code (Bailing V2 / V3, CENO, Arctic, Meditor, Step-3, Beam, Sarvam, Grug, Muse, Kolibri, Mark-SLM; the
    # 0-filled group limit of the first-release DeepSeek-V3 / Kimi-K2 code, Intern-S1-Pro's per-group router, ERNIE-5.5's
    # dual sigmoid, Meditor-2's two-level router), the router_mlp / moe_capacity / block_routing / expert_norm /
    # moe_branch_norm / par_moe_res / router_bias_logit features, the latent-norm / identity-slot / branch / block /
    # Arctic forms in moe_sig, router features only on built MoE layers, SparseMixer at m = 1; 21 specs changed code,
    # round-3 batch 3a; leave-one-out in the commit)
    # 271 -> 277 (round 3, batch 3b: FFN widths, activation clamps and the spot-check fixes -- a gated / MoE target on a
    # vanilla host walled (spot_glu_vanilla), a vanilla target on a gated / MoE host and an up-biased target need the
    # host's up bias (spot_vanilla_glu_bias: +7, the biased GLU targets granite-code / InternLM-7B / basal / hubble /
    # manaca / PowerLM lose their bias-free hosts), no act-width fit on a swigluoai host (spot_act_width_oai: +1), the
    # act_clip host lever (b_act_clip_host: -3), the FFN widths read from code, needle's HadamardMLP, Dory's ffn_norm, the
    # compensators removed (+0); round-3 batch 3; leave-one-out in the commit)
    # 277 -> 283 (round 3, batch 4a: recurrent mixers and loops -- the recurrent families read from code (Qwen4-exp's
    # sigmoid-gated GDN, AliceAI's KDA, fla DeltaNet, Cyrah's LFM2 conv, Solar-Open2's negative-eigenvalue KDA, GPT-X3 /
    # KLM's GDN-2), the hybrid schedules (GPT-X3, KLM, Edge0, Ming-3, Solar-Open2), the fla heads (the fla transformer and
    # DeltaNet placed), the GLA / Jet dims, the all-recurrent stacks without attention features, the loop depths (Ouro,
    # CFRD, LFM3-RD, HRM, the double loop split) and the loop-boundary walls (loop_bnorm / loop_bnorm_cfg, loop_inject,
    # loop_kv_mix, hrm_two_state, early_exit, rd_controller), the depth kind sums and the mixgap CLA-carry routing; 18 specs
    # changed code, round-3 batch 4;
    # leave-one-out in the commit)
    # 283 -> 291 (round 3, batch 4b: residual streams and extra inputs -- the hyper-connection variants by code (hc_dsv4 /
    # dsv4_nano / dsv41 / glm5_next / xing / hy4 / qwen4_exp / motif / needle / san, walled from each other), needle / SAN
    # lanes, CENO post_ln, the residual gains / bias / Muse-Spark gate, backout, Dory's nGPT sphere, the n-gram memories
    # by code (Qwen4-exp's PLE an engram_qwen4), the value-embedding / AttnRes keys and signatures, CFRD's norm shift and
    # residual gate, Bolmo's byte patches; 13 specs changed code, round-3 batch 4; leave-one-out in the commit)
    # 291 -> 289 (round 3, the zoo's classifier fixes -- dist/r3_from_zoo.md, branch zoo-errors: Apertus-1.5's ungated
    # xIELU MLP (-1), TR-MOE-306's legacy top_k 2 (-1), Kanana-2-tiny's q / k norm, Gemma3-MoE's sandwich norms,
    # Phi-3-small's code-fixed biases and the -8k config placed (+0 each); leave-one-out in the commit)
    # 289 -> 297 (round 3, output heads and inputs as slots -- dist/r3_nonlm.md, branch host-nonlm, a design decision on the readout
    # decision: the two score-head rerankers and sundial kept (nonlm_kept, leave-one-out -2), the head / input slots with
    # the readout rule -- mean pooling / L2 embeddings walled from the LMs both ways (nonlm_heads, -5))
    # 297 -> 301 (round 3, kind_order: the layer kinds matched in ORDER, not by count -- granite-4.0-h-350m, Jamba2-3B,
    # Nemotron-3-Super lose their reordered hosts; the PseudoBrain parallel layers split from their sequential twins)
    # 301 -> 300 (round 3, mamba_ffn_none + mamba1_lever: the pure Mamba stacks have no FFN slot -- FFN none, hosted by
    # any FFN by zeroing it -- and a plain Mamba-1 plays Falcon-Mamba by the dt / B / C norm lever, payable, measured:
    # mamba-2.8b leaves the frontier for falcon-mamba-7b)
    # 300 -> 293 (release-rel: the first-wave rules, per step in its commits -- fr-mixer 300 -> 298, fr-resnorm -> 296,
    # fr-positions -> 297, fr-cost -> 297, fr-moe -> 293)
    assert fsize(mode="payable") == 293, "HEADLINE (payable) frontier drifted"
    assert fsize() == 293                                  # the default IS the headline
    assert fsize(mode="mixgap") == 265, "free-mixer-gap slice drifted"   # was 270 (round 3 final, before release-rel), 271 (round 3, mamba_ffn_none / mamba1_lever), 269 (round 3, kind_order), 261 (round 3, output heads as slots), 263 (round 3, zoo classifier fixes), 254 (round 3, batch 4b), 249 (round 3, batch 4a), 242 (round 3, batch 3b), 244 (round 3, batch 3a), 236 (round 3, batch 2d), 231 (round 3, batch 2c), 232 (round 3, batch 2b), 224 (round 3, batch 2a), 217 (round 3, batch 1), 239, 201, 198, 198, 199 (audit), 83, 85
    # the old pure-capacity headline, now the secondary: 41 -> 40 (AR <= diffusion: a diffusion host
    # now dominates one AR model); 51 -> 52 (round 2); 52 -> 46 (round 2 attention / norm pass: KV-head packing);
    # 46 -> 45 (round 2 recurrent / diffusion pass: block <= diffusion, a diffusion host now dominates SDAR-8B-Chat)
    # 45 -> 46 (round 3, batch 1: classification); 46 -> 48 (round 3, batch 2d: each attention shape needs a host shape
    # with its head dim and V width; the new head / latent dims split points); 48 -> 49 (round 3, batch 4a: the new
    # recurrent families and loop points)
    assert fsize(mode="capacity") == 49, "secondary (every wall free) frontier drifted"
    assert fsize(capacity_only=True) == 49                 # legacy flag
    assert fsize(mode="nomix") == 293, "legacy nomix alias = the headline"      # was 300 (round 3 final, before release-rel), 301 (round 3, mamba_ffn_none / mamba1_lever), 297 (round 3, kind_order), 289 (round 3, output heads as slots), 291 (round 3, zoo classifier fixes), 283 (round 3, batch 4b), 277 (round 3, batch 4a), 271 (round 3, batch 3b), 272 (round 3, batch 3a), 264 (round 3, batch 2d), 259 (round 3, batch 2c), 260 (round 3, batch 2b), 254 (round 3, batch 2a), 248 (round 3, batch 1), 271, 258, 221, 219, 218, 219, 104, 106, 101, 84, 79
    assert fsize(mode="exact") == 320, "strictly-exact slice drifted"           # was 321 (round 3 final, before release-rel), 317 (round 3, kind_order), 309 (round 3, output heads as slots), 311 (round 3, zoo classifier fixes), 304 (round 3, batch 4b), 299 (round 3, batch 4a), 292 (round 3, batch 3b), 286 (round 3, batch 2d), 281 (round 3, batch 2c), 279 (round 3, batch 2a), 273 (round 3, batch 1), 275, 262, 231, 233, 233 (audit), 122, 126


def test_context_axis_headline_needs_context_aware_dedup():
    """The opt-in context axis on the secondary (capacity) relation: 80 frontier classes on
    context-aware points (75 before round 3, max_context_unknown -- the context read from config / code, unknown or
    unbounded instead of 4096 --, 72 before round 3, batch 2d -- per-layer head shapes, V widths --, 73 before round 3, batch 2c -- the windows read from code, Llama-4's chunk reach --
    71 before round 3, batch 2b -- the context gates cap the credited context, PLAN decision 3 --
    70 before round 3, batch 1, 72 before the round-2 recurrent / diffusion pass's block <= diffusion, 79 before the round-2
    attention / norm pass's KV-head packing, 81 before the round-2
    classification pass, 68 before the edge-case audit, 69 before AR <= diffusion). Deduping on the
    flags-off key (no max_context / eff_window) merges classes that mix contexts and keeps only the first
    member's (76; 71 before round 3, max_context_unknown, 70 before round 3, batch 3a, 67 before round 3, batch 2d, 68 before round 3, batch 2c, 66 before round 3, batch 2b, 68 before the recurrent / diffusion pass, 75 before packing, 78 before round 2, 67 before the audit). On
    the headline (payable) relation the context axis gives 325
    classes (326 before round 3, mamba_ffn_none / mamba1_lever, 323 before round 3, kind_order, 316 before round 3, output heads as slots, 317 before round 3, the zoo classifier fixes, 316 before round 3, max_context_unknown, 306 before round 3, batch 4b, 304 before round 3, batch 4a, 300 before round 3, batch 3b, 302 before round 3, batch 3a, 296 before round 3, batch 2d, 292 before round 3, batch 2c, 287 before round 3, batch 2a, 282 before round 3, batch 1, 298 before the round-2 rules / rescues pass, 285 before the round-2 recurrent / diffusion pass, 250 before the round-2 MoE / MLA pass, 131 before the
    audit, 127 while the CLA carry was granted free,
    116 with the free mixer gap too, 95 before the activation calibration)."""
    import graph.model_graph as mg
    from graph.currency_curve import n_frontier
    from graph.figures import load
    from graph.model_graph import _dedup_points, emul_matrix
    M = load()
    P0 = _dedup_points(M)                                   # flags-off key
    mg.USE_CONTEXT = True
    try:
        P1 = _dedup_points(M)
        # round 2 recurrent / diffusion pass: block <= diffusion (capacity -2), the recurrent capacity / variant walls
        # round 3, batch 2b (PLAN decision 3): a context-gated point (Qwen-1 logn / dynamic NTK, the Llama-4-style
        # temperatures, Meditor) is credited only up to its gate, as target and host (+2 in both)
        # round 3, batch 2c: the windows read from code move eff_window (Motif 129, gaggle 513, PLaMo-3 / Meditor 2048,
        # Llama-4's chunk 8192, sn38 513, ...), and the 2c features wall / split points
        # round 3 (max_context_unknown): the max context read from config / code -- unknown (BLOOM) supplies and is
        # supplied only by unknown or unbounded, unbounded (Mamba, RecurrentGemma) needs an unbounded host
        assert n_frontier(emul_matrix(P1, mode="capacity")) == 80    # was 75 (round 3, max_context_unknown), 72 (round 3, batch 2d), 73 (round 3, batch 2c), 71 (round 3, batch 2b), 70 (round 3, batch 1), 72, 79, 81 (before round 2)
        assert n_frontier(emul_matrix(P0, mode="capacity")) == 76    # the old dedup artifact (was 71 (round 3, max_context_unknown), 70 (round 3, batch 3a), 67 (round 3, batch 2d), 68 (round 3, batch 2c), 66 (round 3, batch 2b), 68, 75, 78)
        assert n_frontier(emul_matrix(P1, mode="payable")) == 319    # was 325 (round 3 final, before release-rel), 326 (round 3, mamba_ffn_none / mamba1_lever), 323 (round 3, kind_order), 316 (round 3, output heads as slots), 317 (round 3, zoo classifier fixes), 316 (round 3, max_context_unknown), 306 (round 3, batch 4b), 304 (round 3, batch 4a), 300 (round 3, batch 3b), 302 (round 3, batch 3a), 296 (round 3, batch 2d), 292 (round 3, batch 2c), 287 (round 3, batch 2a), 282 (round 3, batch 1), 298, 285, 250, 248, 249, 252 (audit alone), 131, 132, 127
        assert n_frontier(emul_matrix(P1, mode="mixgap")) == 293     # was 297 (round 3 final, before release-rel), 298 (round 3, mamba_ffn_none / mamba1_lever), 297 (round 3, kind_order), 290 (round 3, output heads as slots), 291 (round 3, zoo classifier fixes), 289 (round 3, max_context_unknown), 278 (round 3, batch 4b), 277 (round 3, batch 4a), 272 (round 3, batch 3b), 275 (round 3, batch 3a), 269 (round 3, batch 2d), 265 (round 3, batch 2c), 259 (round 3, batch 2a), 252 (round 3, batch 1), 268, 267, 232, 230, 229, 232 (audit alone), 114, 116
    finally:
        mg.USE_CONTEXT = False
