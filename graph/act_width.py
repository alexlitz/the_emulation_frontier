"""Activation walls paid in FFN width: the k-unit activation swap.

PORTED in round 2, pass 5: graph/model_graph.emul_matrix calls act_k (FFN x k, shifts only with a host FFN bias,
dims ffn_bias); frontier_effect() below now measures it on top of a relation that already has it.

The headline act rule (graph/model_graph.py ACT_FLOORS / ACT_FLOOR_MAX) pays a swap only if ONE host
unit fits the target activation to within ACT_FLOOR_MAX = 0.0005 (calibrated on real models:
act/act_swap_agreement.py). This module proposes a second currency: WIDTH. A target unit Y(a) * b
(GLU; ungated Y(a)) is rebuilt from k host units  sum_i alpha_i X(beta_i a + gamma_i) * b  -- gate rows
scaled (beta_i) and shifted (gamma_i), up rows copied, down-projection columns scaled (alpha_i).
That is exact weight-setting at FFN width x k; the only error is the k-term fit, which must be
<= ACT_FLOOR_MAX on the range real pre-activations reach.

  * shifts gamma_i need a gate bias in the HOST (host_gate_bias). A constant residual channel would
    also do, but none exists under RMSNorm (prenorm_readout_obstruction,
    proofs/PROOFS.md), so a bias-free host is restricted to gamma_i = 0 fits.
  * There is no free linear term (a swish host pays 2 units for one: X(a) - X(-a) = a) and no free
    constant (it needs a shift).
  * k is measured on [-R_TABLE, R_TABLE] with R_TABLE = 160, which covers every model measured
    (max |pre-activation|: gpt2 63, Pythia 98 - 109, Qwen2.5-1.5B 72, nanochat 151).

Data: K_NOSHIFT / K_SHIFT (act/act_width_fit.py -> act/data/act_width_fits.json), AGREEMENT
(act/act_width_agreement.py -> act/data/act_width_agreement.jsonl). Rule: act_width_ok.
This module does not change emul_matrix; frontier_effect() applies the rule on top of it.

Headline effect (integration relation of 2026-11-13, 3,933 models / 842 points, frontier 104): the
rule adds 5,482 point edges and the frontier drops to 98. Six leave -- falcon-11B, parler-tts-large-v1,
Pharia-1-LLM-7B-control-hf, starcoder2-15b (GeLU <- SiLU x6, under Llama-3_1-Nemotron-Ultra-253B /
FlexOlmo-7x7B-1T) and Dory-Beam-4B, llm-jp-3-980m (SiLU <- GeLU x3, under Gemma-4-31B / gemma-7b).
The other GeLU walls (gpt-neox-20b, polyglot-ko-12.8b, RedPajama-7B, Fanar-2-27B, ...) stay: no
SiLU host with 6x their FFN also covers them on every other axis. Sensitivity, one k for both
GeLU <-> SiLU directions: x1 (the swap free) -> 89, x2 -> 94, x3 -> 96, x4 -> 98, x5 - x8 -> 100 (the
same 4 GeLU models leave). Bias-free hosts only: the same 98; if
every host had a gate bias: 97 (DanTagGen-beta, mish <- SiLU x6, also leaves).

  python -m graph.act_width        # the k table and the headline (frontier) effect
"""
from __future__ import annotations

K_MAX = 8                         # widest k searched (act/act_width_fit.py --kmax)
R_TABLE = 160.0                   # fit range of the K tables
ACT_FLOOR_MAX = 0.0005            # = graph.model_graph.ACT_FLOOR_MAX (tests check they agree)

# K_NOSHIFT[(target, host)] = minimal k with sum_{i<=k} alpha_i X(beta_i a) within ACT_FLOOR_MAX of Y
# on [-160, 160]; K_SHIFT the same with gate shifts gamma_i. Absent = no k <= K_MAX found (a wall).
# k = 1 comes from the headline one-unit rule (ONE_UNIT: its exact folds, limits and floors), not from
# these tables. Upper bounds: the fit is a local search (act/act_width_fit.py), so a better optimiser
# could only lower a k.
#
# What falls out (2026-11-13):
#   * the swish cluster is reachable both ways at modest width: SiLU / swigluoai <- GeLU at x3,
#     GeLU <- SiLU at x6 (x5 with a gate bias; k = 5 shift-free misses by 1%: 5.07e-4),
#     GeLU <- swigluoai at x6 (x5 with bias). "GeLU" is the zoo's erf AND tanh GeLU (FIT_ROWS): the
#     larger k of the two.
#   * mish needs shifts both ways: mish's odd part is not linear, and a bias-free swish host's is
#     (X(a) - X(-a) = a), so no shift-free width reaches it (floor 0.10 on [-160, 160]; Lean:
#     shiftfree_silu_obstruction). With a gate bias: SiLU / GeLU <- mish x5, swigluoai <- mish x6,
#     mish <- SiLU x6, mish <- GeLU / swigluoai x7.
#   * ReLU <- relu2 / xIELU at x2 with a gate bias ((X(a + d) - X(a)) / 2d -> relu a: a limit).
#   * walls at every k <= 8: squared ReLU and xIELU targets (quadratic growth: errors ~1e2 - 1e3 on
#     [-160, 160]); situ targets (best: 9e-4 at k = 8 from swigluoai with shifts); situ and xIELU hosts
#     for the swish cluster (>= 3.7e-3 at k = 8); ReLU hosts (piecewise linear, >= 0.08); relu2 hosts
#     for curves (>= 7e-3). At the real models' own ranges too: Apertus' xIELU (|a| <= 38) from SiLU with
#     shifts is 0.061 at k = 8 (default-init xIELU; the trained one is farther), nanochat's relu2
#     (|a| <= 151) 35; SiLU <- situ (the Kimi-K3 walls) 5.1e-3.
FIT_ROWS = {"GeLU": ("GeLU", "gelu_tanh")}      # zoo act -> the fitted function(s) it stands for
TARGET_ALIAS = {"GeLU_tanh": "GeLU"}            # the act slot splits the tanh GeLU off (round 2): its target rows
K_NOSHIFT = {("SiLU", "GeLU"): 3, ("swigluoai", "GeLU"): 3, ("GeLU", "SiLU"): 6, ("GeLU", "swigluoai"): 6}
K_SHIFT = {("SiLU", "GeLU"): 3, ("swigluoai", "GeLU"): 3, ("GeLU", "SiLU"): 5, ("GeLU", "swigluoai"): 5,
           ("GeLU", "mish"): 5, ("SiLU", "mish"): 5, ("swigluoai", "mish"): 6, ("mish", "SiLU"): 6,
           ("mish", "GeLU"): 7, ("mish", "swigluoai"): 7, ("ReLU", "relu2"): 2, ("ReLU", "xielu"): 2}
# Round 2, pass 5: the tanh-GeLU HOSTS (gelu_pytorch_tanh / gelu_new: Gemma, GPT-2) fitted too (python -m
# act.act_width_fit --R 160 --targets SiLU swigluoai mish GeLU --hosts gelu_tanh; act/data/act_width_fits.json
# '<target>|gelu_tanh|160|*'): SiLU 3 (1.9e-4), swigluoai 3 (1.1e-4) -- the erf host's k -- and mish NONE (shift fits
# 5.2e-4 at k = 7 and 8, just over ACT_FLOOR_MAX; the erf host reached 4.7e-4 at k = 7). The relation's GeLU <->
# GeLU_tanh one-unit floor is a 2-cycle, so transitivity forces ONE k for both hosts: a "GeLU" host stands for erf AND
# tanh (TARGET_ALIAS maps the tanh host onto these rows), the larger k -- mish <- GeLU is dropped.
HOST_GELU_TANH = {"SiLU": 3, "swigluoai": 3, "mish": None, "GeLU": 1}
K_SHIFT.pop(("mish", "GeLU"))
# the k-term fit errors behind the tables (and the nearest misses), on [-160, 160]:
# FIT_ERR[(fitted target, host, shift)] = [max error at k = 1, 2, ...]
FIT_ERR = {
    ('SiLU', 'GeLU', False): [0.0247, 0.00172, 0.000172, 2.1e-05, 2.1e-05, 2.1e-05, 2.1e-05, 2.1e-05],
    ('swigluoai', 'GeLU', False): [0.0145, 0.00101, 0.000101, 1.24e-05, 5.36e-06, 5.36e-06, 1.81e-06, 1.81e-06],
    ('GeLU', 'SiLU', False): [0.0139, 0.00518, 0.00184, 0.000959, 0.000507, 0.000334, 0.000233, 0.000232],
    ('gelu_tanh', 'SiLU', False): [0.0142, 0.00554, 0.00194, 0.000989, 0.000557, 0.000365, 0.000265, 0.000264],
    ('GeLU', 'swigluoai', False): [0.0139, 0.00519, 0.00291, 0.000892, 0.000613, 0.00048, 0.000423, 0.000397],
    ('gelu_tanh', 'swigluoai', False): [0.0142, 0.00526, 0.00202, 0.00121, 0.00069, 0.000498, 0.000415, 0.000381],
    ('mish', 'SiLU', False): [0.102, 0.101, 0.101, 0.101],
    ('SiLU', 'mish', False): [0.127, 0.126, 0.0536, 0.00718, 0.0045, 0.00223, 0.00195, 0.00195],
    ('GeLU', 'mish', False): [0.0801, 0.0797, 0.0315, 0.0139, 0.00956, 0.00679, 0.00537, 0.00462],
    ('gelu_tanh', 'mish', False): [0.0802, 0.0798, 0.0316, 0.0142, 0.00947, 0.00654, 0.00531, 0.00469],
    ('situ', 'SiLU', False): [3.49, 1.94],
    ('SiLU', 'GeLU', True): [0.0247, 0.00172, 0.000172, 2.1e-05, 1.84e-05, 1.5e-05, 1.5e-05, 1.5e-05],
    ('swigluoai', 'GeLU', True): [0.0145, 0.00101, 0.000101, 1.24e-05, 2.47e-06, 2.47e-06, 8.29e-07, 8.29e-07],
    ('GeLU', 'SiLU', True): [0.0139, 0.00518, 0.0018, 0.000941, 0.000117, 0.000117, 5.55e-05, 5.55e-05],
    ('gelu_tanh', 'SiLU', True): [0.0142, 0.00552, 0.00193, 0.000979, 0.000119, 0.000119, 0.000119, 0.000119],
    ('GeLU', 'swigluoai', True): [0.0139, 0.00519, 0.00272, 0.000743, 0.000486, 5.12e-05, 3.76e-05, 3.76e-05],
    ('gelu_tanh', 'swigluoai', True): [0.0142, 0.00526, 0.00199, 0.000488, 0.000488, 0.000193, 9.32e-05, 6.21e-05],
    ('GeLU', 'mish', True): [0.0449, 0.0249, 0.0104, 0.000948, 0.000125, 0.000121, 0.000114, 0.000114],
    ('gelu_tanh', 'mish', True): [0.045, 0.025, 0.011, 0.000996, 0.000277, 0.000221, 0.000221, 0.000221],
    ('SiLU', 'mish', True): [0.0707, 0.0388, 0.00702, 0.00197, 0.000127, 7.71e-05, 7.71e-05, 7.71e-05],
    ('swigluoai', 'mish', True): [0.0422, 0.0232, 0.00376, 0.00115, 0.000573, 0.000364, 0.00032, 0.00029],
    ('mish', 'SiLU', True): [0.0628, 0.0288, 0.00723, 0.00492, 0.00142, 0.000395, 0.000384, 0.000384],
    ('mish', 'GeLU', True): [0.0674, 0.0344, 0.015, 0.00846, 0.00298, 0.000647, 0.000471, 0.000471],
    ('mish', 'swigluoai', True): [0.0628, 0.0288, 0.00723, 0.00312, 0.00178, 0.000838, 0.000493, 0.000418],
    ('ReLU', 'relu2', True): [32.2, 6.94e-05, 6.94e-05, 6.94e-05, 6.94e-05, 6.94e-05, 6.94e-05, 6.94e-05],
    ('ReLU', 'xielu', True): [10.9, 7.51e-05, 7.51e-05, 7.51e-05, 7.51e-05, 7.51e-05, 7.51e-05, 7.51e-05],
    ('situ', 'SiLU', True): [1.95, 0.206, 0.058, 0.0176, 0.00294, 0.00145, 0.00145, 0.00145],
    ('situ', 'swigluoai', True): [1.95, 0.206, 0.058, 0.0162, 0.00286, 0.00158, 0.00095, 0.000897],
    ('SiLU', 'xielu', True): [10.9, 0.177, 0.118, 0.11, 0.00573, 0.00447, 0.00415, 0.00372],
    ('SiLU', 'situ', True): [11.4, 1.83, 0.517, 0.349, 0.102, 0.0181, 0.00693, 0.00507],
}

# AGREEMENT[(model, target, host, k, shift)] = (top1, ex32, ex64): every FFN activation of a real
# checkpoint replaced by the K-table k-unit fit (on [-160, 160], which holds the model's own
# pre-activations |a| <= R), fp32, scored as in act/act_swap_agreement.py on the original model's
# greedy 64-token continuations of 24 prompts (strict payable: top1 >= 0.99 and ex32 >= 0.90).
# The threshold carries over: at the table's k every model passes; below it the models split, and
# the boundary sits where the one-unit calibration put it -- fits at 1e-3 fail on gpt2-medium
# (top1 99.7%, 83%), 5.6e-4 passes; 1.7e-3 fails on Qwen2.5-1.5B / Llama / SmolLM2 (79 - 83%), 1.7e-4
# passes. Easier models pass earlier (Pythia-1.4b already at 1.8e-3, k = 3).
AGREEMENT = {
    ('EleutherAI/pythia-1.4b', 'GeLU', 'SiLU', 2, False): (0.9935, 0.7917, 0.75),   # fit err 0.00518, |a| <= 99
    ('EleutherAI/pythia-1.4b', 'GeLU', 'SiLU', 3, False): (0.9993, 0.9583, 0.9583),   # fit err 0.00184, |a| <= 99
    ('EleutherAI/pythia-1.4b', 'GeLU', 'SiLU', 4, False): (0.9987, 0.9583, 0.9167),   # fit err 0.000959, |a| <= 99
    ('EleutherAI/pythia-1.4b', 'GeLU', 'SiLU', 5, False): (1.0, 1.0, 1.0),   # fit err 0.000507, |a| <= 99
    ('EleutherAI/pythia-1.4b', 'GeLU', 'SiLU', 6, False): (1.0, 1.0, 1.0),   # fit err 0.000334, |a| <= 99
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 1, False): (0.9759, 0.4167, 0.2083),   # fit err 0.0139, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 2, False): (0.9948, 0.8333, 0.75),   # fit err 0.00518, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 3, False): (0.998, 0.9583, 0.875),   # fit err 0.00184, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 4, False): (0.998, 0.9583, 0.875),   # fit err 0.000967, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 5, False): (1.0, 0.9583, 0.9583),   # fit err 0.000509, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 6, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000334, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 7, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000233, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 8, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000232, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 1, True): (0.9759, 0.4167, 0.2083),   # fit err 0.0139, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 2, True): (0.9948, 0.8333, 0.75),   # fit err 0.00518, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 3, True): (0.998, 0.9583, 0.875),   # fit err 0.0018, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 4, True): (0.998, 0.9583, 0.875),   # fit err 0.000953, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 5, True): (1.0, 1.0, 1.0),   # fit err 0.000117, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 6, True): (1.0, 1.0, 1.0),   # fit err 0.000117, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 7, True): (1.0, 1.0, 1.0),   # fit err 5.55e-05, |a| <= 109
    ('EleutherAI/pythia-160m', 'GeLU', 'SiLU', 8, True): (1.0, 1.0, 1.0),   # fit err 5.55e-05, |a| <= 109
    ('EleutherAI/pythia-410m', 'GeLU', 'SiLU', 2, False): (0.9915, 0.75, 0.625),   # fit err 0.00518, |a| <= 108
    ('EleutherAI/pythia-410m', 'GeLU', 'SiLU', 3, False): (0.9974, 0.875, 0.875),   # fit err 0.00184, |a| <= 108
    ('EleutherAI/pythia-410m', 'GeLU', 'SiLU', 4, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000959, |a| <= 108
    ('EleutherAI/pythia-410m', 'GeLU', 'SiLU', 5, False): (1.0, 1.0, 1.0),   # fit err 0.000507, |a| <= 108
    ('EleutherAI/pythia-410m', 'GeLU', 'SiLU', 6, False): (1.0, 1.0, 1.0),   # fit err 0.000334, |a| <= 108
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 1, False): (0.9681, 0.3333, 0.1667),   # fit err 0.0142, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 2, False): (0.9896, 0.625, 0.5),   # fit err 0.00554, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 3, False): (0.9954, 0.8333, 0.75),   # fit err 0.00194, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 4, False): (0.998, 0.9583, 0.875),   # fit err 0.000989, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 5, False): (0.9987, 0.9583, 0.9167),   # fit err 0.000559, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 6, False): (0.9987, 0.9583, 0.9167),   # fit err 0.000365, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 7, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000265, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 8, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000264, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 4, True): (0.998, 0.9583, 0.875),   # fit err 0.000979, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 5, True): (0.9993, 1.0, 0.9583),   # fit err 0.000119, |a| <= 12
    ('google/gemma-2-2b', 'GeLU', 'SiLU', 6, True): (0.9993, 1.0, 0.9583),   # fit err 0.000119, |a| <= 12
    ('gpt2', 'GeLU', 'SiLU', 1, False): (0.9714, 0.2917, 0.2083),   # fit err 0.0142, |a| <= 63
    ('gpt2', 'GeLU', 'SiLU', 2, False): (0.9909, 0.5833, 0.5417),   # fit err 0.00554, |a| <= 63
    ('gpt2', 'GeLU', 'SiLU', 3, False): (0.998, 0.9167, 0.875),   # fit err 0.00194, |a| <= 63
    ('gpt2', 'GeLU', 'SiLU', 4, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000989, |a| <= 63
    ('gpt2', 'GeLU', 'SiLU', 5, False): (0.9987, 0.9167, 0.9167),   # fit err 0.000559, |a| <= 63
    ('gpt2', 'GeLU', 'SiLU', 6, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000365, |a| <= 63
    ('gpt2', 'GeLU', 'SiLU', 7, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000265, |a| <= 63
    ('gpt2', 'GeLU', 'SiLU', 8, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000264, |a| <= 63
    ('gpt2-medium', 'GeLU', 'SiLU', 2, False): (0.9844, 0.5, 0.375),   # fit err 0.00554, |a| <= 78
    ('gpt2-medium', 'GeLU', 'SiLU', 3, False): (0.9967, 0.7917, 0.7917),   # fit err 0.00194, |a| <= 78
    ('gpt2-medium', 'GeLU', 'SiLU', 4, False): (0.9967, 0.8333, 0.7917),   # fit err 0.000989, |a| <= 78
    ('gpt2-medium', 'GeLU', 'SiLU', 5, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000557, |a| <= 78
    ('gpt2-medium', 'GeLU', 'SiLU', 6, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000365, |a| <= 78
    ('HuggingFaceTB/SmolLM2-135M', 'SiLU', 'GeLU', 1, False): (0.972, 0.2083, 0.0833),   # fit err 0.0247, |a| <= 50
    ('HuggingFaceTB/SmolLM2-135M', 'SiLU', 'GeLU', 2, False): (0.9961, 0.8333, 0.7917),   # fit err 0.00172, |a| <= 50
    ('HuggingFaceTB/SmolLM2-135M', 'SiLU', 'GeLU', 3, False): (1.0, 1.0, 1.0),   # fit err 0.000172, |a| <= 50
    ('HuggingFaceTB/SmolLM2-135M', 'SiLU', 'GeLU', 4, False): (1.0, 1.0, 1.0),   # fit err 2.1e-05, |a| <= 50
    ('HuggingFaceTB/SmolLM2-135M', 'SiLU', 'mish', 4, True): (0.9974, 0.875, 0.8333),   # fit err 0.00197, |a| <= 50
    ('HuggingFaceTB/SmolLM2-135M', 'SiLU', 'mish', 5, True): (1.0, 1.0, 1.0),   # fit err 0.000127, |a| <= 50
    ('HuggingFaceTB/SmolLM2-135M', 'SiLU', 'mish', 6, True): (1.0, 1.0, 1.0),   # fit err 7.71e-05, |a| <= 50
    ('Qwen/Qwen2.5-0.5B', 'SiLU', 'GeLU', 1, False): (0.9518, 0.1667, 0.0417),   # fit err 0.0247, |a| <= 48
    ('Qwen/Qwen2.5-0.5B', 'SiLU', 'GeLU', 2, False): (0.998, 0.9583, 0.875),   # fit err 0.00172, |a| <= 48
    ('Qwen/Qwen2.5-0.5B', 'SiLU', 'GeLU', 3, False): (0.9993, 0.9583, 0.9583),   # fit err 0.000172, |a| <= 48
    ('Qwen/Qwen2.5-0.5B', 'SiLU', 'GeLU', 4, False): (1.0, 1.0, 1.0),   # fit err 2.1e-05, |a| <= 48
    ('Qwen/Qwen2.5-1.5B', 'SiLU', 'GeLU', 1, False): (0.9551, 0.2917, 0.125),   # fit err 0.0247, |a| <= 73
    ('Qwen/Qwen2.5-1.5B', 'SiLU', 'GeLU', 2, False): (0.9954, 0.7917, 0.7083),   # fit err 0.00172, |a| <= 73
    ('Qwen/Qwen2.5-1.5B', 'SiLU', 'GeLU', 3, False): (1.0, 1.0, 1.0),   # fit err 0.000172, |a| <= 73
    ('Qwen/Qwen2.5-1.5B', 'SiLU', 'GeLU', 4, False): (1.0, 1.0, 1.0),   # fit err 2.1e-05, |a| <= 73
    ('Qwen/Qwen2.5-1.5B', 'SiLU', 'mish', 4, True): (0.9967, 0.8333, 0.7917),   # fit err 0.00197, |a| <= 73
    ('Qwen/Qwen2.5-1.5B', 'SiLU', 'mish', 5, True): (1.0, 1.0, 1.0),   # fit err 0.000127, |a| <= 73
    ('Qwen/Qwen2.5-1.5B', 'SiLU', 'mish', 6, True): (1.0, 1.0, 1.0),   # fit err 7.71e-05, |a| <= 73
    ('meta-llama/Llama-3.2-1B', 'SiLU', 'GeLU', 1, False): (0.9492, 0.125, 0.0833),   # fit err 0.0247, |a| <= 40
    ('meta-llama/Llama-3.2-1B', 'SiLU', 'GeLU', 2, False): (0.9967, 0.8333, 0.7917),   # fit err 0.00172, |a| <= 40
    ('meta-llama/Llama-3.2-1B', 'SiLU', 'GeLU', 3, False): (1.0, 1.0, 1.0),   # fit err 0.000172, |a| <= 40
    ('meta-llama/Llama-3.2-1B', 'SiLU', 'GeLU', 4, False): (1.0, 1.0, 1.0),   # fit err 2.1e-05, |a| <= 40
    ('meta-llama/Llama-3.2-1B', 'SiLU', 'mish', 4, True): (0.9935, 0.7083, 0.625),   # fit err 0.00197, |a| <= 40
    ('meta-llama/Llama-3.2-1B', 'SiLU', 'mish', 5, True): (1.0, 1.0, 1.0),   # fit err 0.000127, |a| <= 40
    ('meta-llama/Llama-3.2-1B', 'SiLU', 'mish', 6, True): (1.0, 1.0, 1.0),   # fit err 7.71e-05, |a| <= 40
    ('AlexKoff88/gpt-oss-tinystories', 'swigluoai', 'GeLU', 1, False): (0.9811, 0.625, 0.25),   # fit err 0.0145, |a| <= 4
    ('AlexKoff88/gpt-oss-tinystories', 'swigluoai', 'GeLU', 2, False): (0.9974, 1.0, 0.875),   # fit err 0.00101, |a| <= 4
    ('AlexKoff88/gpt-oss-tinystories', 'swigluoai', 'GeLU', 3, False): (1.0, 1.0, 1.0),   # fit err 0.000101, |a| <= 4
    ('AlexKoff88/gpt-oss-tinystories', 'swigluoai', 'GeLU', 4, False): (1.0, 1.0, 1.0),   # fit err 1.24e-05, |a| <= 4
    ('AlexKoff88/gpt-oss-tinystories', 'swigluoai', 'mish', 4, True): (1.0, 1.0, 1.0),   # fit err 0.00115, |a| <= 4
    ('AlexKoff88/gpt-oss-tinystories', 'swigluoai', 'mish', 5, True): (1.0, 1.0, 1.0),   # fit err 0.000573, |a| <= 4
    ('AlexKoff88/gpt-oss-tinystories', 'swigluoai', 'mish', 6, True): (1.0, 1.0, 1.0),   # fit err 0.000364, |a| <= 4
}


def _one_unit():
    """(target, host) cells the headline one-unit rule already pays (floor <= ACT_FLOOR_MAX)."""
    from graph.model_graph import ACT_FLOOR_MAX as fm, ACT_FLOORS
    return frozenset((t, h) for t, row in ACT_FLOORS.items() for h, fl in row.items() if fl <= fm)


ONE_UNIT = _one_unit()


def payable(top1, ex32, strict=True):
    """The harness's agreement standard (act/act_swap_agreement.py): strict top1 >= 99% and >= 90% of
    32-token greedy continuations identical; loose top1 >= 98% and >= 50% identical."""
    return top1 >= (0.99 if strict else 0.98) and ex32 >= (0.90 if strict else 0.50)


def _branches(v):
    return v.split("|") if isinstance(v, str) and "|" in v else [v]


def act_k(target, host, gate_bias=False, one_unit=True):
    """Host FFN units per target unit for the activation swap target <- host (1 = same activation or a
    one-unit fold / limit / floor within ACT_FLOOR_MAX), None if no k <= K_MAX fits. gate_bias: the
    host's gate projection has a bias (shifted fits allowed). Branch sets 'a|b' (the universal host):
    a host branch set takes its cheapest branch; a target branch set needs every branch."""
    worst = 1
    for t in _branches(target):
        best = None
        for h in _branches(host):
            if t == h or ((t, h) in ONE_UNIT and (one_unit or {t, h} != {"GeLU", "GeLU_tanh"})):
                k = 1
            else:
                tt = TARGET_ALIAS.get(t, t)            # the zoo's tanh GeLU target: the "GeLU" rows (erf AND tanh)
                hh = TARGET_ALIAS.get(h, h)            # ... and host (equal k: see HOST_GELU_TANH)
                ks = [K_NOSHIFT.get((tt, hh))] + ([K_SHIFT.get((tt, hh))] if gate_bias else [])
                ks = [k for k in ks if k is not None]
                k = min(ks) if ks else None
            if k is not None and (best is None or k < best):
                best = k
        if best is None:
            return None
        worst = max(worst, best)
    return worst


def act_width_ok(target, host, gate_bias=False, target_ffn=None, host_ffn=None):
    """The proposed rule: the swap target <- host is payable at FFN width x k(target, host) -- the
    host's ffn_total AND ffn_active must both be >= k x the target's. target_ffn / host_ffn:
    (ffn_total, ffn_active); omitted, only the activation part is checked (k exists)."""
    k = act_k(target, host, gate_bias)
    if k is None:
        return False
    if target_ffn is None or host_ffn is None:
        return True
    return host_ffn[0] >= k * target_ffn[0] and host_ffn[1] >= k * target_ffn[1]


# --- which hosts have a gate bias -----------------------------------------------------------------
# model_types whose FFN input projection (c_fc / dense_h_to_4h / fc1 / gate_up) carries a bias by
# default in the transformers code; config flags override (below).
GATE_BIAS_TYPES = frozenset({
    "gpt2", "gpt_neox", "gpt_neox_japanese", "gptj", "gpt_neo", "gpt_bigcode", "bloom", "codegen", "opt",
    "phi", "persimmon", "xglm", "biogpt", "starcoder2", "gpt_oss", "jais", "mpt", "falcon",
})
_BIAS_FLAGS = ("mlp_bias", "use_bias", "add_bias_linear", "enable_bias", "bias")


def host_gate_bias(config):
    """Does a checkpoint's FFN gate projection have a bias (so a k-unit fit may shift: gamma_i)?"""
    c = config.get("text_config") or config
    mt = str(c.get("model_type", config.get("model_type", ""))).lower()
    if "no_bias" in c:
        return not bool(c["no_bias"])
    for f in _BIAS_FLAGS:
        if f in c and isinstance(c[f], bool):
            return c[f]
    return mt in GATE_BIAS_TYPES and mt not in ("falcon", "mpt")    # falcon: bias=False default; mpt: no_bias


def width_edges(P, ok, ok_actfree, gate_bias):
    """The edges the rule adds to a relation (pure; numpy). P: points (slots / dims); ok: the relation
    today (emul_matrix); ok_actfree: the same relation with the act slot ignored; gate_bias[j]: host j
    has a gate bias. extra[i, j] = j emulates i only through a k-unit act swap at ffn x k."""
    import numpy as np
    n = len(P)
    acts = [p["slots"]["act"] for p in P]
    fT = np.array([p["dims"]["ffn_total"] for p in P])
    fA = np.array([p["dims"]["ffn_active"] for p in P])
    uniq = sorted(set(acts))
    kk = {}
    for t in uniq:
        for h in uniq:
            for b in (False, True):
                kk[(t, h, b)] = act_k(t, h, b)
    extra = np.zeros((n, n), bool)
    cand = ok_actfree & ~ok
    for i, j in zip(*np.nonzero(cand)):
        k = kk[(acts[i], acts[j], bool(gate_bias[j]))]
        if k is not None and fT[j] >= k * fT[i] and fA[j] >= k * fA[i]:
            extra[i, j] = True
    return extra


def frontier_effect(top=None, bias="config"):
    """Apply the rule to the current headline relation (emul_matrix(mode="payable"), unchanged) and
    return dict(old, new: graph.frontier.reps_of point indices before / after; extra: the added edges;
    P, ok, bias). bias: "config" (host_gate_bias of every member of the point), "none" (shift-free
    fits only), "all" (every host may shift)."""
    import graph.model_graph as mg
    from graph.frontier import load, reps_of
    P = mg._dedup_points(load(top))
    ok = mg.emul_matrix(P, mode="payable")
    real = mg.slot_reaches
    try:
        mg.slot_reaches = lambda slot, a, b, strict=False: True if slot == "act" else real(slot, a, b, strict)
        ok_free = mg.emul_matrix(P, mode="payable")
    finally:
        mg.slot_reaches = real
    if bias == "config":
        cfg = dict(mg.iter_json_configs(mg.ZOO_CONFIGS))
        gb = [all(host_gate_bias(cfg.get(m, {})) for m in p["members"]) for p in P]
    else:
        gb = [bias == "all"] * len(P)
    extra = width_edges(P, ok, ok_free, gb)
    return dict(old=reps_of(ok, P), new=reps_of(ok | extra, P), extra=extra, P=P, ok=ok, bias=gb)


def leavers(eff):
    """[(name, target act, host name, host act, k)] for the frontier models the rule removes, the host
    being the widest-covering new dominator (most models emulated)."""
    import numpy as np
    from graph.model_graph import short
    P, ok, extra, gb = eff["P"], eff["ok"], eff["extra"], eff["bias"]
    cnt = np.array([len(p["members"]) for p in P])
    new_ok = ok | extra
    out = []
    for i in sorted(set(eff["old"]) - set(eff["new"]), key=lambda i: -int(cnt[ok[:, i]].sum())):
        hosts = [j for j in np.nonzero(new_ok[i])[0] if not new_ok[j, i]]
        via = [j for j in hosts if extra[i, j]] or hosts
        j = max(via, key=lambda j: int(cnt[new_ok[:, j]].sum()))
        ta, ha = P[i]["slots"]["act"], P[j]["slots"]["act"]
        out.append((short(P[i]["members"][0]), ta, short(P[j]["members"][0]), ha,
                    act_k(ta, ha, gb[j]) if extra[i, j] else None))
    return out


def main():
    for bias in ("config", "none", "all"):
        eff = frontier_effect(bias=bias)
        print(f"[gate bias: {bias}] {int(eff['extra'].sum())} new point edges; frontier "
              f"{len(eff['old'])} -> {len(eff['new'])}")
        for name, ta, host, ha, k in leavers(eff):
            print(f"  leaves: {name:45s} {ta:>9s} <- {ha:<9s} x{k}  (under {host})")
        for i in sorted(set(eff["new"]) - set(eff["old"])):
            print(f"  joins:  {eff['P'][i]['members'][0]}")


if __name__ == "__main__":
    main()
