# Spot checks of the emulation relation

Random edges of the headline relation (`emul_matrix(mode="payable")` over the zoo's shapes), stratified by the kind of rule each spends, built with `elephantine/emulate.py` where it can and verified end to end. Written by `python -m verify.spot.run_spot_checks` (raw rows: `verify/spot/results.jsonl`).

- edges sampled: **240** (seed [0])
- built and verified: **70** -- pass **70**, FAIL **0** (real checkpoints 19, scaled-down 51)
- not built: unsupported by emulate.py 157, scaled pair not an edge / not shrinkable 8, error 5, timeout 0

**Directly buildable today:** 11,248 of the 46,329 headline edges (**24.3%**) have a member pair that `emulate.py` builds (config level: `emulate.support` on every exact edge, one member per model_type on each side).

## Pass rate per rule kind

A row counts every sampled edge carrying the tag (an edge carries several). *built* = real or scaled build verified; *pass* is over the built ones.

| rule kind | sampled | built | pass | FAIL | unsupported | other |
|---|---|---|---|---|---|---|
| cross_family | 186 | 32 | 32 (100%) | 0 | 148 | 6 |
| feats | 161 | 34 | 34 (100%) | 0 | 121 | 6 |
| approx | 108 | 0 | 0 | 0 | 108 | 0 |
| slot:kv | 88 | 20 | 20 (100%) | 0 | 62 | 6 |
| slot:ffn | 83 | 7 | 7 (100%) | 0 | 75 | 1 |
| rope_table | 68 | 0 | 0 | 0 | 68 | 0 |
| slot:pos | 67 | 0 | 0 | 0 | 67 | 0 |
| eps_lever | 65 | 0 | 0 | 0 | 65 | 0 |
| zero_edge | 59 | 0 | 0 | 0 | 59 | 0 |
| moe_host_for_dense | 54 | 7 | 7 (100%) | 0 | 46 | 1 |
| slot:qknorm | 52 | 0 | 0 | 0 | 52 | 0 |
| slot:window | 50 | 2 | 2 (100%) | 0 | 48 | 0 |
| slot:act | 48 | 0 | 0 | 0 | 48 | 0 |
| window_lever | 39 | 0 | 0 | 0 | 39 | 0 |
| act_width | 38 | 0 | 0 | 0 | 38 | 0 |
| capacity | 34 | 20 | 20 (100%) | 0 | 11 | 3 |
| kv_pack | 34 | 0 | 0 | 0 | 34 | 0 |
| slot:norm | 30 | 0 | 0 | 0 | 30 | 0 |
| ffn_kind | 22 | 0 | 0 | 0 | 22 | 0 |
| slot:block | 17 | 0 | 0 | 0 | 17 | 0 |
| slot:softcap | 15 | 0 | 0 | 0 | 15 | 0 |
| slot:mixer | 14 | 0 | 0 | 0 | 14 | 0 |
| moe<-moe | 11 | 0 | 0 | 0 | 11 | 0 |
| slot:sink | 11 | 0 | 0 | 0 | 11 | 0 |
| mask | 10 | 1 | 1 (100%) | 0 | 8 | 1 |
| slot:gen | 10 | 1 | 1 (100%) | 0 | 8 | 1 |
| slot:mtp | 10 | 0 | 0 | 0 | 10 | 0 |
| prenorm_rescue | 9 | 0 | 0 | 0 | 9 | 0 |
| slot:loop | 9 | 0 | 0 | 0 | 9 | 0 |
| moe_dup | 7 | 0 | 0 | 0 | 7 | 0 |
| recurrent | 6 | 0 | 0 | 0 | 6 | 0 |
| slot:crosslayer_kv | 5 | 0 | 0 | 0 | 5 | 0 |
| start_token | 5 | 0 | 0 | 0 | 5 | 0 |
| dense_host_for_moe | 4 | 0 | 0 | 0 | 4 | 0 |
| looped_host | 4 | 0 | 0 | 0 | 4 | 0 |
| mla | 4 | 0 | 0 | 0 | 4 | 0 |

Per slot transition (built edges only):

| transition | built | pass |
|---|---|---|
| ffn:glu->moe | 7 | 7 |
| gen:ar->diffusion | 1 | 1 |
| kv:GQA->MHA | 3 | 3 |
| kv:MHA->GQA | 9 | 9 |
| kv:MQA->GQA | 6 | 6 |
| kv:MQA->MHA | 2 | 2 |
| window:local->full | 2 | 2 |

## Failures

None: every edge that was built reproduced its target to the standard.
### Failures of the earlier pass on round2-p4 (the same sampler, seed 0, before pass 5)

Each is an `emulate.py` defect (two causes) on an edge the relation still claims; the round2-p5 sample did not draw them (raw rows: `verify/spot/results_round2-p4.jsonl`).

- **MaxJeblick/llama2-0b-unit-test <- AI4PD/ProtGPT3-MSA** (real): fp64 rel 2.02e-04, greedy 8/8; fp32 rel 2.02e-04 top-1 100.0% greedy 8/8. Cause: emulate.py, not the relation: the non-integer head_dim ratio rescales RoPE theta to 4.64e+42, past fp32's range, and inv_freq is computed in fp32 at init, so the host's frequencies past pair 0 are 0 (minimal repro: tests/test_spot_checks.py::test_theta_rescale_overflows_fp32, llama head_dim 6 -> 128). Repro: `python -m verify.spot.run_spot_checks --one MaxJeblick/llama2-0b-unit-test AI4PD/ProtGPT3-MSA`
- **dikiyplayerpig/dpp-gpt-V2.2-Pro-260m <- Qwen/Qwen1.5-0.5B** (real): fp64 rel 5.21e-15, greedy 6/8; fp32 rel 0.00e+00 top-1 100.0% greedy 6/8. Cause: not the construction: every prompt logit matches, yet greedy generation differs -- the target repo's generation_config.json (e.g. a second eos id) is not written to the host's directory by emulate.py, so the target stops where the host continues. Repro: `python -m verify.spot.run_spot_checks --one dikiyplayerpig/dpp-gpt-V2.2-Pro-260m Qwen/Qwen1.5-0.5B`
- **MaxJeblick/llama2-0b-unit-test <- BSC-LT/salamandra-2b** (real): fp64 rel 2.02e-04, greedy 8/8; fp32 rel 2.02e-04 top-1 100.0% greedy 8/8. Cause: emulate.py, not the relation: the non-integer head_dim ratio rescales RoPE theta to 2.15e+85, past fp32's range, and inv_freq is computed in fp32 at init, so the host's frequencies past pair 0 are 0 (minimal repro: tests/test_spot_checks.py::test_theta_rescale_overflows_fp32, llama head_dim 6 -> 128). Repro: `python -m verify.spot.run_spot_checks --one MaxJeblick/llama2-0b-unit-test BSC-LT/salamandra-2b`

## Built, by construction piece

Every piece `emulate.py` used on a built edge, and how many of those edges passed. A large edge composes from these pieces: its plan is covered when each of its pieces passed somewhere at small scale.

| piece | built | pass |
|---|---|---|
| family:granite->llama | 5 | 5 |
| family:llama->granite | 4 | 4 |
| family:llama->llama | 35 | 35 |
| family:llama->mistral | 3 | 3 |
| family:llama->mixtral | 7 | 7 |
| family:llama->qwen2 | 10 | 10 |
| family:mistral->llama | 1 | 1 |
| family:mistral->mistral | 1 | 1 |
| family:mistral->mixtral | 1 | 1 |
| family:mistral->qwen2 | 1 | 1 |
| family:qwen2->qwen2 | 1 | 1 |
| family:qwen3->qwen3 | 1 | 1 |
| ffn:dense | 62 | 62 |
| ffn:tied | 8 | 8 |
| hd:int | 33 | 33 |
| hd:same | 25 | 25 |
| hd:theta | 12 | 12 |
| hidden:pad | 66 | 66 |
| hidden:same | 4 | 4 |
| kv:tile1 | 59 | 59 |
| kv:tilex2 | 7 | 7 |
| kv:tilex3 | 2 | 2 |
| kv:tilex4 | 1 | 1 |
| kv:tilex8 | 1 | 1 |
| layers:same | 10 | 10 |
| layers:spread | 60 | 60 |
| q:pad | 61 | 61 |
| rope:default | 65 | 65 |
| rope:linear | 2 | 2 |
| rope:llama3 | 3 | 3 |
| window:set | 2 | 2 |

Scaled-down edges (pairs too big to run here): 51; of those, 36 use only pieces also verified on a real small checkpoint pair; the rest need a piece verified only at scaled-down size: family:llama->granite, family:llama->mistral, family:mistral->llama, family:mistral->mistral, family:mistral->mixtral, family:mistral->qwen2, family:qwen2->qwen2, kv:tilex4, kv:tilex8, rope:linear, rope:llama3, window:set.

## Why the rest is not built (ranked)

The missing construction pieces behind each unsupported sampled edge (an edge may lack several), most frequent first -- the order in which to extend `emulate.py`.

| missing piece | edges |
|---|---|
| limit / regime / floor edge (v1 writes exact constructions only) | 93 |
| softcap change | 15 |
| family not in v1: gpt_neox | 11 |
| MoE target (expert duplication / MoE <- MoE, dense <- top-1 MoE) | 11 |
| family not in v1: gemma4 | 10 |
| family not in v1: qwen3_5_moe | 8 |
| family not in v1: mimo_v2 | 7 |
| family not in v1: mpt | 6 |
| family not in v1: nanbeige | 6 |
| family not in v1: sdar | 6 |
| family not in v1: gpt2 | 5 |
| qk-norm on one side only | 4 |
| family not in v1: phi | 4 |
| family not in v1: laguna | 4 |
| family not in v1: glm | 4 |
| family not in v1: starcoder2 | 4 |
| family not in v1: gpts2 | 4 |
| family not in v1: qwen3_5 | 4 |
| family not in v1: gemma3 | 3 |
| family not in v1: falcon | 3 |
| family not in v1: qwen3_5_moe_text | 3 |
| family not in v1: bailing_moe | 3 |
| family not in v1: glm4_moe_lite | 3 |
| family not in v1: nae1 | 3 |
| family not in v1: plamo3 | 3 |
| KV-head packing (longrope table per packed head) | 3 |
| family not in v1: ernie4_5_moe | 3 |
| family not in v1: glm4 | 3 |
| family not in v1: mistral3 | 3 |
| family not in v1: qwen3loop | 3 |
| family not in v1: gpt_oss | 3 |
| layer placement (full / sliding kinds) | 2 |
| family not in v1: arkasr | 2 |
| family not in v1: k2_horizon | 2 |
| family not in v1: internlm2 | 2 |
| family not in v1: gpt_neo | 2 |
| family not in v1: glm4_moe | 2 |
| family not in v1: qwen3_vl_moe | 2 |
| family not in v1: mima | 2 |
| family not in v1: bailing_hybrid | 2 |
| family not in v1: ernie4_5 | 2 |
| family not in v1: llava_qwen2 | 2 |
| family not in v1: phimoe | 2 |
| FFN split over routed experts | 2 |
| family not in v1: ethoric | 2 |
| family not in v1: gemma4_unified | 2 |
| family not in v1: mobilint-exaone4 | 2 |
| activation change (FFN rewrite) | 1 |
| sandwich-norm change | 1 |
| family not in v1: gptx3 | 1 |
| family not in v1: monkeyocrv2 | 1 |
| window on a host family without a window setting | 1 |
| family not in v1: viuai | 1 |
| family not in v1: None | 1 |
| family not in v1: zaqeris_axiom | 1 |
| family not in v1: ivme | 1 |
| family not in v1: sundial | 1 |
| family not in v1: nexus | 1 |
| family not in v1: Dream | 1 |
| family not in v1: youtu | 1 |
| family not in v1: youtu_vita | 1 |
| family not in v1: glm_moe_dsa | 1 |
| family not in v1: deepseek_v3 | 1 |
| family not in v1: deepseek_v32 | 1 |
| family not in v1: lfm2 | 1 |
| family not in v1: mamba | 1 |
| family not in v1: edgerock | 1 |
| family not in v1: llava_mistral | 1 |
| family not in v1: opt | 1 |
| family not in v1: fabric | 1 |
| family not in v1: cohere | 1 |
| family not in v1: doge | 1 |
| family not in v1: naive_n05_flash | 1 |
| family not in v1: llava | 1 |
| family not in v1: hy_v3 | 1 |
| family not in v1: mimo | 1 |
| family not in v1: phi4mm | 1 |
| family not in v1: olmo | 1 |
| family not in v1: granitemoehybrid | 1 |
| remote-code family (no stock class to write) | 1 |
| family not in v1: meditor2 | 1 |
| family not in v1: phi3_v | 1 |
| family not in v1: koliber | 1 |
| family not in v1: muse_glimmer | 1 |
| family not in v1: llava-qwen2 | 1 |
| family not in v1: minimax | 1 |
| family not in v1: dots1 | 1 |
| family not in v1: simulvoice_runtime_probe | 1 |
| family not in v1: deci | 1 |
| family not in v1: hyperclovax_vlm | 1 |
| family not in v1: needle | 1 |
| family not in v1: cambrian_qwen | 1 |
| family not in v1: nemotron-nas | 1 |
| family not in v1: spatiallm_qwen | 1 |
| family not in v1: ouro | 1 |
| family not in v1: apriel | 1 |

## Not verified for another reason

| target | host | status | why |
|---|---|---|---|
| fxmarty/tiny-llama-fast-tokenizer | Alibaba-NLP/gte-Qwen2-7B-instruct | scaled_error | the widths share no factor to shrink by (f = 1): the host would keep 6878M parameters |
| yujiepan/llama-2-tiny-3layers-random | meditsolutions/MSH-Lite-7B-v1-Bielik-v2.3-Instruct-Llama-Prune | scaled_error | the widths share no factor to shrink by (f = 1): the host would keep 7785M parameters |
| Xenova/llama2.c-stories15M | llm-jp/llm-jp-3-440m | error | NotImplementedError: Cannot copy out of meta tensor; no data! |
| Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA | meditsolutions/MSH-Lite-7B-v1-Bielik-v2.3-Instruct-Llama-Prune | scaled_error | the widths share no factor to shrink by (f = 2): the host would keep 2047M parameters |
| MaxJeblick/llama2-0b-unit-test | Secbone/llama-33B-instructed | scaled_error | the widths share no factor to shrink by (f = 1): the host would keep 32756M parameters |
| MongoDB/voyage-4-nano | DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder | error | RuntimeError: loading MongoDB/voyage-4-nano: {'unexpected_keys': ['linear.weight']} |
| np-cr/testing-qwen3 | Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex | scaled_error | the widths share no factor to shrink by (f = 1): the host would keep 3885M parameters |
| Xenova/llama2.c-stories15M | saidutta69/EuroLLM-1.7B-Instruct-heretic | error | NotImplementedError: Cannot copy out of meta tensor; no data! |
| afmck/testing-llama-tiny | meta-llama/Llama-Guard-3-1B | error | IndexError: index out of range in self |
| ariG23498/tiny-llama3-test | Qwen/Qwen1.5-14B | scaled_error | the widths share no factor to shrink by (f = 1): the host would keep 13112M parameters |
| MaxJeblick/llama2-0b-unit-test | sarvamai/sarvam-1-v0.5 | scaled_error | the widths share no factor to shrink by (f = 1): the host would keep 2447M parameters |
| sign/utf8-lm-tiny | amd/AMD-Llama-135m | error | ValueError: Tokenizer class UTF8Tokenizer does not exist or is not currently imported. |
| SimpleStories/SimpleStories-35M | Qwen/Qwen1.5-110B | scaled_error | no common power-of-two factor of the widths |

## Every sampled edge against the reduction catalog

Whether or not `emulate.py` builds it, a sampled edge's plan is a composition of reduction types; **214 of 240** sampled edges (89%) use only types with a passing end-to-end catalog instance (below). (Composition is checked type by type, not as one build: the types the rest use without a passing instance --) Over the whole headline relation: **43,964 of 46,329** edges (94.9%) compose from verified types.

| reduction type without a passing instance | sampled edges | headline edges |
|---|---|---|
| slot:ffn:glu->vanilla | 17 | 1,508 |
| slot:norm:LayerNorm->RMSNorm (host without an lm_head bias) | 9 | 804 |
| slot:ffn:vanilla->glu (host without an FFN bias) | 5 | 366 |
| slot:ffn:vanilla->moe (host without an FFN bias) | 0 | 82 |
| slot:ffn:vanilla->moe_shared (host without an FFN bias) | 1 | 52 |
| slot:act:GeLU->swigluoai | 0 | 7 |

## Reduction catalog: one end-to-end instance per reduction type

`verify/spot/reduction_types.py` lists every reduction type the headline relation uses, from the relation's code (164 of 190 listed types; the rest are one-hop orders the headline does not pay). Each has an instance in `verify/spot/catalog/` that builds the host from the target's weights and runs both (8 prompts, 32 greedy tokens, fp64); `tests/test_reduction_catalog.py` runs them and checks that every type has one. *real* = a real checkpoint, else a small synthetic pair (stock transformers classes, or the reference decoder `catalog/_ref.py` where no stock class has the slot value).

### Findings

- **port:comp_ln: no end-to-end instance.** the LayerNorm fold without an lm_head bias, rescued by a 1-D compensator of the token rms (payable, measured on Pythia-160m): the construction is measured only by the scripts of verify/prenorm_rescue/, not built by the host
- **port:comp_par: no end-to-end instance.** parallel <= sequential with the compensator (payable, measured): the construction is measured only by the scripts of verify/prenorm_rescue/, not built by the host
- **slot:ffn:glu->vanilla: no end-to-end instance.** a GLU target on a vanilla (ungated) FFN host is credited at MOE_DEPTH_FACTOR = 2 x depth (FFN_RANK; the factor is labelled 'heuristic' in graph/model_graph.py), but no construction exists in the repo: one vanilla unit act(w.x) cannot form the product act(g) u, and a two-layer route (a product by polarization of squares) must carry act(g) or u across the next pre-norm, which the norm rescales by the token's rms (block/prenorm_carry.py; the relation refuses that carry everywhere else)
- **moe:dup:sparsemixer:m** (`fm_phimoe_dup2`, phimoe E 4, top-2 (SparseMixer) -> phimoe E 8, top-4: each expert twice): SparseMixer is a top-2 in code whatever num_experts_per_tok says, and its second pick masks out only the first expert's index: a duplicate of the top-1 expert wins the second slot. moe_why lets a sparsemixer target take m > 1 copies; it should require m = 1 (no zoo pair uses it: latent). Measured: max|d| 1.46e+00 rel 7.1e-01 top-1 43.8% identical 0/8. Repro: `pytest tests/test_reduction_catalog.py -k fm_phimoe_dup2`
- **port:act_width** (`p5_act_width_gelu_oai`, GeGLU -> swigluoai GLU (gate <= 7, up bias -1 for the (u + 1)) at FFN x 6 (synthetic, reference module)): graph/act_width's k-unit fits for a swigluoai host (act/act_width_fit.py TA['swigluoai'] = z sigmoid(1.702 z)) ignore gpt-oss's gate clamp min(g, 7) (swiglu_limit): the k = 6 GeLU fit's betas reach 3.8, so the clamp binds from |a| ~ 1.8; with the clamp the fit's error on [-6, 6] is 0.54, not 4.8e-4. Every act-width cell with a swigluoai host is affected (7 headline edges GeLU <- swigluoai). Measured: max|d| 5.63e-01 rel 5.6e-02 top-1 97.5% identical 2/8 (k = 6, fit error 4.8e-04 on [-160, 160]). Repro: `pytest tests/test_reduction_catalog.py -k p5_act_width_gelu_oai`
- **slot:ffn:vanilla->glu** (`p5_vanilla_on_glu_nobias`, vanilla SiLU MLP -> bias-free SwiGLU host: the up branch read off a constant residual channel (the only bias-free way to a constant) (synthetic, reference module)): the vanilla <- GLU step is FREE in FFN_ORDER, but the construction (EP.glu_ungated: 'up weights 0, bias 1') needs an up-projection bias; a bias-free host can only read a constant residual channel, which the pre-norm divides by the token's rms. 366 headline edges (and 134 vanilla -> MoE ones) use a host without an FFN bias (dims ffn_bias = 0). Measured: max|d| 5.55e+00 rel 5.0e-01 top-1 76.2% identical 0/8 (the normed constant is 1 / rms(x), not 1). Repro: `pytest tests/test_reduction_catalog.py -k p5_vanilla_on_glu_nobias`
- **slot:norm:LayerNorm->RMSNorm** (`ap_norm_ln_fold_no_lm_bias`, LayerNorm -> RMSNorm host WITHOUT an lm_head bias (synthetic, reference module)): the final LayerNorm's beta has no lm_head bias to fold into: exact only on hosts with one (round 2 pass 5: the LN fold without an lm_head bias is refuted as exact). Measured: max|d| 8.71e-01 rel 6.7e-02 top-1 87.5% identical 0/8 (host without an lm_head bias: the final beta is dropped). Repro: `pytest tests/test_reduction_catalog.py -k ap_norm_ln_fold_no_lm_bias`

### Table

| type | currency | instance (target -> host) | real / synthetic | standard | result |
|---|---|---|---|---|---|
| capacity:eff_layers | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:ffn_active | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:ffn_total | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:head_dim | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:hidden | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:kv_cap | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:kv_pack:nope | capacity | llama GQA (4q/2kv x 8, inv_freq 0 = NoPE) -> llama MQA (8q/1kv x 16, NoPE) | synthetic | exact | pass (max/d/ 2.00e-15 rel 1.0e-15 top-1 100.0% identical 8/8) |
| capacity:kv_pack:rope | capacity | llama GQA (4q/2kv x 8) -> llama MQA (8q/1kv x 16, longrope table): 2 KV heads packed in one | synthetic | exact | pass (max/d/ 8.05e-09 rel 4.0e-09 top-1 100.0% identical 8/8) |
| capacity:layer_kind:FT<-FT | capacity | qwen2 K=V (sliding, sliding) -> qwen2 K=V host (full, sliding, full; reference: tied k / v) | synthetic | exact | pass (max/d/ 2.44e-15 rel 9.6e-16 top-1 100.0% identical 8/8) |
| capacity:layer_kind:FT<-FU | capacity | qwen2 K=V target (sliding, full) -> qwen2 own-V host (reference: K = V as tied k / v projections) | synthetic | exact | pass (max/d/ 2.80e-15 rel 1.1e-15 top-1 100.0% identical 8/8) |
| capacity:layer_kind:FU<-FU | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:layer_kind:FU<-FU | capacity | gemma2 (sliding, full) -> gemma2 (sliding, full, sliding): each layer on a host layer of its kind | synthetic | exact | pass (max/d/ 2.73e-15 rel 8.5e-16 top-1 100.0% identical 8/8) |
| capacity:layer_kind:LT<-FT | capacity | qwen2 K=V (sliding, sliding) -> qwen2 K=V host (full, sliding, full; reference: tied k / v) | synthetic | exact | pass (max/d/ 2.44e-15 rel 9.6e-16 top-1 100.0% identical 8/8) |
| capacity:layer_kind:LT<-FU | capacity | qwen2 K=V (sliding, sliding) -> qwen2 K=V host (full, sliding, full; reference: tied k / v) | synthetic | exact | pass (max/d/ 2.44e-15 rel 9.6e-16 top-1 100.0% identical 8/8) |
| capacity:layer_kind:LT<-LT | capacity | qwen2 K=V (sliding, sliding) -> qwen2 K=V host (full, sliding, full; reference: tied k / v) | synthetic | exact | pass (max/d/ 2.44e-15 rel 9.6e-16 top-1 100.0% identical 8/8) |
| capacity:layer_kind:LT<-LU | capacity | qwen2 K=V target (sliding, full) -> qwen2 own-V host (reference: K = V as tied k / v projections) | synthetic | exact | pass (max/d/ 2.80e-15 rel 1.1e-15 top-1 100.0% identical 8/8) |
| capacity:layer_kind:LU<-FU | capacity | mistral (window 4) -> qwen2 (full layers): the host's window set to the target's | synthetic | exact | pass (max/d/ 2.22e-15 rel 7.4e-16 top-1 100.0% identical 8/8) |
| capacity:layer_kind:LU<-LU | capacity | gemma2 (sliding, full) -> gemma2 (sliding, full, sliding): each layer on a host layer of its kind | synthetic | exact | pass (max/d/ 2.73e-15 rel 8.5e-16 top-1 100.0% identical 8/8) |
| capacity:layer_kind:R<-R | capacity | hybrid target (attn, rec) -> hybrid host (attn, rec, attn, rec), recurrent layer on a recurrent layer, the extra ones identity (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| capacity:layer_kind:S<-S | capacity | CLA target (layer 1 reads layer 0's K / V) -> a deeper CLA host (layers 1, 3 share), the extra layers identity (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| capacity:mla_latent | capacity | deepseek_v2 MLA (kv latent 16) -> deepseek_v2 (kv latent 32 = 2 copies: the latent norm's rms over two copies is the rms of one, same eps) | synthetic | exact | pass (max/d/ 8.88e-16 rel 5.1e-16 top-1 100.0% identical 8/8) |
| capacity:mla_latent | capacity | deepseek_v2 MLA (kv latent 16) -> deepseek_v2 (latent 24, not a multiple): zero-padded, the latent input x sqrt(24 / 16) so the hard-coded eps reads as the target's, the gain / sqrt(24 / 16) | synthetic | limit | pass (max/d/ 1.22e-15 rel 6.3e-16 top-1 100.0% identical 8/8) |
| capacity:moe_depth | capacity | sequential block -> parallel host at 2x depth (attention and FFN in alternate layers) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| capacity:norm_width | capacity | RMSNorm -> LayerNorm at 2x width: the residual mirrored [h; -h] (synthetic, reference module) | synthetic | exact | pass (max/d/ 1.24e-14 rel 1.2e-15 top-1 100.0% identical 8/8) |
| capacity:param_layers | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:q_cap | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:qheads | capacity | llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| capacity:rot_gate | capacity | RoPE (hd 8) -> partial RoPE (hd 16, rot 8): the target's pairs on the host's rotated dims (synthetic, reference module) | synthetic | exact | pass (max/d/ 2.55e-14 rel 2.5e-15 top-1 100.0% identical 8/8) |
| capacity:rot_gate | capacity | llama (RoPE) -> glm (partial RoPE host, stock): hf/partial_rope_host.glm_from_llama | synthetic | exact | pass (max/d/ 1.33e-15 rel 6.2e-16 top-1 100.0% identical 8/8) |
| feature:bias_out:host | feature | llama (tied, no biases) -> llama with attention and MLP biases (set 0), untied head (emulate.py) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| feature:bias_qkv:host | feature | llama -> qwen2_moe: the dense FFN in the shared expert (sigmoid gate 0 -> x 2), routed experts 0, q / k / v biases 0 (emulate.py) | synthetic | exact | pass (max/d/ 2.66e-15 rel 8.9e-16 top-1 100.0% identical 8/8) |
| feature:bias_qkv:host | feature | llama -> qwen2 (q / k / v biases set 0) (emulate.py) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| feature:dsa:host | feature | DSA (index_topk 64 >= the context) <-> dense attention, both ways (synthetic, reference module; DSA is dense attention while the context fits index_topk) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:dsa:target | feature | DSA (index_topk 64 >= the context) <-> dense attention, both ways (synthetic, reference module; DSA is dense attention while the context fits index_topk) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:engram:host | feature | -> a hashed n-gram memory added to the residual, table 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:out_gate:host | feature | no output gate -> a sigmoid output gate with W_g = 0 (0.5, folded into o) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:ple:host | feature | -> per-layer embedding inputs, projections 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:router_bias:host | feature | sigmoid router without a selection bias -> a host with one, set 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:router_group:host | feature | deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| feature:router_group:target | feature | deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| feature:router_sigmoid:host | feature | deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| feature:router_sigmoid:target | feature | deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| feature:router_sqrtsoftplus:host | feature | sqrt(softplus) router (DeepSeek-V4) -> a host whose score is a config value, set to the same; a softmax target on that host likewise (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:router_sqrtsoftplus:target | feature | sqrt(softplus) router (DeepSeek-V4) -> a host whose score is a config value, set to the same; a softmax target on that host likewise (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:selfcond:host | feature | -> a self-conditioning input (previous logits), projection 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:shared_gate:host | feature | qwen2_moe (P, expert 16, shared 24) -> qwen2_moe (expert 24, shared 40): widths zero-padded, the shared expert's sigmoid gate copied | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| feature:shared_gate:host | feature | llama -> qwen2_moe: the dense FFN in the shared expert (sigmoid gate 0 -> x 2), routed experts 0, q / k / v biases 0 (emulate.py) | synthetic | exact | pass (max/d/ 2.66e-15 rel 8.9e-16 top-1 100.0% identical 8/8) |
| feature:switch_lora:host | feature | -> a switched LoRA adapter, B = 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| feature:tied_emb:target | feature | llama (tied, no biases) -> llama with attention and MLP biases (set 0), untied head (emulate.py) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| feature:value_emb:host | feature | -> value embeddings, table 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| moe:dense_host_top1 | moe | mixtral (4 experts, top-1) -> a dense host: per MoE layer a select FFN writing gamma-scaled masks into 4 carried channels and an expert FFN of masked units + erase units (moe/dense_host_prenorm.py 'top1', simulated on the target's own modules) | synthetic | limit | pass (max/d/ 1.34e-05 rel 1.4e-06 top-1 100.0% identical 8/8) |
| moe:dense_layers | moe | deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| moe:dense_layers | moe | a MoE target whose layer 0 is dense -> an all-MoE host: layer 0's experts all hold the dense FFN, router 0 (k experts at weight 1 / k) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| moe:dup:P:1 | moe | qwen2_moe (P, expert 16, shared 24) -> qwen2_moe (expert 24, shared 40): widths zero-padded, the shared expert's sigmoid gate copied | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| moe:dup:P:m | moe | qwen2_moe E 4, top-2, softmax not renormalised -> E 8 (exactly 2 x), top-4 | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| moe:dup:R:1 | moe | mixtral E 4, top-2, expert width 48 -> mixtral E 4, top-2, width 64 (zero-padded) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| moe:dup:R:m | moe | mixtral E 4, top-2 (renormalised) -> mixtral E 8, top-4: each expert twice | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.9e-16 top-1 100.0% identical 8/8) |
| moe:dup:sparsemixer:1 | moe | phimoe E 4, top-2, width 48 -> phimoe E 4, top-2, width 64 | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| moe:dup:sparsemixer:m | moe | phimoe E 4, top-2 (SparseMixer) -> phimoe E 8, top-4: each expert twice | synthetic | exact | FINDING: max/d/ 1.46e+00 rel 7.1e-01 top-1 43.8% identical 0/8 |
| moe:extra_experts | moe | deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| moe:group:max | moe | deepseek_v2 (E 4 in 2 groups, group-limited greedy = max) -> deepseek_v2, wider experts (24) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.8e-16 top-1 100.0% identical 8/8) |
| moe:group:top2sum | moe | deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| moe:host_for_dense | moe | llama (dense SwiGLU) -> mixtral (4 experts, top-2): every expert = the dense FFN, router 0 (emulate.py) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| moe:host_for_dense | moe | llama -> qwen2_moe: the dense FFN in the shared expert (sigmoid gate 0 -> x 2), routed experts 0, q / k / v biases 0 (emulate.py) | synthetic | exact | pass (max/d/ 2.66e-15 rel 8.9e-16 top-1 100.0% identical 8/8) |
| moe:host_for_dense | moe | vanilla SiLU MLP -> MoE host with expert up biases (gpt-oss style): every expert the FFN with its up branch held at 1 by the bias, router 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| moe:latent | moe | MoE with an expert-input latent 12 -> latent 16 (zero-padded) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| moe:score | moe | deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer) | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| moe:shared | moe | qwen2_moe (P, expert 16, shared 24) -> qwen2_moe (expert 24, shared 40): widths zero-padded, the shared expert's sigmoid gate copied | synthetic | exact | pass (max/d/ 6.66e-16 rel 3.6e-16 top-1 100.0% identical 8/8) |
| port:act_width | port | GeLU MLP -> SiLU MLP at FFN x 6 (6 scaled SiLU units per GeLU unit) (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.99e-03 rel 2.6e-04 top-1 100.0% identical 8/8) |
| port:act_width | port | tanh-GeLU MLP -> SiLU MLP at FFN x 6 (synthetic, reference module) | synthetic | approx | pass (max/d/ 3.35e-03 rel 2.9e-04 top-1 100.0% identical 8/8) |
| port:act_width | port | SiLU MLP -> GeLU (erf) MLP at FFN x 3 (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.69e-03 rel 2.4e-04 top-1 100.0% identical 8/8) |
| port:act_width | port | SiLU MLP -> tanh-GeLU MLP at FFN x 3 (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.94e-03 rel 2.6e-04 top-1 100.0% identical 8/8) |
| port:act_width | port | GeGLU -> swigluoai GLU (gate <= 7, up bias -1 for the (u + 1)) at FFN x 6 (synthetic, reference module) | synthetic | approx | FINDING: max/d/ 5.63e-01 rel 5.6e-02 top-1 97.5% identical 2/8 (k = 6, fit error 4.8e-04 on [-160, 160]) |
| port:alibi_labs | port | -- | -- | -- | **not instantiated** |
| port:bos_labs | port | gpt2 (learned positions, LayerNorm) -> llama with inv_freq 0, +2 layers, +5 channels: the start-token rule (hf/bos_constructions.learned_abs_host). A RoPE host at inv_freq 0 is the NoPE host, and a partial-RoPE host at inv_freq 0 on its rotated dims computes the same function | synthetic | exact | pass (max/d/ 8.56e-14 rel 1.2e-13 top-1 100.0% identical 8/8) |
| port:bos_sink | port | gpt-oss (learned sinks) -> sink-free q-bias attention (Qwen2Attention, head_dim + 2 with a zero-frequency spare pair): a host-only marker supplies the sink key (hf/bos_constructions.sink_host) | synthetic | exact | pass (max/d/ 1.86e-14 rel 2.7e-15 top-1 100.0% identical 8/8) |
| port:comp_ln | port | -- | -- | -- | **not instantiated** |
| port:comp_par | port | -- | -- | -- | **not instantiated** |
| port:eps_feats | port | gemma -> gemma2 (sandwich post-norms): the post-norm inputs x 2^-30, the norms linear | synthetic | limit | pass (max/d/ 4.42e-09 rel 9.3e-11 top-1 100.0% identical 8/8) |
| port:eps_feats | port | gemma -> gemma3 (sandwich post-norms and qk-norm), both switched off by the eps lever | synthetic | limit | pass (max/d/ 4.42e-09 rel 9.3e-11 top-1 100.0% identical 8/8) |
| port:eps_latent | port | deepseek_v2 MLA (kv latent 16) -> deepseek_v2 (latent 24, not a multiple): zero-padded, the latent input x sqrt(24 / 16) so the hard-coded eps reads as the target's, the gain / sqrt(24 / 16) | synthetic | limit | pass (max/d/ 1.22e-15 rel 6.3e-16 top-1 100.0% identical 8/8) |
| port:eps_mla_plain | port | gemma MQA -> deepseek_v2 MLA: V in the latent (its kv_a_layernorm linear by the eps lever), K the shared rotary key | synthetic | limit | pass (max/d/ 2.41e-10 rel 5.0e-12 top-1 100.0% identical 8/8) |
| port:eps_mla_plain | port | llama MHA (4 heads x 8) -> deepseek_v2 MLA: 4 rotating keys packed in the shared rotary key (32 dims, a longrope table), values through the latent (eps lever) | synthetic | limit | pass (max/d/ 2.34e-09 rel 1.1e-09 top-1 100.0% identical 8/8) |
| port:eps_mla_plain | port | llama GQA (4 q / 2 kv x 8) -> deepseek_v2 MLA: 2 rotating keys packed in the shared rotary key | synthetic | limit | pass (max/d/ 3.43e-09 rel 1.9e-09 top-1 100.0% identical 8/8) |
| port:eps_qknorm | port | llama -> qwen3 (per-head qk-norm): q / k rows x 2^-30, the norm linear, gain 2^30 sqrt(eps) | synthetic | limit | pass (max/d/ 3.24e-10 rel 9.2e-12 top-1 100.0% identical 8/8) |
| port:eps_qknorm | port | gemma -> gemma3 (sandwich post-norms and qk-norm), both switched off by the eps lever | synthetic | limit | pass (max/d/ 4.42e-09 rel 9.3e-11 top-1 100.0% identical 8/8) |
| port:eps_qknorm | port | no qk-norm -> per-head qk-norm after RoPE (Hunyuan), switched off by the eps lever (synthetic, reference module) | synthetic | limit | pass (max/d/ 1.02e-14 rel 1.0e-15 top-1 100.0% identical 8/8) |
| port:eps_qknorm | port | no qk-norm -> whole-projection qk-norm (OLMo-2 style), switched off by the eps lever (synthetic, reference module) | synthetic | limit | pass (max/d/ 1.33e-14 rel 1.3e-15 top-1 100.0% identical 8/8) |
| port:eps_rec | port | mamba2 (8 heads x 8, gated norm over 64) -> mamba2 (12 heads x 8): zero-padded channels, the shared eps x 64 / 96 and the residual stream x sqrt(64 / 96) so the stream norms read the same | synthetic | limit | pass (max/d/ 4.00e-15 rel 1.3e-15 top-1 100.0% identical 8/8) |
| port:gelu | port | GeLU (erf) MLP -> tanh-GeLU MLP, the same weights (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.62e-03 rel 2.3e-04 top-1 100.0% identical 8/8) |
| port:gelu | port | tanh-GeLU MLP -> GeLU (erf) MLP, the same weights (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.62e-03 rel 2.3e-04 top-1 100.0% identical 8/8) |
| port:ln_beta | port | LayerNorm without a beta (Cohere / OLMo-1) -> RMSNorm host without an lm_head bias: writers centred, nothing to fold (synthetic, reference module) | synthetic | exact | pass (max/d/ 1.15e-14 rel 1.1e-15 top-1 100.0% identical 8/8) |
| port:loop | port | llama (3 untied layers) -> a looped host running its block K = 2 times, +2 channels (counter, start flag), under the start-token rule (hf/looped_host_prenorm.py, variant 'host') | synthetic | limit | pass (max/d/ 1.19e-07 rel 1.6e-08 top-1 100.0% identical 8/8) |
| port:oai_bias | port | SiLU GLU -> swigluoai GLU: gate / 1.702, up scaled by 1e-3 with an up bias -1 inside the clamps (synthetic, reference module) | synthetic | limit | pass (max/d/ 9.99e-13 rel 9.9e-14 top-1 100.0% identical 8/8) |
| port:partial | port | llama (full RoPE) -> phi3 (partial_rotary_factor 0.5): the rotary-fraction lever raises the host's factor to 1, then emulate.py | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| port:qk_switch | port | cohere without qk-norm -> cohere with use_qk_norm (per-head LayerNorm): the flag switched off | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| port:rot_config | port | llama (head_dim 16) -> llama (head_dim 24): a fixed-rotary host sets only rope_theta, theta_h = theta_t^(24 / 16) (emulate.py) | synthetic | exact | pass (max/d/ 2.77e-08 rel 9.3e-09 top-1 100.0% identical 8/8) |
| port:softcap | port | gemma2 without caps -> gemma2 with attention / final caps 50 / 30: the caps set to null in the host's config (exact) | synthetic | exact | pass (max/d/ 3.97e-15 rel 1.2e-15 top-1 100.0% identical 8/8) |
| port:window | port | llama (full layers) -> mistral (sliding window 4 on every layer): the window lever sets the host's window >= the context, so its sliding layers are full ones | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| recurrent:eq | recurrent | rwkv (2 layers x 32) -> rwkv with the same recurrent dims (RC_EQ: every recurrent dim equal, so the construction is the copy) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| recurrent:gdn | recurrent | qwen3_next GDN (2 key heads x 8, 4 value heads x 8) -> qwen3_next (value head dim 16: each value channel twice, the per-head gated RMSNorm sees the same rms; key dim equal) | synthetic | exact | pass (max/d/ 8.88e-16 rel 5.1e-16 top-1 100.0% identical 8/8) |
| recurrent:ge | recurrent | mamba (d_inner 64, d_state 8, dt rank 4, conv 4) -> mamba (d_inner 96, d_state 16, dt rank 8, conv 6): channels / states / rank / taps zero-padded | synthetic | exact | pass (max/d/ 1.11e-15 rel 4.1e-16 top-1 100.0% identical 8/8) |
| recurrent:kda | recurrent | KDA (2 heads x 8, gate rank 4, conv 3) -> KDA (3 heads x 8, gate rank 8, conv 4): heads / rank / taps zero-padded, head_dim equal (synthetic, reference module: no stock KDA class) | synthetic | exact | pass (max/d/ 1.78e-15 rel 1.5e-16 top-1 100.0% identical 8/8) |
| recurrent:m2 | recurrent | mamba2 (8 heads x 8, gated norm over 64) -> mamba2 (12 heads x 8): zero-padded channels, the shared eps x 64 / 96 and the residual stream x sqrt(64 / 96) so the stream norms read the same | synthetic | limit | pass (max/d/ 4.00e-15 rel 1.3e-15 top-1 100.0% identical 8/8) |
| recurrent:m2 | recurrent | mamba2 (8 heads x 8, one group, gated norm over the full width) -> mamba2 (16 heads x 8, d_state 16): every channel twice (the rms over two copies is the rms of one), out_proj reads the first copy | synthetic | exact | pass (max/d/ 1.33e-15 rel 5.0e-16 top-1 100.0% identical 8/8) |
| recurrent:zamba_shared | recurrent | one shared block at 4 positions -> 2 shared blocks alternating (a multiple: each host block a copy of the target's) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:act:GeLU->GeLU_tanh | floor | GeLU (erf) MLP -> tanh-GeLU MLP, the same weights (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.62e-03 rel 2.3e-04 top-1 100.0% identical 8/8) |
| slot:act:GeLU->SiLU | port | GeLU MLP -> SiLU MLP at FFN x 6 (6 scaled SiLU units per GeLU unit) (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.99e-03 rel 2.6e-04 top-1 100.0% identical 8/8) |
| slot:act:GeLU->swigluoai | port | GeGLU -> swigluoai GLU (gate <= 7, up bias -1 for the (u + 1)) at FFN x 6 (synthetic, reference module) | synthetic | approx | FINDING: max/d/ 5.63e-01 rel 5.6e-02 top-1 97.5% identical 2/8 (k = 6, fit error 4.8e-04 on [-160, 160]) |
| slot:act:GeLU_tanh->GeLU | floor | tanh-GeLU MLP -> GeLU (erf) MLP, the same weights (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.62e-03 rel 2.3e-04 top-1 100.0% identical 8/8) |
| slot:act:GeLU_tanh->SiLU | port | tanh-GeLU MLP -> SiLU MLP at FFN x 6 (synthetic, reference module) | synthetic | approx | pass (max/d/ 3.35e-03 rel 2.9e-04 top-1 100.0% identical 8/8) |
| slot:act:ReLU->GeLU | mag | ReLU MLP -> GeLU (erf) MLP, beta = 1e9 (synthetic, reference module) | synthetic | limit | pass (max/d/ 6.22e-15 rel 5.5e-16 top-1 100.0% identical 8/8) |
| slot:act:ReLU->GeLU_tanh | mag | ReLU MLP -> GeLU (tanh) MLP, beta = 1e9 (synthetic, reference module) | synthetic | limit | pass (max/d/ 6.22e-15 rel 5.5e-16 top-1 100.0% identical 8/8) |
| slot:act:ReLU->SiLU | mag | ReLU MLP -> SiLU MLP: SiLU(beta x) / beta, beta = 1e9 (synthetic, reference module) | synthetic | limit | pass (max/d/ 6.22e-15 rel 5.5e-16 top-1 100.0% identical 8/8) |
| slot:act:ReLU->mish | mag | ReLU MLP -> mish MLP, beta = 1e9 (synthetic, reference module) | synthetic | limit | pass (max/d/ 6.22e-15 rel 5.5e-16 top-1 100.0% identical 8/8) |
| slot:act:ReLU->swigluoai | mag | ReLU MLP -> swigluoai GLU (gpt-oss: gate <= 7, /up/ <= 7, gate and up biases): gate held at 7 by its bias, the up clamp's lower kink at the target's kink, slope s = 1e-3 (synthetic, reference module) | synthetic | limit | pass (max/d/ 1.82e-11 rel 1.7e-12 top-1 100.0% identical 8/8) |
| slot:act:SiLU->GeLU | port | SiLU MLP -> GeLU (erf) MLP at FFN x 3 (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.69e-03 rel 2.4e-04 top-1 100.0% identical 8/8) |
| slot:act:SiLU->GeLU_tanh | port | SiLU MLP -> tanh-GeLU MLP at FFN x 3 (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.94e-03 rel 2.6e-04 top-1 100.0% identical 8/8) |
| slot:act:SiLU->swigluoai | mag | SiLU GLU -> swigluoai GLU: gate / 1.702, up scaled by 1e-3 with an up bias -1 inside the clamps (synthetic, reference module) | synthetic | limit | pass (max/d/ 9.99e-13 rel 9.9e-14 top-1 100.0% identical 8/8) |
| slot:act:relu2->xielu | mag | relu^2 MLP -> xIELU MLP: xIELU(c x) / (alpha_p c^2), c = 1e8 (synthetic, reference module) | synthetic | limit | pass (max/d/ 6.28e-08 rel 4.7e-09 top-1 100.0% identical 8/8) |
| slot:act:swigluoai->SiLU | mag | swigluoai GLU -> SiLU GLU with an up bias +1: gate x 1.702, exact while the target's clamps do not bind (synthetic, reference module) | synthetic | limit | pass (max/d/ 7.99e-15 rel 7.0e-16 top-1 100.0% identical 8/8) |
| slot:block:sequential->parallel | size (capacity rows) | sequential block -> parallel host at 2x depth (attention and FFN in alternate layers) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:crosslayer_kv:no->yes | depth (ZERO_EDGES) | own-KV target (2 layers) -> CLA host (layers 1, 3 reuse the previous layer's K / V): the sharing layers zeroed (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:ffn:glu->moe | cap | llama (dense SwiGLU) -> mixtral (4 experts, top-2): every expert = the dense FFN, router 0 (emulate.py) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| slot:ffn:glu->moe_shared | cap,free | llama -> qwen2_moe: the dense FFN in the shared expert (sigmoid gate 0 -> x 2), routed experts 0, q / k / v biases 0 (emulate.py) | synthetic | exact | pass (max/d/ 2.66e-15 rel 8.9e-16 top-1 100.0% identical 8/8) |
| slot:ffn:glu->vanilla | size (capacity rows) | -- | -- | -- | **not instantiated** |
| slot:ffn:moe->glu | size (capacity rows) | mixtral (4 experts, top-1) -> a dense host: per MoE layer a select FFN writing gamma-scaled masks into 4 carried channels and an expert FFN of masked units + erase units (moe/dense_host_prenorm.py 'top1', simulated on the target's own modules) | synthetic | limit | pass (max/d/ 1.34e-05 rel 1.4e-06 top-1 100.0% identical 8/8) |
| slot:ffn:moe->moe_shared | free | mixtral (E 4, top-2) -> qwen2_moe (E 4, top-2, renormalised, a gated shared expert set to 0) | synthetic | exact | pass (max/d/ 8.33e-16 rel 4.8e-16 top-1 100.0% identical 8/8) |
| slot:ffn:vanilla->glu | free | vanilla GeLU MLP -> GeGLU host with an up bias: the up path held at 1 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:ffn:vanilla->glu | free | vanilla SiLU MLP -> bias-free SwiGLU host: the up branch read off a constant residual channel (the only bias-free way to a constant) (synthetic, reference module) | synthetic | exact | FINDING: max/d/ 5.55e+00 rel 5.0e-01 top-1 76.2% identical 0/8 (the normed constant is 1 / rms(x), not 1) |
| slot:ffn:vanilla->moe | free,cap | vanilla SiLU MLP -> MoE host with expert up biases (gpt-oss style): every expert the FFN with its up branch held at 1 by the bias, router 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:ffn:vanilla->moe_shared | free,cap | vanilla SiLU MLP -> MoE host with expert up biases (gpt-oss style): every expert the FFN with its up branch held at 1 by the bias, router 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:gen:ar->block | mask | causal LM -> block-causal host given block size 1 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:gen:ar->diffusion | mask | causal LM -> bidirectional (diffusion) host given the causal mask (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:gen:block->diffusion | mask | block-causal (4) -> bidirectional host given the block mask (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:gen:diffusion->block | mask | bidirectional -> block-causal host given an all-ones mask (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:kv:GQA->MHA | kv | llama GQA (4q/2kv) -> llama MHA (6q/6kv) | synthetic | exact | pass (max/d/ 2.33e-15 rel 7.8e-16 top-1 100.0% identical 8/8) |
| slot:kv:GQA->MLA | size (capacity rows) | llama GQA (4 q / 2 kv x 8) -> deepseek_v2 MLA: 2 rotating keys packed in the shared rotary key | synthetic | limit | pass (max/d/ 3.43e-09 rel 1.9e-09 top-1 100.0% identical 8/8) |
| slot:kv:GQA->MQA | size (capacity rows) | llama GQA (4q/2kv x 8) -> llama MQA (8q/1kv x 16, longrope table): 2 KV heads packed in one | synthetic | exact | pass (max/d/ 8.05e-09 rel 4.0e-09 top-1 100.0% identical 8/8) |
| slot:kv:MHA->GQA | size (capacity rows) | llama MHA (3q/3kv) -> llama GQA (6q/3kv): one query per host group used | synthetic | exact | pass (max/d/ 2.22e-15 rel 1.0e-15 top-1 100.0% identical 8/8) |
| slot:kv:MHA->MLA | size (capacity rows) | llama MHA (4 heads x 8) -> deepseek_v2 MLA: 4 rotating keys packed in the shared rotary key (32 dims, a longrope table), values through the latent (eps lever) | synthetic | limit | pass (max/d/ 2.34e-09 rel 1.1e-09 top-1 100.0% identical 8/8) |
| slot:kv:MHA->MQA | size (capacity rows) | llama MHA (1q/1kv) -> llama MQA (6q/1kv) | synthetic | exact | pass (max/d/ 7.22e-16 rel 4.8e-16 top-1 100.0% identical 8/8) |
| slot:kv:MQA->GQA | kv | llama MQA (4q/1kv) -> llama GQA (6q/3kv) | synthetic | exact | pass (max/d/ 2.44e-15 rel 8.6e-16 top-1 100.0% identical 8/8) |
| slot:kv:MQA->MHA | kv | llama MQA (4q/1kv) -> llama MHA (6q/6kv) | synthetic | exact | pass (max/d/ 2.44e-15 rel 8.6e-16 top-1 100.0% identical 8/8) |
| slot:kv:MQA->MLA | size (capacity rows) | gemma MQA -> deepseek_v2 MLA: V in the latent (its kv_a_layernorm linear by the eps lever), K the shared rotary key | synthetic | limit | pass (max/d/ 2.41e-10 rel 5.0e-12 top-1 100.0% identical 8/8) |
| slot:loop:looped->untied | depth | a looped block (2 layers x 2) -> 4 untied layers (unrolled) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:loop:untied->looped | size (capacity rows) | 2 untied layers -> a looped host (block of 2) with its loop count set to 1 (synthetic, reference module; see the finding: a fixed loop count cannot) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:loop:untied->looped | size (capacity rows) | llama (3 untied layers) -> a looped host running its block K = 2 times, +2 channels (counter, start flag), under the start-token rule (hf/looped_host_prenorm.py, variant 'host') | synthetic | limit | pass (max/d/ 1.19e-07 rel 1.6e-08 top-1 100.0% identical 8/8) |
| slot:mixer:softmax->hybrid | depth (ZERO_EDGES) | softmax (2 layers) -> hybrid host (attn, rec, attn, rec): recurrent layers zeroed (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:mtp:no->yes | free | no MTP head -> a host with an MTP head (the main logits do not read it) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:mtp:yes->no | free | an MTP head -> no MTP head (dropped) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:norm:LayerNorm->RMSNorm | fold | LayerNorm (biased, GeLU MLP) -> RMSNorm host with an lm_head bias (Phi-style): writers centred, betas folded (synthetic, reference module) | synthetic | exact | pass (max/d/ 1.15e-14 rel 8.9e-16 top-1 100.0% identical 8/8) |
| slot:norm:LayerNorm->RMSNorm | fold | LayerNorm -> RMSNorm host WITHOUT an lm_head bias (synthetic, reference module) | synthetic | exact | FINDING: max/d/ 8.71e-01 rel 6.7e-02 top-1 87.5% identical 0/8 (host without an lm_head bias: the final beta is dropped) |
| slot:norm:LayerNorm->RMSNorm | fold | LayerNorm without a beta (Cohere / OLMo-1) -> RMSNorm host without an lm_head bias: writers centred, nothing to fold (synthetic, reference module) | synthetic | exact | pass (max/d/ 1.15e-14 rel 1.1e-15 top-1 100.0% identical 8/8) |
| slot:norm:RMSNorm->LayerNorm | width | RMSNorm -> LayerNorm at 2x width: the residual mirrored [h; -h] (synthetic, reference module) | synthetic | exact | pass (max/d/ 1.24e-14 rel 1.2e-15 top-1 100.0% identical 8/8) |
| slot:pos:alibi->additive_bias | free | ALiBi -> an additive per-head bias table m_h (j - i) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:pos:alibi->partial_rope | mag | ALiBi -> partial RoPE (one rotated pair, q / k biases): the small-angle limit (synthetic, reference module) | synthetic | limit | pass (max/d/ 3.80e-11 rel 3.2e-12 top-1 100.0% identical 8/8) |
| slot:pos:alibi->rope | mag | ALiBi -> RoPE (head_dim + 2, q / k biases): the small-angle limit (synthetic, reference module) | synthetic | limit | pass (max/d/ 3.80e-11 rel 3.2e-12 top-1 100.0% identical 8/8) |
| slot:pos:learned_abs->nope | port | gpt2 (learned positions, LayerNorm) -> llama with inv_freq 0, +2 layers, +5 channels: the start-token rule (hf/bos_constructions.learned_abs_host). A RoPE host at inv_freq 0 is the NoPE host, and a partial-RoPE host at inv_freq 0 on its rotated dims computes the same function | synthetic | exact | pass (max/d/ 8.56e-14 rel 1.2e-13 top-1 100.0% identical 8/8) |
| slot:pos:learned_abs->partial_rope | port | gpt2 (learned positions, LayerNorm) -> llama with inv_freq 0, +2 layers, +5 channels: the start-token rule (hf/bos_constructions.learned_abs_host). A RoPE host at inv_freq 0 is the NoPE host, and a partial-RoPE host at inv_freq 0 on its rotated dims computes the same function | synthetic | exact | pass (max/d/ 8.56e-14 rel 1.2e-13 top-1 100.0% identical 8/8) |
| slot:pos:learned_abs->rope | port | gpt2 (learned positions, LayerNorm) -> llama with inv_freq 0, +2 layers, +5 channels: the start-token rule (hf/bos_constructions.learned_abs_host). A RoPE host at inv_freq 0 is the NoPE host, and a partial-RoPE host at inv_freq 0 on its rotated dims computes the same function | synthetic | exact | pass (max/d/ 8.56e-14 rel 1.2e-13 top-1 100.0% identical 8/8) |
| slot:pos:nope->alibi | free | NoPE -> ALiBi with the bias switched off (slopes 0) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:pos:nope->learned_abs | free | NoPE -> learned absolute positions, table 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:pos:nope->partial_rope | free | NoPE -> partial RoPE (rot hd / 2) with inv_freq = 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:pos:nope->rope | free | NoPE -> RoPE with inv_freq = 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:pos:partial_rope->rope | free | partial RoPE (rot 4 of 8) -> full RoPE: pass dims on zero-frequency pairs of a settable table (synthetic, reference module) | synthetic | exact | pass (max/d/ 1.27e-14 rel 1.3e-15 top-1 100.0% identical 8/8) |
| slot:pos:rope->partial_rope | rot | RoPE (hd 8) -> partial RoPE (hd 16, rot 8): the target's pairs on the host's rotated dims (synthetic, reference module) | synthetic | exact | pass (max/d/ 2.55e-14 rel 2.5e-15 top-1 100.0% identical 8/8) |
| slot:pos:rope->partial_rope | rot | llama (RoPE) -> glm (partial RoPE host, stock): hf/partial_rope_host.glm_from_llama | synthetic | exact | pass (max/d/ 1.33e-15 rel 6.2e-16 top-1 100.0% identical 8/8) |
| slot:pos:t5_relbias->additive_bias | free | T5 bucketed relative bias -> an additive bias table (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:qknorm:head->head_distinct | fold | per-head qk RMSNorm (one gain) -> per-head gains, all equal (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:qknorm:head_free->head | fold | weight-free qk-norm -> gained per-head qk-norm, gain 1 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:qknorm:head_free->head_distinct | fold | weight-free qk-norm -> per-head gains 1 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:qknorm:head_free->head_post | fold | weight-free qk-norm (before RoPE) -> gained qk-norm after RoPE, gain 1: the rms is rotation-invariant (synthetic, reference module) | synthetic | exact | pass (max/d/ 1.07e-14 rel 9.9e-16 top-1 100.0% identical 8/8) |
| slot:qknorm:head_ln->head_ln_bias | fold | per-head LayerNorm qk-norm -> with biases, bias 0 (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:qknorm:ln_shared_bias->head_ln_bias | fold | shared-gain LayerNorm qk-norm -> per-head gains and biases, repeated (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:qknorm:no->head | approx | llama -> qwen3 (per-head qk-norm): q / k rows x 2^-30, the norm linear, gain 2^30 sqrt(eps) | synthetic | limit | pass (max/d/ 3.24e-10 rel 9.2e-12 top-1 100.0% identical 8/8) |
| slot:qknorm:no->head_ln | approx | cohere without qk-norm -> cohere with use_qk_norm (per-head LayerNorm): the flag switched off | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:qknorm:no->head_post | approx | no qk-norm -> per-head qk-norm after RoPE (Hunyuan), switched off by the eps lever (synthetic, reference module) | synthetic | limit | pass (max/d/ 1.02e-14 rel 1.0e-15 top-1 100.0% identical 8/8) |
| slot:qknorm:no->proj | approx | no qk-norm -> whole-projection qk-norm (OLMo-2 style), switched off by the eps lever (synthetic, reference module) | synthetic | limit | pass (max/d/ 1.33e-14 rel 1.3e-15 top-1 100.0% identical 8/8) |
| slot:sink:no->yes | mag | plain attention -> learned per-head sinks at logit -1e4 (exp underflows to 0) (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| slot:softcap:final->attn | cap-large | final cap 30 -> attention cap 1e4 + the same final cap (synthetic, reference module) | synthetic | approx | pass (max/d/ 1.33e-06 rel 1.4e-07 top-1 100.0% identical 8/8) |
| slot:softcap:final->no | argmax | final-logit cap 10 -> no cap: argmax unchanged (synthetic, reference module) | synthetic | approx | pass (max/d/ 2.43e+00 rel 3.2e-01 top-1 100.0% identical 8/8) |
| slot:softcap:no->attn | cap-large | no cap -> attention cap set large (1e4) (synthetic, reference module) | synthetic | approx | pass (max/d/ 1.34e-06 rel 1.3e-07 top-1 100.0% identical 8/8) |
| slot:softcap:no->final | cap-large | no cap -> final-logit cap set large (1e4) (synthetic, reference module) | synthetic | approx | pass (max/d/ 3.42e-06 rel 3.4e-07 top-1 100.0% identical 8/8) |
| slot:window:full->local | depth (ZERO_EDGES) | qwen2 (2 full layers) -> qwen2 (sliding, full, full): full target layers only on host full layers | synthetic | exact | pass (max/d/ 3.05e-15 rel 1.3e-15 top-1 100.0% identical 8/8) |
| slot:window:local->full | mask | mistral (window 4) -> qwen2 (full layers): the host's window set to the target's | synthetic | exact | pass (max/d/ 2.22e-15 rel 7.4e-16 top-1 100.0% identical 8/8) |
| zero:crosslayer_kv:no<-yes | zero | own-KV target (2 layers) -> CLA host (layers 1, 3 reuse the previous layer's K / V): the sharing layers zeroed (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| zero:mixer:softmax<-hybrid | zero | softmax (2 layers) -> hybrid host (attn, rec, attn, rec): recurrent layers zeroed (synthetic, reference module) | synthetic | exact | pass (max/d/ 0.00e+00 rel 0.0e+00 top-1 100.0% identical 8/8) |
| zero:window:full<-local | zero | qwen2 (2 full layers) -> qwen2 (sliding, full, full): full target layers only on host full layers | synthetic | exact | pass (max/d/ 3.05e-15 rel 1.3e-15 top-1 100.0% identical 8/8) |
