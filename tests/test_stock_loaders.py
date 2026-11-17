"""The universal host against each stock `transformers` architecture's OWN code (whole models).

Every case builds a small stock model (synthetic config, random weights, fp64), loads it into the one
`UniversalElephantine` with its `from_*` loader and compares logits: against the target as it runs
(its fp32 casts in place: fp32 noise) and with those casts lifted to fp64 (`_lift_fp32`: fp64 noise).
`flips` switch a family-specific branch off in the host and check the logits move: the branch is
doing real work, not sitting at a value where it is inert. `test_real_checkpoints` runs released
checkpoints when they are in the HF cache.

Run:  pytest tests/test_stock_loaders.py -q
"""
import math

import pytest
import torch

from tests.test_universal_elephantine import _ids, _lift_fp32, _rand_

F64 = torch.float64

B = dict(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3, num_attention_heads=6,
         num_key_value_heads=2)
LT = ["sliding_attention", "full_attention", "sliding_attention"]
MOE = dict(num_experts_per_tok=2)
MLA = dict(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3, num_attention_heads=4,
           num_key_value_heads=4, kv_lora_rank=16, q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=12,
           v_head_dim=16, n_routed_experts=8, moe_intermediate_size=24, n_shared_experts=1,
           first_k_dense_replace=1, num_experts_per_tok=2)
G = dict(vocab_size=97, n_embd=48, n_layer=3, n_head=4, n_positions=32)
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}


def _layers(h, f):
    for ul in h.layers:
        f(ul)


def attn_(name, val):
    return lambda h: _layers(h, lambda ul: getattr(ul.attn, name).fill_(val))


def ffn_(name, val):
    return lambda h: _layers(h, lambda ul: getattr(ul.ffn, name).fill_(val))


def layer_(name, val):
    return lambda h: _layers(h, lambda ul: getattr(ul, name).fill_(val))


def host_(name, val):
    return lambda h: getattr(h, name).fill_(val)


def nowin(h):
    _layers(h, lambda ul: ul.attn.window.fill_(float("inf")))


def rope_off(h):
    _layers(h, lambda ul: ul.attn.inv_freq.zero_())


def norm_on(name):
    return lambda h: _layers(h, lambda ul: getattr(ul, name).ident.fill_(0.0))


def qk_off(h):
    _layers(h, lambda ul: (ul.attn.q_coef.copy_(torch.tensor([1.0, 0.0])), ul.attn.k_coef.copy_(torch.tensor([1.0, 0.0]))))


def grp_off(h):
    _layers(h, lambda ul: ul.ffn.grp.copy_(torch.tensor([1.0, 1.0, 0.0, 0.0])))


def bias_off(*names):
    return lambda h: _layers(h, lambda ul: [getattr(ul.ffn, n).zero_() for n in names])


def emb_scale(c):
    return lambda h: h.embed.weight.mul_(c)


def nb_off(h):                                          # SSM normalized features read raw (no RMS)
    _layers(h, lambda ul: (ul.ssm.n_avg.zero_(), ul.ssm.n_eps.fill_(1.0)))


HY = dict(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
          mamba_n_heads=4, mamba_d_head=16, mamba_n_groups=2, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_chunk_size=4)
FH = dict(HY, head_dim=8, mamba_d_ssm=64, embedding_multiplier=2.0, lm_head_multiplier=0.5, mlp_multipliers=[0.7, 1.3],
          attention_in_multiplier=0.8, attention_out_multiplier=1.2, key_multiplier=0.6, ssm_in_multiplier=1.1,
          ssm_out_multiplier=0.9, ssm_multipliers=[0.5, 1.5, 0.8, 1.2, 0.9], use_mamba_kernels=False)


# id: (config class, model class, loader, config kwargs, options); options: flips (branch -> off),
# lift_only (the stock module cannot run in fp64 with its fp32 casts: compare lifted only), attn
# (attention implementation), tol (relative, fp64-lifted), tol32 (relative, as the target runs)
CASES = {
    # --- Llama-shaped bodies (from_llama) ---------------------------------------------------------
    "llama-biases": ("LlamaConfig", "LlamaForCausalLM", "from_llama", dict(B, mlp_bias=True, attention_bias=True), {}),
    "llama-yarn": ("LlamaConfig", "LlamaForCausalLM", "from_llama",
                   dict(B, rope_scaling=YARN, max_position_embeddings=128), dict(flips=[attn_("scale", 0.3)])),
    "llama-longrope": ("LlamaConfig", "LlamaForCausalLM", "from_llama",
                       dict(B, max_position_embeddings=64, rope_scaling={"rope_type": "longrope", "short_factor": [1.0 + 0.1 * i for i in range(4)],
                                                                      "long_factor": [2.0] * 4, "original_max_position_embeddings": 32,
                                                                      "factor": 2.0}), {}),
    "mistral-window": ("MistralConfig", "MistralForCausalLM", "from_llama", dict(B, sliding_window=4), dict(flips=[nowin])),
    "ministral": ("MinistralConfig", "MinistralForCausalLM", "from_llama", dict(B, head_dim=8, sliding_window=4, layer_types=LT),
                  dict(flips=[nowin])),
    "qwen2-window": ("Qwen2Config", "Qwen2ForCausalLM", "from_llama",
                     dict(B, use_sliding_window=True, sliding_window=4, max_window_layers=1), dict(flips=[nowin])),
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM", "from_llama", dict(B, head_dim=16), dict(flips=[qk_off])),
    "qwen3-yarn-window": ("Qwen3Config", "Qwen3ForCausalLM", "from_llama",
                          dict(B, head_dim=16, rope_scaling=YARN, max_position_embeddings=128, use_sliding_window=True,
                               sliding_window=4, max_window_layers=1), dict(flips=[nowin])),
    "helium": ("HeliumConfig", "HeliumForCausalLM", "from_llama", dict(B, head_dim=8), {}),
    "seed_oss": ("SeedOssConfig", "SeedOssForCausalLM", "from_llama", dict(B, attention_bias=True, attention_out_bias=True), {}),
    "ernie4_5": ("Ernie4_5Config", "Ernie4_5ForCausalLM", "from_llama", dict(B, use_bias=True), {}),
    "smollm3-nope": ("SmolLM3Config", "SmolLM3ForCausalLM", "from_llama", dict(B, no_rope_layers=[1, 0, 1], pad_token_id=0), {}),
    "arcee-relu2": ("ArceeConfig", "ArceeForCausalLM", "from_llama", dict(B), dict(flips=[ffn_("c_relu2", 0.0)])),
    "mixtral": ("MixtralConfig", "MixtralForCausalLM", "from_llama", dict(B, num_local_experts=4, sliding_window=4, **MOE),
                dict(flips=[nowin, ffn_("renorm", 0.0)])),
    "qwen2_moe": ("Qwen2MoeConfig", "Qwen2MoeForCausalLM", "from_llama",
                  dict(B, num_experts=4, moe_intermediate_size=24, shared_expert_intermediate_size=40, mlp_only_layers=[1], **MOE),
                  dict(flips=[lambda h: _layers(h, lambda ul: ul.ffn.s_coef.copy_(torch.tensor([1.0, 0.0])))])),
    "qwen3_moe": ("Qwen3MoeConfig", "Qwen3MoeForCausalLM", "from_llama",
                  dict(B, head_dim=8, num_experts=4, moe_intermediate_size=24, norm_topk_prob=True, mlp_only_layers=[1], **MOE),
                  dict(flips=[ffn_("renorm", 0.0)])),
    "olmoe-clip": ("OlmoeConfig", "OlmoeForCausalLM", "from_llama", dict(B, num_experts=4, clip_qkv=0.5, pad_token_id=0, **MOE),
                   dict(flips=[attn_("qkv_clip", float("inf")), attn_("qk_full", 0.0)])),
    "ernie4_5_moe": ("Ernie4_5_MoeConfig", "Ernie4_5_MoeForCausalLM", "from_llama",
                     dict(B, moe_num_experts=4, moe_intermediate_size=24, moe_num_shared_experts=1, moe_layer_start_index=1,
                          moe_k=2, use_bias=True), dict(lift_only=True)),
    "glm4_moe": ("Glm4MoeConfig", "Glm4MoeForCausalLM", "from_llama",
                 dict(B, head_dim=8, n_routed_experts=8, moe_intermediate_size=24, n_shared_experts=1, first_k_dense_replace=1,
                      n_group=4, topk_group=2, routed_scaling_factor=1.5, use_qk_norm=True, partial_rotary_factor=0.5, **MOE),
                 dict(lift_only=True, flips=[grp_off, ffn_("route_scale", 1.0)])),
    "dots1": ("Dots1Config", "Dots1ForCausalLM", "from_llama",
              dict(B, head_dim=8, n_routed_experts=8, moe_intermediate_size=24, n_shared_experts=1, first_k_dense_replace=1,
                   n_group=4, topk_group=2, routed_scaling_factor=1.5, **MOE), dict(lift_only=True, flips=[grp_off, qk_off])),
    "qwen2_5_vl-text": ("Qwen2_5_VLConfig", "Qwen2_5_VLForConditionalGeneration", "from_qwen_vl",
                        dict(text_config=dict(B, rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]}),
                             vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=48)),
                        dict(vl=True)),
    "qwen3_vl-text": ("Qwen3VLConfig", "Qwen3VLForConditionalGeneration", "from_qwen_vl",
                      dict(text_config=dict(B, head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1],
                                                                         "mrope_interleaved": True}),
                           vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=48,
                                              deepstack_visual_indexes=[0])), dict(vl=True)),
    "qwen3_vl_moe-text": ("Qwen3VLMoeConfig", "Qwen3VLMoeForConditionalGeneration", "from_qwen_vl",
                          dict(text_config=dict(B, head_dim=8, num_experts=4, moe_intermediate_size=24, **MOE,
                                                rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1],
                                                              "mrope_interleaved": True}),
                               vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                                                  out_hidden_size=48, deepstack_visual_indexes=[0])), dict(vl=True, lift_only=True)),
    "phimoe-sparsemixer": ("PhimoeConfig", "PhimoeForCausalLM", "from_phimoe",
                           dict(B, num_local_experts=6, num_experts_per_tok=2, router_jitter_noise=0.3, attention_bias=True,
                                lm_head_bias=True),
                           # its rotary builds the angle table per call in fp32 (an fp32 arange, no cast to lift)
                           dict(tol=1e-8, flips=[ffn_("smix_eps", 0.0)])),
    # --- other dense families ---------------------------------------------------------------------
    "granite": ("GraniteConfig", "GraniteForCausalLM", "from_granite",
                dict(B, embedding_multiplier=3.0, residual_multiplier=0.3, attention_multiplier=0.2, logits_scaling=4.0),
                dict(flips=[emb_scale(1 / 3.0)])),
    "granitemoe": ("GraniteMoeConfig", "GraniteMoeForCausalLM", "from_granite",
                   dict(B, num_local_experts=4, embedding_multiplier=3.0, residual_multiplier=0.3, attention_multiplier=0.2,
                        logits_scaling=4.0, **MOE), {}),
    "granitemoeshared": ("GraniteMoeSharedConfig", "GraniteMoeSharedForCausalLM", "from_granite",
                         dict(B, num_local_experts=4, shared_intermediate_size=40, embedding_multiplier=3.0,
                              residual_multiplier=0.3, attention_multiplier=0.2, logits_scaling=4.0, **MOE), {}),
    "phi3-partial-window": ("Phi3Config", "Phi3ForCausalLM", "from_phi3", dict(B, partial_rotary_factor=0.75, sliding_window=4, pad_token_id=0),
                            dict(flips=[nowin])),
    "phi3-longrope": ("Phi3Config", "Phi3ForCausalLM", "from_phi3",
                      dict(B, pad_token_id=0, max_position_embeddings=64, original_max_position_embeddings=32,
                           rope_scaling={"type": "longrope", "short_factor": [1.0 + 0.1 * i for i in range(4)], "long_factor": [2.0] * 4}),
                      dict(flips=[attn_("scale", 1 / math.sqrt(8))])),
    "glm": ("GlmConfig", "GlmForCausalLM", "from_glm", dict(B, head_dim=8, pad_token_id=0), {}),
    "glm4-sandwich": ("Glm4Config", "Glm4ForCausalLM", "from_glm", dict(B, head_dim=8, pad_token_id=0),
                      dict(flips=[lambda h: _layers(h, lambda ul: ul.post_coef.copy_(torch.tensor([[1.0, 0.0]] * 3)))])),
    "olmo-clip": ("OlmoConfig", "OlmoForCausalLM", "from_olmo", dict(B, clip_qkv=0.5, pad_token_id=0),
                  dict(flips=[attn_("qkv_clip", float("inf"))])),
    "olmo2": ("Olmo2Config", "Olmo2ForCausalLM", "from_olmo", dict(B, pad_token_id=0),
              dict(flips=[attn_("qk_full", 0.0), norm_on("norm_a")])),
    "olmo3": ("Olmo3Config", "Olmo3ForCausalLM", "from_olmo", dict(B, sliding_window=4, layer_types=LT, pad_token_id=0),
              dict(flips=[nowin, attn_("qk_full", 0.0)])),
    "flex_olmo": ("FlexOlmoConfig", "FlexOlmoForCausalLM", "from_olmo", dict(B, num_experts=4, pad_token_id=0, **MOE), {}),
    "exaone4": ("Exaone4Config", "Exaone4ForCausalLM", "from_olmo", dict(B, head_dim=8, sliding_window=4, layer_types=LT),
                dict(flips=[nowin])),
    "cohere2": ("Cohere2Config", "Cohere2ForCausalLM", "from_cohere2", dict(B, head_dim=8, sliding_window=4, layer_types=LT, logit_scale=0.5),
                dict(flips=[layer_("par", 0.0), nowin])),
    "hunyuan-qk-post": ("HunYuanDenseV1Config", "HunYuanDenseV1ForCausalLM", "from_hunyuan",
                        dict(B, head_dim=8, use_qk_norm=True, pad_token_id=0), dict(flips=[attn_("qk_post", 0.0)])),
    "hunyuan_moe": ("HunYuanMoEV1Config", "HunYuanMoEV1ForCausalLM", "from_hunyuan",
                    dict(B, head_dim=8, num_experts=4, moe_topk=2, use_qk_norm=True, pad_token_id=0), {}),
    "starcoder2": ("Starcoder2Config", "Starcoder2ForCausalLM", "from_starcoder2", dict(B, sliding_window=4), dict(flips=[nowin])),
    "nemotron": ("NemotronConfig", "NemotronForCausalLM", "from_nemotron", dict(B, head_dim=8, partial_rotary_factor=0.5),
                 dict(flips=[ffn_("c_relu2", 0.0)])),
    "stablelm-qk-ln": ("StableLmConfig", "StableLmForCausalLM", "from_stablelm",
                       dict(B, partial_rotary_factor=0.25, use_qkv_bias=True, qk_layernorm=True, use_parallel_residual=False),
                       dict(flips=[lambda h: _layers(h, lambda ul: ul.attn.qk_centre.zero_())], stablelm_init=True)),
    "stablelm-parallel": ("StableLmConfig", "StableLmForCausalLM", "from_stablelm",
                          dict(B, partial_rotary_factor=0.25, use_parallel_residual=True), dict(flips=[layer_("par", 0.0)])),
    "persimmon": ("PersimmonConfig", "PersimmonForCausalLM", "from_persimmon",
                  dict(B, partial_rotary_factor=0.5, qk_layernorm=True, num_key_value_heads=6),
                  dict(flips=[lambda h: _layers(h, lambda ul: ul.attn.qk_centre.zero_())])),
    "phi-qk-ln": ("PhiConfig", "PhiForCausalLM", "from_phi", dict(B, partial_rotary_factor=0.5, qk_layernorm=True, num_key_value_heads=6),
                  dict(flips=[layer_("par", 0.0)])),
    "phi": ("PhiConfig", "PhiForCausalLM", "from_phi", dict(B, partial_rotary_factor=0.5, num_key_value_heads=6),
            dict(flips=[lambda h: h.lm_head.bias.zero_()])),
    "codegen": ("CodeGenConfig", "CodeGenForCausalLM", "from_gptj", dict(G, rotary_dim=4, n_ctx=32),
                dict(sincos=True, flips=[layer_("par", 0.0)])),
    "gptj": ("GPTJConfig", "GPTJForCausalLM", "from_gptj", dict(G, rotary_dim=4), dict(sincos=True, flips=[rope_off])),
    "gpt_bigcode-mqa": ("GPTBigCodeConfig", "GPTBigCodeForCausalLM", "from_gpt_bigcode", dict(G, multi_query=True), {}),
    "gpt_bigcode-mha": ("GPTBigCodeConfig", "GPTBigCodeForCausalLM", "from_gpt_bigcode", dict(G, multi_query=False), {}),
    "bloom-alibi": ("BloomConfig", "BloomForCausalLM", "from_bloom", dict(vocab_size=97, hidden_size=48, n_layer=3, n_head=4),
                    dict(tol=1e-9, flips=[attn_("alibi", 0.0), host_("emb_norm_c", 0.0)])),   # BloomGelu: 0.79788456
    "opt": ("OPTConfig", "OPTForCausalLM", "from_opt",
            dict(vocab_size=97, hidden_size=48, num_hidden_layers=3, ffn_dim=80, num_attention_heads=6, max_position_embeddings=32,
                 word_embed_proj_dim=48), dict(flips=[ffn_("c_relu", 0.0)])),
    "opt-post-ln": ("OPTConfig", "OPTForCausalLM", "from_opt",
                    dict(vocab_size=97, hidden_size=48, num_hidden_layers=3, ffn_dim=80, num_attention_heads=6,
                         max_position_embeddings=32, word_embed_proj_dim=24, do_layer_norm_before=False),
                    dict(flips=[lambda h: _layers(h, lambda ul: ul.sn_c.zero_())])),
    "cohere-qk-norm": ("CohereConfig", "CohereForCausalLM", "from_cohere",
                       dict(B, use_qk_norm=True, logit_scale=0.5), dict(flips=[lambda h: _layers(h, lambda ul: ul.attn.qk_centre.zero_())])),
    "opt-project": ("OPTConfig", "OPTForCausalLM", "from_opt",
                    dict(vocab_size=97, hidden_size=48, num_hidden_layers=3, ffn_dim=80, num_attention_heads=6,
                         max_position_embeddings=32, word_embed_proj_dim=24), {}),
    # Falcon-ALiBi under SDPA: transformers 4.57's eager path adds the alibi bias twice (once in the
    # scores, once more through the causal mask); SDPA adds it once, as the original Falcon-RW code does
    "falcon-alibi": ("FalconConfig", "FalconForCausalLM", "from_falcon",
                     dict(vocab_size=97, hidden_size=48, num_hidden_layers=3, num_attention_heads=4, alibi=True,
                          multi_query=False, parallel_attn=False, bias=True), dict(attn="sdpa", flips=[attn_("alibi", 0.0)])),
    "falcon-mqa-parallel": ("FalconConfig", "FalconForCausalLM", "from_falcon",
                            dict(vocab_size=97, hidden_size=48, num_hidden_layers=3, num_attention_heads=4, alibi=False,
                                 multi_query=True, parallel_attn=True, bias=False), dict(flips=[layer_("par", 0.0)])),
    "falcon-new-arch": ("FalconConfig", "FalconForCausalLM", "from_falcon",
                        dict(vocab_size=97, hidden_size=48, num_hidden_layers=3, num_attention_heads=4, alibi=False,
                             new_decoder_architecture=True, num_kv_heads=2, bias=False), {}),
    "gemma2-softcaps": ("Gemma2Config", "Gemma2ForCausalLM", "from_gemma",
                        dict(B, num_hidden_layers=4, num_attention_heads=4, head_dim=16, sliding_window=4,
                             attn_logit_softcapping=0.5, final_logit_softcapping=0.7, query_pre_attn_scalar=16),
                        dict(flips=[attn_("cap", float("inf")), host_("final_cap", float("inf")), nowin])),
    "vaultgemma": ("VaultGemmaConfig", "VaultGemmaForCausalLM", "from_gemma",
                   dict(B, num_hidden_layers=4, num_attention_heads=4, head_dim=16, sliding_window=4,
                        attn_logit_softcapping=0.5, final_logit_softcapping=0.7, query_pre_attn_scalar=16),
                   dict(flips=[attn_("cap", float("inf"))])),
    "gemma3-yarn": ("Gemma3TextConfig", "Gemma3ForCausalLM", "from_gemma",
                    dict(B, num_hidden_layers=4, num_attention_heads=4, head_dim=16, sliding_window=4, query_pre_attn_scalar=16,
                         rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32},
                         max_position_embeddings=128), {}),
    # --- Mamba hybrids -----------------------------------------------------------------------------
    "bamba": ("BambaConfig", "BambaForCausalLM", "from_bamba", dict(HY, attn_layer_indices=[1], use_mamba_kernels=False),
              dict(flips=[rope_off])),
    "granitemoehybrid-moe": ("GraniteMoeHybridConfig", "GraniteMoeHybridForCausalLM", "from_bamba",
                             dict(HY, layer_types=["mamba", "attention", "mamba"], num_local_experts=4, num_experts_per_tok=2,
                                  shared_intermediate_size=40, embedding_multiplier=3.0, residual_multiplier=0.3,
                                  attention_multiplier=0.2, logits_scaling=4.0, position_embedding_type="rope"),
                             dict(flips=[emb_scale(1 / 3.0)])),
    "granitemoehybrid-dense-nope": ("GraniteMoeHybridConfig", "GraniteMoeHybridForCausalLM", "from_bamba",
                                    dict(HY, layer_types=["mamba", "attention", "mamba"], num_local_experts=0,
                                         shared_intermediate_size=40, position_embedding_type="nope"), {}),
    "falcon_h1-norm-gate": ("FalconH1Config", "FalconH1ForCausalLM", "from_falcon_h1",
                            dict(FH, mamba_rms_norm=True, mamba_norm_before_gate=True), dict(flips=[layer_("seq", 1.0)])),
    "falcon_h1-gate-norm": ("FalconH1Config", "FalconH1ForCausalLM", "from_falcon_h1",
                            dict(FH, mamba_rms_norm=True, mamba_norm_before_gate=False), {}),
    "falcon_h1-no-norm": ("FalconH1Config", "FalconH1ForCausalLM", "from_falcon_h1", dict(FH, mamba_rms_norm=False), {}),
    "jamba": ("JambaConfig", "JambaForCausalLM", "from_jamba",
              dict(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                   num_key_value_heads=2, attn_layer_period=2, attn_layer_offset=1, expert_layer_period=2, expert_layer_offset=1,
                   num_experts=4, num_experts_per_tok=2, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_dt_rank=6,
                   use_mamba_kernels=False), dict(flips=[nb_off, ffn_("renorm", 1.0)])),
    "falcon_mamba": ("FalconMambaConfig", "FalconMambaForCausalLM", "from_falcon_mamba",
                     dict(vocab_size=97, hidden_size=32, num_hidden_layers=3, state_size=8, expand=2, conv_kernel=4,
                          time_step_rank=6, use_mambapy=False, mixer_rms_eps=1e-6), dict(flips=[nb_off])),
    "recurrent_gemma": ("RecurrentGemmaConfig", "RecurrentGemmaForCausalLM", "from_recurrent_gemma",
                        dict(vocab_size=97, hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=1, head_dim=12, lru_width=48, attention_window_size=4,
                             block_types=["recurrent", "attention", "recurrent"], logits_soft_cap=3.0),
                        # its prefill with use_cache keeps only the last attention_window_size keys when T > window
                        # (both paths agree for T <= window): compared against the cache-free forward
                        dict(nocache=True, flips=[host_("final_cap", float("inf"))])),
    "minimax": ("MiniMaxConfig", "MiniMaxForCausalLM", "from_minimax",
                dict(B, head_dim=8, num_local_experts=4, layer_types=["linear_attention", "full_attention", "linear_attention"],
                     full_attn_alpha_factor=1.7, full_attn_beta_factor=0.6, linear_attn_alpha_factor=1.3,
                     linear_attn_beta_factor=0.8, mlp_alpha_factor=1.5, mlp_beta_factor=0.7, **MOE),
                # the lightning layer's block decay tables are built in fp32 at init (no cast to lift)
                dict(tol=1e-8, flips=[lambda h: _layers(h, lambda ul: ul.res_c.fill_(1.0))])),
    # --- MLA / grouped routing / sinks + swigluoai ---------------------------------------------------
    "deepseek_v3": ("DeepseekV3Config", "DeepseekV3ForCausalLM", "from_deepseek",
                    dict(MLA, n_group=4, topk_group=2, routed_scaling_factor=2.5),
                    dict(lift_only=True, flips=[grp_off, ffn_("route_scale", 1.0), ffn_("sel_bias", 0.0)])),
    "deepseek_v3-yarn-noninterleaved": ("DeepseekV3Config", "DeepseekV3ForCausalLM", "from_deepseek",
                                        dict(MLA, n_group=1, topk_group=1, rope_interleave=False, max_position_embeddings=128,
                                             rope_scaling=dict(YARN, mscale=1.0, mscale_all_dim=1.0, beta_fast=32, beta_slow=1)),
                                        dict(lift_only=True)),
    "deepseek_v2-group-limited": ("DeepseekV2Config", "DeepseekV2ForCausalLM", "from_deepseek",
                                  dict(MLA, n_group=4, topk_group=1, topk_method="group_limited_greedy", routed_scaling_factor=2.0),
                                  dict(flips=[grp_off, ffn_("route_scale", 1.0)])),
    "deepseek_v2-lite": ("DeepseekV2Config", "DeepseekV2ForCausalLM", "from_deepseek",
                         dict(MLA, q_lora_rank=None, topk_method="greedy"), {}),
    "gpt_oss": ("GptOssConfig", "GptOssForCausalLM", "from_gpt_oss",
                dict(vocab_size=97, hidden_size=48, intermediate_size=24, num_hidden_layers=3, num_attention_heads=6,
                     num_key_value_heads=2, head_dim=8, num_local_experts=4, num_experts_per_tok=2, sliding_window=4,
                     layer_types=LT), dict(lift_only=True, gate_scale=40.0,
                                           flips=[attn_("null_logit", float("-inf")), ffn_("up_shift", 0.0), ffn_("c_clip", 0.0),
                                                  bias_off("b_gate", "b_up", "b_down"), lambda h: _layers(h, lambda ul: ul.ffn.router.bias.zero_())])),
}


def _build(cfg_name, model_name, kw, opt):
    import transformers as tr
    extra = {"experts_implementation": "eager"} if int(tr.__version__.split(".")[0]) >= 5 else {}
    cfg = getattr(tr, cfg_name)(**kw, attn_implementation=opt.get("attn", "eager"), **extra)
    if opt.get("stablelm_init"):                        # tf 4.57 _init_weights trips on bias-free per-head LayerNorms
        from transformers.models.stablelm import modeling_stablelm as sm
        old = sm.StableLmPreTrainedModel._init_weights
        sm.StableLmPreTrainedModel._init_weights = lambda self, m: None if getattr(m, "bias", 0) is None else old(self, m)
        try:
            t = getattr(tr, model_name)(cfg)
        finally:
            sm.StableLmPreTrainedModel._init_weights = old
    else:
        t = getattr(tr, model_name)(cfg)
    t = _rand_(t.to(F64).eval())
    if opt.get("sincos"):                               # GPT-J / CodeGen build their sin / cos table in fp32 at init
        from elephantine.universal import _sincos_inv_freq   # (create_sinusoidal_positions): rebuild it with
        for blk in t.transformer.h:                     # the same fp32 frequencies, the angles in fp64
            a = blk.attn
            n, rot = a.embed_positions.shape
            ang = torch.arange(n, dtype=F64)[:, None] * _sincos_inv_freq(rot)[None]
            a.embed_positions = torch.cat((ang.sin(), ang.cos()), 1)
    torch.manual_seed(1)
    with torch.no_grad():                               # selection biases are buffers: give them values
        for n, buf in t.named_buffers():
            if n.endswith("e_score_correction_bias"):
                buf.copy_(torch.randn_like(buf) * 0.05)
    if opt.get("gate_scale"):                           # GPT-OSS: push gate / up past the swigluoai limit
        with torch.no_grad():
            for blk in t.model.layers:
                blk.mlp.experts.gate_up_proj.mul_(opt["gate_scale"])
    return t


@torch.no_grad()
def _run_case(case, spec):
    """Build the stock model, load the host, compare; returns the gaps and each flip's effect."""
    import elephantine.universal as U
    cfg_name, model_name, loader, kw, opt = spec
    t = _build(cfg_name, model_name, kw, opt)
    x = _ids(getattr(t.config, "text_config", t.config).vocab_size, T=12)
    host = getattr(U, loader)(t)
    out = host(x)
    fw = (lambda: t(x, use_cache=False).logits) if opt.get("nocache") else (lambda: t(x).logits)
    with _lift_fp32():
        ref64 = fw()
    r = dict(case=case, mt=t.config.model_type, sc=ref64.abs().max().item(), d64=(ref64 - out).abs().max().item(),
             d=None, flips=[])
    if not opt.get("lift_only"):
        r["d"] = (fw() - out).abs().max().item()
    for flip in opt.get("flips", []):                   # each family branch is doing work
        h2 = getattr(U, loader)(t)
        flip(h2)
        r["flips"].append((h2(x) - out).abs().max().item())
    return r


def _check(r, opt):
    sc = r["sc"]
    assert r["d64"] < opt.get("tol", 1e-12) * sc, r
    if r["d"] is not None:
        assert r["d"] < 1e-5 * sc, r
    for i, df in enumerate(r["flips"]):
        assert df > 1e-6 * sc, (r["case"], "flip", i, df)


@pytest.mark.parametrize("case", list(CASES))
def test_same_host_carries_stock_family(case):
    _check(_run_case(case, CASES[case]), CASES[case][4])


# released checkpoints (CPU, fp64 with the fp32 casts lifted, then in their own fp32); skipped when not cached.
# repo: (loader, model_type, options)
REAL = {
    "HuggingFaceTB/SmolLM2-135M": ("from_llama", "llama", {}),  # llama
    "Qwen/Qwen2.5-0.5B": ("from_llama", "qwen2", {}),  # qwen2: q / k / v biases
    "Qwen/Qwen3-0.6B": ("from_llama", "qwen3", {}),  # qwen3: per-head qk-norm
    "allenai/OLMo-2-0425-1B": ("from_olmo", "olmo2", {}),  # olmo2: post-norms, full-projection qk-norm
    "allenai/OLMo-1B-0724-hf": ("from_olmo", "olmo", {}),  # olmo: non-parametric LayerNorm, clip_qkv
    "ibm-granite/granite-3.0-1b-a400m-base": ("from_granite", "granitemoe", {}),  # granitemoe: multipliers + MoE
    "stabilityai/stablelm-2-1_6b": ("from_stablelm", "stablelm", {}),  # stablelm: partial rotary, LN biases
    "microsoft/phi-1_5": ("from_phi", "phi", {}),  # phi: parallel block, lm_head bias
    "Salesforce/codegen-350M-mono": ("from_gptj", "codegen", dict(sincos=True)),  # codegen: mp_num = 4 qkv, rotate_every_two
    "bigcode/tiny_starcoder_py": ("from_gpt_bigcode", "gpt_bigcode", {}),  # gpt_bigcode: MQA, learned positions
    # bloom: ALiBi, embedding LayerNorm; Bloom builds the ALiBi bias in fp32 (slope * position rounded per
    # element, 16 heads: inexact slopes) and its GeLU constant is 0.79788456: agreement ~2e-8 rel
    "bigscience/bloom-560m": ("from_bloom", "bloom", dict(tol=1e-7)),
    "facebook/opt-125m": ("from_opt", "opt", {}),  # opt: positions offset 2, ReLU
    "facebook/opt-350m": ("from_opt", "opt", {}),  # opt: post-LayerNorm (two host layers per layer), project_in / out
    "google/gemma-2-2b": ("from_gemma", "gemma2", {}),  # gemma2: both softcaps, sandwich, SWA
    # Falcon-RW: ALiBi whose bias Falcon rounds to bf16 (slopes AND slope * position, per element); the host's
    # slope_bf16 * (s - t) is exact, so agreement is at that bf16-bias level, not fp noise
    "tiiuae/falcon-rw-1b": ("from_falcon", "falcon", dict(attn="sdpa", tol=3e-4, tol32=3e-4)),
}


@pytest.mark.parametrize("repo", list(REAL))
@torch.no_grad()
def test_real_checkpoints(repo):
    import elephantine.universal as U
    from transformers import AutoModelForCausalLM
    loader, _, opt = REAL[repo]
    try:
        t = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True, dtype=F64,
                                                 attn_implementation=opt.get("attn", "eager")).eval()
    except OSError:
        pytest.skip(f"{repo} not cached")
    if opt.get("sincos"):
        from elephantine.universal import _sincos_inv_freq
        for blk in t.transformer.h:
            a = blk.attn
            n, rot = a.embed_positions.shape
            ang = torch.arange(n, dtype=F64)[:, None] * _sincos_inv_freq(rot)[None]
            a.embed_positions = torch.cat((ang.sin(), ang.cos()), 1)
    x = _ids(min(t.config.vocab_size, 50000), T=16)
    host = getattr(U, loader)(t)
    out = host(x)
    with _lift_fp32():
        ref64 = t(x).logits
    sc = ref64.abs().max().item()
    d64 = (ref64 - out).abs().max().item()
    print(f"{repo}: fp64-lifted max |diff| {d64:.3e} (rel {d64 / sc:.1e})")
    assert d64 < opt.get("tol", 1e-12) * sc, (repo, d64, sc)
    del ref64
    ref32 = t.float()(x).logits.double()
    d32 = (ref32 - out).abs().max().item()
    print(f"{repo}: vs its own fp32 max |diff| {d32:.3e} (rel {d32 / sc:.1e})")
    assert d32 < opt.get("tol32", 1e-4) * sc, (repo, d32, sc)
    assert (ref32.argmax(-1) == out.argmax(-1)).float().mean().item() > 0.95


# --- transformers-5-only families: the same runner, in a transformers 5.18 subprocess (4.57 is installed) ---------
NH = dict(vocab_size=97, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8, intermediate_size=48,
          mamba_num_heads=4, mamba_head_dim=16, n_groups=2, ssm_state_size=8, conv_kernel=4, chunk_size=4,
          n_routed_experts=8, moe_intermediate_size=16, moe_shared_expert_intermediate_size=24, num_experts_per_tok=2,
          n_group=4, topk_group=2, routed_scaling_factor=2.0, norm_topk_prob=True, mlp_hidden_act="relu2")
NH_LT = ["linear_attention", "moe", "full_attention", "mlp", "linear_attention", "moe"]

TF5_TYPES = {'mistral4': 'mistral4', 'mimo_v2_flash': 'mimo_v2_flash', 'axk1': 'axk1', 'youtu': 'youtu', 'minicpm3': 'minicpm3', 'granite_swa': 'granite_swa',
             'granitemoe_swa': 'granitemoe_swa', 'hyperclovax-post-norm': 'hyperclovax', 'hyperclovax': 'hyperclovax',
             'kimi_linear': 'kimi_linear', 'olmo_hybrid': 'olmo_hybrid', 'glm_moe_dsa': 'glm_moe_dsa', 'deepseek_v32': 'deepseek_v32',
             'gemma4-ple-kvshare-keqv': 'gemma4_text', 'gemma4_unified': 'gemma4_unified_text', 'gemma4-moe': 'gemma4_text',
             'nemotron_h': 'nemotron_h',
             'nemotron_h-latent': 'nemotron_h',
             'ministral3-llama4-temp': 'ministral3',
             'hy_v3': 'hy_v3',
             'exaone_moe': 'exaone_moe',
             'solar_open': 'solar_open',
             'mellum': 'mellum',
             'minimax_m2': 'minimax_m2',
             'afmoe': 'afmoe',
             'laguna': 'laguna',
             'glm4_moe_lite': 'glm4_moe_lite',
             'deepseek_v3-tf5': 'deepseek_v3'}   # case -> model_type (the tf5 config classes are not importable here)

G4 = dict(vocab_size=97, hidden_size=48, intermediate_size=40, num_hidden_layers=6, num_attention_heads=4,
          num_key_value_heads=2, head_dim=8, global_head_dim=16, sliding_window=4, final_logit_softcapping=3.0,
          layer_types=["sliding_attention", "full_attention"] * 3)
TF5_CASES = {
    "gemma4-ple-kvshare-keqv": ("Gemma4TextConfig", "Gemma4ForCausalLM", "from_gemma4",
                                dict(G4, hidden_size_per_layer_input=8, vocab_size_per_layer_input=97, num_kv_shared_layers=2,
                                     attention_k_eq_v=True, num_global_key_value_heads=1, use_double_wide_mlp=True),
                                dict(flips=[host_("final_cap", float("inf")), attn_("v_c", 0.0), layer_("ple_c", 0.0),
                                            attn_("kv_read", 0.0), nowin,
                                            lambda h: _layers(h, lambda ul: ul.attn.qk_width.fill_(16.0))])),
    "gemma4_unified": ("Gemma4UnifiedTextConfig", "Gemma4UnifiedForCausalLM", "from_gemma4",
                       dict(G4, num_kv_shared_layers=2, attention_k_eq_v=True, num_global_key_value_heads=1),
                       dict(flips=[attn_("kv_read", 0.0)])),
    "glm_moe_dsa": ("GlmMoeDsaConfig", "GlmMoeDsaForCausalLM", "from_dsa",
                    dict(MLA, n_group=4, topk_group=2, routed_scaling_factor=1.5, index_topk=5, index_head_dim=16,
                         index_n_heads=8, indexer_types=["full", "shared", "full"]),
                    dict(lift_only=True, flips=[attn_("dsa_topk", float("inf")), grp_off,
                                                lambda h: h.layers[1].attn.dsa_share.fill_(0.0)])),
    "deepseek_v32": ("DeepseekV32Config", "DeepseekV32ForCausalLM", "from_dsa",
                     dict(MLA, n_group=4, topk_group=2, routed_scaling_factor=1.5, index_topk=5, index_head_dim=16,
                          index_n_heads=8), dict(lift_only=True, flips=[attn_("dsa_topk", float("inf"))])),
    "kimi_linear": ("KimiLinearConfig", "KimiLinearForCausalLM", "from_kimi_linear",
                    dict(MLA, linear_attn_config={"kda_layers": [1, 3], "full_attn_layers": [2], "head_dim": 8, "num_heads": 4,
                                                  "short_conv_kernel_size": 4},
                         mlp_layer_types=["dense", "sparse", "sparse"], n_group=4, topk_group=2, routed_scaling_factor=1.5,
                         pad_token_id=0),
                    dict(lift_only=True, flips=[grp_off])),
    "olmo_hybrid": ("OlmoHybridConfig", "OlmoHybridForCausalLM", "from_olmo_hybrid",
                    dict(B, layer_types=["linear_attention", "full_attention", "linear_attention"], linear_num_key_heads=2,
                         linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8, pad_token_id=0),
                    dict(lift_only=True, flips=[lambda h: _layers(h, lambda ul: ul.ssm.cg.mul_(0.5))])),
    "youtu": ("YoutuConfig", "YoutuForCausalLM", "from_deepseek", dict(MLA, pad_token_id=0), dict(lift_only=True)),
    "minicpm3": ("MiniCPM3Config", "MiniCPM3ForCausalLM", "from_deepseek",
                 dict(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, kv_lora_rank=16, q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=12,
                      v_head_dim=16, scale_emb=3.0, scale_depth=1.4, dim_model_base=16, pad_token_id=0),
                 dict(lift_only=True, flips=[emb_scale(1 / 3.0)])),
    "granite_swa": ("GraniteSWAConfig", "GraniteSWAForCausalLM", "from_granite",
                    dict(B, sliding_window=4, layer_types=LT, embedding_multiplier=3.0, residual_multiplier=0.3, attention_multiplier=0.2, logits_scaling=4.0, pad_token_id=0),
                    dict(flips=[nowin, attn_("null_logit", float("-inf"))])),
    "granitemoe_swa": ("GraniteMoeSWAConfig", "GraniteMoeSWAForCausalLM", "from_granite",
                       dict(B, sliding_window=4, layer_types=LT, num_local_experts=4, embedding_multiplier=3.0, residual_multiplier=0.3, attention_multiplier=0.2, logits_scaling=4.0, pad_token_id=0, **MOE),
                       dict(lift_only=True, flips=[attn_("null_logit", float("-inf"))])),
    "hyperclovax-post-norm": ("HyperCLOVAXConfig", "HyperCLOVAXForCausalLM", "from_hyperclovax",
                              dict(B, embedding_multiplier=3.0, residual_multiplier=0.3, attention_multiplier=0.2, logits_scaling=4.0, use_post_norm=True, pad_token_id=0),
                              dict(flips=[lambda h: _layers(h, lambda ul: ul.post_coef.copy_(torch.tensor([[1.0, 0.0]] * 3)))])),
    "hyperclovax": ("HyperCLOVAXConfig", "HyperCLOVAXForCausalLM", "from_hyperclovax",
                    dict(B, embedding_multiplier=3.0, residual_multiplier=0.3, attention_multiplier=0.2, logits_scaling=4.0, use_post_norm=False, pad_token_id=0), {}),
    "axk1": ("AXK1Config", "AXK1ForCausalLM", "from_deepseek", dict(MLA, n_group=4, topk_group=2, pad_token_id=0),
             dict(lift_only=True)),
    "mimo_v2_flash": ("MiMoV2FlashConfig", "MiMoV2FlashForCausalLM", "from_mimo_v2_flash",
                      dict(B, num_key_value_heads=1, head_dim=12, v_head_dim=8, attention_value_scale=0.7, sliding_window=4, layer_types=LT,
                           n_routed_experts=8, moe_intermediate_size=24, n_group=4, topk_group=2, routed_scaling_factor=1.5,
                           mlp_layer_types=["dense", "sparse", "sparse"], pad_token_id=0, **MOE),
                      dict(lift_only=True, flips=[nowin, attn_("null_logit", float("-inf"))])),
    "mistral4": ("Mistral4Config", "Mistral4ForCausalLM", "from_deepseek",
                 dict(MLA, head_dim=8, n_group=4, topk_group=2, pad_token_id=0,
                      rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "llama_4_scaling_beta": 0.3,
                                       "original_max_position_embeddings": 4}),
                 dict(lift_only=True, tol=1e-8, flips=[attn_("qtemp", 0.0)])),
    "gemma4-moe": ("Gemma4TextConfig", "Gemma4ForCausalLM", "from_gemma4",
                   dict(G4, hidden_size_per_layer_input=0, enable_moe_block=True, num_experts=4, top_k_experts=2,
                        moe_intermediate_size=16), dict(lift_only=True, flips=[ffn_("exp_scale", 1.0), layer_("ffn_split", 0.0)])),
    "nemotron_h": ("NemotronHConfig", "NemotronHForCausalLM", "from_nemotron_h",
                   dict(NH, layers_block_type=NH_LT, time_step_min=0.4, time_step_max=0.5),   # the dt floor bites
                   dict(lift_only=True, flips=[grp_off, ffn_("route_scale", 1.0),
                                               lambda h: _layers(h, lambda ul: ul.ssm.dt_min.fill_(0.0))])),
    "nemotron_h-latent": ("NemotronHConfig", "NemotronHForCausalLM", "from_nemotron_h",
                          dict(NH, layers_block_type=NH_LT, moe_latent_size=12), dict(lift_only=True)),
    "ministral3-llama4-temp": ("Ministral3Config", "Ministral3ForCausalLM", "from_llama",
                               dict(B, head_dim=8, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                                     "llama_4_scaling_beta": 0.3,
                                                                     "original_max_position_embeddings": 4}),
                               # the target computes the temperature log in fp32 (default dtype, no cast to lift)
                               dict(tol=1e-8, flips=[attn_("qtemp", 0.0)])),
    "hy_v3": ("HYV3Config", "HYV3ForCausalLM", "from_llama",
              dict(B, head_dim=8, num_experts=8, moe_intermediate_size=24, num_shared_experts=1, first_k_dense_replace=1,
                   router_scaling_factor=2.0, **MOE), dict(lift_only=True, flips=[ffn_("sel_bias", 0.0), ffn_("route_scale", 1.0)])),
    "exaone_moe": ("ExaoneMoeConfig", "ExaoneMoeForCausalLM", "from_llama",
                   dict(B, head_dim=8, num_experts=8, moe_intermediate_size=24, num_shared_experts=1, n_group=4, topk_group=2,
                        routed_scaling_factor=2.0, sliding_window=4, layer_types=LT, mlp_layer_types=["dense", "sparse", "sparse"], **MOE),
                   dict(lift_only=True, flips=[grp_off, nowin])),
    "solar_open": ("SolarOpenConfig", "SolarOpenForCausalLM", "from_llama",
                   dict(B, head_dim=8, n_routed_experts=8, moe_intermediate_size=24, n_shared_experts=1, n_group=4, topk_group=2,
                        routed_scaling_factor=2.0, **MOE), dict(lift_only=True, flips=[grp_off])),
    "mellum": ("MellumConfig", "MellumForCausalLM", "from_llama",
               dict(B, head_dim=8, num_experts=4, moe_intermediate_size=24, norm_topk_prob=True, **MOE), dict(flips=[ffn_("renorm", 0.0)])),
    "minimax_m2": ("MiniMaxM2Config", "MiniMaxM2ForCausalLM", "from_llama",
                   dict(B, head_dim=8, num_local_experts=4, **MOE), dict(lift_only=True, flips=[attn_("qk_full", 0.0)])),
    "afmoe": ("AfmoeConfig", "AfmoeForCausalLM", "from_afmoe",
              dict(B, head_dim=8, num_experts=8, moe_intermediate_size=24, num_shared_experts=1, num_dense_layers=1,
                   sliding_window=4, layer_types=LT, route_scale=2.0, mup_enabled=True, **MOE),
              dict(lift_only=True, flips=[nowin, ffn_("route_scale", 1.0), attn_("gate_coef", 0.0)])),
    "laguna": ("LagunaConfig", "LagunaForCausalLM", "from_laguna",
               dict(B, head_dim=8, num_attention_heads_per_layer=[6, 4, 6], num_experts=8, moe_intermediate_size=24,
                    shared_expert_intermediate_size=24, mlp_layer_types=["dense", "sparse", "sparse"], sliding_window=4,
                    layer_types=LT, moe_router_logit_softcapping=0.5, moe_routed_scaling_factor=2.0, gating="per-head", **MOE),
               dict(lift_only=True, flips=[nowin, ffn_("router_cap", float("inf")), attn_("gate_sp", 0.0)])),
    "glm4_moe_lite": ("Glm4MoeLiteConfig", "Glm4MoeLiteForCausalLM", "from_deepseek",
                      dict(MLA, n_group=4, topk_group=2, routed_scaling_factor=1.5), dict(lift_only=True, flips=[grp_off])),
    "deepseek_v3-tf5": ("DeepseekV3Config", "DeepseekV3ForCausalLM", "from_deepseek",
                        dict(MLA, n_group=4, topk_group=2, routed_scaling_factor=2.5), dict(lift_only=True, flips=[grp_off])),
}


@pytest.fixture(scope="module")
def tf5_results():
    """Run every TF5_CASES entry in ONE transformers-5.18 subprocess; case -> result dict."""
    import json, os, subprocess, sys
    from tests.test_universal_elephantine import TF5, TF5_PRELUDE
    if not os.path.isdir(TF5):
        pytest.skip("transformers 5.18 not unpacked (python -m elephantine.fetch_upstream_refs)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    body = TF5_PRELUDE + """
import json, traceback
from tests.test_stock_loaders import TF5_CASES, _run_case
for k, v in TF5_CASES.items():
    try:
        print("RESULT " + json.dumps(_run_case(k, v)), flush=True)
    except Exception as e:
        print("RESULT " + json.dumps(dict(case=k, error=traceback.format_exc()[-3000:])), flush=True)
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, root]))
    r = subprocess.run([sys.executable, "-c", body], env=env, capture_output=True, text=True, timeout=1800, cwd=root)
    out = {}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            d = json.loads(line[7:])
            out[d["case"]] = d
    assert out, r.stdout[-2000:] + r.stderr[-4000:]
    return out


@pytest.mark.parametrize("case", list(TF5_CASES))
def test_same_host_carries_tf5_family(case, tf5_results):
    r = tf5_results.get(case)
    assert r is not None and "error" not in r, (r or {}).get("error", "no result")
    assert r["mt"] == TF5_TYPES[case], (r["mt"], TF5_TYPES[case])
    _check(r, TF5_CASES[case][4])


def test_tf5_types_cover_the_cases():
    assert set(TF5_TYPES) == set(TF5_CASES)


# --- Gemma-4 K = V (attention_k_eq_v): both directions, on the stock transformers-5 code ---------------------
G4_KEQV_BODY = '''
import json, torch
from tests.test_stock_loaders import G4, _build, _ids, _lift_fp32
import elephantine.universal as U
F64 = torch.float64
out = {}
# 1. a stock K = V target on the separate-K / V host: from_gemma4 copies k_proj into the host's v_proj on the
#    full layers (the target has v_proj = None there; v = v_norm(k_proj x), no k-norm, no rotary on V)
t = _build("Gemma4TextConfig", "Gemma4ForCausalLM", dict(G4, hidden_size_per_layer_input=0, attention_k_eq_v=True,
           num_global_key_value_heads=1), {})
x = _ids(97, T=12)
host = U.from_gemma4(t)
with torch.no_grad():
    o = host(x)
    with _lift_fp32():
        ref = t(x).logits
out["keqv_gap"], out["keqv_scale"] = (ref - o).abs().max().item(), ref.abs().max().item()
full = [i for i, lt in enumerate(t.config.layer_types) if lt == "full_attention"]
out["full_v_none"] = all(t.model.layers[i].self_attn.v_proj is None for i in full)
out["host_v_is_k"] = all(torch.equal(host.layers[i].attn.v_proj.weight, host.layers[i].attn.k_proj.weight) for i in full)
# 2. a K = V layer cannot play a layer with an independent V: the same-shape target WITHOUT the tie, hosted
#    with the host's full layers tied v_proj := k_proj (all a K = V layer holds), misses by O(1)
t2 = _build("Gemma4TextConfig", "Gemma4ForCausalLM", dict(G4, hidden_size_per_layer_input=0, attention_k_eq_v=False), {})
h2 = U.from_gemma4(t2)
with torch.no_grad():
    o2 = h2(x)
    with _lift_fp32():
        ref2 = t2(x).logits
    out["untied_gap"] = (ref2 - o2).abs().max().item()
    for i in full:
        a = h2.layers[i].attn
        a.v_proj.weight.copy_(a.k_proj.weight); a.v_proj.bias.copy_(a.k_proj.bias)
    out["tied_host_gap"], out["untied_scale"] = (ref2 - h2(x)).abs().max().item(), ref2.abs().max().item()
print("RESULT " + json.dumps(out))
'''


def test_gemma4_k_eq_v_both_directions():
    """Gemma-4 attention_k_eq_v: (1) a separate-K / V host plays the K = V target exactly (k_proj copied into
    v_proj); (2) tying the host's V to its K (all a K = V layer can hold) breaks a target with its own V --
    the graph's per-layer kinds (FT / LT reached only from FU / LU, model_graph.KIND_REACH) and the *_u
    capacities encode both directions."""
    import json, os, subprocess, sys
    from tests.test_universal_elephantine import TF5, TF5_PRELUDE
    if not os.path.isdir(TF5):
        pytest.skip("transformers 5.18 not unpacked (python -m elephantine.fetch_upstream_refs)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, root]))
    r = subprocess.run([sys.executable, "-c", TF5_PRELUDE + G4_KEQV_BODY], env=env, capture_output=True, text=True,
                       timeout=900, cwd=root)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, r.stdout[-2000:] + r.stderr[-4000:]
    d = json.loads(line[7:])
    assert d["full_v_none"] and d["host_v_is_k"]
    assert d["keqv_gap"] < 1e-12 * d["keqv_scale"], d                  # exact (fp64, casts lifted)
    assert d["untied_gap"] < 1e-12 * d["untied_scale"], d              # the untied target is carried too ...
    assert d["tied_host_gap"] > 1e-2 * d["untied_scale"], d            # ... but not with V tied to K


def test_gemma4_k_eq_v_in_the_graph():
    """The headline relation reads the tie: a Gemma-4 K = V model is hosted by a separate-KV twin of the same
    shape, and does not host it; a sliding-window K = V host does not count its K = V full layers for a target's
    independent-V full layers."""
    import copy
    from graph.model_graph import classify, dims_of, emulates
    cfg = {"model_type": "gemma4_text", "architectures": ["Gemma4ForCausalLM"], "hidden_size": 5376,
           "num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 256,
           "global_head_dim": 512, "num_global_key_value_heads": 4, "attention_k_eq_v": True,
           "intermediate_size": 21504, "sliding_window": 1024, "rms_norm_eps": 1e-6, "vocab_size": 262144,
           "hidden_activation": "gelu_pytorch_tanh", "rope_theta": 1e6,
           "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10}
    untied = {**copy.deepcopy(cfg), "attention_k_eq_v": False, "num_key_value_heads": 16}
    untied["num_global_key_value_heads"] = 4
    pt = lambda c: {"slots": classify(c), "dims": dims_of(c)}
    T, U_ = pt(cfg), pt(untied)
    assert T["dims"]["kFT_eff"] == 10 and T["dims"]["kLU_eff"] == 50 and T["dims"]["kFU_eff"] == 0
    assert U_["dims"]["kFU_eff"] == 10 and U_["dims"]["kFT_eff"] == 0
    # the independent-V capacity of the K = V model is its sliding layers' (16 x 256), not the global 512-dim heads
    assert T["dims"]["head_dim"] == 512 and T["dims"]["head_dim_u"] == 256
    assert emulates(U_, T)                       # separate K, V hosts K = V: copy k_proj into v_proj
    assert not emulates(T, U_)                   # K = V hosts no independent-V full layer

