transformers main 5.23.0.dev0

| model_type | points | sliding layers | code reads the window | verdict | examples |
|---|---:|---:|---|---|---|
| qwen3 | 26 | 133 | `modeling_qwen3.py:231: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | Qwen3.6-27B-DFlash, NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark |
| gemma4_text | 16 | 313 | `modeling_gemma4.py:1159: self.sliding_window = config.sliding_window if self.is_sliding else None` | window >= context = full attention | Wald-26B-A4B, Gemma-4-31B-IT-NVFP4 |
| mistral | 13 | 303 | `modeling_mistral.py:353: mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask` | window >= context = full attention | Abel-7B-002, Ministral-3b-instruct |
| gemma3_text | 10 | 293 | `modeling_gemma3.py:324: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | GaMS3-12B-Instruct, anon-1b |
| gpt_oss | 6 | 66 | `modeling_gpt_oss.py:284: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | gpt-oss-20b, gpt-oss-120b |
| deepseek_v4 | 6 | 206 | `modeling_deepseek_v4.py:182: super().__init__(sliding_window=config.sliding_window)` | compressed layers stay local (only compress-ratio-0 layers convert) | DeepSeek-V4-Flash-MXFP4, deepseek-v4-flash-0731-spark |
| phimoe | 4 | 100 | `modeling_phimoe.py:635: mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask` | window >= context = full attention | Phi-tiny-MoE-instruct, Phi-3.5-MoE-instruct |
| deepseek_v41_text | 3 | 44 | - | compressed layers stay local (only compress-ratio-0 layers convert) | DeepSeek-V4.1-Flash, DeepSeek-V4.1-Flash-1017-obliterated |
| gemma2 | 3 | 57 | `modeling_gemma2.py:238: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | gemma-2-9b, medicus |
| gemma4_unified_text | 3 | 80 | `modeling_gemma4_unified.py:339: self.sliding_window = config.sliding_window if self.is_sliding else None` | window >= context = full attention | gemma-4-12B-it-assistant, humanizer |
| mimo_v2 | 3 | 135 | `modeling_mimo_v2_flash.py:358: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | MiMo-V2.5, MiMo-V2.5-Pro |
| starcoder2 | 3 | 102 | `modeling_starcoder2.py:356: mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask` | window >= context = full attention | starcoder2-3b, starcoder2-7b |
| afmoe | 3 | 111 | `modeling_afmoe.py:333: self.sliding_window = config.sliding_window if self.is_local_attention else None` | window >= context = full attention | Trinity-Mini, Trinity-Large-Base |
| nanochat_gpt | 3 | 36 | - | remote code: not checked | d16_optimized_537m_seed3, d20_optimized_897m_seed3 |
| laguna | 2 | 66 | `modeling_laguna.py:350: self.sliding_window = config.sliding_window if self.is_local_attention else None` | window >= context = full attention | Laguna-XS.2, Laguna-S-2.1 |
| phi3 | 2 | 64 | `modeling_phi3.py:384: mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask` | window >= context = full attention | Triplex, lexi-coder-v4.3 |
| step3p5 | 2 | 67 | - | remote code: not checked | Step-3.5-Flash, Step-3.5-Flash-Base-Midtrain |
| needle | 2 | 47 | - | remote code: not checked | needle3, needle2 |
| olmo3 | 2 | 72 | `modeling_olmo3.py:164: self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None` | window >= context = full attention | Olmo-3-1025-7B, Olmo-3-1125-32B |
| exaone_moe | 2 | 94 | `modeling_exaone_moe.py:149: self.sliding_window = config.sliding_window` | window >= context = full attention | K-EXAONE-2.0-750B-A37B, K-EXAONE-236B-A23B |
| lfm3 | 2 | 10 | `modeling_lfm3.py:237: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | LFM3-1.7B-Base, LFM3-1.7B-Frontier-SmartAgent |
| mixtral | 2 | 34 | `modeling_mixtral.py:443: mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask` | window >= context = full attention | Smaug-Mixtral-v0.1, tiny-mixtral |
| spark2_5 | 2 | 48 | - | remote code: not checked | Laplace-1-Vector, Spark-X2.6-1.7B-Base |
| gemma3n_text | 2 | 32 | `modeling_gemma3n.py:1168: self.sliding_window = config.sliding_window if self.is_sliding else None` | window >= context = full attention | gemma-3n-E2B-it-lm-4bit, gemma-3n-E4B-it-lm-4bit |
| plamo2 | 2 | 24 | - | remote code: not checked | plamo-2-1b, plamo-2-translate |
| llama | 2 | 2 | - | in library, no window read (the window never binds) | qwen3-1.7b-eagle3-sharegpt-sw768-epoch9-step467800, llama-3.2-3b-instruct-eagle3-sharegpt-sw512-epoch9-step231810 |
| cost_wise_gemma | 1 | 42 | - | remote code: not checked | bge-reranker-v2.5-gemma2-lightweight |
| plamo2vl | 1 | 16 | - | remote code: not checked | plamo-2.1-2b-vl |
| diffusion_gemma_text | 1 | 25 | `modeling_diffusion_gemma.py:277: self.sliding_window = config.sliding_window if self.is_sliding else None` | window >= context = full attention | DiffusionGemma-26B-A4B-it-ORL |
| exaone4 | 1 | 48 | `modeling_exaone4.py:199: self.sliding_window = config.sliding_window` | window >= context = full attention | EXAONE-4.0-32B |
| beam | 1 | 43 | `modeling_beam.py:229: self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None` | window >= context = full attention | Beam-1.0-501B-A23B |
| muse_glimmer_text | 1 | 39 | `modeling_muse_glimmer.py:310: self.sliding_window = config.sliding_window if self.is_local_attention else None` | window >= context = full attention | wayfinder-glimmer-30b |
| naive_n05_flash | 1 | 39 | - | remote code: not checked | Naive-N0.5-Flash |
| gpt_oss_puzzle | 1 | 26 | - | remote code: not checked | gpt-oss-puzzle-88B |
| sn38-nanochrono | 1 | 21 | - | remote code: not checked | sn38-r16-c12-step687-alpha05 |
| dory | 1 | 27 | - | remote code: not checked | Dory-Beam-4B |
| dots3_note | 1 | 33 | `modeling_dots3_note.py:495: self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None` | window >= context = full attention | dots3-note-prev |
| cyrah | 1 | 0 | - | remote code: not checked | my-model-v01 |
| flint | 1 | 35 | `modeling_flint.py:262: self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None` | window >= context = full attention | flint-oQ4e |
| grug_moe | 1 | 26 | - | remote code: not checked | snowball-67b-a2b-base-262k-qk175 |
| iquest_q1 | 1 | 88 | - | remote code: not checked | IQuest-Q1 |
| mellum | 1 | 21 | `modeling_mellum.py:228: self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None` | window >= context = full attention | Mellum2-12B-A2.5B-Base |
| moshi | 1 | 32 | `modeling_moshi.py:510: self.sliding_window = config.sliding_window` | window >= context = full attention | hf-moshika |
| lfm3_rd | 1 | 5 | - | remote code: not checked | LFM3-1.7B-RD |
| granite_swa | 1 | 17 | `modeling_granite_swa.py:149: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | granite-swash-2b |
| granitemoe_swa | 1 | 20 | `modeling_granitemoe_swa.py:229: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | granite-swash-3b-a600m |
| recurrent_gemma | 1 | 8 | `modeling_recurrent_gemma.py:212: self.sliding_window = config.sliding_window` | window >= context = full attention | recurrentgemma-2b |
| phi4flash | 1 | 0 | - | remote code: not checked | Phi-4-mini-flash-reasoning |
| muse_spark_text | 1 | 46 | `modeling_muse_spark.py:378: self.sliding_window = config.sliding_window if self.is_local_attention else None` | window >= context = full attention | Muse-Spark-Open-816B-A42B-GGUF |
| Motif | 1 | 53 | - | remote code: not checked | Motif-3 |
| gemma3_moe | 1 | 15 | - | remote code: not checked | gemma3moe-16x4-270m-starter |
| kanana2_tiny | 1 | 24 | - | remote code: not checked | kanana-2-1.3b-instruct |
| vaultgemma | 1 | 0 | `modeling_vaultgemma.py:188: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | vaultgemma-1b |
| ernie5_5_moe | 1 | 43 | - | compressed layers stay local (only compress-ratio-0 layers convert) | ERNIE-5.5-Flash-Base |
| kolibri1 | 1 | 40 | `modeling_kolibri1.py:149: self.sliding_window = config.sliding_window` | window >= context = full attention | Kolibri-1-mlx-mixed-4-8-bit |
| ministral | 1 | 27 | `modeling_ministral.py:154: self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None` | window >= context = full attention | Ministral-8B-Instruct-heretic |
| muse_glimmer | 1 | 39 | `modeling_muse_glimmer.py:310: self.sliding_window = config.sliding_window if self.is_local_attention else None` | window >= context = full attention | Muse-Glimmer-q4-MLX |
| hymba | 1 | 14 | - | remote code: not checked | Hymba-1.5B-Base |
| jet_nemotron | 1 | 2 | - | remote code: not checked | Jet-Nemotron-2B |
| cohere2 | 1 | 24 | `modeling_cohere2.py:190: self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None` | window >= context = full attention | c4ai-command-r7b-12-2024-heretic |
| minicpm | 1 | 32 | - | remote code: not checked | MiniCPM4-8B-marlin-vLLM |
| meditor2 | 1 | 0 | - | remote code: not checked | Meditor-2-3.4B-A1.8B-HMoE |
| domynedge | 1 | 24 | - | remote code: not checked | Domyn-Edge-Base |
| mm200 | 1 | 16 | - | remote code: not checked | Ying-K1.5-Lite-200M |
| meditor | 1 | 0 | - | remote code: not checked | Meditor-1-2.2B-A500M-MoE |

Points per verdict: window >= context = full attention: 121; remote code: not checked: 32; compressed layers stay local (only compress-ratio-0 layers convert): 10; in library, no window read (the window never binds): 2

Windows that cannot bind as published (window >= own max_position_embeddings): CloneLM-Heavy (4096 >= 2048); nano-mistral (4096 >= 2048); Lite-Mistral-150M-v2-Instruct (4096 >= 2048); Triplex (262144 >= 131072); lexi-coder-v4.3 (262144 >= 131072); rnj-1-instruct (32768 >= 32768); Phi-3.5-MoE-instruct (131072 >= 131072); Phi-3.5-MoE-0.8B-A0.2B (131072 >= 131072); Smaug-Mixtral-v0.1 (32768 >= 32768); hf-moshika (3000 >= 3000); Ministral-8B-Instruct-heretic (32768 >= 32768)
