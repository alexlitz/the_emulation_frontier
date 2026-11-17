# Vendored from the_emulation_frontier/graph/model_graph.py by `python -m recipes.vendor`. Do not edit: edit graph/model_graph.py.
"""Definitions from the_emulation_frontier's graph package that the vendored construction code reads.

Any other name is part of the emulation relation, which tef_recipes does not ship (a recipe's pair was checked
against it when the recipe was generated): importing one raises ImportError.
"""

# graph/model_graph.py: ROPE_PAIRS_FREE_TYPES
ROPE_PAIRS_FREE_TYPES = frozenset({
    "afmoe", "apertus", "arcee", "axk1", "axk2", "bailing_hybrid", "bamba", "beam", "cohere", "cohere2",
    "deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4", "diffusion_gemma_text", "doge", "dots1", "ernie4_5",
    "ernie4_5_moe", "exaone4", "exaone_moe", "falcon", "falcon_h1", "flex_olmo", "flint", "gemma", "gemma2",
    "gemma3", "gemma3_text", "gemma4", "gemma4_text", "gemma4_unified_text", "glm", "glm4", "glm4_moe",
    "glm4_moe_lite", "glm_moe_dsa", "gpt_neox", "gpt_neox_japanese", "gpt_oss", "granite", "granite_swa",
    "granitemoe", "granitemoe_swa", "granitemoehybrid", "helium", "hrm_text", "hunyuan_v1_dense", "hunyuan_v1_moe",
    "hy_v3", "hy_v4", "hyperclovax", "jetmoe", "kolibri1", "laguna", "lfm2", "lfm2_moe", "llama", "llama4",
    "llama4_text", "mellum", "mimo_v2_flash", "minicpm3", "minimax", "minimax_m2", "minimax_m3_vl", "ministral",
    "ministral3", "mistral", "mistral4", "mistral_large4_text", "mixtral", "moshi", "muse_glimmer",
    "muse_glimmer_text", "muse_spark_text", "nemotron", "olmo", "olmo2", "olmo3", "olmoe", "persimmon", "phi",
    "phi3", "phimoe", "qwen2", "qwen2_5_vl", "qwen2_moe", "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text",
    "qwen3_5_text", "qwen3_moe", "qwen3_next", "qwen3_vl", "qwen3_vl_moe", "qwen3_vl_moe_text", "qwen3_vl_text",
    "qwen4_exp", "qwen4_exp_text", "seed_oss", "smollm3", "solar_open", "stablelm", "starcoder2", "vaultgemma",
    "youtu", "zamba2",
    # remote modeling files that build their rotary through ROPE_INIT_FUNCTIONS[rope_type] (hub code, round 2 pass 2)
    "bailing_moe", "k2_horizon", "mimo_v2", "nemotron-nas", "llada2_moe", "exaone", "ouro", "sdar", "sdar_moe",
    "mask_moe", "step3p5", "dream", "xing4_0", "bolmo", "joyai_llm_flash", "internlm3", "step3_text", "param2moe",
    "ming_v3_asr", "sarvam_moe", "apriel", "youtu_vita_text", "youtu_vl", "motif", "kanana2_tiny", "jet_nemotron",
    "iquestloopcoder", "interns1_pro_text",
    # remote modeling files that run a native transformers LM body (its rotary): MiMo (Qwen2Attention), FastVLM /
    # VideoLLaMA3 / Hulu-Med (Qwen2 / Qwen3 models), the bge Gemma-2 reranker (Gemma2RotaryEmbedding), Typhoon2-audio
    # (Llama), Naive-N0.5 (GPTNeoXRotaryEmbedding), gpt-oss-Puzzle, qwen3loop, Isaac (Qwen3), gemma3moe, Magma
    # (AutoModelForCausalLM on a Llama-3 text config), AllGDN / Qwen3.8-Longbook (Qwen3.5), simulvoice (Qwen2)
    "mimo", "llava_qwen2", "videollama3_qwen2", "hulumed_qwen2", "hulumed_qwen3", "cost_wise_gemma",
    "typhoon2audio", "naive_n05_flash", "gpt_oss_puzzle", "qwen3loop", "isaac", "gemma3_moe", "magma", "allgdn",
    "qwen3_5_gdn24", "simulvoice_runtime_probe"})

# graph/model_graph.py: kv_hosts_needed
def kv_hosts_needed(kt, gt, gh, q):
    """Host KV heads (each with gh query heads, holding up to q target KV heads side by side) that a target of
    kt KV heads with gt query heads each needs: the target's query heads laid end to end, each host head takes
    the longest run it can -- <= gh query heads from <= q distinct target KV heads (a run split across host
    heads replicates that target head's K, V). The reach of a head is monotone in its start, so the greedy run
    is optimal among contiguous layouts. q = 1 is the replication rule kt * ceil(gt / gh)."""
    if kt <= 0 or gt <= 0 or gh <= 0:
        return 0
    if q <= 1:
        return kt * -(-gt // gh)
    y, heads, end = 0, 0, kt * gt
    while y < end:
        y = min(y + gh, (y // gt + q) * gt)
        heads += 1
    return heads


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    raise ImportError(f"{name} belongs to the emulation relation (the_emulation_frontier's graph package), which "
                      "tef_recipes does not ship: run the relation gate (python -m elephantine.emulate) from a "
                      "checkout of the repository")
