"""Mechanical classification rules for sweep hunks.

Each rule returns (label, slot, reason) or None. A hunk no rule decides needs a review entry (reviews/*.jsonl).
The rules only ever say IRRELEVANT (or MAPPED for config defaults the classifier already reads through the config
class); a hunk that changes what the forward pass computes is never decided by a rule.
"""
import re

IRRELEVANT, MAPPED, UNMAPPED = "IRRELEVANT", "MAPPED", "UNMAPPED"

# members that are not on the forward path of fixed weights
NON_FORWARD_MEMBERS = {
    # init / training
    "_init_weights": "weight initialisation (training only; the weights are loaded)",
    "init_weights": "weight initialisation (training only)",
    "_initialize_weights": "weight initialisation (training only)",
    "reset_parameters": "weight initialisation (training only)",
    "_set_gradient_checkpointing": "gradient checkpointing (training memory, same function)",
    "gradient_checkpointing_enable": "gradient checkpointing", "gradient_checkpointing_disable": "gradient checkpointing",
    "supports_gradient_checkpointing": "gradient checkpointing flag",
    "_backward_compatibility_gradient_checkpointing": "gradient checkpointing",
    "enable_input_require_grads": "training helper", "load_balancing_loss_func": "auxiliary training loss",
    "_supports_param_buffer_assignment": "loading plumbing",
    # module plumbing
    "get_input_embeddings": "accessor", "set_input_embeddings": "accessor", "get_output_embeddings": "accessor",
    "set_output_embeddings": "accessor", "get_decoder": "accessor", "set_decoder": "accessor", "get_encoder": "accessor",
    "get_model": "accessor", "tie_weights": "weight tying at load time (the tie itself is the tied_emb feature, read "
                                            "from the config)", "_tie_weights": "weight tying at load time",
    "_tie_or_clone_weights": "weight tying at load time", "resize_token_embeddings": "vocab resizing helper",
    "_resize_token_embeddings": "vocab resizing helper", "post_init": "init hook", "extra_repr": "repr string",
    "__repr__": "repr string", "_from_config": "constructor plumbing", "from_pretrained": "loading plumbing",
    "_load_pretrained_model": "loading plumbing", "_get_no_split_modules": "device-map plumbing",
    "base_model_prefix": "state-dict prefix", "config_class": "config class pointer", "config": "type annotation",
    "main_input_name": "input name", "input_modalities": "modality list", "_no_split_modules": "device-map plumbing",
    "_skip_keys_device_placement": "device-map plumbing", "_keys_to_ignore_on_load_missing": "loading plumbing",
    "_keys_to_ignore_on_load_unexpected": "loading plumbing", "_keys_to_ignore_on_save": "saving plumbing",
    "_checkpoint_conversion_mapping": "state-dict key renames", "_tied_weights_keys": "weight-tying key list (the tie "
                                                                                        "is read from the config)",
    "_tp_plan": "tensor-parallel plan", "_pp_plan": "pipeline-parallel plan", "_ep_plan": "expert-parallel plan",
    "base_model_tp_plan": "tensor-parallel plan", "base_model_pp_plan": "pipeline-parallel plan",
    "base_model_ep_plan": "expert-parallel plan", "_supports_flash_attn": "attention-backend flag",
    "_supports_flash_attn_2": "attention-backend flag", "_supports_flash_attn_3": "attention-backend flag",
    "_supports_sdpa": "attention-backend flag", "_supports_flex_attn": "attention-backend flag",
    "_supports_cache_class": "cache flag", "_supports_static_cache": "cache flag", "_supports_quantized_cache": "cache flag",
    "_supports_attention_backend": "attention-backend flag", "_can_compile_fullgraph": "torch.compile flag",
    "_can_record_outputs": "output-capture hooks", "_is_stateful": "generation flag (stateful cache)",
    "_supports_default_dynamic_cache": "cache flag", "_supports_static_cache_class": "cache flag",
    "is_parallelizable": "model-parallel flag", "model_parallel": "model-parallel flag", "parallelize": "model-parallel",
    "deparallelize": "model-parallel", "device_map": "model-parallel", "_is_hf_initialized": "init flag",
    "_input_embed_layer": "accessor name", "_supports_cache": "cache flag", "_supports_mtp": "generation flag",
    "_keep_in_fp32_modules_strict": None, "_keep_in_fp32_modules": None,       # numerics: reviewed
    "can_generate": "generation flag", "_supports_static_cache_shapes": "cache flag",
    "_no_split_modules_list": "device-map plumbing", "accepts_loss_kwargs": "training loss plumbing",
    "_compile_dynamic": "torch.compile flag", "_supports_attention_mask": "flag",
    # generation helpers (not the forward function)
    "prepare_inputs_for_generation": "generation helper", "_reorder_cache": "generation cache reorder",
    "generate": "generation loop", "chat": "chat helper", "stream_chat": "chat helper", "stream_generate": "generation",
    "build_inputs": "chat helper", "_build_chat_input": "chat helper", "process_response": "chat helper",
    "batch_chat": "chat helper", "_update_model_kwargs_for_generation": "generation helper",
    "_prepare_cache_for_generation": "generation helper", "_get_initial_cache_position": "generation helper",
    "_supports_logits_to_keep": "generation flag", "_validate_model_kwargs": "generation helper",
    "_cache_dependant_input_preparation": "generation helper", "_prepare_generation_config": "generation helper",
    "get_rope_index": None,
    "_create_cache": "cache plumbing", "_setup_cache": "cache plumbing", "_reset_cache": "cache plumbing",
    "setup_cache": "cache plumbing", "reset_cache": "cache plumbing", "_clear_cache": "cache plumbing",
}

# identifiers whose lines only move plumbing (cache, output flags, logging, return containers, kwargs typing)
PLUMBING = set("""
self None True False if else elif not and or is in return pass for return_dict use_return_dict output_attentions
output_hidden_states use_cache past_key_value past_key_values cache_position all_hidden_states all_self_attns
all_attentions all_router_logits next_decoder_cache next_cache present_key_value presents layer_past
output_router_logits router_logits kwargs Unpack FlashAttentionKwargs TransformersKwargs Optional Tuple Union List
Cache DynamicCache torch Tensor LongTensor FloatTensor BoolTensor bool int float str tuple list dict Callable
logger warning warning_once info debug warn warnings logging use_reentrant gradient_checkpointing training
BaseModelOutputWithPast CausalLMOutputWithPast MoeModelOutputWithPast MoeCausalLMOutputWithPast ModelOutput
last_hidden_state hidden_states attentions loss logits output outputs v isinstance to_tuple can_return_tuple
auto_docstring check_model_inputs capture_outputs merge_with_config_defaults deprecate_kwarg add_start_docstrings
add_start_docstrings_to_model_forward replace_return_docstrings is_torchdynamo_compiling layer_outputs
attn_weights self_attn_weights position_embeddings new_name version raise ValueError NotImplementedError
DynamicLayer from_legacy_cache to_legacy_cache return_legacy_cache get_seq_length config num_logits_to_keep
logits_to_keep slice_indices def forward hidden_states_out output_type _CONFIG_FOR_DOC _CHECKPOINT_FOR_DOC
device default_device dtype to type_as float32 float16 bfloat16 half jit is_tracing is_torchdynamo_exporting
_attn_implementation AttentionMaskConverter _unmask_unattended min_dtype finfo npu xpu cuda mps device_type
""".split())
LINE_IRRELEVANT = re.compile(r"^[-+]\s*(logger\.|warnings\.warn|raise |assert |@|#|print\(|\)|\]|\}|pass$|\.\.\.$)")
TOKEN = re.compile(r"[A-Za-z_]\w*")


def plumbing_line(line):
    body = line[1:].strip()
    if not body or LINE_IRRELEVANT.match(line):
        return True
    if re.match(r"^(def|async def) \w+\(", body):
        return True                                  # a signature: what the body computes is in the body hunks
    if LOSS_LINE.match(body):
        return True                                  # the training loss: computed beside the logits, never into them
    idents = set(TOKEN.findall(re.sub(r"(['\"]).*?\1", "", body)))
    return idents <= PLUMBING


LOSS_LINE = re.compile(r"^(\w*loss\w*\s*[+\-*/]?=|\w*loss\w*\s*,|if (labels|\w*loss\w*) is not None|"
                       r"(shift_logits|shift_labels|labels|loss_fct)\s*=|aux_loss\b|if output_router_logits)")


# config keys whose default never changes the forward function on fixed weights
CONFIG_IRRELEVANT = {
    "initializer_range": "init only", "initializer_factor": "init only", "init_std": "init only",
    "use_cache": "cache flag", "output_attentions": "output flag", "output_hidden_states": "output flag",
    "return_dict": "output flag", "torchscript": "export flag", "bos_token_id": "special-token id (data, not the "
    "function)", "eos_token_id": "special-token id", "pad_token_id": "special-token id (the embedding row is a "
    "weight)", "sep_token_id": "special-token id", "decoder_start_token_id": "special-token id",
    "router_aux_loss_coef": "training loss weight", "output_router_logits": "training loss flag",
    "aux_loss_alpha": "training loss weight", "seq_aux": "training loss flag", "z_loss_coef": "training loss weight",
    "router_z_loss_coef": "training loss weight", "moe_aux_loss_coeff": "training loss weight",
    "aux_loss_coef": "training loss weight", "gradient_checkpointing": "training memory",
    "pretraining_tp": "tensor-parallel sliced matmuls: the same sums (Llama-2 pretraining_tp)",
    "transformers_version": "metadata", "architectures": "metadata", "auto_map": "metadata",
    "tokenizer_class": "metadata", "torch_dtype": "storage dtype", "dtype": "storage dtype",
    "attention_dropout": "dropout (eval: off)", "hidden_dropout": "dropout (eval: off)", "dropout": "dropout (eval: off)",
    "resid_pdrop": "dropout", "embd_pdrop": "dropout", "attn_pdrop": "dropout", "hidden_dropout_prob": "dropout",
    "attention_probs_dropout_prob": "dropout", "activation_dropout": "dropout", "ffn_dropout": "dropout",
    "classifier_dropout": "classification head", "summary_type": "classification head", "summary_use_proj":
    "classification head", "summary_activation": "classification head", "summary_proj_to_labels": "classification head",
    "summary_first_dropout": "classification head", "num_labels": "classification head", "problem_type": "head",
    "is_decoder": "flag", "is_encoder_decoder": "flag", "use_return_dict": "output flag",
    "max_position_embeddings": "context length (USE_CONTEXT = False: not priced; the RoPE scaling that reads it is a "
                               "rope_scaling key, reviewed separately)",
    "vocab_size": "vocabulary size (USE_VOCAB = False: not priced)",
    "_attn_implementation": "attention backend (same function: eager / sdpa agree)",
    "attn_implementation": "attention backend", "use_flash_attn": "attention backend", "use_flash_attention": "backend",
    "use_flash_attention_2": "backend", "_flash_attn_2_enabled": "backend", "use_sdpa": "backend",
    "fused_bias_fc": "kernel fusion", "fused_dropout_add_ln": "kernel fusion", "fused_mlp": "kernel fusion",
    "fuse_cross_entropy": "loss kernel", "fuse_norm": "kernel fusion (same function)", "fuse_swiglu": "kernel fusion",
    "use_kernels": "kernel choice", "use_fused_moe": "kernel choice", "moe_layer_freq_offset": None,
    "mlp_impl": None, "model_type": "type tag", "name_or_path": "metadata", "_name_or_path": "metadata",
    "tie_encoder_decoder": "encoder-decoder flag", "chunk_size_feed_forward": "chunked FFN: the same function",
    "use_mem_eff_path": "kernel path", "use_triton": "kernel choice", "use_bias_dropout_fusion": "kernel fusion",
    "flash_attn": "backend", "use_liger": "kernel choice", "mamba_chunk_size": "chunked scan: the same function",
    "chunk_size": "chunked scan: the same function", "use_mamba_kernels": "kernel choice",
    "router_jitter_noise": "training-time router noise (eval: off)", "jitter_noise": "training-time noise",
    "bos_token": "token", "eos_token": "token", "pad_token": "token", "image_token_id": "vision token id",
    "image_token_index": "vision token id", "video_token_id": "vision token id", "vision_start_token_id": "vision",
    "vision_end_token_id": "vision", "vision_token_id": "vision", "audio_token_id": "audio token id",
    "mask_token_id": None, "boi_token_index": "vision token", "eoi_token_index": "vision token",
}


def rule(h):
    role, member, kind = h["role"], h["member"], h["kind"]
    if role == "vision":
        return IRRELEVANT, None, ("rule:vision -- a non-text modality class (vision / audio tower, projector, "
                                  "tokenizer); the graph classifies the text decoder (model_graph._lm_unwrap)")
    if role in ("output", "cache"):
        return IRRELEVANT, None, f"rule:{role} -- a {role} container (no arithmetic on the hidden states)"
    if role == "other_head":
        return IRRELEVANT, None, ("rule:other_head -- a classification / QA / reward head; the graph compares "
                                  "the causal-LM next-token logits")
    if h["occ"] and not any(o.get("reached", True) for o in h["occ"]):
        return IRRELEVANT, None, ("rule:unreached -- no causal-LM entry class (the auto_map AutoModel* class, or the "
                                  "ForCausalLM / Model classes) references this class or function, directly or "
                                  "through the classes it builds (vision towers are not entered): it never runs in "
                                  "the text forward")
    if h["occ"] and all(BACKEND.search(o["cls"]) for o in h["occ"]):
        return IRRELEVANT, None, ("rule:backend -- a flash / sdpa / flex / npu attention-backend subclass; the "
                                  "reference function is the eager class (every loader here runs "
                                  "attn_implementation='eager'), whose hunks are reviewed")
    if h.get("placed_models", 1) == 0:
        return IRRELEVANT, None, ("rule:unplaced -- every model carrying this code is dropped by classify() "
                                  "(non-decoder, drafter / assistant, or no LM dims); the unplaced types are listed "
                                  "in SWEEP.md")
    if kind == "config_default":
        r = CONFIG_IRRELEVANT.get(member, "")
        if r:
            return IRRELEVANT, None, f"rule:config_key -- {member}: {r}"
        if all(o.get("configs_without_key", 1) == 0 for o in h["occ"]):
            return IRRELEVANT, None, ("rule:config_default_unused -- every snapshot config of the type sets "
                                      f"{member} explicitly, so the class default never runs")
        if re.search(r"dropout|_pdrop$|token_id$|token_index$|^use_cache$|loss|initializer|_std$", member):
            return IRRELEVANT, None, f"rule:config_key -- {member}: dropout / token id / loss / init key"
        return None
    if member in NON_FORWARD_MEMBERS and NON_FORWARD_MEMBERS[member]:
        return IRRELEVANT, None, f"rule:member -- {member}: {NON_FORWARD_MEMBERS[member]}"
    if MULTIMODAL_MEMBER.search(member):
        return IRRELEVANT, None, (f"rule:multimodal -- {member}: the image / video / audio input path, which a "
                                  "text-only forward does not run")
    if re.match(r"^_?(test|example|demo)", member):
        return IRRELEVANT, None, f"rule:member -- {member}: test / demo helper"
    if role == "pretrained" and member not in ("_keep_in_fp32_modules", "_keep_in_fp32_modules_strict",
                                               "forward", "__init__") and kind != "new_class":
        is_method = any(l.lstrip("+- ").startswith(("def ", "async def ", "@")) for l in h["text"].splitlines()[1:3])
        if not is_method:
            return IRRELEVANT, None, (f"rule:pretrained -- {member}: PreTrainedModel plumbing (loading, init, "
                                      "backend flags); the forward is in the model classes")
    lines = [l for l in h["text"].splitlines()[1:] if l[:1] in "+-"]
    if lines and all(plumbing_line(l) for l in lines):
        return IRRELEVANT, None, ("rule:plumbing -- every changed line moves only cache / output-flag / logging / "
                                  "return-container / kwargs-typing plumbing or a signature")
    return None


BACKEND = re.compile(r"(FlashAttention2?|FlashAttn|SdpaAttention|SDPAAttention|FlexAttention|NpuAttention|"
                     r"XformersAttention|TritonAttention|FA2Attention|AttentionFA2|AttentionSdpa|AttentionFlash)")


MULTIMODAL_MEMBER = re.compile(r"image|video|audio|vision|visual|pixel|ts_feature|speech|placeholder_mask|multimodal|"
                               r"^get_mm_|_mm_|deepstack|patch_embed|merge_input_ids_with")
