# tie_default_unread
live 14 / placed 15 (<none> 3, cfrd 3, bitsy_parallel 1, cyrah 1, gemma3_moe 1, inferno 1, kanana2_tiny 1, mima 1,
nexus 1, phi3small 1, tinymixtral 1); direction over-credit; batch a2_dims

## 1. What the code computes
Remote config classes defaulting `tie_word_embeddings=True`, with configs that omit the key (so lm_head = E):
- MiniCPM (no model_type in the snapshot; `MiniCPMForCausalLM`, auto_map `configuration_minicpm.MiniCPMConfig`):
  `openbmb/MiniCPM4-0.5B/configuration_minicpm.py:128` `tie_word_embeddings=True`, `modeling_minicpm.py:1154`
  `_tied_weights_keys = ['lm_head.weight']`; same in BitCPM-CANN-0.5B (`:128`, `:1154`), BitCPM-CANN-0.5B-unquantized
  (`:128`, `:1255`), and `openbmb/MiniCPM-2B-sft-bf16/configuration_minicpm.py:134` (also no model_type, not a carrier).
- Phi-3-small: `microsoft/Phi-3-small-128k-instruct/configuration_phi3_small.py:30,234` subclasses PretrainedConfig
  without a tie argument -> `tf4:configuration_utils.py:220` default True; `modeling_phi3_small.py:900`
  `_tied_weights_keys = ["lm_head.weight"]` (8k: `:899`).
- `Lilgarca/my-model-v01/configuration_cyrah.py:281` True, `modeling_cyrah.py:662` tied keys.
- `ifx-pse-sys-ml/flame-27m-grounded/configuration_nexus.py:36` True, `modeling_nexus.py:287,294`.
- `kakaocorp/kanana-2-1.3b-instruct/configuration_kanana2_tiny.py:59` True, `modeling_kanana2_tiny.py:553`.
- `mikecovlee/tinymixtral-v1.1-1b/configuration_tinymixtral.py:27` True, `modeling_tinymixtral.py:242-244`.
- `zarko1321/Bitsy500M/configuration_bitsy.py:13` True, `modeling_bitsy.py:102`.
- gemma3_moe: `tardellirs/gemma3moe-16x4-270m-starter/modeling_gemma3_moe.py:28,33` `Gemma3MoeConfig(Gemma3TextConfig)`
  -> `tf4:models/gemma3/configuration_gemma3.py:186` default True.
Tied in code whatever the config says:
- `AlgoDriveAI/Inferno-Exp-1.0/modeling_inferno.py:111` `self.lm_head.weight = self.embed.weight` (config default
  True at `:20`).
- `root-sys-n/Mima-Pico-15m/modeling_mima.py:139` `self.lm_head.weight = self.wte.weight`.
- CFRD: `alice-noa-chan/haru_1.1/model.py:668-671` `_logits = F.linear(final_norm(x), token_embedding.weight)` (haru:
  `model.py:580`), although haru / haru_1.1 save `tie_word_embeddings: false`; the input is
  `E[t] + surface_gain * P(surface_table[t])` (`haru_1.1/model.py:655-664`).
- p5 (`p5:model_graph.py` `_tied` -> `_cfg_value` -> `_class_default`): none of these types is in transformers 4.57's
  CONFIG_MAPPING nor in `_CLASS_DEFAULT_FALLBACK["tie_word_embeddings"]`, and a config with no model_type has no
  class at all, so all 16 classify untied (haru_2 is tied by its key). Not fixed in p5.

## 2. Classifier change
- reads: model_type; architectures + auto_map when model_type is empty; the tie keys
- new: FEATURES key tied_emb (existing), via `_CLASS_DEFAULT_FALLBACK["tie_word_embeddings"]` entries and a new
  `TIED_BY_CODE` set checked first in `_tied`
- detection:
```python
TIE_DEFAULT_TRUE = frozenset({"minicpm", "inferno", "cyrah", "nexus", "kanana2_tiny", "phi3small", "tinymixtral",
                              "bitsy_parallel", "gemma3_moe"})
TIED_BY_CODE = frozenset({"cfrd", "mima", "inferno"})

def _tie_type(cfg, model_type):
    if model_type:
        return model_type
    if any(a == "MiniCPMForCausalLM" for a in archs(cfg)) and "configuration_minicpm" in auto_map(cfg):
        return "minicpm"
    return ""

def detect_tie_default_unread(cfg, model_type):
    lm = lm_of(cfg)
    mt = _tie_type(cfg, model_type)
    if mt in TIED_BY_CODE:
        return True
    if mt in TIE_DEFAULT_TRUE and not any(k in lm or k in (cfg or {})
                                          for k in ("tie_word_embeddings", "tie_embedding", "weight_tying")):
        if mt == "minicpm" and not any(a == "MiniCPMForCausalLM" for a in archs(cfg)):
            return None
        return True
    return None
```
- apply note: in `_CLASS_DEFAULT_FALLBACK["tie_word_embeddings"]` add True for minicpm, cyrah, nexus, kanana2_tiny,
  phi3small, tinymixtral, bitsy_parallel, gemma3_moe; in `_tied` return True first when the type is in a new
  `TIED_BY_CODE = {"cfrd", "mima", "inferno"}`; in `_class_default` (or `_tied`) map an empty model_type with
  architectures `MiniCPMForCausalLM` to "minicpm" (a `_ARCH_TYPE_FALLBACK = {"MiniCPMForCausalLM": "minicpm"}`).

## 3. Relation rule
- target side: a tied target is played by any host (`tied_emb` is free on the target side): more in-edges
  (tied hosts now qualify).
- host side: a tied host cannot play an untied target (wall): fewer out-edges.
- kind: exact (target side); wall (host side)
- cost: none
- free by the config rules: tie_word_embeddings is shape-changing (never free); CFRD / Mima / Inferno tie in code.
  CFRD's input surface term is an extra per-token input table the untied host absorbs into its input embedding.
- theorem: `tied_sub_untied` (Every allowed rule: tied target <- untied host); the host-side wall is OPEN
  (Walls of the edge-case audit: tied host -> untied target).

## 4. Test fixtures
- flag (every live repo + the same-class siblings): AlgoDriveAI/Inferno-Exp-1.0, Lilgarca/my-model-v01,
  alice-noa-chan/haru, alice-noa-chan/haru_1.1, ifx-pse-sys-ml/flame-27m-grounded, kakaocorp/kanana-2-1.3b-instruct,
  microsoft/Phi-3-small-128k-instruct, microsoft/Phi-3-small-8k-instruct, mikecovlee/tinymixtral-v1.1-1b,
  openbmb/BitCPM-CANN-0.5B, openbmb/BitCPM-CANN-0.5B-unquantized, openbmb/MiniCPM4-0.5B, openbmb/MiniCPM-2B-sft-bf16,
  root-sys-n/Mima-Pico-15m, tardellirs/gemma3moe-16x4-270m-starter, zarko1321/Bitsy500M -> True
- not flag: openbmb/MiniCPM4-8B (minicpm, explicit false), BAAI/bge-reranker-v2-minicpm-layerwise (minicpm type, a
  layerwise reranker class), microsoft/Phi-3-mini-4k-instruct (phi3, 4.57 class), google/gemma-3-270m (gemma3_text,
  4.57 class default already read), Qwen/Qwen3-30B-A3B
- scan: 18 repos (incl. haru_2, already tied by its key, and MiniCPM-1B-sft-bf16, same class as 2B).

## Verdict
real. Mostly fewer edges: ~18 models move from untied to tied hosts and lose their out-edges to untied targets
(small models, many potential targets each: plausibly hundreds of edges); they gain in-edges from tied hosts.
