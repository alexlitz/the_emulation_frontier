# mpt_remote_alibi_default_false
live 1 / placed 5 (mpt 5); direction both; batch a5_positions

## 1. What the code computes
- Remote llm-foundry config `explosion-testing/mpt-test/configuration_mpt.py`:
  - `:4`: `attn_config_defaults` has `'alibi': False, 'alibi_bias_max': 8`.
  - `:10,73`: `learned_pos_emb` defaults to True.
  - `:90-97`: `_validate_config -> _set_config_defaults(attn_config, attn_config_defaults)` fills every missing attn_config key from those defaults.
- Model `explosion-testing/mpt-test/modeling_mpt.py`:
  - `:41`: `self.alibi = config.attn_config['alibi']`.
  - `:54-55`: `if not self.alibi: self.wpe = nn.Embedding(max_seq_len, d_model)`.
  - `:184`: `x = wte + wpe(pos)` when there is no alibi; the ALiBi bias is built only when alibi (`:91,169`).
  - So a remote MPT config whose attn_config omits `alibi` runs learned absolute positions and no ALiBi.
- The native `MptAttentionConfig` defaults alibi to True. round2-p5 `_normalize` (p5:model_graph.py:2560-2563) uses `ac.get("alibi", True)` even when auto_map routes to the remote code. explosion-testing/mpt-test (`attn_config {attn_impl: torch}`, auto_map remote, learned_pos_emb true) is therefore classed pos=alibi.
- The other 4 placed MPT carriers (replit-code-v1-3b, PhoGPT-4B / -Chat, anas-awadalla/mpt-7b) set alibi true explicitly, so the copy has no effect on them. Not fixed in p5.

## 2. Classifier change
- reads: model_type mpt, auto_map to modeling_mpt (remote), `attn_config` without an `alibi` key, `learned_pos_emb`.
- new: pos slot `learned_abs` (instead of alibi), in `_normalize`: `alibi = ac.get("alibi", False if remote else True)`. classify's LEARNED_ABS test needs `mt == "mpt" and d.get("learned_pos_emb") and not alibi`, since mpt is not in LEARNED_ABS.
- detection:
```python
def detect_mpt_remote_alibi_default_false(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "mpt" or "modeling_mpt" not in auto_map(cfg):
        return None
    ac = lm.get("attn_config")
    ac = ac if isinstance(ac, dict) else {}
    if "alibi" in ac:
        return None
    return "learned_abs" if lm.get("learned_pos_emb", True) is not False else "nope"
```
- apply note: in `_normalize`'s `mpt` branch, `remote = bool(g("auto_map"))` (the same test `_alibi_dims` uses) and `alibi=ac.get("alibi", not remote)`. In classify, before `elif mt in LEARNED_ABS`, add `elif mt == "mpt" and not d.get("alibi") and d.get("learned_pos_emb", True): pos = "learned_abs"`. A remote MPT with both off would hit the config's own ValueError (`configuration_mpt.py:117-118`), so it is not a case.

## 3. Relation rule
- target side: learned_abs. Hosts need a settable absolute table (learned_abs hosts with n_positions ≥ the context), or a RoPE host with the start-token construction (`learned_abs_two_block`, round2-p5's pos_table rule).
- host side: a learned_abs host plays NoPE (table 0) and learned_abs targets. It no longer claims ALiBi targets by slope nesting (remote MPT alibi_bias_max lever); `alibi_off` drops to 0 for this point.
- kind: exact (learned_abs <- learned_abs table copy; nope <- learned_abs `nope_sub_learned_abs`).
- cost: the existing learned_abs costs (pos_table length).
- free by the config rules: the attn_config.alibi flag is a feature switch (not free).
- theorem: `nope_sub_learned_abs`, `learned_abs_two_block`, `learned_abs_model` (PROOFS.md pos rows).

## 4. Test fixtures
- flag: explosion-testing/mpt-test -> learned_abs
- not flag:
  - replit/replit-code-v1-3b, vinai/PhoGPT-4B-Chat, vinai/PhoGPT-4B, anas-awadalla/mpt-7b (remote MPT with alibi true set explicitly)

## Verdict
Real, but one test point.
- explosion-testing/mpt-test moves from the ALiBi class to learned_abs: it loses ALiBi hosts and targets, and gains learned_abs ones.
- Count: about 0 real models; a test repo of the junk kind.
