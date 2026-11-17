# switch_cache_layers_counted
live 4 / placed 4 (granite_switch 4); direction both; batch a2_dims

## 1. What the code computes
- `ext:granite_switch/gh__generative-computing__granite-switch/config.py:25-31` `SWITCH_CACHE_LAYERS = 2`: "the
  composer inflates num_hidden_layers by this count and the models subtract it to recover the physical decoder-layer
  count" (MultiSwitch: a counting slot and a memory slot); `config.py:482` `num_decoder_layers = num_hidden_layers -
  SWITCH_CACHE_LAYERS`; `config.py:229-238` layer_types default `["full_attention"] * num_hidden_layers` (inflated
  length).
- `.../modeling_granite_switch.py:439-444` `if config.num_adapters > 0: layer_offset = self.switch.num_cache_layers;
  num_decoder_layers = config.num_hidden_layers - layer_offset` (`multi.py:332` num_cache_layers = 2); vLLM backend
  `.../granite_switch_model.py:232-243` the same.
- Snapshot: num_adapters 2 (4.1-8b, 4.1-30b) / 12 (the previews), num_hidden_layers 42 / 66, layer_types 42 / 66
  entries, all full_attention -> 40 / 64 decoder layers.
- p5 (`p5:model_graph.py` dims_of): `n_layers = g("num_hidden_layers", ...)` = 42 / 66 -> layers, param_layers,
  eff_layers, ffn_units (eff_layers x ffn_total), and `_attn_counts` over the 42-entry layer_types (kFU_* counts).
  Not fixed in p5.

## 2. Classifier change
- reads: model_type granite_switch, num_adapters, num_hidden_layers, layer_types
- new: dims layers / param_layers / eff_layers / ffn_units / attn_* / k*_eff counts = num_hidden_layers - 2, in dims_of
  (via `_normalize`)
- detection:
```python
def detect_switch_cache_layers_counted(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "granite_switch" or num(lm.get("num_adapters")) <= 0:
        return None
    n = num(lm.get("num_hidden_layers"))
    return n - 2 if n > 2 else None
```
- apply note: in `_normalize` add `elif mt == "granite_switch" and _num(g("num_adapters")) > 0:` set
  `u["num_hidden_layers"] = n - 2` and, when layer_types is a list of length n, `u["layer_types"] = lt[2:]` (all
  full_attention here, so which two are dropped does not matter); dims_of and `_attn_counts` then read 40 / 64.

## 3. Relation rule
- target side: a Granite-Switch target needs 40 / 64 host layers, not 42 / 66 (more hosts; but switch_lora walls it
  from every non-switch host, so only switch hosts count).
- host side: a switch host offers 40 / 64 layers: it no longer plays 41-42 / 65-66-layer targets (fewer edges).
- kind: exact (depth capacity)
- cost: none
- free by the config rules: not free (layer count is shape)
- theorem: `zeroed_sublayer` (depth: host deeper)

## 4. Test fixtures
- flag: ibm-granite/granite-switch-4.1-30b -> 64; granite-switch-4.1-3b-preview, -4.1-8b, -4.1-8b-preview -> 40
- not flag: ibm-granite/granite-4.0-h-tiny (granitemoehybrid, no switch), ibm-granite/granite-4.0-micro
- scan: exactly the 4.

## Verdict
real, small: 4 hosts lose targets with 41-42 (65-66) layers within their width, and the switch-to-switch target
edges shift. Few edges (the 8b / 3b-preview / 8b-preview share 40 layers already; the 30b changes most).
