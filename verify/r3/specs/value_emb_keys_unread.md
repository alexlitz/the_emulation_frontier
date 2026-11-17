# value_emb_keys_unread
live 5 / placed 5 (arloop 2, arloop_baseline 1, klm 1, sn38-nanochrono 1); direction over-credit; batch a12_extra_inputs

## 1. What the code computes
- ARLoop (osunlp/ARLoop-1.2B/modeling_arloop.py, ARLoop-0.6B byte-identical): `ve_gate = Linear(32, n_kv)` on layers with `layer % 2 == (physical_layers - 1) % 2` (:37-38); `v += 2*sigmoid(ve_gate(x[..., :32])) * value_embedding` (:46-48); `shared_value_embedding = Embedding(vocab, n_kv*head_dim)` (:110-111), `ve = shared_value_embedding(input_ids)` passed to every layer (:188, :199).
- PreNormLoop (osunlp/PreNormLoop-0.6B/modeling_baseline.py): the same ve_gate / add (:37-48), shared table (:110), `ve` fed to every block of a plain pre-norm residual (:163-170).
- sn38-nanochrono (ext:sn38-nanochrono/github__chronollm__sn38/modeling_nanochrono.py): `g_proj = Linear(aux_gate_dim, n_kv)` on `aux_layers` (:36-39); `v += 3*sigmoid(g_proj(x[..., :gate_dim])) * aux` (:46-48); per-layer tables `aux_embeds[i] = Embedding(vocab, kv_dim)` (:124-125, :189). `aux_layers` None -> every other layer (configuration_nanochrono.py:46-47); snapshot 14 odd layers, aux_gate_dim 12. Also smear (`x += mix_scale*sigmoid(mix_gate(x[:24])) * x_prev`, :169-178) and x0 lambdas (`x = r_i x + s_i h0`, :187): nanochat's other value_emb components.
- KLM-1.5 (MinimaLabs/KLM-1.5/modeling_klm.py): `value_mix = Parameter([0.5, 0.5])` (:150); `values = m0*v + m1*v_first`, v_first = the first attention layer's V (:157-160, :340-345): a value residual (no table).
- p5 `_feats` value_emb keys: `value_embedding_layers`, `ve_gate_channels`, `use_x0_lambdas`, `use_smear`, `value_residual` (p5:model_graph.py:1765-1767): none present. p5 feats: ARLoop / PreNormLoop `tied_emb`, sn38 `''`, KLM `tied_emb`.

## 2. Classifier change
- reads: model_type; sn38 `aux_layers`
- new: FEATURES `value_emb` (existing key) for these types via a code table; optional dims `value_emb_kind` in {ve_shared, ve_layer, v_residual}, in `_feats` / `_pass5_dims`
- detection:
```python
VALUE_EMB_BY_CODE = {"arloop": "ve_shared", "arloop_baseline": "ve_shared", "klm": "v_residual"}

def detect_value_emb_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "sn38-nanochrono":
        aux = lm.get("aux_layers")
        return "ve_layer" if aux is None or (isinstance(aux, list) and aux) else None
    return VALUE_EMB_BY_CODE.get(model_type)
```
- apply note: add `VALUE_EMB_BY_CODE` near VNORM_TYPES (p5:model_graph.py:1941); extend the value_emb condition in `_feats` (:1765) with `or mt in VALUE_EMB_BY_CODE or (mt == "sn38-nanochrono" and d.get("aux_layers", True))`. If the kinds are recorded, the edge rule on value_emb pairs is: equal kind, or ve_layer host <- ve_shared target (copy the shared table per layer); gate scale 2 vs 3 folds into the table; host gate channels >= target's (zero the rest); host VE layers a superset of the target's after layer alignment. v_residual and ve_* do not nest.

## 3. Relation rule
- target side: a host with a value embedding / value residual of a nesting kind; a plain host is a wall (an extra per-token input into V).
- host side: free -- ve: zero the table (gate * 0 = 0); v_residual: value_mix = (1, 0).
- kind: target wall; host exact.
- cost: none.
- free by the config rules: none (gate channels / layer set are shape / switch facts).
- theorem: host `value_emb_off` (PROOFS.md "value_emb host -> plain"); target OPEN ("switch_lora / engram / ple / value_emb target <- plain host").

## 4. Test fixtures
- flag: osunlp/ARLoop-0.6B, osunlp/ARLoop-1.2B, osunlp/PreNormLoop-0.6B -> "ve_shared"; kingbear329/sn38-r16-c12-step687-alpha05 -> "ve_layer"; MinimaLabs/KLM-1.5 -> "v_residual".
- not flag: jkminder/d16_optimized_537m_seed5 (nanochat_gpt: already by keys), paradigma-inc/limite-1b-violetto (limite: `ve_gate_channels` key path), castorini/gaggle-nanochat-pretrained-climbmix-20260924 (gaggle: keys), AxiomicLabs/GPT-S3-3M (AttnRes, no VE), Qwen/Qwen3-0.6B.
- scan: exactly the 5 live repos.

## Verdict
real. Fewer edges: 5 targets lose every host without value_emb (ARLoop / PreNormLoop are looped, so few hosts reached them anyway; sn38 and KLM lose most of theirs). Host side unchanged. Note nanochat_port_unlisted (a1) also re-labels sn38-nanochrono's FFN / qk-norm; independent.
