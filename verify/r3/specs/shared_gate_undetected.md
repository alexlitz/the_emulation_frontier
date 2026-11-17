# shared_gate_undetected
live 2 / placed 2 (alice_ai 1, memt_qwen3_5_moe 1); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
- AliceAI: `AliceAISparseMoEBlock` builds `shared_expert = AliceAIMLP(hidden, shared_expert_intermediate_size)` and `shared_expert_gate = Linear(hidden, 1, bias=False)`. forward: `routed + shared(x) * sigmoid(shared_expert_gate(x))` (AMAImedia/AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF/modeling_alice_ai.py:558-579).
- MemT-1.0-Nano: `MemTDecoderLayer(qwen.Qwen3_5MoeDecoderLayer)` with `self.mlp = qwen.Qwen3_5MoeSparseMoeBlock(config)` (MemTensor/MemT-1.0-Nano/modeling_memt.py:270-279). Qwen3_5MoeSparseMoeBlock computes `sigmoid(shared_expert_gate(x)) * shared_expert(x)` (tf5:models/qwen3_5_moe/modeling_qwen3_5_moe.py:~909-918).
- Snapshot: both have shared_expert_intermediate_size 512 (AliceAI 512 x 512 experts, MemT 256 experts).
- p5 `SHARED_GATE_TYPES` (p5:model_graph.py:1943) lacks `alice_ai` and `memt_qwen3_5_moe`, so `_feats` (~1791) does not add `shared_gate`. Both classify as moe_shared with an ungated shared expert.

## 2. Classifier change
- reads: model_type + `shared_expert_intermediate_size`.
- new: table entries `alice_ai`, `memt_qwen3_5_moe` in `SHARED_GATE_TYPES`.
- detection:
```python
SHARED_GATE_CODE = {"qwen2_moe", "qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_omni_moe",
                    "qwen4_exp", "qwen4_exp_text", "alice_ai", "memt_qwen3_5_moe"}

def detect_shared_gate_undetected(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in SHARED_GATE_CODE and (num(lm.get("shared_expert_intermediate_size")) > 0
                                           or model_type == "qwen3_next"):
        return True
    return None
```
- apply note: p5:model_graph.py `SHARED_GATE_TYPES` (~1943): add `"alice_ai", "memt_qwen3_5_moe"`. Nothing else: the `_feats` test and the FEATURES rule `shared_gate` ("wall", "free") then apply.

## 3. Relation rule
- target side: sigmoid(w . x) * shared(x) is a data-dependent product no ungated shared expert forms (FEATURES shared_gate, target wall). A gated host is needed.
- host side: a gated host plays an ungated target exactly: w = 0 gives gate 1/2, and the shared down-projection doubles.
- kind: target wall (OPEN in PROOFS, "out_gate / shared_gate target <- ungated host"); host exact.
- cost: none.
- free by the config rules: the gate is code-fixed, so it is not a config switch.
- theorem: host `moe_shared_gate_off`, `shared_gate_half`, `ffn_smul` (PROVED exact). Target wall: OPEN (PROOFS "Walls of the edge-case audit").

## 4. Test fixtures
- flag: AMAImedia/AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF, MemTensor/MemT-1.0-Nano (the live repos); Qwen/Qwen1.5-MoE-A2.7B, Qwen/Qwen3-Next-80B-A3B-Instruct, nvidia/Qwen3.6-35B-A3B-NVFP4 (the same code fact, already in the table) -> true.
- not flag: MemTensor/MemT-1.0-Flash (memt_deepseek_v4: ungated shared expert), Qwen/Qwen3-30B-A3B (qwen3_moe, no shared expert), moonshotai/Kimi-Linear-48B-A3B-Instruct, HIT-SCIR/Chinese-Mixtral-8x7B.
- scan: alice_ai 1, memt_qwen3_5_moe 1, plus the existing qwen2_moe 6, qwen3_5_moe 2, qwen3_5_moe_text 57, qwen3_next 12, qwen4_exp_text 24 (already carried).

## Verdict
real, small. Fewer edges for the 2 points as targets (ungated moe_shared hosts drop). AliceAI is masked today by its attn_res wall (block_attn_res_block_size 4), so only MemT-1.0-Nano actually changes: it loses ungated hosts, keeping the gated Qwen3.5 / Qwen-Next family (GDN hosts only, rf_gdn). That is roughly a handful of edges.
