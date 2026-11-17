# parallel_by_code_unlisted
live 1 / placed 1 (bitsy_parallel 1); direction both; batch a1_typetables

## 1. What the code computes
- zarko1321/Bitsy500M/modeling_bitsy.py:81 BitsyBlock: :84-85 two RMSNorms `attn_norm`, `mlp_norm` (weighted, rms_norm_eps); :92 `a = attn(attn_norm(x))`; :93 `z = mlp_norm(x)`; :95 `return x + a + down(silu(gate(z)) * up(z))`. Attention and MLP both read the block INPUT x (separate norms, as GPT-NeoX use_parallel_residual): a parallel block, fixed in code.
- config.json:17 `model_type: "bitsy_parallel"`; no use_parallel_residual / parallel_attn key.
- p5 classify (p5:model_graph.py:2311-2316) sets block=parallel from `use_parallel_residual` / `parallel_attn` or `mt in PARALLEL_BY_CODE` (:501 = gptj, codegen, progen, phi, bunny-phi, cohere, cohere2): Bitsy500M gets block=sequential.
- (same model, other batch items: per-head q / k norm before an interleaved RoPE -> qknorm_by_code_unlisted "head".)

## 2. Classifier change
- reads: model_type
- new: block slot "parallel"; table entry in PARALLEL_BY_CODE
- detection:
```python
def detect_parallel_by_code_unlisted(cfg, model_type):
    return "parallel" if model_type == "bitsy_parallel" else None
```
- apply note: add "bitsy_parallel" to `PARALLEL_BY_CODE` (p5:model_graph.py:501). Two separate norms (not Phi / GPT-J's one shared input norm): the parallel block value covers both (a shared-norm target is the two-norm block with tied gains).

## 3. Relation rule
- target side: Bitsy500M as a parallel target: exact on parallel hosts (GPT-NeoX-style two-norm hosts directly; a one-norm host GPT-J / Phi cannot untie the two gains -- the relation does not split this today, so note it); on sequential hosts only by the 2x-width carry, which fails under a pre-norm, or the pass-5 RMS compensator for measured families (`rms_ok`, PRENORM_RMS_RANGE): bitsy_parallel is not measured, so walled
- host side: Bitsy500M as a host plays sequential targets at 2x depth (attention-only then MLP-only blocks, exact) and parallel targets directly
- kind: exact (parallel <- parallel; seq target <- parallel host at 2x depth); wall (parallel target <- sequential host, unmeasured family)
- cost: depth x2 for sequential targets on it (emul_matrix depth2 `par_emu_seq`)
- free by the config rules: no (a block layout is a feature switch, not a config value)
- theorem: `seq_sub_parallel`, `seq_stack_sub_parallel`, `zeroed_sublayer` (PROOFS.md "block: seq -> parallel at 2x depth"); `parallel_sub_seq_2x` with `parallel_sub_seq_prenorm_fails` (PROOFS.md "block: parallel -> seq at 2x width": REFUTED under a pre-norm); pass-5 compensator `vanishing_carry_limit` for measured families only

## 4. Test fixtures
- flag: zarko1321/Bitsy500M -> "parallel"
- not flag: EleutherAI/gpt-j-6b (gptj: already in PARALLEL_BY_CODE); Qwen/Qwen3-0.6B (sequential)

## Verdict
real, one model, both directions: as a target it loses its sequential hosts (Llama / Qwen-sized sequential hosts it was credited with exactly) and gains the parallel ones (Pythia / GPT-J / Phi-sized, subject to its other slots: its new qknorm=head will wall most of them); as a host it keeps sequential targets but at 2x depth (fewer fit). Net fewer edges, tens.
