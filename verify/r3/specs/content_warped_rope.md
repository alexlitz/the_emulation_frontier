# content_warped_rope
live 1 / placed 1 (gptx2 1); direction over-credit; batch a5_positions

## 1. What the code computes
All paths are under `reaperdoesntknow/GPT-X2-125M-CIx-Long-Context/` (snapshot 9daf85cad2f7).
- `configuration_gptx2.py:58,93`: `use_symplectic_rope` defaults to False. The snapshot sets it true.
- `modeling_gptx2.py:706-744` (`compute_metric_warped_freqs_cis`):
  - `base_phase = t·inv_freq` and `radial = 1/(1 + t/radial_base)`.
  - `phase = base_phase·(1 + mass_delta·radial) + drag`, with drag = spin_delta·(t/(T−1))².
  - The rotation of position t depends on the sequence length T and on content-driven deltas.
- `:96` (`masked_sequence_mean`) and `:869` (`clock_feature`): the deltas come from a SymplecticClock (`:754`, a leapfrog integrator over a softplus-diagonal Hamiltonian). It is driven by the masked mean of the hidden states over the WHOLE sequence (non-causal: position t's encoding reads later tokens), pooled over the batch, and its state is carried across forward calls.
- Each delta is `scale·tanh(W z)` (project_metric_delta, hunk 4c42f9763d09).
- `:1065-1067,1100,1198-1261`: blocks and the LM build the clocks and warped tables only when `use_symplectic_rope`; otherwise the plain table.
- round2-p5: nothing reads use_symplectic_rope. classify gives pos=rope, feats tied_emb, rope_sched (64,) (checked). Not fixed in p5.

## 2. Classifier change
- reads: model_type gptx2 with remote modeling_gptx2 and `use_symplectic_rope` true.
- new: FEATURES `rope_warp` ("wall", "free"), in `_feats`.
- detection:
```python
def detect_content_warped_rope(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "gptx2" and lm.get("use_symplectic_rope") is True and "modeling_gptx2" in auto_map(cfg):
        return "rope_warp"
    return None
```
- apply note: in `_feats`, `if mt == "gptx2" and d.get("use_symplectic_rope") is True: f.add("rope_warp")`, and add the FEATURES entry beside nope_layers.

## 3. Relation rule
- target side: wall. The phase depends on the content of the whole sequence (non-causal) and on persistent state. No causal fixed-table host, and no causal host at all, reproduces it (`causal_ne_bidirectional`-type: position t reads t' > t).
- host side: free. Zeroing the metric projections (W = 0: weights) makes mass_delta = spin_delta = 0, so phase = base_phase, the plain table, whatever the clock state.
- kind: wall (target) / exact (host).
- cost: none.
- free by the config rules: use_symplectic_rope is a feature switch (not free). The zeroed projections are weights.
- theorem: host side trivial (`scalar_fold`: delta 0). Target side: `causal_ne_bidirectional` covers the non-causal read; no RoPE-specific theorem is needed.

## 4. Test fixtures
- flag: reaperdoesntknow/GPT-X2-125M-CIx-Long-Context -> rope_warp
- not flag:
  - AxiomicLabs/GPT-X2.5-135M (gptx2 type, different remote code, no symplectic key)
  - AxiomicLabs/GPT-X3-150M (gptx3)

## Verdict
Real (over-credit), 1 point (a 125M experimental model).
- As a target it loses every host: fewer edges, all its current incoming edges.
- As a host it is unchanged.
