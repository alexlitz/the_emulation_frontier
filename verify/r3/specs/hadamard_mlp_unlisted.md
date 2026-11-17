# hadamard_mlp_unlisted
live 2 / placed 2 (needle 2); direction both; batch a12_extra_inputs

## 1. What the code computes
(ext:needle/gh__cactus-compute__cactus/architecture.py, config.py)
- Every block's FFN is `HadamardMLP(d_model, split_perms=hada_split)` (architecture.py:368-369); there is no other FFN class and no config switch (the snapshot's `"mlp": "hadamard"` is descriptive).
- HadamardMLP (:298-330): `n = next_pow2(d_model)` (needle2 512, needle3 768 -> 1024); factors `_hada_blocks(n)` = (2^(floor(log2 n)/2), n / that) -> 16 x 32 (512) / 32 x 32 (1024) (config.py:328-330); three learned Kronecker-factored transforms `_kron_apply(z, A_s, B_s)` = (A_s kron B_s) z (:286-290), Walsh-initialised (:264-269, :293-295); fixed seeded permutations P1, P2 (RandomState 11, 13; per-half if split) (:272-283); diagonals d1..d4, bias b2; conditioner `cond = 1 + softmax(x V) U`, rank 8 (:271, :323).
- Forward: `z = pad(x, n)`; `z = P1 (A1 kron B1)(d1 z)`; `z = P2 (A2 kron B2) silu(d2 * cond * z + b2)`; `z = (A3 kron B3)(d3 z)`; return `(d4 z)[:d_model]` (:305, :323-329).
- So: one ungated SiLU at width n (not 2-4 x d), a data-dependent per-channel input scale (cond), an inner bias, structured (not dense) n x n maps. No gate / up / down matrices.
- p5: ffn `glu`, act `SiLU`, inter 4 x hidden fallback (2048 / 3072) (classify :2276-2282).

## 2. Classifier change
- reads: model_type needle (code-fixed)
- new: ffn slot value `hadamard` (no FFN_ORDER edges), in `classify`; dims `inter` = next_pow2(hidden), in `dims_of`
- detection:
```python
def detect_hadamard_mlp_unlisted(cfg, model_type):
    return "hadamard" if model_type == "needle" else None
```
- apply note: in `classify` FFN block (p5:model_graph.py:2275) add `elif mt in HADAMARD_FFN_TYPES: ffn = "hadamard"` (HADAMARD_FFN_TYPES = {"needle"}) before the GATED / default branches; leave `FFN_ORDER` (:93) without a "hadamard" key (isolated: only hadamard <- hadamard by equality); `FFN_RANK["hadamard"] = 1` (:3432), COLOR entry; in `dims_of` set `inter = 1 << (hidden - 1).bit_length()` for these types (:3284). Equal `inter` (n) is required for the exact hadamard <- hadamard edge (n fixes the Kronecker blocks and permutations); the `_ffn_widths_fit` multiple rule must not apply to hadamard pairs. act stays SiLU.

## 3. Relation rule
- target side: only a hadamard host with equal n and split_perms: a dense GLU / vanilla FFN has no softmax conditioner multiplying the pre-activation and no inner bias (and a hadamard host of another n has other block shapes / permutations).
- host side: a hadamard host cannot play a dense FFN target: its maps are Kronecker-structured rank-limited products, not arbitrary d x inter matrices.
- kind: wall both ways; exact hadamard <- hadamard at equal n (same code).
- cost: none.
- free by the config rules: none (FFN shape / activation path).
- theorem: needs theorem (both OPEN): "a GLU / vanilla FFN without an input-dependent pre-activation scale cannot equal silu(cond(x) * Mx + b)", and "a Kronecker-structured n x n map does not span dense d x inter maps".

## 4. Test fixtures
- flag: Cactus-Compute/needle2, Cactus-Compute/needle3 -> "hadamard".
- not flag: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit (qwen3_5_text `hadamard_config`: a quantisation rotation, dense FFN), Thox-ai/ThoxMicroprocessor-8M-Base-TinyStories (`use_hadamard_mixer`: token mixer, not this FFN; unread), baidu/ERNIE-5.5-Flash-Base (`use_fast_hadamard`), meta-llama/Llama-3.1-8B-Instruct.
- scan: exactly the 2 needle repos.

## Verdict
real. Fewer edges both ways: needle2 / needle3 lose every glu edge as target and as host (2 small models, so most of the change is host side: small glu targets they were credited with); needle2 <-> needle3 stay walled (n 512 vs 1024).
