# Related work

Prior art per axis, from a literature sweep. The individual edges below are
scattered across papers; the contribution here is (a) collecting them under one
**emulation-cost currency** vocabulary, (b) a size/capacity-aware graph over the
real model zoo, and (c) two edges I could not find published.

## Attention

- **GQA ⊆ MLA** — *TransMLA: Multi-Head Latent Attention Is All You Need*
  (Meng, Yao, Zhang; NeurIPS 2025 Spotlight). https://arxiv.org/abs/2502.07864
  Proves GQA is a special case of MLA (Theorem 1: MLA is strictly more expressive
  at equal KV cache) via the same SVD/low-rank construction used in `attn/mla.py`.
  This *is* our MLA edge, already published. Adopted at scale (Ling-2.5-1T).
- **MQA / GQA** — Shazeer 2019 (MQA), Ainslie et al. 2023 (GQA). The tie-weights
  restriction is folklore; the *wide-shared-KV* reverse construction in
  `attn/mha_mqa_gqa.py` (MQA→MHA, exact, KV-memory cost) I did not find written up.
- **softmax vs linear/kernel** — Performer/FAVOR+ (Choromanski et al. 2021),
  Linear Transformers (Katharopoulos et al. 2020). The infinite-feature wall and
  the ~2-layer depth for induction/associative-recall are established; `attn/
  softmax_linear.py` reproduces the width and magnitude currencies.

## Normalization

- **LayerNorm → RMSNorm by folding centering** — *Enjoy Your Layer Normalization
  with the Computational Efficiency of RMSNorm* (2026), https://arxiv.org/html/2605.14521 ;
  and *FlashNorm* (Graef 2024), https://arxiv.org/html/2407.09577v4 . Both prove the
  mean-centering folds into an adjacent linear layer **exactly** (column-centered
  weight constraint) — our "fold" currency, confirmed exact for train + inference,
  with graph-based detection of "foldable" LNs. This is essentially `norm/`.
- **RMSNorm** — Zhang & Sennrich 2019. https://arxiv.org/abs/1910.07467

## FFN / MoE

- **dense ⊆ MoE** (route-to-one-expert) — folklore; e.g. Grootendorst's
  "A Visual Guide to MoE". The capacity condition (`k·expert_width ≥ dense_width`)
  and the E×/k× reverse cost are standard.
- **GLU variants** — Shazeer 2020 ("GLU Variants Improve Transformer"). The
  bilinear-GLU ⊆ squared-activation-FFN and SwiGLU ⊆ 2-layer identities in
  `act/relu_swiglu.py` I did not find stated as emulation edges.

## Positional

- **ALiBi vs RoPE** — the literature groups them only weakly: same *shift-invariant
  equivalence class* (measure-theoretic analysis, https://arxiv.org/pdf/2605.19944),
  or both as *limiting cases of a wavelet parent* (MoPE, https://arxiv.org/pdf/2606.01258).
  "Round and Round We Go" (https://arxiv.org/pdf/2410.06205) argues they behave
  *differently*. The exact **ALiBi = small-angle limit of a fixed RoPE pair** (with
  the additive-bias/magnitude trade) — the seed result — appears unpublished.

## Framework-level (closest in spirit, different object)

- **Hasse Diagrams for Attention** (2026), https://arxiv.org/pdf/2606.09951 — a
  partial-order/Hasse framework, but over attention *masks* / token reachability,
  not architecture components.
- **Kernel-based** unifications of attention variants; **universal-approximation**
  frameworks — narrower (attention-only or expressivity-only), no cost accounting.
- Catalog blogs (lists, not graphs): Lilian Weng, *The Transformer Family v2*
  https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/ ;
  kipply, *Transformer Taxonomy* https://kipp.ly/transformer-taxonomy/ .

## Novel here (not found in the sweep)

1. The unified **currency** accounting (magnitude / KV-memory / width / depth /
   fold / capacity) applied across *all* components at once.
2. The **ALiBi→RoPE** exact small-angle construction.
3. The **activation graph** as emulation edges (ReLU as β→∞ limit of Swish/
   Softplus/GeLU with β folded into the linear layer; Maxout as the PWL supremum).
4. The **capacity-aware graph over real checkpoints** + the minimal emulator
   frontier (which real models emulate all others, given component sizes).
