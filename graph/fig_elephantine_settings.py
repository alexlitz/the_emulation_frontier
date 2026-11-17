"""Blog figure: one elephantine layer and four settings of it (the hf.co/alexlitz/elephantine demo).

  python -m graph.fig_elephantine_settings      # -> graph/fig_elephantine_settings.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
ON, OFFF, GRID = "#d6e6f8", "#f0efec", "#e4e3df"
models = ["SmolLM2-135M", "Pythia-160M", "Falcon-H1-Tiny-90M", "Qwen3-0.6B diffusion"]
kinds = ["Llama-style transformer", "GPT-NeoX transformer", "attention + Mamba-2 hybrid", "masked-diffusion LM"]
OFF = None
rows = [  # (group, branch, values per model)
    ("block", "norm", ["RMSNorm", "LayerNorm (centre on)", "RMSNorm", "RMSNorm"]),
    ("block", "residual layout", ["sequential", "parallel: attn ∥ FFN", "parallel: attn ∥ SSM", "sequential"]),
    ("attention", "positions", ["RoPE", "partial RoPE (25%)", "RoPE", "RoPE"]),
    ("attention", "heads", ["GQA, KV tiled", "MHA", "GQA, KV tiled", "GQA, KV tiled"]),
    ("attention", "qk-norm", [OFF, OFF, OFF, "on"]),
    ("attention", "mask", ["causal", "causal", "causal", "full (bidirectional)"]),
    ("SSM", "gated-delta recurrence", [OFF, OFF, "Mamba-2 + short conv", OFF]),
    ("FFN", "activation", ["SwiGLU", "GeLU (ungated)", "SwiGLU", "SwiGLU"]),
    ("FFN", "biases", [OFF, "on", OFF, OFF]),
    ("FFN", "MoE routing", [OFF, OFF, OFF, OFF]),
]
nr, nc = len(rows), len(models)
cw, rh, lw, gw = 2.35, 0.46, 2.15, 1.0
W, H = gw + lw + nc * cw + 0.3, (nr + 3.2) * rh + 1.9
fig = plt.figure(figsize=(W, H), dpi=200, facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
top = H - 0.35
ax.text(0.15, top, "One elephantine layer, four settings", fontsize=15, color=INK, weight="bold", va="top")
ax.text(0.15, top - 0.42,
        "x ← x + Attn(norm(x))   ·   x ← x + SSM(norm(x))   ·   x ← x + FFN(norm(x))      "
        "the same code in every column; only weights and branch coefficients differ",
        fontsize=9.5, color=INK2, va="top")
y0 = top - 1.05
x0 = gw + lw
for j, (m, k) in enumerate(zip(models, kinds)):
    cx = x0 + j * cw + cw / 2
    ax.text(cx, y0, m, ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold")
    ax.text(cx, y0 - 0.02, k, ha="center", va="top", fontsize=8.5, color=INK2)
yy = y0 - 0.42
prev = None
for i, (g, b, vals) in enumerate(rows):
    y = yy - (i + 1) * rh
    if g != prev:
        ax.plot([0.15, W - 0.15], [y + rh, y + rh], color=GRID, lw=1)
        ax.text(0.2, y + rh / 2, g, fontsize=9.5, color=INK, weight="bold", va="center")
        prev = g
    ax.text(gw, y + rh / 2, b, fontsize=9.5, color=INK2, va="center")
    for j, v in enumerate(vals):
        x = x0 + j * cw
        on = v is not None
        ax.add_patch(FancyBboxPatch((x + 0.06, y + 0.05), cw - 0.12, rh - 0.10,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=ON if on else OFFF, ec="none"))
        ax.text(x + cw / 2, y + rh / 2, v if on else "off", ha="center", va="center",
                fontsize=9, color=INK if on else MUTED)
yb = yy - (nr + 1) * rh - 0.15
ax.add_patch(FancyBboxPatch((gw, yb - 0.11), 0.32, 0.22, boxstyle="round,pad=0,rounding_size=0.05", fc=ON, ec="none"))
ax.text(gw + 0.42, yb, "branch switched on / option set by the loader", fontsize=8.5, color=INK2, va="center")
ax.add_patch(FancyBboxPatch((gw + 3.6, yb - 0.11), 0.32, 0.22, boxstyle="round,pad=0,rounding_size=0.05", fc=OFFF, ec="none"))
ax.text(gw + 4.02, yb, "branch off (coefficient 0, weights zero)", fontsize=8.5, color=INK2, va="center")
ax.text(0.15, yb - 0.42, "Every setting reproduces its original model through the original's own transformers code "
        "(exact in float64; checked from hf.co/alexlitz/elephantine).", fontsize=8.5, color=MUTED, va="center")
fig.savefig("graph/fig_elephantine_settings.png", facecolor=SURF)
