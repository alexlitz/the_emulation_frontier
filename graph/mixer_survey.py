"""Sequence-mixer survey: how many of the top models are NOT plain attention?

We classify the emulation graph over softmax-attention transformers, but the
zoo also has state-space (Mamba/RWKV) and hybrid (Jamba/Zamba/Nemotron-H) models
that have no attention slot and get dropped. This counts them, to size the gap."""
import json
from collections import Counter

SSM = {"mamba", "mamba2", "falcon_mamba", "rwkv", "rwkv5", "rwkv6", "rwkv7",
       "rwkv_hybrid", "retnet", "based"}
# LFM2/3 (gated short conv) and RecurrentGemma (RG-LRU) interleave softmax layers: hybrids
HYBRID = {"jamba", "zamba", "zamba2", "nemotron_h", "falcon_h1", "granitemoehybrid",
          "plamo2", "bamba", "hymba", "minimax", "minimax_text_01", "phi4flash", "samba",
          "lfm2", "lfm2_moe", "lfm3", "lfm3_rd", "recurrent_gemma", "recurrentgemma",
          "nemotron_h_puzzle", "ceno", "jet_nemotron"}


def main():
    rows = json.load(open("graph/hub_10k.json"))
    cat = Counter()
    ssm_types, hyb_types = Counter(), Counter()
    for r in rows:
        mt = r["config"].get("model_type", "?")
        if mt in SSM:
            cat["ssm"] += 1; ssm_types[mt] += 1
        elif mt in HYBRID:
            cat["hybrid"] += 1; hyb_types[mt] += 1
        elif r["config"].get("num_attention_heads") or r["config"].get("n_head"):
            cat["attention"] += 1
        else:
            cat["other/skip"] += 1

    with open("graph/mixer_survey.txt", "w") as f:
        f.write(f"sequence-mixer breakdown of {sum(cat.values())} fetched configs:\n\n")
        for k, v in cat.most_common():
            f.write(f"  {k:14} {v:5}  ({100*v/sum(cat.values()):.1f}%)\n")
        f.write(f"\nSSM types:    {dict(ssm_types)}\n")
        f.write(f"hybrid types: {dict(hyb_types)}\n")
    print("done")


if __name__ == "__main__":
    main()
