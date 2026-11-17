# elephantine

The current demo, [hf.co/alexlitz/elephantine](https://huggingface.co/alexlitz/elephantine), is built by
`push_hub_demo.py` from `universal.py`: the remote code in `hub/` is a vendored copy of the universal
host, and each Hub subfolder is one real model loaded into it by a `from_*` loader and checked against
the original (`python -m elephantine.push_hub_demo --out DIR`).

The older two scripts below build the August 2026 demo (since retired from the Hub in favour of
the one above):
the "elephantine" universal-emulator architecture (a hybrid Jamba SSM + attention + MoE
model) weighted so its forward pass reproduces the smallest SiLU/RMSNorm decoder LM in
this dataset **token-for-token**. `build_elephantine.py` constructs and verifies it (fp64 +
fp32); `save_and_push_elephantine.py` saves the safetensors, writes the model card, and
pushes to the Hub. Run from inside this directory: `python build_elephantine.py`.
