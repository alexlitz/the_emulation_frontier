"""A recipe's build stage: turn a Hub target into a run directory for a framework reference.

    build_native(target, out)                  the target's own config + safetensors (+ tokenizer)
    build_exact(target, host, out)             a Llama-style host checkpoint that computes the target exactly
                                               (elephantine/emulate.py v1 constructions)
    build_universal(target, loader, out)       the universal host set by its from_* loader, exported as tef-ucore

Needs torch + transformers (the targets are read with their stock transformers class; no remote code).
"""
from __future__ import annotations

import json
import os
import shutil

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json")


def _copy_tokenizer(target, out):
    from huggingface_hub import hf_hub_download
    for f in TOKENIZER_FILES:
        try:
            p = os.path.join(target, f) if os.path.isdir(target) else hf_hub_download(target, f)
        except Exception:
            continue
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(out, f))


def build_native(target, out, revision=None):
    """Download config.json, the safetensors (all shards) and the tokenizer files into out."""
    from huggingface_hub import snapshot_download
    os.makedirs(out, exist_ok=True)
    src = target if os.path.isdir(target) else snapshot_download(
        target, revision=revision, allow_patterns=["config.json", "*.safetensors", "model.safetensors.index.json",
                                                   *TOKENIZER_FILES])
    for f in os.listdir(src):
        if f == "config.json" or f.endswith(".safetensors") or f == "model.safetensors.index.json" or f in TOKENIZER_FILES:
            dst = os.path.join(out, f)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.realpath(os.path.join(src, f)), dst)
    return out


def build_exact(target, host, out, log=print):
    """The exact-host route: emulate.py's v1 construction of `target` in `host`'s architecture (the pair was
    checked to be an exact edge of the headline relation when the recipe was generated)."""
    import torch
    from transformers import AutoModelForCausalLM
    from tef_recipes import emulate as E
    try:
        out = E.prepare_out(out)         # an existing directory is deleted only if an earlier build wrote it
    except E.OutDirError as e:
        raise SystemExit(str(e))
    traw, hraw = E.fetch_config(target), E.fetch_config(host)
    con = E.plan_construction(traw, hraw)
    tc = E.resolved(traw)
    hdict = E.host_config_dict(hraw, con)
    for x in E.construction_lines(con, tc, E.resolved(hdict)):
        log("  " + x)
    with E.warnings_off():
        model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.float64, attn_implementation="eager").eval()
    T = E.read_target(model, tc, con.tfam)
    sd = E.build_host(T, tc, hdict, con, log)
    del T, model
    E.save_host(out, hdict, sd, None)
    _copy_tokenizer(target, str(out))
    (out / "recipe.json").write_text(json.dumps({"route": "exact-host", "target": target, "host": host,
                                                 "config_changes": {k: v for k, v in con.config_changes.items()
                                                                    if hraw.get(k) != v}}, indent=1, default=str))
    return str(out)


def build_universal(target, loader, out, dtype="float64"):
    """The universal route: the target loaded with transformers, the universal host set by `loader`, exported."""
    import torch
    from transformers import AutoModelForCausalLM
    from tef_recipes import universal as U
    from tef_recipes.ucore_export import save
    dt = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(target, dtype=dt).eval()
    with torch.no_grad():
        host = getattr(U, loader)(model).eval()
    del model
    cfg = save(out, host, extra={"target": target, "loader": loader})
    _copy_tokenizer(target, out)
    return cfg
