"""Fidelity of a student against the teacher on held-out tokens: KL(teacher || student) per token
(full vocabulary), top-1 agreement, and both perplexities -- all streamed layer by layer, so it
runs for the 26B / 21B pair on one 24 GB GPU."""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from approx.arch import SCfg, s_embed, s_layer, s_logits, student_rope
from approx.importance import Teacher, stream


class Student:
    """Student weights: a canonical-name dict (tiny / in-memory) or a directory written by
    `save_student` / a stock gpt-oss checkpoint (MXFP4 experts are dequantized)."""

    def __init__(self, scfg: SCfg, weights=None, path=None, device="cpu", dtype=None):
        self.scfg, self.w, self.path, self.device, self.dtype = scfg, weights, path, device, dtype
        self.files = {}
        if path is not None:
            cj = os.path.join(path, "config.json")
            if os.path.exists(cj) and "rms_norm_eps" in json.load(open(cj)):
                scfg.eps = json.load(open(cj))["rms_norm_eps"]   # a converted student's eps (eps_plan)
            idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
            for n, f in idx.items():
                c = n[len("model."):] if n.startswith("model.") else n
                self.files[c] = (os.path.join(path, f), n)
        self._open = {}

    def _t(self, name):
        from safetensors import safe_open
        f, n = self.files[name]
        if f not in self._open:
            self._open[f] = safe_open(f, framework="pt", device="cpu")
        return self._open[f].get_tensor(n)

    def _cast(self, v):
        v = v.to(self.device)
        return v.to(self.dtype) if (self.dtype is not None and v.is_floating_point()) else v

    def layer(self, li):
        p = f"layers.{li}."
        if self.w is not None:
            d = {k[len(p):]: self._cast(v) for k, v in self.w.items() if k.startswith(p)}
        else:
            d = {}
            for k in self.files:
                if not k.startswith(p):
                    continue
                kk = k[len(p):]
                if kk.endswith("_blocks"):
                    base = kk[: -len("_blocks")]
                    d[base] = self._cast(dequant_mxfp4(self._t(k).to(self.device), self._t(p + base + "_scales").to(self.device)))
                elif kk.endswith("_scales"):
                    continue
                else:
                    d[kk] = self._cast(self._t(k))
            for kk in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                if kk in d and d[kk].shape[-1] != (2 * self.scfg.I if "gate_up" in kk else self.scfg.h):
                    d[kk] = d[kk].transpose(-1, -2).contiguous()
        d["_type"] = self.scfg.layer_types[li]
        return d

    def glob(self):
        src = self.w if self.w is not None else None
        g = lambda k: self._cast(src[k] if src is not None else self._t(k))
        return {k: g(k) for k in ("embed_tokens.weight", "norm.weight", "lm_head.weight")}


class RandomStudent(Student):
    """baseline (a): gpt-oss's own init, generated layer by layer (a 21B random model does not fit
    in RAM next to everything else)."""

    def __init__(self, scfg: SCfg, seed=0, device="cpu", dtype=torch.bfloat16):
        super().__init__(scfg, weights={}, device=device, dtype=dtype)
        self.seed = seed

    def _gen(self, li):
        from approx.arch import random_student
        import dataclasses
        one = dataclasses.replace(self.scfg, L=1, layer_types=[self.scfg.layer_types[max(li, 0)]])
        return random_student(one, seed=self.seed * 1000 + li + 1, dtype=self.dtype)

    def layer(self, li):
        w = self._gen(li)
        d = {k[len("layers.0."):]: self._cast(v) for k, v in w.items() if k.startswith("layers.0.")}
        d["_type"] = self.scfg.layer_types[li]
        return d

    def glob(self):
        w = self._gen(-1)
        return {k: self._cast(w[k]) for k in ("embed_tokens.weight", "norm.weight", "lm_head.weight")}


def dequant_mxfp4(blocks, scales):
    """MXFP4 (gpt-oss) -> bf16, via transformers' own unpacker; returns HF's (E, in, out) layout
    transposed back to how the stored blocks are indexed (E, out, in)."""
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    return convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)


@torch.no_grad()
def student_final(stu: Student, ids, bs=4):
    sc, dev = stu.scfg, stu.device
    g = stu.glob()
    hs = torch.cat([s_embed(g, ids[i:i + bs].to(dev)).cpu() for i in range(0, ids.shape[0], bs)])
    rope = student_rope(sc, ids.shape[1], dev)
    for li in range(sc.L):
        lw = stu.layer(li)
        hs = torch.cat([s_layer(lw, sc, hs[i:i + bs].to(dev), rope).cpu() for i in range(0, hs.shape[0], bs)])
        del lw
    return hs


@torch.no_grad()
def compare(teacher: Teacher | None, stu: Student, ids, bs=4, chunk=512, hT=None):
    """KL(teacher||student) (nats/token), top-1 agreement, teacher / student perplexity. Positions
    0..T-2 predict ids[:, 1:]. If teacher is None, only the student's perplexity is returned."""
    from approx.arch import t_logits
    hS = student_final(stu, ids, bs)
    gS = stu.glob()
    tgt = ids[:, 1:].reshape(-1)
    HS = hS[:, :-1].reshape(-1, hS.shape[-1])
    if teacher is not None:
        if hT is None:
            hT = stream(teacher, ids, bs)[-1]
        gT = teacher.glob()
        HT = hT[:, :-1].reshape(-1, hT.shape[-1])
    kl = agree = nllT = nllS = 0.0
    n = HS.shape[0]
    dev = stu.device
    for i in range(0, n, chunk):
        zs = s_logits(gS, stu.scfg, HS[i:i + chunk].to(dev)).log_softmax(-1)
        y = tgt[i:i + chunk].to(dev)
        nllS += F.nll_loss(zs, y, reduction="sum").item()
        if teacher is not None:
            zt = t_logits(gT, teacher.cfg, HT[i:i + chunk].to(teacher.device)).log_softmax(-1).to(dev)
            kl += (zt.exp() * (zt - zs)).sum().item()
            agree += (zt.argmax(-1) == zs.argmax(-1)).sum().item()
            nllT += F.nll_loss(zt, y, reduction="sum").item()
    out = {"ppl_student": math.exp(nllS / n), "tokens": n}
    if teacher is not None:
        out.update(kl=kl / n, top1=agree / n, ppl_teacher=math.exp(nllT / n))
    return out


def save_student(path, scfg: SCfg, glob, layers, dtype=torch.bfloat16, tokenizer_from=None, extra=None):
    """Write a stock-loadable GptOssForCausalLM checkpoint (bf16, unquantized experts), one
    safetensors shard per layer."""
    w = LayerWriter(path, dtype)
    for i, lw in (layers.items() if isinstance(layers, dict) else layers):
        w(i, lw)
    w.finish(scfg, glob, tokenizer_from, extra)


class LayerWriter:
    """sink for Converter.run that writes each student layer to its own shard as it is produced."""

    def __init__(self, path, dtype=torch.bfloat16, keep=False):
        from safetensors.torch import save_file
        self.path, self.dtype, self.keep, self.save_file = path, dtype, keep, save_file
        self.wm, self.mem = {}, {}
        os.makedirs(path, exist_ok=True)

    def existing(self):
        """{i: canonical weights} of layer shards already written (resume); registers them."""
        import glob as _g
        from safetensors import safe_open
        out = {}
        for f in sorted(_g.glob(os.path.join(self.path, "model-layer*.safetensors"))):
            i = int(os.path.basename(f)[len("model-layer"):-len(".safetensors")])
            with safe_open(f, "pt") as s:
                p = f"model.layers.{i}."
                out[i] = {k[len(p):]: s.get_tensor(k) for k in s.keys()}
                self.wm.update({k: os.path.basename(f) for k in s.keys()})
        return out

    def __call__(self, i, lw):
        d = {f"model.layers.{i}.{k}": v.to(self.dtype).contiguous() for k, v in lw.items()}
        fname = f"model-layer{i:02d}.safetensors"
        self.save_file(d, os.path.join(self.path, fname), metadata={"format": "pt"})
        self.wm.update({k: fname for k in d})
        if self.keep:
            self.mem[i] = lw

    def finish(self, scfg: SCfg, glob, tokenizer_from=None, extra=None):
        d = {(k if k.startswith("lm_head") else "model." + k): v.to(self.dtype).contiguous() for k, v in glob.items()}
        self.save_file(d, os.path.join(self.path, "model-globals.safetensors"), metadata={"format": "pt"})
        self.wm.update({k: "model-globals.safetensors" for k in d})
        json.dump({"metadata": {}, "weight_map": self.wm}, open(os.path.join(self.path, "model.safetensors.index.json"), "w"), indent=1)
        json.dump(scfg.hf_config(torch_dtype="bfloat16"), open(os.path.join(self.path, "config.json"), "w"), indent=1)
        if extra is not None:
            json.dump(extra, open(os.path.join(self.path, "conversion_info.json"), "w"), indent=1, default=str)
        if tokenizer_from:                       # the TEACHER's tokenizer (vocab_size = teacher's)
            import shutil
            for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                if os.path.exists(os.path.join(tokenizer_from, f)):
                    shutil.copy(os.path.join(tokenizer_from, f), self.path)
