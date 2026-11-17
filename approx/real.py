"""Driver for the real pair google/gemma-4-26B-A4B (teacher) -> openai/gpt-oss-20b architecture.

  python -m approx.real prep                         # tokenize calib / held-out / distill windows (both domains)
  python -m approx.real stats    --domain code       # teacher stream + layer-ablation dKL + grad x act importance
  python -m approx.real convert  --cal code --name conv-code
  python -m approx.real convert  --cal general --name naive --naive
  python -m approx.real eval     --name conv-code --data code
  python -m approx.real gptoss   --data general      # the real gpt-oss-20b with ITS tokenizer (context only)
  python -m approx.real distill  --name conv-code --data code --steps 150

Everything is streamed layer by layer on one GPU (--dev); outputs go to $APPROX_OUT
(default $TEF_SCRATCH/approx_convert; TEF_SCRATCH defaults to ~/.cache/the_emulation_frontier).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from dataclasses import fields, replace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from approx.arch import SCfg, TCfg
from approx.importance import Teacher, attribution, layer_ablation, stream

OUT = os.environ.get("APPROX_OUT", os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "approx_convert"))
TEACHER = "google/gemma-4-26B-A4B"
STUDENT = "openai/gpt-oss-20b"
T_LEN = 512
SIZES = {"calib": 32, "heldout": 32, "train": 800}


def snap(repo):
    s = sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/")))
    s = [x for x in s if glob.glob(x + "*.safetensors")]
    return s[-1].rstrip("/")


def tcfg():
    return TCfg.from_hf(json.load(open(os.path.join(snap(TEACHER), "config.json"))))


def scfg_target():
    """gpt-oss-20b's architecture with the free config values set for the teacher."""
    s = SCfg.from_hf(json.load(open(os.path.join(snap(STUDENT), "config.json"))))
    tc = tcfg()
    # eps: a placeholder; Converter.run sets the real one (approx.convert.eps_plan: eps h_t / (h_s max C^2),
    # with the eps channel), saved in the student's config.json, which evaluate.Student reads back
    s.vocab, s.rope_scaling, s.eps, s.window = tc.vocab, None, tc.eps, tc.window
    s.rope_theta = tc.rope["sliding_attention"]["theta"]
    return s


def P(*a):
    return os.path.join(OUT, *a)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cmd_prep(a):
    from approx.data import load_tok, texts, windows
    os.makedirs(P("data"), exist_ok=True)
    ttok = load_tok(snap(TEACHER))
    stok = load_tok(snap(STUDENT))
    for dom in ("general", "code"):
        tr = texts(dom, "train")
        ho = texts(dom, "heldout")
        log(dom, "docs", len(tr), len(ho))
        # calibration and distillation windows come from disjoint document ranges of the train split
        cut = len(tr) // 4
        for split, docs, n in (("calib", tr[:cut], SIZES["calib"]), ("train", tr[cut:], SIZES["train"]),
                               ("heldout", ho, SIZES["heldout"])):
            ids, nb = windows(ttok, docs, n, T_LEN, bos=ttok.convert_tokens_to_ids("<bos>"), seed=1)
            torch.save({"ids": ids, "bytes": nb}, P("data", f"{dom}_{split}.pt"))
            log(dom, split, tuple(ids.shape), nb, "bytes")
        ids, nb = windows(stok, ho, SIZES["heldout"], T_LEN, bos=None, seed=1)
        torch.save({"ids": ids, "bytes": nb}, P("data", f"{dom}_heldout_gptoss.pt"))


def data(dom, split):
    return torch.load(P("data", f"{dom}_{split}.pt"))


def teacher(dev):
    return Teacher(tcfg(), path=snap(TEACHER), device=dev)


def cmd_stats(a):
    T = teacher(a.dev)
    cal = data(a.domain, "calib")["ids"]
    t0 = time.time()
    hs = stream(T, cal, bs=4)
    torch.save(hs, P(f"hs_calib_{a.domain}.pt"))
    log("stream", time.time() - t0)
    sub = cal[: a.imp_seqs]
    dkl = layer_ablation(T, sub, bs=4)
    log("layer ablation dKL", [round(x, 4) for x in dkl.tolist()], time.time() - t0)
    imp = attribution(T, sub, hs=[h[: a.imp_seqs] for h in hs], bs=2)
    log("attribution", time.time() - t0)
    torch.save({"dkl": dkl, "imp": imp}, P(f"imp_{a.domain}.pt"))


def cmd_convert(a):
    from approx.convert import Converter, Opts, naive_opts
    from approx.evaluate import LayerWriter
    T = teacher(a.dev)
    o = naive_opts() if a.naive else Opts()
    for kv in a.opt or []:
        k, v = kv.split("=")
        typ = {f.name: f.type for f in fields(Opts)}[k]
        o = replace(o, **{k: (v == "True") if typ in ("bool", bool) else (int(v) if typ in ("int", int) else
                                                                        (float(v) if typ in ("float", "float | None") else v))})
    st = torch.load(P(f"imp_{a.cal}.pt"))
    hs = torch.load(P(f"hs_calib_{a.cal}.pt"))
    cal = data(a.cal, "calib")["ids"]
    n = a.cal_seqs or cal.shape[0]
    cal, hs = cal[:n], [h[:n] for h in hs]
    sc = scfg_target()
    out = P("students", a.name)
    w = LayerWriter(out) if not a.no_save else (lambda i, lw: None)
    conv = Converter(T, sc, o, imp=None if a.naive else st["imp"], dkl=None if a.naive else st["dkl"], log=log)
    if a.layers:
        conv.limit = int(a.layers)
    done = w.existing() if (a.resume and not a.no_save) else None
    if done:
        log("resuming: layers", sorted(done))
    t0 = time.time()
    glob_, info = conv.run(cal, hs=hs, sink=w, done=done)
    info["opts"] = {f.name: getattr(o, f.name) for f in fields(Opts)}
    info["time_s"] = time.time() - t0
    if not a.no_save:
        w.finish(sc, glob_, tokenizer_from=snap(TEACHER), extra=info)
    json.dump(info, open(P(f"convinfo_{a.name}.json"), "w"), indent=1, default=str)
    log("done", time.time() - t0)


def teacher_final(dom, split, dev, n=None):
    ids = data(dom, split)["ids"]
    n = n or ids.shape[0]
    f = P(f"hT_{dom}_{split}_{n}.pt")
    if os.path.exists(f):
        return torch.load(f)
    ids = ids[:n]
    T = teacher(dev)
    hs = stream_last(T, ids)
    torch.save(hs, f)
    return hs


@torch.no_grad()
def stream_last(T, ids, bs=8):
    """final teacher hidden only (memory: one copy of the activations)."""
    from approx.arch import t_embed, t_layer
    cfg, dev = T.cfg, T.device
    g = T.glob()
    h = torch.cat([t_embed(g, cfg, ids[i:i + bs].to(dev)).cpu() for i in range(0, ids.shape[0], bs)])
    for li in range(cfg.L):
        lw = T.layer(li)
        h = torch.cat([t_layer(lw, cfg, li, h[i:i + bs].to(dev)).cpu() for i in range(0, h.shape[0], bs)])
        del lw
        log(f"  teacher layer {li}")
    return h


def cmd_eval(a):
    from approx.arch import random_student
    from approx.evaluate import Student, compare
    d = data(a.data, "heldout")
    ids = d["ids"][: a.n] if a.n else d["ids"]
    hT = teacher_final(a.data, "heldout", a.dev)[: ids.shape[0]]
    T = teacher(a.dev)
    sc = scfg_target()
    if a.name == "random":
        from approx.evaluate import RandomStudent
        S = RandomStudent(sc, device=a.dev, dtype=getattr(torch, a.act_dtype))
    else:   # stored weights are bf16; activations in a.act_dtype (fp32 by default: see README, bf16 diverges)
        S = Student(sc, path=P("students", a.name), device=a.dev, dtype=getattr(torch, a.act_dtype))
    r = compare(T, S, ids, hT=hT)
    nb = d["bytes"] * ids.shape[0] / d["ids"].shape[0]
    r["bpb_student"] = math.log(r["ppl_student"]) * r["tokens"] / math.log(2) / nb
    r["bpb_teacher"] = math.log(r["ppl_teacher"]) * r["tokens"] / math.log(2) / nb
    os.makedirs(P("eval"), exist_ok=True)
    suffix = "" if a.act_dtype == "float32" else f"-{a.act_dtype}"
    json.dump(r, open(P("eval", f"{a.name}{suffix}_{a.data}.json"), "w"), indent=1)
    log(a.name, a.data, r)


def cmd_gptoss(a):
    from approx.evaluate import Student, compare
    d = data(a.data, "heldout_gptoss")
    ids = d["ids"][: a.n] if a.n else d["ids"]
    sc = SCfg.from_hf(json.load(open(os.path.join(snap(STUDENT), "config.json"))))
    S = Student(sc, path=snap(STUDENT), device=a.dev, dtype=torch.bfloat16)
    r = compare(None, S, ids)
    r["bpb_student"] = math.log(r["ppl_student"]) * r["tokens"] / math.log(2) / (d["bytes"] * ids.shape[0] / d["ids"].shape[0])
    os.makedirs(P("eval"), exist_ok=True)
    json.dump(r, open(P("eval", f"gpt-oss-20b_{a.data}.json"), "w"), indent=1)
    log("gpt-oss-20b (own tokenizer)", a.data, r)


def cmd_distill(a):
    from approx.distill import PipeStudent, TeacherHead, distill
    from approx.evaluate import Student
    sc = scfg_target()
    tr_ids = data(a.data, "train")["ids"][: a.train_n]
    trh = teacher_final(a.data, "train", "cuda:0", a.train_n)
    evals = {d: (data(d, "heldout")["ids"][: a.eval_n], teacher_final(d, "heldout", "cuda:0")[: a.eval_n])
             for d in ("general", "code")}
    S = Student(sc, path=P("students", a.name), device="cpu", dtype=torch.bfloat16)
    devs = tuple(a.devices.split(","))
    model = PipeStudent(S, devices=devs, split=a.split if len(devs) > 1 else None, r=a.rank, r_exp=a.rank_exp, log=log,
                        offload=not a.no_offload, dtype=torch.float32,       # fp32 activations (bf16 diverges)
                        probe=tr_ids[:1] if a.probe else None)
    th = TeacherHead(tcfg(), Teacher(tcfg(), path=snap(TEACHER)).glob(), devs[-1])
    curve = distill(model, th, tr_ids, trh, evals, steps=a.steps, bs=a.bs, lr=a.lr, lr_other=a.lr_other,
                    eval_every=a.eval_every, log=log)
    os.makedirs(P("distill"), exist_ok=True)
    tag = f"{a.name}_{a.data}_{a.steps}{a.tag}"
    json.dump(curve, open(P("distill", tag + ".json"), "w"), indent=1)
    torch.save({i: l.export() for i, l in enumerate(model.layers)} | {"norm.weight": model.n_f.detach().cpu()},
               P("distill", tag + "_trained.pt"))
    log("curve", curve)


def cmd_report(a):
    """markdown tables from the eval / conversion / distillation JSONs."""
    ev = {}
    for f in sorted(glob.glob(P("eval", "*.json"))):
        n, d = os.path.basename(f)[:-5].rsplit("_", 1)
        ev[(n, d)] = json.load(open(f))
    names = [n for n in ["random", "naive", "conv-general", "conv-code"] if any(k[0] == n for k in ev)]
    print("| student | data | KL(T||S) nats/tok | top-1 agree | ppl student | ppl teacher | bits/byte student | bits/byte teacher |")
    print("|---|---|---|---|---|---|---|---|")
    for n in names:
        for d in ("general", "code"):
            r = ev.get((n, d))
            if r:
                print(f"| {n} | {d} | {r['kl']:.3f} | {r['top1']:.3f} | {r['ppl_student']:.1f} | {r['ppl_teacher']:.2f} | "
                      f"{r['bpb_student']:.3f} | {r['bpb_teacher']:.3f} |")
    for d in ("general", "code"):
        r = ev.get(("gpt-oss-20b", d))
        if r:
            print(f"| gpt-oss-20b (own tokenizer, real weights) | {d} | - | - | {r['ppl_student']:.1f} | - | {r['bpb_student']:.3f} | - |")
    for f in sorted(glob.glob(P("convinfo_*.json"))):
        info = json.load(open(f))
        print(f"\n{os.path.basename(f)}: layer map {info['pi']}  time {info.get('time_s', 0) / 60:.0f} min")
        tl = tcfg().layer_types
        f3 = lambda v: "-" if v is None else f"{v:.3f}"
        print("| student layer | teacher layer | type | local attn err | local FFN err | stream err (fit seqs) | stream err (held-out seqs) | routing mass recall | dense units | grad refit FFN err |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for x in info["layers"]:
            gr = x.get("grad_refit")
            print(f"| {x['i']} | {x['l']} | {tl[x['l']][:4]} | {f3(x.get('attn_err'))} | {f3(x.get('ffn_err'))} | {f3(x.get('stream_err'))} | "
                  f"{f3(x.get('stream_err_ho'))} | {f3(x.get('mass_recall'))} | {x.get('dense_units', '-')} | "
                  f"{'%.3f -> %.3f' % (gr['before'], gr['after']) if gr else '-'} |")
    for f in sorted(glob.glob(P("distill", "*.json"))):
        c = json.load(open(f))
        print(f"\n{os.path.basename(f)}")
        keys = [k for k in c[0] if k != "step"]
        print("| step | " + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for r in c:
            print(f"| {r['step']} | " + " | ".join(f"{r.get(k, float('nan')):.3f}" for k in keys) + " |")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("prep")
    x = sp.add_parser("stats"); x.add_argument("--domain", default="general"); x.add_argument("--imp-seqs", type=int, default=4)
    x.add_argument("--dev", default="cuda:0")
    x = sp.add_parser("convert"); x.add_argument("--cal", default="general"); x.add_argument("--name", required=True)
    x.add_argument("--naive", action="store_true"); x.add_argument("--opt", action="append")
    x.add_argument("--cal-seqs", type=int, default=0); x.add_argument("--dev", default="cuda:0")
    x.add_argument("--layers", default=None); x.add_argument("--no-save", action="store_true")
    x.add_argument("--resume", action="store_true")
    x = sp.add_parser("eval"); x.add_argument("--name", required=True); x.add_argument("--data", default="general")
    x.add_argument("--n", type=int, default=0); x.add_argument("--dev", default="cuda:0")
    x.add_argument("--act-dtype", default="float32")
    x = sp.add_parser("gptoss"); x.add_argument("--data", default="general"); x.add_argument("--n", type=int, default=0)
    x.add_argument("--dev", default="cuda:0")
    x = sp.add_parser("distill"); x.add_argument("--name", required=True); x.add_argument("--data", default="general")
    x.add_argument("--steps", type=int, default=150); x.add_argument("--bs", type=int, default=4)
    x.add_argument("--rank", type=int, default=16); x.add_argument("--rank-exp", type=int, default=0)
    x.add_argument("--lr", type=float, default=2e-4); x.add_argument("--lr-other", type=float, default=2e-5)
    x.add_argument("--eval-every", type=int, default=25); x.add_argument("--train-n", type=int, default=400)
    x.add_argument("--eval-n", type=int, default=16); x.add_argument("--split", type=int, default=12)
    x.add_argument("--no-offload", action="store_true"); x.add_argument("--devices", default="cuda:0")
    x.add_argument("--tag", default=""); x.add_argument("--probe", action="store_true")
    x = sp.add_parser("report")
    a = ap.parse_args()
    {"prep": cmd_prep, "stats": cmd_stats, "convert": cmd_convert, "eval": cmd_eval, "gptoss": cmd_gptoss,
     "distill": cmd_distill, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
