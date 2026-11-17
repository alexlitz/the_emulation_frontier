"""The universal host (elephantine/universal.py) in plain PyTorch, run from its tef-ucore export (no
transformers, no repo code): every core branch of UniversalElephantine (tef_recipes.ucore_export.CORE_BRANCHES)
for one sequence, with a KV cache for attention and a conv + recurrent-state cache for the SSM. fp32."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from recipes.common.runtime import Generator, load_run_dir  # tef:inline

F32 = torch.float32
NEG = float("-inf")


def _rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


class UCoreModel(Generator):
    def __init__(self, run_dir):
        cfg, sd = load_run_dir(run_dir)
        if cfg.get("format") != "tef-ucore":
            raise ValueError("not a tef-ucore run directory")
        self.c = cfg
        self.w = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in sd.items()}
        torch.set_grad_enabled(False)

    # --- pieces ---------------------------------------------------------------------------------------
    def _lin(self, name, x):
        y = x @ self.w[name + ".weight"].T
        b = self.w.get(name + ".bias")
        return y if b is None else y + b

    def _norm(self, spec, name, x):
        if spec["ident"]:
            return x
        if spec["centre"] != 0:
            x = x - spec["centre"] * x.mean(-1, keepdim=True)
        y = _rms(x, spec["eps"]) * self.w[name + ".weight"]
        b = self.w.get(name + ".bias")
        return y if b is None else y + b

    def _qk_norm(self, s, p, t, i):
        coef = s["q_coef"] if i == 0 else s["k_coef"]
        if coef[1] == 0:
            return coef[0] * t
        dims = (-2, -1) if s["qk_full"] else (-1,)
        n = t - t.mean(dims, keepdim=True) if s["qk_centre"][i] else t
        width = s["qk_width"] * (t.shape[-2] if s["qk_full"] else 1)
        nw = self.w[p + ("q_nw" if i == 0 else "k_nw")]
        n = n * torch.rsqrt(n.pow(2).sum(dims, keepdim=True) / width + s["qk_eps"][i]) * nw
        if s["qk_centre"][i]:
            n = n + self.w[p + ("q_nb" if i == 0 else "k_nb")]
        return coef[0] * t + coef[1] * n

    def _attn(self, s, p, x, cache, pos, reg):
        c, w = self.c, self.w
        T, H, hd = x.shape[0], c["heads"], c["head_dim"]
        if s["Dn"]:
            n = x @ w[p + "pn.weight"].T
            n = n * torch.rsqrt(n.pow(2) @ w[p + "n_avg"] + s["n_eps"]) * w[p + "n_w"]
            x = torch.cat((x, n), -1)
        q = self._lin(p + "q_proj", x).view(T, H, hd)
        k = self._lin(p + "k_proj", x).view(T, H, hd)
        v = self._lin(p + "v_proj", x).view(T, H, hd)
        if s["v_norm"]:
            v = v * torch.rsqrt(v.pow(2).sum(-1, keepdim=True) / s["qk_width"] + s["v_eps"]) * w[p + "v_nw"]
        if s["rope"]:
            ang = torch.arange(pos, pos + T, dtype=F32)[:, None] * w[p + "inv_freq"][None]
            cos = torch.cat((ang.cos(), ang.cos()), -1)[:, None]
            sin = torch.cat((ang.sin(), ang.sin()), -1)[:, None]
            rope = lambda t: t * cos + _rot_half(t) * sin
        else:
            rope = lambda t: t
        if s["qk_post"]:
            q, k = rope(q), rope(k)
        q, k = self._qk_norm(s, p, q, 0), self._qk_norm(s, p, k, 1)
        if s["clip"] is not None:
            cl = s["clip"]
            q, k, v = q.clamp(-cl, cl), k.clamp(-cl, cl), v.clamp(-cl, cl)
        if not s["qk_post"]:
            q, k = rope(q), rope(k)
        if s.get("qtemp"):
            beta, M = s["qtemp"]
            t_ = torch.arange(pos, pos + T, dtype=F32)
            q = q * (1 + beta * torch.log1p(torch.floor(t_ / M)))[:, None, None]
        if s.get("kv_read"):                              # the KV register: an earlier layer's whole cache
            k, v = reg[s["kv_slot"]]
        else:
            if cache.get("k") is not None:
                k = torch.cat((cache["k"], k), 0)
                v = torch.cat((cache["v"], v), 0)
            cache["k"], cache["v"] = k, v
            if s.get("kv_write"):
                reg[s["kv_slot"]] = (k, v)
        S = k.shape[0]
        sc = torch.einsum("thd,shd->hts", q, k) * s["scale"]
        if s["cap"] is not None:
            sc = s["cap"] * torch.tanh(sc / s["cap"])
        tt = torch.arange(pos, pos + T)[:, None]
        ss = torch.arange(S)[None]
        if s["alibi"]:
            sc = sc + w[p + "alibi"][:, None, None] * (ss - tt).to(F32)
        win = math.inf if s["window"] is None else s["window"]
        allowed = ((tt - ss) <= win) | (ss < s["n_sink"])
        allowed = allowed & (ss <= tt)
        sc = sc.masked_fill(~allowed, NEG)
        if s["null"]:
            sc = torch.cat((sc, w[p + "null_logit"][:, None, None].expand(H, T, 1)), -1)
        o = torch.einsum("hts,shd->thd", sc.softmax(-1)[..., :S], v).reshape(T, H * hd)
        if s["o_coef"][1] != 0:
            oc = s["o_coef"]
            o = oc[0] * o + oc[1] * o * torch.rsqrt(o.pow(2) @ w[p + "o_avg"] + s["o_eps"]) * w[p + "o_w"]
        g = s["gate_coef"]
        if s.get("gate_sp"):
            gx = self._lin(p + "g_proj", x)
            o = o * (g[0] + g[1] * torch.sigmoid(gx) + s["gate_sp"] * F.softplus(gx))
        elif g[1] != 0:
            o = o * (g[0] + g[1] * torch.sigmoid(self._lin(p + "g_proj", x)))
        elif g[0] != 1:
            o = o * g[0]
        return self._lin(p + "o_proj", o)

    def _ssm(self, s, p, x, cache, pos):
        w = self.w
        T = x.shape[0]
        H, dk, dv, K = s["H"], s["dk"], s["dv"], s["K"]
        u = self._lin(p + "in_proj", x)                                   # (T, D)
        hist = cache.get("conv")
        if hist is None:
            hist = torch.zeros(K - 1, s["D"], dtype=F32)
        up = torch.cat((hist, u), 0)
        cache["conv"] = up[up.shape[0] - (K - 1):] if K > 1 else hist
        cw = w[p + "conv_w"]
        cv = sum(cw[:, j] * up[j:j + T] for j in range(K)) + w[p + "conv_b"]
        cv = w[p + "act_silu"] * F.silu(cv) + w[p + "act_lin"] * cv
        r = torch.cat((u, cv), -1)
        if s["Dn"]:
            n = self._lin(p + "pn", r)
            r = torch.cat((r, n * torch.rsqrt(n.pow(2) @ w[p + "n_avg"] + s["n_eps"]) * w[p + "n_w"]), -1)
        if s["Dh"]:
            r = torch.cat((r, torch.sigmoid(self._lin(p + "ph", r))), -1)
        q = self._lin(p + "pq", r).view(T, H, dk)
        k = self._lin(p + "pk", r).view(T, H, dk)
        if s["l2_eps_out"] != 0:
            l2 = lambda t: t / ((t.pow(2).sum(-1, keepdim=True) + s["l2_eps"]).sqrt() + s["l2_eps_out"])
        else:
            l2 = lambda t: t * torch.rsqrt(t.pow(2).sum(-1, keepdim=True) + s["l2_eps"])
        q = s["q_coef"][0] * q + s["q_coef"][1] * l2(q)
        k = s["k_coef"][0] * k + s["k_coef"][1] * l2(k)
        v = self._lin(p + "pv", r).view(T, H, dv)
        sp = F.softplus(self._lin(p + "p1", r).view(T, H, dk))
        if s["dt_min"] > 0:
            sp = sp.clamp(min=s["dt_min"])
        loga = -(w[p + "A1"] * sp + w[p + "A2"] * torch.sigmoid(self._lin(p + "p2", r).view(T, H, dk)) + w[p + "A3"])
        dt = sp[..., 0]                                                   # (T, H)
        a = w[p + "mem"] * loga.exp()                                     # (T, H, dk)
        sb, si = torch.sigmoid(self._lin(p + "pb", r)), torch.sigmoid(self._lin(p + "pi", r))
        gamma = w[p + "cg"] * sb + w[p + "cg0"]
        lru = torch.sqrt((1 - a[..., 0] ** 2).clamp_min(0))
        first = (torch.arange(pos, pos + T) == 0).to(F32)[:, None]
        lru = lru + w[p + "reset"] * first * (1 - lru)
        c6 = w[p + "cw"]
        wr = (c6[0] * dt + c6[1] * sb + c6[2] * self._lin(p + "pe", r).exp() + c6[3] * lru * si + c6[4]
              + c6[5] * dt * sb)
        S = cache.get("S")
        if S is None:
            S = torch.zeros(H, dv, dk, dtype=F32)
        ys = []
        for t in range(T):
            S = S * a[t][:, None, :]
            Sk = (S * k[t][:, None, :]).sum(-1)
            S = S - gamma[t][:, None, None] * Sk[..., None] * k[t][:, None, :]
            S = S + wr[t][:, None, None] * v[t][:, :, None] * k[t][:, None, :]
            ys.append((S * q[t][:, None, :]).sum(-1))
        cache["S"] = S
        y = torch.stack(ys, 0) + w[p + "Dskip"] * v                       # (T, H, dv)
        y = y.reshape(T, H * dv) @ w[p + "mix"]
        z = self._lin(p + "pz", r)
        g = w[p + "gate"]
        G = g[0] * F.silu(z) + g[1] * torch.sigmoid(z) + g[2] * F.gelu(z, approximate="tanh") + g[3] * z + g[4]
        nrm = lambda t: t * torch.rsqrt(t.pow(2) @ w[p + "avg"] + s["norm_eps"]) * w[p + "norm_w"] + w[p + "norm_b"]
        oc = w[p + "out_coef"]
        out = oc[0] * nrm(y) * G + oc[1] * nrm(y * G) + oc[2] * y * G
        return self._lin(p + "out_proj", out)

    @staticmethod
    def _act(a, g, u):
        if a["relu2"]:
            return F.relu(u) ** 2
        if a["relu"]:
            return F.relu(u)
        if a["gelu"] or a["geglu"]:
            ge = lambda t: F.gelu(t, approximate="tanh" if a["gelu_tanh"] else "none")
            return ge(u) if a["gelu"] else ge(g) * u
        if a["mish"]:
            return F.mish(g) * u
        if a.get("xielu"):
            ap, an, beta, eps = a["xielu"]
            return torch.where(u > 0, ap * u * u + beta * u, (torch.expm1(u.clamp(max=eps)) - u) * an + beta * u)
        if a["c_clip"] != 0:
            lg, lu = a["clip"]
            if lg is not None:
                g = g + a["c_clip"] * (g.clamp(max=lg) - g)
            if lu is not None:
                u = u + a["c_clip"] * (u.clamp(-lu, lu) - u)
        u = u + a["up_shift"]
        return F.silu(g) * u

    def _ffn(self, s, p, x, xe=None, xg=None, parts=False):
        """routed(xe) (router on xg) + shared(x) * gate; parts: (routed, gated shared) (Gemma-4's split FFN)."""
        w, a = self.w, s["act"]
        xs, xe, xg = x, (x if xe is None else xe), (x if xg is None else xg)
        out = routed = 0
        r = s["routed"]
        if r is not None:
            x = xe
            logits = self._lin(p + "router", xg)                         # (T, E)
            if r["router_cap"] is not None:
                logits = r["router_cap"] * torch.tanh(logits / r["router_cap"])
            sc_ = r["score_c"]
            pr = logits.softmax(-1) if sc_ == 0 else (1 - sc_) * logits.softmax(-1) + sc_ * torch.sigmoid(logits)
            sel = (logits if r["sel_logit"] else pr) + w[p + "sel_bias"]
            G, kg = int(r["grp"][0]), int(r["grp"][1])
            if G > 1 and kg < G:
                sg = sel.view(sel.shape[0], G, -1)
                gs = sg.max(-1).values if r["grp"][2] != 0 else sg.topk(2, -1).values.sum(-1)
                keep = torch.zeros_like(gs).scatter_(-1, gs.topk(kg, -1).indices, 1.0)
                sel = sel.masked_fill(keep.repeat_interleave(sel.shape[-1] // G, -1) == 0, float(r["grp"][3]))
            topi = sel.topk(r["top_k"], -1).indices
            topv = pr.gather(-1, topi)
            topv = (topv + r["renorm"] * (topv / (topv.sum(-1, keepdim=True) + r["renorm_eps"]) - topv)) * r["route_scale"]
            topv = topv * w[p + "exp_scale"][topi]
            wts = torch.zeros_like(pr).scatter_(-1, topi, topv)          # (T, E), zero off the top-k
            h = self._act(a, torch.einsum("efd,td->tef", w[p + "w_gate"], x) + w[p + "b_gate"],
                          torch.einsum("efd,td->tef", w[p + "w_up"], x) + w[p + "b_up"])
            eo = torch.einsum("edf,tef->ted", w[p + "w_down"], h) + w[p + "b_down"]
            out = routed = (eo * wts[..., None]).sum(1)
        x = xs
        sh = s["shared"]
        shared = 0
        if sh is not None:
            u = self._lin(p + "s_up", x)
            g = self._lin(p + "s_gate", x) if (p + "s_gate.weight") in w else u
            y = self._lin(p + "s_down", self._act(a, g, u))
            c = sh["s_coef"]
            sgt = c[0] + (c[1] * torch.sigmoid(x @ w[p + "s_router.weight"].T) if c[1] != 0 else 0)
            shared = y * sgt
            out = out + shared
        return (routed, shared) if parts else out

    def _layer(self, i, L, x, cache, pos, reg, pli):
        p = "layers.%d." % i

        def post(j, o, nm):
            pc = L["post_coef"][j]
            return pc[0] * o + (pc[1] * self._norm(L[nm], p + nm, o) if pc[1] != 0 else 0)
        if L["sn_in"] is not None:
            x = self._norm(L["sn_in"], p + "sn_in", x)
        a = s = 0
        if L["attn"] is not None:
            a = post(0, self._attn(L["attn"], p + "attn.", self._norm(L["norm_a"], p + "norm_a", x), cache, pos, reg),
                     "post_a")
        if L["ssm"] is not None:
            s = post(1, self._ssm(L["ssm"], p + "ssm.", self._norm(L["norm_s"], p + "norm_s", x + L["seq"] * a),
                                  cache, pos), "post_s")
        x_in = x
        x = (x if L["res_c"][0] == 1 else L["res_c"][0] * x) + a + s
        xf = x_in if L["par"] != 0 else x
        if L["res_c"][1] != 1:
            x = L["res_c"][1] * x
        if L["ffn"] is not None and L.get("ffn_split"):
            nf = lambda nm: self._norm(L[nm], p + nm, xf)
            r_, s_ = self._ffn(L["ffn"], p + "ffn.", nf("norm_f"), xe=nf("norm_f2"), xg=nf("norm_r"), parts=True)
            x = x + post(2, self._norm(L["post_f1"], p + "post_f1", s_) + self._norm(L["post_f2"], p + "post_f2", r_),
                         "post_f")
        elif L["ffn"] is not None:
            x = x + post(2, self._ffn(L["ffn"], p + "ffn.", self._norm(L["norm_f"], p + "norm_f", xf)), "post_f")
        if L.get("ple"):                                  # per-layer embedding input
            g = F.gelu(x @ self.w[p + "ple_gate.weight"].T, approximate="tanh" if L["ple"]["tanh"] else "none")
            y = (g * pli[:, i]) @ self.w[p + "ple_proj.weight"].T
            x = x + L["ple"]["c"] * self._norm(L["ple"]["post"], p + "ple_post", y)
        if L["sn_out"] is not None:
            x = self._norm(L["sn_out"], p + "sn_out", x)
        return x * L["out_scale"] if L["out_scale"] != 1 else x

    # --- the model --------------------------------------------------------------------------------------
    def new_cache(self, max_len=None):
        return [{} for _ in range(self.c["L"])]

    def step(self, ids, cache, pos):
        c, w = self.c, self.w
        x = w["embed.weight"][torch.tensor(ids)]
        if c["max_pos"]:
            x = x + w["pos"][pos:pos + len(ids)]
        if c["emb_norm"] is not None:
            x = self._norm(c["emb_norm"], "emb_norm", x)
        pli = None
        if c.get("ple"):                                  # pli = rms(P x0) w + E[ids], per layer
            dp, T = c["ple"]["dim"], x.shape[0]
            pp = (x @ w["ple_P.weight"].T).view(T, c["L"], dp)
            pli = _rms(pp, c["ple"]["eps"]) * w["ple_w"] + w["ple_E"][torch.tensor(ids)].view(T, c["L"], dp)
        reg = {}
        for i, L in enumerate(c["layers"]):
            x = self._layer(i, L, x, cache[i], pos, reg, pli)
        out = self._lin("lm_head", self._norm(c["norm"], "norm", x))
        if c["final_cap"] is not None:
            out = c["final_cap"] * torch.tanh(out / c["final_cap"])
        return out.numpy()


def load_model(run_dir):
    return UCoreModel(run_dir)
