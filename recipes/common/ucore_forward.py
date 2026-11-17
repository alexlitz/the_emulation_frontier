"""The universal host's forward over a small array backend, for the frameworks whose arrays are numpy-like
(JAX, TensorFlow, tinygrad). It is recipes/frameworks/pytorch/universal.py (checked against the torch
UniversalElephantine on every stock-loader case) with torch calls replaced by backend calls `B.*`.

Every position-dependent input (RoPE angles, masks, cache writes) is computed in numpy and passed as data,
and the KV cache is preallocated to the generation length, so every decode step has the same shapes.
Inlined into the JAX / TensorFlow / tinygrad recipes."""
import math

import numpy as np

from recipes.common.runtime import Generator, load_run_dir  # tef:inline


class UCoreForward(Generator):
    """B: a backend (see recipes/frameworks/<fw>/universal.py) with linear (x @ w.T), mean_sq (the mean of x^2
    over the last axis, kept), asarray, numpy, zeros, cat, stack, reshape, transpose, sum, mean, max, exp, tanh,
    sigmoid, silu, softplus, gelu, mish, relu, sqrt, rsqrt, clip, softmax, einsum, take, topk, one_hot,
    take_along_last, where_pos, realize."""
    B = None

    def __init__(self, run_dir):
        cfg, sd = load_run_dir(run_dir)
        if cfg.get("format") != "tef-ucore":
            raise ValueError("not a tef-ucore run directory")
        self.c = cfg
        self.w = {k: self.B.asarray(np.ascontiguousarray(v, dtype=np.float32)) for k, v in sd.items()}
        self.npw = {k: v for k, v in sd.items() if k.endswith("inv_freq")}

    # --- pieces ---------------------------------------------------------------------------------------
    def _lin(self, name, x):
        y = self.B.linear(x, self.w[name + ".weight"])
        b = self.w.get(name + ".bias")
        return y if b is None else y + b

    def _rms(self, x, eps):
        B = self.B
        return x * B.rsqrt(B.mean_sq(x) + eps)

    def _norm(self, spec, name, x):
        B = self.B
        if spec["ident"]:
            return x
        if spec["centre"] != 0:
            x = x - spec["centre"] * B.mean(x, -1, True)
        y = self._rms(x, spec["eps"]) * self.w[name + ".weight"]
        b = self.w.get(name + ".bias")
        return y if b is None else y + b

    def _qk_norm(self, s, p, t, i):
        B = self.B
        coef = s["q_coef"] if i == 0 else s["k_coef"]
        if coef[1] == 0:
            return coef[0] * t
        dims = (-2, -1) if s["qk_full"] else (-1,)
        red = lambda v: B.sum(B.sum(v, -1, True), -2, True) if s["qk_full"] else B.sum(v, -1, True)
        n = t - red(t) / float(t.shape[-1] * (t.shape[-2] if s["qk_full"] else 1)) if s["qk_centre"][i] else t
        width = s["qk_width"] * (t.shape[-2] if s["qk_full"] else 1)
        nw = self.w[p + ("q_nw" if i == 0 else "k_nw")]
        n = n * B.rsqrt(red(n * n) / width + s["qk_eps"][i]) * nw
        if s["qk_centre"][i]:
            n = n + self.w[p + ("q_nb" if i == 0 else "k_nb")]
        return coef[0] * t + coef[1] * n

    @staticmethod
    def _rot_half(B, x):
        h = x.shape[-1] // 2
        return B.cat((-x[..., h:], x[..., :h]), -1)

    def _attn(self, s, p, x, cache, pos, tab):
        B, c, w = self.B, self.c, self.w
        T, H, hd = x.shape[0], c["heads"], c["head_dim"]
        n_max = tab["n"]
        if s["Dn"]:
            n = B.linear(x, w[p + "pn.weight"])
            n = n * B.rsqrt((n * n) @ w[p + "n_avg"] + s["n_eps"]) * w[p + "n_w"]
            x = B.cat((x, n), -1)
        q = B.reshape(self._lin(p + "q_proj", x), (T, H, hd))
        k = B.reshape(self._lin(p + "k_proj", x), (T, H, hd))
        v = B.reshape(self._lin(p + "v_proj", x), (T, H, hd))
        if s["v_norm"]:
            v = v * B.rsqrt(B.sum(v * v, -1, True) / s["qk_width"] + s["v_eps"]) * w[p + "v_nw"]
        if s["rope"]:
            inv = self.npw[p + "inv_freq"]
            ang = np.arange(pos, pos + T, dtype=np.float32)[:, None] * inv[None]
            cos = B.asarray(np.concatenate((np.cos(ang), np.cos(ang)), -1)[:, None].astype(np.float32))
            sin = B.asarray(np.concatenate((np.sin(ang), np.sin(ang)), -1)[:, None].astype(np.float32))
            rope = lambda t: t * cos + self._rot_half(B, t) * sin
        else:
            rope = lambda t: t
        if s["qk_post"]:
            q, k = rope(q), rope(k)
        q, k = self._qk_norm(s, p, q, 0), self._qk_norm(s, p, k, 1)
        if s["clip"] is not None:
            cl = s["clip"]
            q, k, v = B.clip(q, -cl, cl), B.clip(k, -cl, cl), B.clip(v, -cl, cl)
        if not s["qk_post"]:
            q, k = rope(q), rope(k)
        if s.get("qtemp"):
            beta, M = s["qtemp"]
            t_ = np.arange(pos, pos + T, dtype=np.float64)
            q = q * B.asarray((1 + beta * np.log1p(np.floor(t_ / M))).astype(np.float32)[:, None, None])
        reg = tab["reg"]
        if s.get("kv_read"):                                         # the KV register: an earlier layer's cache
            K, V = reg[s["kv_slot"]]
        else:
            if "k" not in cache:
                cache["k"] = B.zeros((n_max, H, hd))
                cache["v"] = B.zeros((n_max, H, hd))
            place = tab["place"]                                     # (n, T) one-hot: cache row <- token
            K = cache["k"] + B.einsum("st,thd->shd", place, k)
            V = cache["v"] + B.einsum("st,thd->shd", place, v)
            cache["k"], cache["v"] = K, V = B.realize(K), B.realize(V)
            if s.get("kv_write"):
                reg[s["kv_slot"]] = (K, V)
        sc = B.einsum("thd,shd->hts", q, K) * s["scale"]
        if s["cap"] is not None:
            sc = s["cap"] * B.tanh(sc / s["cap"])
        tt = np.arange(pos, pos + T)[:, None]
        ss = np.arange(n_max)[None]
        if s["alibi"]:
            sc = sc + B.reshape(w[p + "alibi"], (H, 1, 1)) * B.asarray((ss - tt).astype(np.float32))
        win = math.inf if s["window"] is None else s["window"]
        ok = (((tt - ss) <= win) | (ss < s["n_sink"])) & (ss <= tt)
        sc = sc + B.asarray(np.where(ok, 0.0, -np.inf).astype(np.float32))
        if s["null"]:
            nl = B.reshape(w[p + "null_logit"], (H, 1, 1)) + B.zeros((H, T, 1))
            sc = B.cat((sc, nl), -1)
            pr = B.softmax(sc, -1)[..., :n_max]
        else:
            pr = B.softmax(sc, -1)
        o = B.reshape(B.einsum("hts,shd->thd", pr, V), (T, H * hd))
        if s["o_coef"][1] != 0:
            oc = s["o_coef"]
            o = oc[0] * o + oc[1] * o * B.rsqrt((o * o) @ w[p + "o_avg"] + s["o_eps"]) * w[p + "o_w"]
        g = s["gate_coef"]
        if s.get("gate_sp"):
            gx = self._lin(p + "g_proj", x)
            o = o * (g[0] + g[1] * B.sigmoid(gx) + s["gate_sp"] * B.softplus(gx))
        elif g[1] != 0:
            o = o * (g[0] + g[1] * B.sigmoid(self._lin(p + "g_proj", x)))
        elif g[0] != 1:
            o = o * g[0]
        return self._lin(p + "o_proj", o)

    def _ssm(self, s, p, x, cache, pos):
        B, w = self.B, self.w
        T = x.shape[0]
        H, dk, dv, K = s["H"], s["dk"], s["dv"], s["K"]
        u = self._lin(p + "in_proj", x)                                   # (T, D)
        if "conv" not in cache:
            cache["conv"] = B.zeros((K - 1, s["D"])) if K > 1 else None
            cache["S"] = B.zeros((H, dv, dk))
        up = B.cat((cache["conv"], u), 0) if K > 1 else u
        if K > 1:
            cache["conv"] = B.realize(up[up.shape[0] - (K - 1):])
        cw = w[p + "conv_w"]
        cv = w[p + "conv_b"]
        for j in range(K):
            cv = cv + cw[:, j] * up[j:j + T]
        cv = w[p + "act_silu"] * B.silu(cv) + w[p + "act_lin"] * cv
        r = B.cat((u, cv), -1)
        if s["Dn"]:
            n = self._lin(p + "pn", r)
            r = B.cat((r, n * B.rsqrt((n * n) @ w[p + "n_avg"] + s["n_eps"]) * w[p + "n_w"]), -1)
        if s["Dh"]:
            r = B.cat((r, B.sigmoid(self._lin(p + "ph", r))), -1)
        q = B.reshape(self._lin(p + "pq", r), (T, H, dk))
        k = B.reshape(self._lin(p + "pk", r), (T, H, dk))
        if s["l2_eps_out"] != 0:
            l2 = lambda t: t / (B.sqrt(B.sum(t * t, -1, True) + s["l2_eps"]) + s["l2_eps_out"])
        else:
            l2 = lambda t: t * B.rsqrt(B.sum(t * t, -1, True) + s["l2_eps"])
        q = s["q_coef"][0] * q + s["q_coef"][1] * l2(q)
        k = s["k_coef"][0] * k + s["k_coef"][1] * l2(k)
        v = B.reshape(self._lin(p + "pv", r), (T, H, dv))
        sp = B.softplus(B.reshape(self._lin(p + "p1", r), (T, H, dk)))
        if s["dt_min"] > 0:
            sp = B.clip(sp, s["dt_min"], None)
        loga = -(w[p + "A1"] * sp + w[p + "A2"] * B.sigmoid(B.reshape(self._lin(p + "p2", r), (T, H, dk))) + w[p + "A3"])
        dt = sp[..., 0]                                                   # (T, H)
        a = w[p + "mem"] * B.exp(loga)                                    # (T, H, dk)
        sb, si = B.sigmoid(self._lin(p + "pb", r)), B.sigmoid(self._lin(p + "pi", r))
        gamma = w[p + "cg"] * sb + w[p + "cg0"]
        a0 = a[..., 0]
        lru = B.sqrt(B.clip(1 - a0 * a0, 0, None))
        first = B.asarray((np.arange(pos, pos + T) == 0).astype(np.float32)[:, None])
        lru = lru + w[p + "reset"] * first * (1 - lru)
        c6 = w[p + "cw"]
        wr = (c6[0] * dt + c6[1] * sb + c6[2] * B.exp(self._lin(p + "pe", r)) + c6[3] * lru * si + c6[4]
              + c6[5] * dt * sb)
        S = cache["S"]
        ys = []
        for t in range(T):
            kt = B.reshape(k[t], (H, 1, dk))
            S = S * B.reshape(a[t], (H, 1, dk))
            Sk = B.sum(S * kt, -1, True)                                  # (H, dv, 1)
            S = S - B.reshape(gamma[t], (H, 1, 1)) * Sk * kt
            S = S + B.reshape(wr[t], (H, 1, 1)) * B.reshape(v[t], (H, dv, 1)) * kt
            S = B.realize(S)
            ys.append(B.sum(S * B.reshape(q[t], (H, 1, dk)), -1))
        cache["S"] = S
        y = B.stack(ys, 0) + w[p + "Dskip"] * v                           # (T, H, dv)
        y = B.reshape(y, (T, H * dv)) @ w[p + "mix"]
        z = self._lin(p + "pz", r)
        g = w[p + "gate"]
        G = g[0] * B.silu(z) + g[1] * B.sigmoid(z) + g[2] * B.gelu(z, True) + g[3] * z + g[4]
        nrm = lambda t: t * B.rsqrt((t * t) @ w[p + "avg"] + s["norm_eps"]) * w[p + "norm_w"] + w[p + "norm_b"]
        oc = w[p + "out_coef"]
        out = oc[0] * nrm(y) * G + oc[1] * nrm(y * G) + oc[2] * y * G
        return self._lin(p + "out_proj", out)

    def _act(self, a, g, u):
        B = self.B
        if a["relu2"]:
            r = B.relu(u)
            return r * r
        if a["relu"]:
            return B.relu(u)
        if a["gelu"] or a["geglu"]:
            return B.gelu(u, a["gelu_tanh"]) if a["gelu"] else B.gelu(g, a["gelu_tanh"]) * u
        if a["mish"]:
            return B.mish(g) * u
        if a.get("xielu"):
            ap, an, beta, eps = a["xielu"]
            neg = (B.exp(B.clip(u, None, eps)) - 1 - u) * an + beta * u
            return B.where_pos(u, ap * u * u + beta * u, neg, 0.0)
        if a["c_clip"] != 0:
            lg, lu = a["clip"]
            if lg is not None:
                g = g + a["c_clip"] * (B.clip(g, None, lg) - g)
            if lu is not None:
                u = u + a["c_clip"] * (B.clip(u, -lu, lu) - u)
        u = u + a["up_shift"]
        return B.silu(g) * u

    def _ffn(self, s, p, x, xe=None, xg=None, parts=False):
        """routed(xe) (router on xg) + shared(x) * gate; parts: (routed, gated shared) (Gemma-4's split FFN)."""
        B, w, a = self.B, self.w, s["act"]
        xs, xe, xg = x, (x if xe is None else xe), (x if xg is None else xg)
        out = routed = shared = None
        r = s["routed"]
        if r is not None:
            E = r["E"]
            x = xe
            logits = self._lin(p + "router", xg)                         # (T, E)
            if r["router_cap"] is not None:
                logits = r["router_cap"] * B.tanh(logits / r["router_cap"])
            sc_ = r["score_c"]
            sm = B.softmax(logits, -1)
            pr = sm if sc_ == 0 else (1 - sc_) * sm + sc_ * B.sigmoid(logits)
            sel = (logits if r["sel_logit"] else pr) + w[p + "sel_bias"]
            G, kg = int(r["grp"][0]), int(r["grp"][1])
            if G > 1 and kg < G:
                T = x.shape[0]
                sg = B.reshape(sel, (T, G, E // G))
                if r["grp"][2] != 0:
                    gs = B.max(sg, -1)
                else:
                    gs = B.sum(B.topk(sg, 2)[0], -1)
                gi = B.topk(gs, kg)[1]                                    # (T, kg)
                keep = B.sum(B.one_hot(gi, G), -2)                        # (T, G)
                keep = B.reshape(B.reshape(keep, (T, G, 1)) + B.zeros((T, G, E // G)), (T, E))
                sel = B.where_pos(keep, sel, B.zeros((T, E)) + float(r["grp"][3]))
            topv_sel, topi = B.topk(sel, r["top_k"])
            topv = B.take_along_last(pr, topi)
            topv = (topv + r["renorm"] * (topv / (B.sum(topv, -1, True) + r["renorm_eps"]) - topv)) * r["route_scale"]
            oh = B.one_hot(topi, E)                                       # (T, k, E)
            topv = topv * B.sum(oh * w[p + "exp_scale"], -1)
            wts = B.sum(oh * B.reshape(topv, tuple(topv.shape) + (1,)), -2)      # (T, E), zero off the top-k
            h = self._act(a, B.einsum("efd,td->tef", w[p + "w_gate"], x) + w[p + "b_gate"],
                          B.einsum("efd,td->tef", w[p + "w_up"], x) + w[p + "b_up"])
            eo = B.einsum("edf,tef->ted", w[p + "w_down"], h) + w[p + "b_down"]
            out = routed = B.einsum("ted,te->td", eo, wts)
        x = xs
        sh = s["shared"]
        if sh is not None:
            u = self._lin(p + "s_up", x)
            g = self._lin(p + "s_gate", x) if (p + "s_gate.weight") in w else u
            y = self._lin(p + "s_down", self._act(a, g, u))
            c = sh["s_coef"]
            if c[1] != 0:
                y = y * (c[0] + c[1] * B.sigmoid(B.linear(x, w[p + "s_router.weight"])))
            elif c[0] != 1:
                y = y * c[0]
            shared = y
            out = y if out is None else out + y
        return (routed, shared) if parts else out

    def _layer(self, i, L, x, cache, pos, tab):
        p = "layers.%d." % i

        def post(j, o, nm):
            pc = L["post_coef"][j]
            r = pc[0] * o if pc[0] != 1 else o
            return r + pc[1] * self._norm(L[nm], p + nm, o) if pc[1] != 0 else r
        if L["sn_in"] is not None:
            x = self._norm(L["sn_in"], p + "sn_in", x)
        a = s = None
        if L["attn"] is not None:
            a = post(0, self._attn(L["attn"], p + "attn.", self._norm(L["norm_a"], p + "norm_a", x), cache, pos, tab),
                     "post_a")
        if L["ssm"] is not None:
            xs = x if a is None else x + L["seq"] * a
            s = post(1, self._ssm(L["ssm"], p + "ssm.", self._norm(L["norm_s"], p + "norm_s", xs), cache, pos), "post_s")
        x_in = x
        x = x if L["res_c"][0] == 1 else L["res_c"][0] * x
        if a is not None:
            x = x + a
        if s is not None:
            x = x + s
        xf = x_in if L["par"] != 0 else x
        if L["res_c"][1] != 1:
            x = L["res_c"][1] * x
        B = self.B
        if L["ffn"] is not None and L.get("ffn_split"):
            nf = lambda nm: self._norm(L[nm], p + nm, xf)
            r_, s_ = self._ffn(L["ffn"], p + "ffn.", nf("norm_f"), xe=nf("norm_f2"), xg=nf("norm_r"), parts=True)
            y = None
            for part, nm in ((s_, "post_f1"), (r_, "post_f2")):
                if part is not None:
                    yy = self._norm(L[nm], p + nm, part)
                    y = yy if y is None else y + yy
            x = x + post(2, y, "post_f")
        elif L["ffn"] is not None:
            x = x + post(2, self._ffn(L["ffn"], p + "ffn.", self._norm(L["norm_f"], p + "norm_f", xf)), "post_f")
        if L.get("ple"):                                                  # per-layer embedding input
            g = B.gelu(B.linear(x, self.w[p + "ple_gate.weight"]), L["ple"]["tanh"])
            y = B.linear(g * tab["pli"][:, i], self.w[p + "ple_proj.weight"])
            x = x + L["ple"]["c"] * self._norm(L["ple"]["post"], p + "ple_post", y)
        if L["sn_out"] is not None:
            x = self._norm(L["sn_out"], p + "sn_out", x)
        x = x * L["out_scale"] if L["out_scale"] != 1 else x
        return self.B.realize(x)

    # --- the model --------------------------------------------------------------------------------------
    def new_cache(self, max_len=None):
        return {"n": int(max_len or 1), "layers": [{} for _ in range(self.c["L"])]}

    def step(self, ids, cache, pos):
        B, c, w = self.B, self.c, self.w
        T, n = len(ids), cache["n"]
        if pos + T > n:
            raise ValueError("cache too short")
        place = (np.arange(n)[:, None] == np.arange(pos, pos + T)[None]).astype(np.float32)
        tab = {"n": n, "place": B.asarray(place), "reg": {}, "pli": None}
        x = B.take(w["embed.weight"], np.asarray(ids, np.int32))
        if c["max_pos"]:
            x = x + w["pos"][pos:pos + T]
        if c["emb_norm"] is not None:
            x = self._norm(c["emb_norm"], "emb_norm", x)
        if c.get("ple"):                                                  # pli = rms(P x0) w + E[ids], per layer
            dp = c["ple"]["dim"]
            pp = B.reshape(B.linear(x, w["ple_P.weight"]), (T, c["L"], dp))
            pe = B.reshape(B.take(w["ple_E"], np.asarray(ids, np.int32)), (T, c["L"], dp))
            tab["pli"] = self._rms(pp, c["ple"]["eps"]) * w["ple_w"] + pe
        for i, L in enumerate(c["layers"]):
            x = self._layer(i, L, x, cache["layers"][i], pos, tab)
        out = self._lin("lm_head", self._norm(c["norm"], "norm", x))
        if c["final_cap"] is not None:
            out = c["final_cap"] * B.tanh(out / c["final_cap"])
        return B.numpy(out)
