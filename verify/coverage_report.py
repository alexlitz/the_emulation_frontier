"""The coverage dashboard: the 15 metrics of verify/COVERAGE_PLAN.md, each computed from its
source of truth.

  python -m verify.coverage_report                     # write verify/COVERAGE.md and verify/coverage.json
  python -m verify.coverage_report --update-baseline   # also write verify/coverage_baseline.json
  python -m verify.coverage_report --run-z3            # run the Z3 module first and cache its outputs (M9, M10)
  python -m verify.coverage_report --only M3,M5        # a subset (prints; writes nothing)

Some sources still live on unmerged branches. SOURCES (below) lists,
per source, the places to look, in order: a path (relative to the repo root, or absolute), or a file or
directory at a branch tip (`git show <branch>:<path>`). The first place that has every file the metric
needs is used. Local paths come first, so a source is read from disk as soon as its branch is merged. A
metric whose source is found nowhere reports "n/a (source missing: ...)" instead of failing.

tests/test_coverage_ratchet.py holds every available metric at or above verify/coverage_baseline.json,
and holds M1, M10 and M15 at 100 % once their source is read from disk (merged); a source read from a
branch is provisional, and those hard gates skip with the reason. M3 and M6 were to be hard gates too; their
remaining gaps need new Lean theorems / constructions (after the release), so they are ratchets whose
uncovered items may only shrink (DEFERRED_GATES).

Nothing here changes the relation; the draft (M14) is only read.
"""
from __future__ import annotations

import argparse
import ast
import fractions
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFT = os.environ.get("TEF_DRAFT")          # the post's draft (markdown); M14 is skipped when it is not set
OUT_MD = os.path.join(ROOT, "verify", "COVERAGE.md")
OUT_JSON = os.path.join(ROOT, "verify", "coverage.json")
BASELINE = os.path.join(ROOT, "verify", "coverage_baseline.json")
Z3_CACHE = "verify/z3_results.json"

# --- where each source lives -------------------------------------------------------------------------------
# name -> candidates, tried in order: ("path", p) with p relative to ROOT or absolute; ("git", branch, p).
# A candidate is used if it has every file the metric asks for. Edit this table when a branch is merged,
# renamed or moved.

SOURCES = {
    "sweep":        [("path", "verify/sweep"), ("git", "nocode-hunt", "verify/sweep"),
                     ("git", "code-diff-sweep", "verify/sweep")],
    "edge_just":    [("path", "verify/EDGE_JUSTIFICATION.md"), ("git", "z3-final", "verify/EDGE_JUSTIFICATION.md"),
                     ("git", "z3-relation", "verify/EDGE_JUSTIFICATION.md")],
    # the Lean library and its index (proofs/lean/check.sh builds and audits it)
    "proofs":       [("path", "proofs/PROOFS.md")],
    "lean":         [("path", "proofs/lean/EmulationProofs")],
    "bridges":      [("path", "verify/BRIDGES.md"), ("git", "bridges-2", "verify/BRIDGES.md"),
                     ("git", "lean-bridges", "verify/BRIDGES.md")],
    "spot_results": [("path", "verify/spot/results.jsonl"), ("git", "spot-checks", "verify/spot/results.jsonl")],
    "z3_cache":     [("path", Z3_CACHE), ("git", "z3-final", Z3_CACHE), ("git", "z3-relation", Z3_CACHE)],
    "z3_module":    [("path", "verify/z3_relation.py"), ("git", "z3-final", "verify/z3_relation.py"),
                     ("git", "z3-relation", "verify/z3_relation.py")],
    "lipschitz":    [("path", "verify/lipschitz/out/summary.json"),
                     ("git", "lipschitz", "verify/lipschitz/out/summary.json")],
    "zoo_browser":  [("path", "tests/test_zoo_browser.py"), ("git", "zoo-tests", "tests/test_zoo_browser.py"),
                     ("git", "zoo-browser", "tests/test_zoo_browser.py")],
    "zoo_template": [("path", "graph/templates/zoo.html")],
    # live metrics run the code of a branch (its tree extracted with `git archive`) until it is merged, then
    # this checkout ("path", ""). A branch already contained in HEAD is skipped.
    "emulate_code": [("git", "emulate-v3", ""), ("path", "")],
    "loader_code":  [("git", "host-gaps", ""), ("path", "")],
    "spot_code":    [("git", "spot-checks", ""), ("path", "")],         # M6: the reduction catalog and its types
    "check_post":   [("path", "graph/check_post.py")],
    "post_numbers": [("path", "dist/post_numbers.json")],
    "draft":        [("path", DRAFT)] if DRAFT else [],
    "review_tests": [("path", "tests/test_review_findings.py"), ("git", "tooling-fixes", "tests/test_review_findings.py"),
                     ("git", "code-review", "tests/test_review_findings.py")],
}

# M1, M10, M15 must reach 100 % once their source is merged (read from disk)
HARD_GATED = ("M1", "M10", "M15")
# M3 and M6 were planned as hard gates. On the release relation (baseline c7d0381) their gaps need new Lean theorems or
# constructions, post-publication work: they are ratchets (coverage_baseline.json), and tests/test_coverage_ratchet.py
# holds their uncovered items to these lists (a gap may close, no new one may open). Closing one: drop it here.
DEFERRED_GATES = {
    # M3, headline rules proved (164 / 189): the 25 rules still justified by CODE / CATALOG, not by a Lean theorem
    "M3": (
        "gen: ar -> prefix (CODE)", "gen: block -> prefix (CODE)", "gen: diffusion -> prefix (CODE)",
        "gen: prefix -> block (CODE)", "head: embedding -> lm (CODE)", "head: score -> lm (CODE)",
        "qknorm: head_free -> head_free_gain (CODE)", "canon: host has it, target lacks it (CATALOG)",
        "doc_mask: host has it, target lacks it (CATALOG)", "enc_input: host has it, target lacks it (CODE)",
        "exit_heads: host has it, target lacks it (CODE)", "fake_quant: target has it, host lacks it (CODE)",
        "loop_bnorm_cfg: host has it, target lacks it (CODE)", "mem_attn: host has it, target lacks it (CATALOG)",
        "mla_q_raw: host has it, target lacks it (CATALOG)", "q_temp: host has it, target lacks it (CODE)",
        "res_gain: host has it, target lacks it (CATALOG)", "res_gate: host has it, target lacks it (CATALOG)",
        "rope_warp: host has it, target lacks it (CODE)", "vha: host has it, target lacks it (CATALOG)",
        "xsa_scaled: host has it, target lacks it (CATALOG)", "config: softcap values (a null cap is no cap) (CODE)",
        "config: a q / k norm switched off by its flag (CODE)",
        "HadamardMLP target on the same HadamardMLP (equal n) (CODE)", "recurrent RC_EQ: identical recurrent dims (CODE)"),
    # M6, headline reduction types with an end-to-end instance (158 / 164): two whose catalog instance is a standing
    # FINDING (xfail: fm_phimoe_dup2, p5_act_width_gelu_oai) and four with no catalog entry (no construction yet)
    "M6": ("moe:dup:sparsemixer:m", "slot:act:GeLU->swigluoai", "port:alibi_labs", "port:comp_ln", "port:comp_par",
           "slot:ffn:glu->vanilla"),
}


class Missing(Exception):
    """A source is found nowhere: the metric reports n/a."""


def _git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    return r.returncode, r.stdout


class Source:
    """One resolved place: a directory or file on disk, or a tree / blob at a branch tip."""

    def __init__(self, name, cand):
        self.name, self.cand = name, cand
        self.kind = cand[0]
        if self.kind == "path":
            p = cand[1]
            self.root = p if os.path.isabs(p) else os.path.join(ROOT, p)
            self.label = cand[1] or "this checkout"
        else:
            self.branch, self.root = cand[1], cand[2]
            code, sha = _git("rev-parse", "--verify", "--quiet", "--short", self.branch)
            self.label = f"git:{self.branch}:{cand[2]}" + (f" @ {sha.decode().strip()}" if code == 0 else
                                                           " (no such branch)")

    @property
    def merged(self):
        """read from disk inside this repo (not a branch, not another worktree)"""
        r = os.path.abspath(self.root)
        return self.kind == "path" and (r == ROOT or r.startswith(ROOT + os.sep))

    def _p(self, rel):
        if not rel:
            return self.root
        return os.path.join(self.root, rel) if self.kind == "path" else "/".join(x for x in (self.root, rel) if x)

    def exists(self, rel=""):
        if self.kind == "path":
            return os.path.exists(self._p(rel))
        if _git("rev-parse", "--verify", "--quiet", self.branch)[0]:
            return False
        if _git("merge-base", "--is-ancestor", self.branch, "HEAD")[0] == 0:
            return False                                # merged: read it from disk instead
        return _git("cat-file", "-e", f"{self.branch}:{self._p(rel)}")[0] == 0

    def read(self, rel=""):
        if self.kind == "path":
            with open(self._p(rel), encoding="utf-8") as f:
                return f.read()
        code, out = _git("show", f"{self.branch}:{self._p(rel)}")
        if code:
            raise Missing(f"{self.label}/{rel}")
        return out.decode("utf-8")

    def listdir(self, rel=""):
        if self.kind == "path":
            return sorted(os.listdir(self._p(rel)))
        out = _git("ls-tree", "--name-only", f"{self.branch}:{self._p(rel)}")[1].decode()
        return sorted(out.split())


_USED = []          # the sources the current metric read (compute() resets it)


def source(name, need=("",)):
    """The first candidate of SOURCES[name] that has every relative path in `need`; else Missing."""
    tried = []
    for cand in SOURCES[name]:
        s = Source(name, cand)
        if all(s.exists(r) for r in need):
            _USED.append(s)
            return s
        tried.append(s.label)
    raise Missing(f"{name}: " + ("; ".join(tried) or "not configured"))


# --- markdown helpers ----------------------------------------------------------------------------------------

def md_tables(text):
    """[(heading, header cells, rows of cells)] for every pipe table, under the nearest heading."""
    out, head, lines = [], "", text.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            head = ln.lstrip("#").strip()
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):
            hdr = _cells(ln)
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            out.append((head, hdr, rows))
            continue
        i += 1
    return out


def _cells(line):
    parts = re.split(r"(?<!\\)\|", line.strip())
    return [c.strip() for c in parts[1:-1]]


def table(text, heading_re, col_re=None):
    """rows (as dicts keyed by lower-case header) of the first table under a heading matching heading_re"""
    for head, hdr, rows in md_tables(text):
        if re.search(heading_re, head, re.I) and (col_re is None or any(re.search(col_re, h, re.I) for h in hdr)):
            keys = [h.lower() for h in hdr]
            return [dict(zip(keys, r)) for r in rows]
    return None


def ticks(s):
    return re.findall(r"`([A-Za-z_][A-Za-z0-9_.']*)`", s or "")


def metric(num, den, src, detail="", items=None):
    return {"numerator": num, "denominator": den, "source": src, "status": "ok", "detail": detail,
            **({"items": items} if items else {})}


# --- M1, M2: the code-diff sweep --------------------------------------------------------------------------------

def m1():
    """types whose code diff vs parent is fully classified (every hunk carrying the type has a label) /
    every model type in the snapshot with code; types with no code anywhere are the plan's stated
    exclusion, counted in the detail."""
    s = source("sweep", ("inventory.json", "hunks.jsonl"))
    inv = json.loads(s.read("inventory.json"))["types"]
    labels = {"MAPPED", "IRRELEVANT", "UNMAPPED"}
    bad, n_hunks, unmapped = {}, 0, 0
    for ln in s.read("hunks.jsonl").splitlines():
        if not ln.strip():
            continue
        h = json.loads(ln)
        n_hunks += 1
        unmapped += h.get("label") == "UNMAPPED"
        if h.get("label") not in labels:
            for t in h.get("types") or []:
                bad[t] = bad.get(t, 0) + 1
    no_code = sorted(t for t, v in inv.items() if v.get("no_code") or not v.get("code"))
    with_code = [t for t in inv if t not in set(no_code)]
    done = [t for t in with_code if t not in bad]
    return metric(len(done), len(with_code), s.label,
                  f"{len(inv)} snapshot model types; {len(no_code)} with no code anywhere (excluded, the plan's "
                  f"'no code' list); {n_hunks:,} hunks, {unmapped:,} labelled UNMAPPED (classified: a sweep finding)"
                  + (f"; unclassified hunks in {len(bad)} types" if bad else ""),
                  items=sorted(bad)[:50])


def _handled_rule(s, models):
    s = (s or "").strip()
    if s in ("-", "") :
        return models == 0
    low = s.lower()
    return not (low.startswith(("none", "unclear", "?")) or "no theorem" in low)


def m2():
    """feature variants with read ∧ (host branch ∨ wall) ∧ rule as target ∧ rule as host / every variant in
    the sweep's coverage matrix (verify/sweep/coverage.json)."""
    s = source("sweep", ("coverage.json",))
    rows = json.loads(s.read("coverage.json"))
    ok, fails = 0, {"read": 0, "host": 0, "target": 0, "as host": 0}
    bad = []
    for r in rows:
        n = r.get("models", 0)
        uh = (r.get("universal_host") or "").lower()
        rt, rh = r.get("rule_target", ""), r.get("rule_host", "")
        read = bool(r.get("read_by_classifier"))
        host = (uh.startswith(("yes", "abstract", "n/a")) or rt.lower().startswith("wall")
                or rh.lower().startswith("wall"))
        tgt, hst = _handled_rule(rt, n), _handled_rule(rh, n)
        for k, v in (("read", read), ("host", host), ("target", tgt), ("as host", hst)):
            fails[k] += not v
        if read and host and tgt and hst:
            ok += 1
        else:
            bad.append(r["key"])
    return metric(ok, len(rows), s.label,
                  "variants failing: " + ", ".join(f"{k} {v}" for k, v in fails.items())
                  + " (a rule counts if it states a status and does not say 'no theorem' / 'none' / 'unclear'; "
                    "'-' only for 0-model abstract values)", items=bad[:50])


# --- M3: rules proved -----------------------------------------------------------------------------------------

def m3():
    """allowed headline rules with a Lean theorem or a recorded measurement / all allowed headline rules, from
    EDGE_JUSTIFICATION.md (generated from the code by verify/z3_relation.py). With the verdict column (z3-final
    on): rules whose modes include payable (the headline relation), over the slot, ZERO_EDGES, feature and
    capacity tables; justified = verdict PROVED or MEASURED (UNJUSTIFIED and REFUTED are not). The older format
    without verdicts: slot transitions + capacity rules, justified if the theorem column names a theorem and
    the note does not say REFUTED."""
    s = source("edge_just")
    text = s.read()
    tabs = [(h, [dict(zip([c.lower() for c in hdr], r)) for r in rows]) for h, hdr, rows in md_tables(text)
            if h.startswith(("Every ", "ZERO_EDGES")) and "verdict" in [c.lower() for c in hdr]]
    if tabs:
        num = den = 0
        per, bad, all_ = [], [], {}
        for h, rows in tabs:
            head = [r for r in rows if "payable" in r.get("modes", "")]
            ok = [r for r in head if r["verdict"].split()[0] in ("PROVED", "MEASURED")]
            num, den = num + len(ok), den + len(head)
            per.append(f"{h.split(' (')[0].replace('Every allowed ', '').replace('Every ', '')} {len(ok)}/{len(head)}")
            bad += [f"{r['rule']} ({r['verdict']})" for r in head if r not in ok]
            for r in rows:
                all_[r["verdict"]] = all_.get(r["verdict"], 0) + 1
        return metric(num, den, s.label, "headline rules: " + ", ".join(per) + "; every mode (mixgap-only rules "
                      "too): " + ", ".join(f"{k} {v}" for k, v in sorted(all_.items())), items=bad)
    slot = table(text, r"^Every allowed slot transition")
    cap = table(text, r"^Every capacity rule")
    if slot is None or cap is None:
        raise Missing(f"{s.label}: the 'Every allowed slot transition' / 'Every capacity rule' tables")

    def just(r):
        th = r.get("theorem(s)", "")
        note = r.get("note", "")
        if "MEASURED" in th or re.search(r"\bMEASURED\b", note):
            return True
        return bool(th) and "UNJUSTIFIED" not in th and th != "—" and "REFUTED" not in note

    js, jc = sum(map(just, slot)), sum(map(just, cap))
    bad = [f"{r['slot']} {r['target']} -> {r['host']}" for r in slot if not just(r)] + \
          [r["rule"] for r in cap if not just(r)]
    return metric(js + jc, len(slot) + len(cap), s.label,
                  f"slot transitions {js}/{len(slot)}, capacity rules {jc}/{len(cap)} (pre-verdict format)", items=bad)


# --- M4, M12: PROOFS.md -----------------------------------------------------------------------------------------

def _lean_names():
    """(theorem names, theorem-or-lemma names) of the Lean library, or None"""
    try:
        s = source("lean")
    except Missing:
        return None
    th, all_ = set(), set()
    pat = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+)?(theorem|lemma)\s+([A-Za-z0-9_.']+)", re.M)
    for f in s.listdir():
        if f.endswith(".lean"):
            for kind, name in pat.findall(s.read(f)):
                all_.add(name)
                if kind == "theorem":
                    th.add(name)
    return th, all_, s.label


def _theorem_count(text):
    m = re.search(r"on all ([\d,]+) theorems and lemmas", text)
    return int(m.group(1).replace(",", "")) if m else None


def m4():
    """walls labelled proved-impossible (or another PROVED / REFUTED status) or OPEN with a measurement / all
    walls in PROOFS.md's walls table."""
    s = source("proofs")
    text = s.read()
    rows = table(text, r"^Walls", r"wall")
    if rows is None:
        raise Missing(f"{s.label}: the walls table")
    ok, bad = 0, []
    for r in rows:
        st, note = r.get("status", ""), r.get("note", "")
        good = bool(re.search(r"IMPOSSIBLE|PROVED|REFUTED", st)) or (
            "OPEN" in st and re.search(r"measur", st + " " + note, re.I) is not None)
        ok += good
        if not good:
            bad.append(r[next(iter(r))])
    n = _theorem_count(text)
    ln = _lean_names()
    det = f"{len(rows) - ok} OPEN without a measurement"
    if n:
        det += f"; library: {n} theorems and lemmas (PROOFS.md header)"
    if ln:
        det += f", {len(ln[1])} declarations counted in the Lean sources ({len(ln[0])} theorems)"
    return metric(ok, len(rows), s.label, det, items=bad)


_SUFFIX = re.compile(r"_(model_bound|model_limit|layer_bound|greedy|bound|limit)$")


def _stems(names):
    out = set()
    for t in names:
        s = _SUFFIX.sub("", t)
        out.add(s)
        if "_sub_" in s:
            a, b = s.split("_sub_", 1)
            out.add(f"{a}_{b}")
        if "_to_" in s:
            out.add(s.split("_to_", 1)[0])
        out.add(s.split("_")[0])
    return out


# verify/lipschitz/measure.py's rule names -> the PROOFS.md theorem stems (see _stems) whose constants they measure
LIPSCHITZ_STEMS = {"relu_silu": ("relu_silu",), "act_swap": ("relu", "relu2", "relu_silu", "relu_gelu", "relu_mish"),
                   "alibi_rope": ("alibi_rope",), "softcap_host": ("softcap", "cap"), "mask": ("mask",),
                   "sink": ("sink",), "soft_router": ("soft_router",), "lever_qk": ("leverAttn", "epsLever"),
                   "lever_post": ("postNorm",)}


def _constants():
    """(stem -> True for every rule whose constants are measured on at least one real model, label), or None.
    verify/lipschitz/out/summary.json: model -> {"rules": {rule: {...measured constants, threshold...}}}."""
    try:
        s = source("lipschitz")
    except Missing:
        return None
    d = json.loads(s.read())
    out = {}
    for model, v in d.items():
        for rule in (v.get("rules") or {}):
            for st in LIPSCHITZ_STEMS.get(rule, (rule,)):
                out[st] = True
    return out, s.label


def m12():
    """limit rules (PROOFS.md 'Every allowed rule' rows whose status is a limit or a bound) with a per-op bound,
    a whole-model bound, a greedy threshold and measured or derived constants / all limit rules. The parts are
    found by theorem name: a row's theorems, and for each stem X (a theorem name minus _bound / _limit /
    _greedy, a_sub_b -> a_b) the library's X_bound / X_layer_bound, X_model_bound / X_model_limit, X_greedy."""
    s = source("proofs")
    text = s.read()
    rows = table(text, r"^Every allowed rule", r"rule")
    if rows is None:
        raise Missing(f"{s.label}: the 'Every allowed rule' table")
    ln = _lean_names()
    universe = set(ln[1]) if ln else set(ticks(text))
    consts = _constants()
    lim = [r for r in rows if re.search(r"\blimit\b|\bBOUND\b", r.get("status", ""), re.I)]
    parts = {"per-op bound": 0, "whole-model bound": 0, "greedy threshold": 0, "constants": 0}
    ok, items = 0, []
    for r in lim:
        names = ticks(r.get("theorem(s)", ""))
        st = _stems(names)
        has = {
            "per-op bound": any("bound" in t and "model" not in t for t in names)
            or any(f"{x}_bound" in universe or f"{x}_layer_bound" in universe for x in st),
            "whole-model bound": any("_model_" in t for t in names)
            or any(f"{x}_model_bound" in universe or f"{x}_model_limit" in universe for x in st),
            "greedy threshold": any("greedy" in t for t in names) or any(f"{x}_greedy" in universe for x in st),
            "constants": bool(consts) and any(consts[0].get(k) for k in [r["rule"], *st]),
        }
        for k, v in has.items():
            parts[k] += v
        if all(has.values()):
            ok += 1
        else:
            items.append(f"{r['rule']}: missing " + ", ".join(k for k, v in has.items() if not v))
    det = f"{len(lim)} limit / bound rules; parts present: " + ", ".join(f"{k} {v}" for k, v in parts.items())
    det += ("; constants source: " + consts[1]) if consts else \
        "; constants: no measurement source (PROOFS.md: 'The Lipschitz constants ... are hypotheses, not derived')"
    return metric(ok, len(lim), s.label + (f" + {ln[2]}" if ln else ""), det, items=items)


# --- M5: bridges -------------------------------------------------------------------------------------------------

def m5():
    """construction theorems with a test on the repo function that builds them (BRIDGES.md BRIDGED rows) / all
    construction theorems: the Lean library's `theorem` declarations (lemmas are helpers) minus the ones
    BRIDGES.md lists as not counted (helpers, impossibility theorems, negative controls)."""
    s = source("bridges")
    text = s.read()
    bridged, other, skip = set(), set(), set()
    for head, hdr, rows in md_tables(text):
        keys = [h.lower() for h in hdr]
        if "theorem" not in keys or "status" not in keys:
            continue
        for r in rows:
            d = dict(zip(keys, r))
            names, st = ticks(d["theorem"]), d["status"]
            if st.startswith("BRIDGED"):
                bridged.update(names)
            elif "not counted" in st:
                skip.update(names)
            else:
                other.update(names)
    ln = _lean_names()
    if ln:
        den = ln[0] - skip
        src = f"{s.label} + {ln[2]}"
        det = (f"{len(ln[0])} theorem declarations, {len(ln[0] & skip)} not counted by BRIDGES.md; "
               f"{len(bridged)} BRIDGED names ({len(bridged - ln[0])} not a `theorem` in the library: lemmas or "
               f"renamed), {len(other)} NO CODE / MISMATCH")
    else:
        den = bridged | other
        src = s.label + " (Lean sources missing: denominator = BRIDGES.md's counted rows)"
        det = f"{len(other)} NO CODE / MISMATCH"
    num = len(bridged & den)
    return metric(num, len(den), src, det)


# --- M6, M7: spot checks and the reduction catalog -----------------------------------------------------------------

def _passed(e):
    for k in ("pass", "passed", "ok", "verified"):
        if k in e:
            return bool(e[k])
    return str(e.get("status", e.get("result", ""))).lower() in ("pass", "passed", "ok", "verified")


def m7():
    """spot-checked edges passing / sampled (verify/spot/results.jsonl; the plan asks >= 200, stratified)."""
    s = source("spot_results")
    rows = [json.loads(ln) for ln in s.read().splitlines() if ln.strip()]
    ok = sum(map(_passed, rows))
    return metric(ok, len(rows), s.label, ("" if len(rows) >= 200 else "fewer than the 200 samples the plan asks; ")
                  + f"{len(rows) - ok} failing")


# --- M8, M11: computed live from the relation and the loaders ---------------------------------------------------------

CODE_ROOT = os.environ.get("COVERAGE_CODE_ROOT", ROOT)     # the tree whose code a live metric runs


LIVE_NEEDS = {"spot_code": ("verify/spot/reduction_types.json", "tests/test_reduction_catalog.py")}


def live(mid, key):
    """Run metric `mid` on the code of SOURCES[key]: in this process for this checkout, else in a child process
    inside the branch's tree (git archive into a temporary directory)."""
    s = source(key, LIVE_NEEDS.get(key, ("",)))
    fn = {m[0]: m[2] for m in METRICS}[mid]
    if s.kind == "path":
        r = fn.__wrapped__()
        r["source"] = f"live, {s.label}: " + r["source"]
        return r
    sha = _git("rev-parse", s.branch)[1].decode().strip()
    cache = os.path.join(ROOT, "dist", "coverage_cache", f"{mid}-{sha}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    with tempfile.TemporaryDirectory() as tmp:
        arc = subprocess.run(["git", "archive", s.branch], cwd=ROOT, capture_output=True)
        if arc.returncode:
            raise Missing(f"{s.label}: git archive failed")
        subprocess.run(["tar", "-x", "-C", tmp], input=arc.stdout, check=True)
        env = dict(os.environ, COVERAGE_CODE_ROOT=tmp, PYTHONPATH=tmp)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--live", mid], cwd=tmp, env=env,
                           capture_output=True, text=True)
        out = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        if p.returncode or not out:
            raise Missing(f"{s.label}: the live run failed: {(p.stderr.strip().splitlines() or ['?'])[-1]}")
        r = json.loads(out[-1])
    r["source"] = f"live, {s.label}: " + r["source"]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:                      # a branch tip's result never changes: cache it by commit
        json.dump(r, f)
    return r


def _code_entry(key):
    """decorator: the metric runs on SOURCES[key]'s code; fn.__wrapped__ is the in-process body"""
    def deco(body):
        def run():
            return live(run.mid, key)
        run.__wrapped__, run.__doc__, run.__name__ = body, body.__doc__, body.__name__
        return run
    return deco


def _enter_code_root():
    if CODE_ROOT not in sys.path:
        sys.path.insert(0, CODE_ROOT)
    os.chdir(CODE_ROOT)


@_code_entry("spot_code")
def m6():
    """headline reduction types with a non-skipped catalog test that builds and verifies / every headline type
    of verify/spot/reduction_types.json. Runs tests/test_reduction_catalog.py -m "not slow" (the synthetic
    entries; the meta-test requires one per type) and maps each passing entry to its types through
    verify.spot.catalog.CATALOG."""
    _enter_code_root()
    path = os.path.join(CODE_ROOT, "verify", "spot", "reduction_types.json")
    if not os.path.exists(path):
        raise Missing("verify/spot/reduction_types.json")
    T = json.load(open(path))
    head = [t["id"] for t in T if t.get("headline", True)]
    from verify.spot.catalog import CATALOG, load_all
    load_all()
    with tempfile.TemporaryDirectory() as tmp:
        xml = os.path.join(tmp, "junit.xml")
        env = dict(os.environ, PYTHONPATH=CODE_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        subprocess.run([sys.executable, "-m", "pytest", "tests/test_reduction_catalog.py", "-q", "-m", "not slow",
                        "-p", "no:cacheprovider", f"--junitxml={xml}"], cwd=CODE_ROOT, env=env, capture_output=True)
        if not os.path.exists(xml):
            raise Missing("tests/test_reduction_catalog.py: pytest produced no report")
        res = {}
        for tc in ET.parse(xml).getroot().iter("testcase"):
            kids = {c.tag for c in tc}
            m = re.search(r"\[(.*)\]$", tc.get("name", ""))
            res[m.group(1) if m else tc.get("name")] = ("failed" if kids & {"failure", "error"} else
                                                         "skipped" if "skipped" in kids else "passed")
    ok = {t for t in head if any(res.get(e.check.__name__) == "passed" for e in CATALOG.get(t, []))}
    cnt = {k: sum(v == k for v in res.values()) for k in ("passed", "skipped", "failed")}
    return metric(len(ok), len(head), "verify/spot/reduction_types.json + tests/test_reduction_catalog.py",
                  f"{len(T)} reduction types ({len(head)} headline); catalog run (not slow): "
                  + ", ".join(f"{k} {v}" for k, v in cnt.items()),
                  items=[t for t in head if t not in ok])


@_code_entry("emulate_code")
def m8():
    """headline edges (shape pairs, j != i, emul_matrix(mode='payable')) that elephantine.emulate v1 builds for the
    shapes' command repos (graph.edge_plan.cmd_repo; exact edge, both sides supported, support() is None) / all
    headline edges. The counting of graph.model_graph.zoo_data / graph.edge_plan.zoo_edges, over every edge."""
    _enter_code_root()
    import numpy as np

    from elephantine.emulate import side_support, support
    from graph import model_graph as G
    from graph.edge_plan import cmd_repo
    from graph.frontier import load
    M = load()
    rank = {m["name"]: k for k, m in enumerate(M)}                  # snapshot order = downloads (zoo_data)
    P = G._dedup_points(M)
    ok = G.emul_matrix(P, mode="payable")
    ex = G.emul_matrix(P, mode="exact")
    np.fill_diagonal(ok, False)
    configs = dict(G.iter_json_configs(G.ZOO_CONFIGS))
    repos = [cmd_repo(sorted(p["members"], key=lambda m: rank.get(m, len(rank))), configs) for p in P]
    cnt = np.array([len(p["members"]) for p in P])

    def side(i, role):
        c = configs.get(repos[i])
        return c is not None and side_support(c, role) is None

    tok = [side(i, "target") for i in range(len(P))]
    hok = [side(j, "host") for j in range(len(P))]
    build = n_ex = 0
    w_all = w_build = 0
    cache = {}
    for i, j in zip(*np.where(ok)):
        w = int(cnt[i] * cnt[j])
        w_all += w
        if not ex[i, j]:
            continue
        n_ex += 1
        if not (tok[i] and hok[j]):
            continue
        key = (repos[i], repos[j])
        if key not in cache:
            cache[key] = support(configs[repos[i]], configs[repos[j]]) is None
        if cache[key]:
            build += 1
            w_build += w
    n = int(ok.sum())
    return metric(build, n, "graph.model_graph.emul_matrix(payable) + elephantine.emulate.support",
                  f"{len(P)} shapes; {n_ex:,} of the {n:,} headline shape edges are exact; weighted by models "
                  f"(target members x host members): {w_build:,} / {w_all:,} buildable. Every headline edge, not "
                  f"only the zoo page's edge view (covering pairs + frontier hosts)")


@_code_entry("loader_code")
def m11():
    """models whose family loader runs against its own code (graph.minimal_emulator.loaded_split: loaded and
    tested) / all models in the snapshot (graph.minimal_emulator.load)."""
    _enter_code_root()
    from graph.minimal_emulator import load, loaded_split
    models = load()
    carried, loaded = loaded_split(models)
    return metric(loaded, len(models), "graph.minimal_emulator.loaded_split",
                  f"{carried:,} carried by the universal host; the rest priced by slot constructions (remote code "
                  f"we may not execute is the plan's stated exclusion)")


# --- M9, M10: the Z3 module's outputs (cached) ------------------------------------------------------------------------

MODES = ("exact", "payable", "mixgap", "capacity")


def _z3_props_from_md():
    """M9 / M10 inputs from EDGE_JUSTIFICATION.md's 'Properties of the relation' section (z3-final on), in the
    shape of the --run-z3 cache; Missing if the section is absent."""
    s = source("edge_just")
    text = s.read()
    i = text.find("## Properties of the relation")
    if i < 0:
        raise Missing(f"{s.label}: no 'Properties of the relation' section")
    sec = text[i:]
    m = re.search(r"([\d,]+) disagreements? over every ordered pair of the ([\d,]+) real shapes \((\d+) modes x "
                  r"(\d+) flag settings(, the joined host as a host column)?", sec)
    if not m:
        raise Missing(f"{s.label}: no differential sentence")
    n = int(m.group(2).replace(",", ""))
    pairs = n * (n + (1 if m.group(5) else 0)) * int(m.group(3)) * int(m.group(4))
    h = re.search(r"and ([\d,]+) Hypothesis pairs", sec)
    props = {}
    rows = table(sec, r"^Properties of the relation", r"property") or []
    names = {"reflexive": "refl", "transitive": "trans", "the joined host dominates": "join"}
    for r in rows:
        key = next((v for k, v in names.items() if r["property"].startswith(k)), None)
        for mode in MODES:
            if key and mode in r:
                props[f"{key}:{mode}"] = r[mode]
    for a, b, st in re.findall(r"(\w+) <= (\w+) (PROVED|REFUTED|UNKNOWN[^;.]*)", sec):
        props[f"mono:{a}:{b}"] = st
    return {"differential": [{"pairs": pairs, "disagreements": int(m.group(1).replace(",", ""))}],
            "hypothesis": int(h.group(1).replace(",", "")) if h else None, "properties": props,
            "generated": "EDGE_JUSTIFICATION.md", "commit": s.label}, s.label


def _z3_results():
    try:
        s = source("z3_cache")
        return json.loads(s.read()), s.label
    except Missing as e:
        try:
            return _z3_props_from_md()
        except Missing as e2:
            raise Missing(f"{e}; {e2}")


def m9():
    """real ordered shape pairs x modes x flag sets where the Z3 model equals emul_matrix / all of them: the
    differential check of verify/z3_relation.py, cached in verify/z3_results.json by --run-z3, else read from
    EDGE_JUSTIFICATION.md's properties section (pairs = shapes x (shapes + the joined host) x modes x flags)."""
    d, label = _z3_results()
    rows = d.get("differential") or []
    if not rows:
        raise Missing(f"{label}: no differential rows")
    tot = sum(r["pairs"] for r in rows)
    bad = sum(r["disagreements"] for r in rows)
    hyp = d.get("hypothesis", d.get("hypothesis_disagreements", "?"))
    return metric(tot - bad, tot, label, f"{bad:,} disagreements on the real pairs; Hypothesis: {hyp} "
                  f"({'pairs, 0 disagreements' if 'hypothesis' in d else 'disagreements'}); from {d.get('generated')}")


def m10():
    """{reflexive, transitive, monotone, join} x 4 modes proved by Z3 on the final code / 16. Monotone for mode
    m is exact <= payable <= mixgap <= capacity at the link from m; for capacity (the top) every link."""
    d, label = _z3_results()
    props = d.get("properties") or {}
    if not props:
        raise Missing(f"{label}: no property results")
    mono = {f"{a}<={b}": props.get(f"mono:{a}:{b}", "") for a, b in zip(MODES, MODES[1:])}
    got, bad = 0, []
    for k, m in enumerate(MODES):
        st = {"reflexive": props.get(f"refl:{m}", ""), "transitive": props.get(f"trans:{m}", ""),
              "join": props.get(f"join:{m}", "")}
        st["monotone"] = (mono[f"{m}<={MODES[k + 1]}"] if k + 1 < len(MODES)
                          else ("PROVED" if all(v == "PROVED" for v in mono.values()) else "not all links"))
        for p, v in st.items():
            if str(v).startswith("PROVED"):
                got += 1
            else:
                bad.append(f"{p} {m}: {v or 'missing'}")
    return metric(got, 16, label, f"from {d.get('generated', '?')}", items=bad)


def run_z3(timeout_s=120, skip_hypothesis=False):
    """Run verify/z3_relation.run against this checkout's relation and cache what M9 / M10 read."""
    import importlib.util
    import time
    s = source("z3_module")
    if s.kind == "path":
        path = s._p("")
        tmp = None
    else:
        tmp = tempfile.NamedTemporaryFile("w", suffix="_z3_relation.py", delete=False)
        tmp.write(s.read())
        tmp.close()
        path = tmp.name
    sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location("z3_relation_cov", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    diff = []
    pat = re.compile(r"real pairs\s+ctx=(\w+)\s+vocab=(\w+)\s+(\w+)\s*:\s*([\d,]+) pairs, ([\d,]+) edges, (\d+) disagree")
    lines = []

    def log(*a):
        msg = " ".join(map(str, a))
        lines.append(msg)
        print(msg)
        m = pat.search(msg)
        if m:
            diff.append({"ctx": m.group(1) == "True", "vocab": m.group(2) == "True", "mode": m.group(3),
                         "pairs": int(m.group(4).replace(",", "")), "edges": int(m.group(5).replace(",", "")),
                         "disagreements": int(m.group(6))})

    r = mod.run(MODES, timeout_s * 1000, skip_hypothesis=skip_hypothesis, log=log)
    props = {}
    for k, v in r.items():
        if isinstance(k, tuple):
            if k[0] == "trans":
                v = v["status"]
            elif k[0] == "join":
                v = v[0]
            props[":".join(k)] = v
    real_bad = sum(x["disagreements"] for x in diff)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "module": s.label,
           "commit": _git("rev-parse", "--short", "HEAD")[1].decode().strip(),
           "differential": diff, "hypothesis_disagreements": r["disagreements"] - real_bad, "properties": props}
    with open(os.path.join(ROOT, Z3_CACHE), "w") as f:
        json.dump(out, f, indent=1)
    if tmp:
        os.unlink(tmp.name)
    print(f"wrote {Z3_CACHE}")


# --- M13: zoo browser tests ---------------------------------------------------------------------------------------------

def m13():
    """page features with an automated browser test / all page features. The page template declares its
    features, as data-feature="name" attributes or one comment <!-- zoo-features: a, b, c -->; a feature is
    tested if tests/test_zoo_browser.py names it in a string literal."""
    st = source("zoo_template")
    html = st.read()
    feats = set(re.findall(r'data-feature="([^"]+)"', html))
    for m in re.finditer(r"<!--\s*zoo-features:(.*?)-->", html, re.S):
        feats.update(x.strip() for x in re.split(r"[,\n]", m.group(1)) if x.strip())
    if not feats:
        raise Missing(f"{st.label}: no declared feature list (data-feature attributes or <!-- zoo-features: -->)")
    sb = source("zoo_browser")
    lits = {n.value for n in ast.walk(ast.parse(sb.read())) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    hit = sorted(f for f in feats if f in lits)
    return metric(len(hit), len(feats), f"{st.label} + {sb.label}", items=sorted(feats - set(hit)))


# --- M14: the post's numbers --------------------------------------------------------------------------------------------

def m14(compute_numbers=False):
    """draft numbers matching the computed values / every number the strict checker sees in the draft: the
    agreeing occurrences, over agreeing + mismatching occurrences + uncovered numbers + missing anchors
    (graph/check_post.py; --strict fails on a missing anchor). The draft is only read."""
    sd = source("draft")
    try:
        sn = source("post_numbers")
    except Missing:
        if not compute_numbers:
            raise Missing("dist/post_numbers.json (generate it with --post-numbers, or python -m graph.post_numbers)")
        out = os.path.join(ROOT, "dist", "post_numbers.json")
        r = subprocess.run([sys.executable, "-m", "graph.post_numbers", "--out", out], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode:
            last = [x for x in r.stderr.splitlines() if x.strip()][-1:] or ["?"]
            raise Missing(f"dist/post_numbers.json (graph.post_numbers failed on this checkout: {last[0].strip()})")
        sn = source("post_numbers")
    sys.path.insert(0, ROOT)
    from graph.check_post import check
    res = check(sd.read(), json.loads(sn.read())["numbers"])
    ok, mm, un, mi = (len(res[k]) for k in ("ok", "mismatches", "uncovered", "missing"))
    return metric(ok, ok + mm + un + mi, f"the post draft, via TEF_DRAFT (read only) vs {sn.label}",
                  f"{ok} agree, {mm} mismatches, {un} uncovered numbers, {mi} missing anchors",
                  items=[f"L{r['line']} {r['key']}: draft {r['draft_value']!r}, computed {r['value']!r}"
                         for r in res["mismatches"]][:50])


# --- M15: known bugs fixed ------------------------------------------------------------------------------------------------

def m15():
    """confirmed bugs with a passing regression test / all confirmed bugs in tests/test_review_findings.py. Static:
    a bug is fixed when none of its tests carries the xfail(strict=True) marker (strict mode makes a fixed bug's
    marker fail, so the marker goes with the fix). A test's bug number is its section comment '# --- N.', or
    'REVIEW N' in its xfail reason."""
    s = source("review_tests")
    src = s.read()
    lines = src.split("\n")
    tree = ast.parse(src)
    bugs = {}
    for node in tree.body:
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")):
            continue
        xfail, num = False, None
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        for d in node.decorator_list:
            seg = ast.get_source_segment(src, d) or ""
            if "xfail" in seg:
                xfail = True
                m = re.search(r"REVIEW (\d+)", seg)
                num = num or (m and int(m.group(1)))
        for ln in reversed(lines[:first - 1]):
            m = re.match(r"#\s*-+\s*(\d+)\.", ln)
            if m:
                num = int(m.group(1)) if num is None else num
                break
        bugs.setdefault(num, []).append(xfail)
    fixed = sorted(k for k, v in bugs.items() if not any(v))
    n_x = sum(sum(v) for v in bugs.values())
    n_t = sum(len(v) for v in bugs.values())
    return metric(len(fixed), len(bugs), s.label,
                  f"{n_t} tests: {n_x} xfail (unfixed), {n_t - n_x} plain (fixed); fixed bugs: {fixed or 'none'}",
                  items=[f"REVIEW {k}" for k in sorted(bugs, key=lambda k: (k is None, k)) if any(bugs[k])])


# --- the table ------------------------------------------------------------------------------------------------------------

METRICS = [
    ("M1", "Model types reviewed", m1),
    ("M2", "Feature variants handled", m2),
    ("M3", "Rules proved", m3),
    ("M4", "Walls classified", m4),
    ("M5", "Theorems bridged to code", m5),
    ("M6", "Reduction types with an end-to-end instance", m6),
    ("M7", "Random edges verified", m7),
    ("M8", "Edges directly buildable", m8),
    ("M9", "Relation model fidelity", m9),
    ("M10", "Relation properties proved", m10),
    ("M11", "Models carried end to end", m11),
    ("M12", "Limits fully grounded", m12),
    ("M13", "Zoo behaviour tested", m13),
    ("M14", "Post numbers verified", m14),
    ("M15", "Known bugs fixed", m15),
]


m6.mid, m8.mid, m11.mid = "M6", "M8", "M11"


def compute(only=None, compute_numbers=False, log=print):
    out = {}
    for mid, name, fn in METRICS:
        if only and mid not in only:
            continue
        log(f"{mid} {name} ...")
        _USED.clear()
        try:
            r = fn(compute_numbers) if fn is m14 else fn()
            num, den = r["numerator"], r["denominator"]
            r["pct"] = round(100.0 * num / den, 2) if den else None
            r["merged"] = all(x.merged for x in _USED)      # every source read from disk here (or live)
        except Missing as e:
            r = {"numerator": None, "denominator": None, "pct": None, "source": None,
                 "status": f"n/a (source missing: {e})", "detail": "", "merged": False}
        r = {"name": name, "hard_gate": mid in HARD_GATED, **r}
        out[mid] = r
        log(f"   {_fmt(r)}")
    return out


def _fmt(r):
    if r["status"] != "ok":
        return r["status"]
    pct = "—" if r["pct"] is None else f"{r['pct']:.1f} %"
    return f"{r['numerator']:,} / {r['denominator']:,} ({pct})"


def ge(cur, base):
    """cur's fraction >= base's (exact arithmetic; an empty denominator counts as 100 %)"""
    f = lambda r: fractions.Fraction(r["numerator"], r["denominator"]) if r["denominator"] else fractions.Fraction(1)
    return f(cur) >= f(base)


def markdown(res):
    sha = _git("rev-parse", "--short", "HEAD")[1].decode().strip()
    L = ["# Coverage dashboard", "",
         "Generated by `python -m verify.coverage_report` (do not edit by hand). The metrics are defined in "
         "verify/COVERAGE_PLAN.md; each one's source is listed with the branch tip it was read from "
         "while it is unmerged. `tests/test_coverage_ratchet.py` keeps every available metric at or above "
         "`verify/coverage_baseline.json`; M1, M10 and M15 are gated at 100 % once their source is merged; M3 and M6 "
         "are ratchets whose open items may only shrink (their gaps need post-release theorems / constructions).",
         "", f"Computed at commit `{sha}`.", "",
         "| # | metric | value | % | gate | source |", "|---|---|---:|---:|---|---|"]
    for mid, r in res.items():
        val = _fmt(r).split(" (")[0] if r["status"] == "ok" else "n/a"
        pct = f"{r['pct']:.1f}" if r.get("pct") is not None else "—"
        gate = ("100 % (merged)" if r["merged"] else "100 % once merged") if r["hard_gate"] else "ratchet"
        src = r["source"] if r["status"] == "ok" else r["status"]
        L.append(f"| {mid} | {r['name']} | {val} | {pct} | {gate} | {src} |")
    L += ["", "## Details", ""]
    for mid, r in res.items():
        L.append(f"**{mid} {r['name']}.** " + (r["detail"] or r["status"]))
        if r.get("items"):
            L.append("")
            L.append("Not yet covered: " + "; ".join(str(x) for x in r["items"][:30])
                     + (f"; … ({len(r['items'])} listed in coverage.json)" if len(r["items"]) > 30 else ""))
        L.append("")
    L += ["## Definitions", ""]
    for mid, name, fn in METRICS:
        L.append(f"- **{mid}** " + " ".join((fn.__doc__ or "").split()))
    return "\n".join(L) + "\n"


def update_baseline(res):
    base = {}
    if os.path.exists(BASELINE):
        base = json.load(open(BASELINE))["metrics"]
    for mid, r in res.items():
        if r["status"] == "ok":
            base[mid] = {k: r[k] for k in ("numerator", "denominator", "pct", "source")}
        else:
            base.pop(mid, None)                        # n/a now: no stale value to ratchet against
    with open(BASELINE, "w") as f:
        json.dump({"note": "coverage may only go up: tests/test_coverage_ratchet.py compares each metric's fraction "
                           "with this; update with python -m verify.coverage_report --update-baseline",
                   "commit": _git("rev-parse", "--short", "HEAD")[1].decode().strip(),
                   "metrics": dict(sorted(base.items(), key=lambda kv: int(kv[0][1:])))}, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(BASELINE, ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--update-baseline", action="store_true", help="write verify/coverage_baseline.json from this run")
    ap.add_argument("--only", default=None,
                    help="comma-separated metric ids: recompute these, keep the others from verify/coverage.json")
    ap.add_argument("--run-z3", action="store_true", help="run verify/z3_relation first and cache its outputs")
    ap.add_argument("--z3-timeout", type=int, default=120, help="seconds per solver call (--run-z3)")
    ap.add_argument("--live", default=None, help=argparse.SUPPRESS)          # internal: one live metric, as JSON
    ap.add_argument("--post-numbers", action="store_true",
                    help="generate dist/post_numbers.json (graph.post_numbers, ~15 min) when missing; "
                         "else M14 is n/a")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.run_z3:
        run_z3(a.z3_timeout)
    only = set(a.only.split(",")) if a.only else None
    if a.live:
        fn = {m[0]: m[2] for m in METRICS}[a.live]
        print(json.dumps(fn.__wrapped__()))
        return 0
    res = compute(only, compute_numbers=a.post_numbers)
    if only and os.path.exists(OUT_JSON):
        old = json.load(open(OUT_JSON))
        res = {mid: res.get(mid, old.get(mid)) for mid, _, _ in METRICS if mid in res or mid in old}
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(OUT_MD, "w") as f:
        f.write(markdown(res))
    print(f"wrote {os.path.relpath(OUT_JSON, ROOT)}, {os.path.relpath(OUT_MD, ROOT)}")
    if a.update_baseline:
        update_baseline(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
