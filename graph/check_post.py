"""Check the numbers in a blog-post draft against post_numbers.json (graph/post_numbers.py).

For every key, each anchor (a regex on the draft's phrasing, or a cell of a markdown table) is located
in the draft and the number found there is compared with the computed value. Reported:

  MISMATCH        the draft's value differs from the computed one (key, line, draft text, computed value)
  MISSING ANCHOR  the anchor matches nowhere: the phrasing changed (or the sentence was cut)
  UNCOVERED       numbers in the prose that no anchor covers (best effort: tables, code, links,
                  exponents, years and numbers glued to a name, like Qwen-72B or fp32, are skipped)

Only the draft part is read: everything before "Blog post (published):", minus the front matter.
Exit status 1 on any mismatch (with --strict, on a missing anchor too). --fix-preview prints a unified
diff of the replacements it would make; the draft is NEVER written.

Run:  python -m graph.check_post DRAFT.md [dist/post_numbers.json] [--fix-preview] [--strict]
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys

END_MARK = "Blog post (published):"
NUM = r"(\d{1,3}(?:(?:\{,\}|,)\d{3})+|\d+(?:\.\d+)?)"
PRE = r"(?:\$|\*\*|~|\\mathbf\{|\\sim\\!|\\sim)*"
CELL_NUM = re.compile(PRE + NUM)


def compile_re(p):
    return re.compile(p.replace("<N>", PRE + NUM), re.M)


def draft_text(text):
    """(text cut at the published marker, start offset of the body after any front matter)"""
    cut = text.find(END_MARK)
    if cut >= 0:
        text = text[:cut]
    start = 0
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end >= 0:
            start = end + 4
    return text, start


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


# --- markdown tables --------------------------------------------------------------------------------


def _split_cells(line, off):
    """cells of a table line as (start, end, stripped text), absolute offsets"""
    bars = [m.start() for m in re.finditer(r"(?<!\\)\|", line)]
    out = []
    for a, b in zip(bars, bars[1:]):
        raw = line[a + 1:b]
        ls = len(raw) - len(raw.lstrip())
        txt = raw.strip()
        out.append((off + a + 1 + ls, off + a + 1 + ls + len(txt), txt))
    return out


def tables(text, header):
    """every table whose header line matches `header`: a list of body rows, each a list of cells"""
    out = []
    hre = re.compile(header)
    lines = text.split("\n")
    offs, o = [], 0
    for ln in lines:
        offs.append(o)
        o += len(ln) + 1
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and hre.search(lines[i]):
            rows = []
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                if not re.fullmatch(r"\s*\|[\s:|-]*\|?\s*", lines[j]):          # the --- separator
                    rows.append(_split_cells(lines[j], offs[j]))
                j += 1
            out.append(rows)
            i = j
        else:
            i += 1
    return out


def find(anchor, text):
    """[(start, end, raw value text, covered span)] for every occurrence of the anchor"""
    if "re" in anchor and "table" not in anchor:
        return [(m.start(1), m.end(1), m.group(1), m.span(0)) for m in compile_re(anchor["re"]).finditer(text)]
    hits = []
    for rows in tables(text, anchor["table"]):
        sel = anchor["row"]
        if isinstance(sel, int):
            picked = rows[sel:sel + 1] if sel < len(rows) else []
        else:
            rre = re.compile(sel["re"])
            picked = [r for r in rows if len(r) > sel["col"] and rre.search(r[sel["col"]][2])]
        for r in picked:
            if len(r) <= anchor["col"]:
                continue
            a, b, txt = r[anchor["col"]]
            if "re" in anchor:
                m = compile_re(anchor["re"]).search(txt)
                if m:
                    hits.append((a + m.start(1), a + m.end(1), m.group(1), (a, b)))
            elif anchor.get("whole"):                    # a name / shape: the whole cell
                hits.append((a, b, txt, (a, b)))
            else:                                        # a number: the cell's first one
                m = CELL_NUM.search(txt)
                if m:
                    hits.append((a + m.start(1), a + m.end(1), m.group(1), (a, b)))
    return hits


# --- values -------------------------------------------------------------------------------------------


def _num(raw):
    return float(raw.replace("{,}", "").replace(",", ""))


def _unit(raw, kind):
    s = raw.replace("{,}", "").replace(",", "")
    if "." in s:
        return 10.0 ** -len(s.split(".")[1])
    if kind == "approx" and len(s) > 1:
        return 10.0 ** (len(s) - len(s.rstrip("0")))
    return 1.0


def _tokens(name):
    return [t for t in re.split(r"[-_.\s/]+", name.lower()) if t]


def _clean_name(raw):
    return raw.replace("*", "").replace("`", "").strip()


def name_ok(raw, aliases):
    """the draft's name (or a parenthesised part of it) is a whole alias -- an accepted repo name, with or
    without its org -- up to case and separators. A token SUBSET is not enough: 'MiMo-V2.5' is a sibling of
    'MiMo-V2.5-Pro', not a short name for it."""
    raw = _clean_name(raw)
    parts = [raw] + [p.strip() for p in re.split(r"[()]", raw) if p.strip()]
    whole = set()
    for a in aliases:
        whole.add(tuple(_tokens(a)))
        whole.add(tuple(_tokens(a.rsplit("/", 1)[-1])))
    return any(tk and tuple(tk) in whole for tk in map(_tokens, parts))


def compare(entry, raw):
    """(ok, the draft's value as read)"""
    kind, v = entry["kind"], entry["value"]
    if kind == "int":
        d = _num(raw)
        return d == v, int(d)
    if kind == "approx":
        # a decimal may be rounded or truncated at its last digit ("0.009" for 0.00955); an integer must be
        # rounded correctly, at its trailing zeros ("2,550" for 2,546), so "80" never passes for 89.9
        d, u = _num(raw), _unit(raw, kind)
        ok = math.isclose(d, math.floor(v / u + 0.5) * u, abs_tol=u * 1e-6)
        if "." in raw:
            ok = ok or math.isclose(d, math.floor(v / u + 1e-9) * u, abs_tol=u * 1e-6)
        return ok, d
    if kind == "name":
        return name_ok(raw, entry.get("aliases") or [v]), _clean_name(raw)
    if kind == "bool":
        d = raw.strip() == "✅"
        return d == v, d
    if kind == "tokens":
        return all(t in raw for t in v), raw.strip()
    raise ValueError(kind)


def render(entry, raw):
    """the replacement text for a mismatched occurrence, in the draft's own style; None if not fixable"""
    kind, v = entry["kind"], entry["value"]
    if kind in ("int", "approx"):
        sep = "{,}" if "{,}" in raw else ("," if "," in raw else "")
        u = _unit(raw, kind)
        x = math.floor(v / u + 0.5) * u
        dec = max(0, -int(round(math.log10(u)))) if u < 1 else 0
        s = f"{x:,.{dec}f}" if sep else f"{x:.{dec}f}"
        return s.replace(",", sep) if sep else s
    if kind == "name":
        return v
    if kind == "bool":
        return "✅" if v else "❌"
    return None


# --- the check ----------------------------------------------------------------------------------------


def check(text, numbers):
    """-> dict(mismatches, missing, ok, uncovered, replacements)"""
    body, start = draft_text(text)
    res = {"mismatches": [], "missing": [], "ok": [], "uncovered": [], "replacements": []}
    covered = []
    for key, e in numbers.items():
        for k, anchor in enumerate(e["anchors"]):
            a = dict(anchor)
            if e["kind"] in ("name", "tokens") and "table" in a and "re" not in a:
                a["whole"] = True
            hits = [h for h in find(a, body) if h[0] >= start]
            if not hits:
                if not e.get("optional"):
                    res["missing"].append({"key": key, "anchor": anchor, "value": e["value"]})
                continue
            for s, t, raw, cov in hits:
                covered.append(cov)
                ok, d = compare(e, raw)
                rec = {"key": key, "line": line_of(body, s), "draft": raw, "draft_value": d,
                       "value": e["value"], "source": e["source"],
                       "context": body[max(cov[0] - 30, 0):min(cov[1] + 30, len(body))].replace("\n", " ")}
                if ok:
                    res["ok"].append(rec)
                else:
                    res["mismatches"].append(rec)
                    new = render(e, raw)
                    if new is not None and new != raw:
                        res["replacements"].append((s, t, new))
    res["uncovered"] = uncovered(body, start, covered)
    res["placeholders"] = placeholders(body, start, numbers)
    return res


PLACEHOLDER = re.compile(r"⟨N:([A-Za-z0-9_.\-]+)⟩")


def placeholders(body, start, numbers):
    """every <N:key> placeholder in the draft, with the computed value of the key of that name (None: no such key, or
    a key whose value this tree cannot compute -- its source says where it lives)"""
    out = []
    for m in PLACEHOLDER.finditer(body):
        if m.start() < start:
            continue
        e = numbers.get(m.group(1))
        out.append({"line": line_of(body, m.start()), "key": m.group(1), "known": e is not None,
                    "value": e["value"] if e else None, "source": e["source"] if e else ""})
    return out


_SKIP_SPANS = [r"https?://\S+", r"\]\([^)]*\)", r"!\[\[[^\]]*\]\]", r"`[^`\n]*`", r"\^\{[^}]*\}", r"\^-?\d+",
               r"10\^\{?-?\d+\}?", r"\\[a-zA-Z]+"]
_BARE = re.compile(r"(?<![\w.,\-/\\^{}])" + NUM + r"(?![\w/]|\.\d)")


def uncovered(body, start, covered):
    """bare numbers in prose lines outside every covered span"""
    out = []
    pos, in_code = 0, False
    for ln in body.split("\n"):
        a = pos
        pos += len(ln) + 1
        st = ln.strip()
        if st.startswith("```"):
            in_code = not in_code
            continue
        if a < start or in_code or st.startswith("|") or not st:
            continue
        masked = ln
        for p in _SKIP_SPANS:
            masked = re.sub(p, lambda m: "x" * len(m.group(0)), masked)    # letters: glues \sim1 to a name
        for m in _BARE.finditer(masked):
            s, t = a + m.start(1), a + m.end(1)
            raw = m.group(1)
            n = raw.replace("{,}", "").replace(",", "")
            if re.fullmatch(r"(19|20)\d\d", n) or n in ("0", "0.0"):
                continue
            if any(cs <= s and t <= ce for cs, ce in covered):
                continue
            out.append({"line": line_of(body, s), "number": raw,
                        "context": ln[max(m.start(1) - 35, 0):m.end(1) + 35].strip()})
    return out


def preview(text, replacements):
    new = text
    for s, t, r in sorted(set(replacements), reverse=True):
        new = new[:s] + r + new[t:]
    return new


def report(res, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(f"== MISMATCHES ({len(res['mismatches'])}) ==")
    for r in sorted(res["mismatches"], key=lambda r: r["line"]):
        p(f"  L{r['line']:<4} {r['key']}: draft {r['draft_value']!r} ({r['draft']!r}), computed {r['value']!r}")
        p(f"        ...{r['context']}...")
    p(f"\n== MISSING ANCHORS ({len(res['missing'])}) ==")
    for r in res["missing"]:
        a = r["anchor"]
        where = a.get("re") if "table" not in a else f"table {a['table']!r} row {a['row']!r} col {a['col']}"
        p(f"  {r['key']}: computed {r['value']!r}; anchor {where}")
    p(f"\n== UNCOVERED NUMBERS ({len(res['uncovered'])}) ==")
    for r in res["uncovered"]:
        p(f"  L{r['line']:<4} {r['number']:>8}   {r['context']}")
    ph = res.get("placeholders", [])
    p(f"\n== PLACEHOLDERS ({len(ph)}; {sum(1 for r in ph if r['value'] is None)} without a computed value) ==")
    for r in ph:
        v = r["value"] if r["known"] else "(no key)"
        p(f"  L{r['line']:<4} ⟨N:{r['key']}⟩ = {v!r}" + (f"   [{r['source']}]" if r["value"] is None and r["known"] else ""))
    keys_ok = {r["key"] for r in res["ok"]} - {r["key"] for r in res["mismatches"]}
    p(f"\n== OK: {len(res['ok'])} occurrences agree ({len(keys_ok)} keys fully consistent) ==")


def main(argv=None):
    ap = argparse.ArgumentParser(description="check a draft's numbers against post_numbers.json")
    ap.add_argument("draft")
    ap.add_argument("numbers", nargs="?", default="dist/post_numbers.json")
    ap.add_argument("--fix-preview", action="store_true",
                    help="print a unified diff of the replacements (never writes the draft)")
    ap.add_argument("--strict", action="store_true", help="a missing anchor also fails")
    args = ap.parse_args(argv)
    with open(args.draft, encoding="utf-8") as f:
        text = f.read()
    with open(args.numbers, encoding="utf-8") as f:
        numbers = json.load(f)["numbers"]
    res = check(text, numbers)
    report(res)
    if args.fix_preview:
        new = preview(text, res["replacements"])
        diff = difflib.unified_diff(text.splitlines(keepends=True), new.splitlines(keepends=True),
                                    fromfile=args.draft, tofile=args.draft + " (preview, not written)")
        print("\n== FIX PREVIEW ==")
        sys.stdout.writelines(diff)
    return 1 if res["mismatches"] or (args.strict and res["missing"]) else 0


if __name__ == "__main__":
    sys.exit(main())
