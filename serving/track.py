"""Daily price tracker over serving.arbitrage: dated, gzipped snapshots plus a spread diff report.

  python -m serving.track run       # fetch -> serving/data/history/YYYY-MM-DD/*.json.gz, run the books,
                                    #   write report.md vs the previous snapshot, prune > 60 days
  python -m serving.track diff A B  # compare two dated snapshots (YYYY-MM-DD or a snapshot dir path)

The uncompressed latest pull stays in serving/data/ (what `python -m serving.arbitrage` reads). A spread
is keyed by (book, target, workload): book1/book2 are priced at the 3:1 blend, book3 per WORKLOADS entry.
Pruned snapshots keep their report as serving/data/history/reports/YYYY-MM-DD.md.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import re
import shutil
import sys

from serving import arbitrage

HISTORY = os.path.join(arbitrage.DATA, "history")
RAW = ("or_models.json", "or_endpoints.json", "fl_models.json")
BOOKS = "arbitrage.json.gz"
MOVE = 0.10                                             # "moved" threshold: 10 percentage points
KEEP_DAYS = 60
BLEND = "blend 3:1"
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------- snapshots

def snapshots(history=HISTORY):
    """Dated snapshot dir names, oldest first."""
    if not os.path.isdir(history):
        return []
    return sorted(n for n in os.listdir(history) if _DATE.match(n) and os.path.isdir(os.path.join(history, n)))


def snap_dir(x, history=HISTORY):
    return x if os.path.isdir(x) and not _DATE.match(x) else os.path.join(history, x)


def write_gz(obj, path):
    with gzip.open(path, "wt", compresslevel=6) as f:
        json.dump(obj, f)


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def snapshot_books(d):
    """Books of a snapshot dir (computed and cached as arbitrage.json.gz if missing)."""
    p = os.path.join(d, BOOKS)
    if not os.path.exists(p):
        write_gz(arbitrage.books(d), p)
    return read_gz(p)


# ---------------------------------------------------------------- diff

def spreads(books):
    """{(book, target, workload): spread} from an arbitrage.books() dict."""
    out = {}
    for k in ("book1", "book2"):
        for r in books.get(k, []):
            out[(k, r["target"], BLEND)] = r["spread"]
    for r in books.get("book3", []):
        for w, v in r["workloads"].items():
            out[("book3", r["target"], w)] = v["spread"]
    return out


def diff(a, b, move=MOVE):
    """Changes from spreads dict a (older) to b (newer). Each list holds (key, old, new)."""
    res = {"new": [], "gone": [], "flipped": [], "moved": []}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None:
            if y > 0:
                res["new"].append((k, None, y))
        elif y is None:
            if x > 0:
                res["gone"].append((k, x, None))
        elif (x > 0) != (y > 0):
            res["flipped"].append((k, x, y))
        elif abs(y - x) > move:
            res["moved"].append((k, x, y))
    return res


def summary(res):
    return ", ".join(f"{len(v)} {k}" for k, v in res.items())


def report(res, new_name, old_name, books=None):
    pct = lambda v: "-" if v is None else f"{v:+.1%}"
    lines = [f"# Price spreads {new_name} vs {old_name or '(no previous snapshot)'}", ""]
    if books is not None:
        lines += [f"book1 {len(books.get('book1', []))} rows, book2 {len(books.get('book2', []))} rows, "
                  f"book3 {len(books.get('book3', []))} targets with a positive spread on some workload.", ""]
    lines += [f"Changes: {summary(res)} (moved = |change| > {MOVE:.0%} points).", ""]
    workloads = sorted({k[2] for v in res.values() for k, _, _ in v}, key=lambda w: (w != BLEND, w))
    for w in workloads:
        lines += [f"## {w}", ""]
        for cat, rows in res.items():
            rows = [r for r in rows if r[0][2] == w]
            if not rows:
                continue
            lines += [f"### {cat} ({len(rows)})", "", "| book | target | old | new | change |", "|---|---|---|---|---|"]
            for (book, tgt, _), x, y in sorted(rows, key=lambda r: -abs((r[2] or 0) - (r[1] or 0))):
                ch = "-" if x is None or y is None else f"{y - x:+.1%}"
                lines.append(f"| {book} | {tgt} | {pct(x)} | {pct(y)} | {ch} |")
            lines.append("")
    if not workloads:
        lines += ["No changes.", ""]
    return "\n".join(lines)


def compare(a, b, history=HISTORY):
    da, db = snap_dir(a, history), snap_dir(b, history)
    return diff(spreads(snapshot_books(da)), spreads(snapshot_books(db)))


# ---------------------------------------------------------------- retention

def prune(history=HISTORY, today=None, keep_days=KEEP_DAYS):
    """Delete dated snapshot dirs older than keep_days, moving each report.md to reports/<date>.md first."""
    today = today or dt.date.today()
    gone = []
    for n in snapshots(history):
        if (today - dt.date.fromisoformat(n)).days > keep_days:
            d = os.path.join(history, n)
            rep = os.path.join(d, "report.md")
            if os.path.exists(rep):
                os.makedirs(os.path.join(history, "reports"), exist_ok=True)
                shutil.move(rep, os.path.join(history, "reports", f"{n}.md"))
            shutil.rmtree(d)
            gone.append(n)
    return gone


# ---------------------------------------------------------------- run

def run(history=HISTORY, today=None, do_fetch=True):
    today = today or dt.date.today()
    name = today.isoformat()
    d = os.path.join(history, name)
    os.makedirs(d, exist_ok=True)
    if do_fetch:
        arbitrage.fetch()                               # latest, uncompressed, in serving/data/
    for f in RAW:
        write_gz(arbitrage.read_json(f), os.path.join(d, f + ".gz"))
    if os.path.exists(os.path.join(d, BOOKS)):
        os.remove(os.path.join(d, BOOKS))               # re-run on the same day: recompute
    books = snapshot_books(d)
    prev = [n for n in snapshots(history) if n < name and os.path.exists(os.path.join(history, n, BOOKS))]
    old = prev[-1] if prev else None
    res = diff(spreads(read_gz(os.path.join(history, old, BOOKS))) if old else {}, spreads(books))
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write(report(res, name, old, books))
    pruned = prune(history, today)
    print(f"{name} vs {old or 'none'}: {summary(res)}; book1 {len(books['book1'])}, book2 {len(books['book2'])}, "
          f"book3 {len(books['book3'])}; pruned {len(pruned)}; report {os.path.join(d, 'report.md')}")
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        run(do_fetch="--no-fetch" not in argv)
    elif argv[:1] == ["diff"] and len(argv) == 3:
        res = compare(argv[1], argv[2])
        print(report(res, os.path.basename(argv[2].rstrip("/")), os.path.basename(argv[1].rstrip("/"))))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
