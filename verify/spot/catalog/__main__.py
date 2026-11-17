"""Run every catalog entry and write verify/spot/catalog_results.json (the SPOT.md catalog table reads it).

  python -m verify.spot.catalog [--fast] [--only SUBSTRING]
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from verify.spot.catalog import CATALOG, cached, load_all

OUT = Path(__file__).resolve().parents[1] / "catalog_results.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the real-checkpoint (slow) entries")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    load_all()
    seen, rows = {}, []
    for t, es in sorted(CATALOG.items()):
        for e in es:
            seen.setdefault(e.check.__name__, e)
    old = {r["entry"]: r for r in json.loads(OUT.read_text())} if OUT.exists() else {}
    for name, e in sorted(seen.items()):
        if a.only not in name or (a.fast and e.slow):
            if name in old:
                rows.append(old[name])
            continue
        row = {"entry": name, "types": list(e.types), "instance": e.instance, "real": e.real, "standard": e.standard}
        if [r for r in e.needs if not cached(r)]:
            row.update(status="skipped (not cached)")
        else:
            t0 = time.time()
            try:
                torch.manual_seed(0)
                r = e.check()
                st = ("pass" if r.passed else "FAIL") if not e.finding else ("FINDING" if not r.passed else
                                                                              "finding-no-longer-reproduces")
                row.update(status=st, finding=e.finding, result=r.line(), rel=r.rel, top1=r.top1,
                           identical=f"{r.identical}/{r.n}")
            except Exception as ex:                       # noqa: BLE001 -- recorded
                row.update(status="error", result=f"{type(ex).__name__}: {ex}"[:300],
                           trace=traceback.format_exc()[-1200:])
            row["seconds"] = round(time.time() - t0, 1)
        print(f"{row['status']:8} {name:45} {row.get('result', '')}", flush=True)
        rows.append(row)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
