"""Merge the sweep: hunks_raw.jsonl + rules.py + reviews/*.jsonl -> hunks.jsonl, inventory.json, findings.json, SWEEP.md.

    python verify/sweep/build_report.py

Every hunk gets exactly one classification: a rule's (rules.py, mechanical) or a review line's (reviews/*.jsonl,
read by hand). A hunk with neither is reported and the script exits non-zero.
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from rules import rule  # noqa: E402
from findings_live import LIVE, NOTES, TEXT  # noqa: E402


def _live(pred, cfg, cls):
    from graph import model_graph as MG
    lm = MG._lm_unwrap(cfg)
    lm = lm if isinstance(lm, dict) else {}
    try:
        return bool(pred(cfg, lm, cls or {}, MG.dims_of(cfg) if cls else {}))
    except Exception:
        return True                     # a predicate that cannot read the config counts the carrier

TEXT_MAX = 800
OCC_MAX = 12
# finding keys merged after review (synonyms from different reviewers -> one key)
ALIASES = json.load(open(os.path.join(HERE, "finding_aliases.json"))) if os.path.exists(
    os.path.join(HERE, "finding_aliases.json")) else {}


def reviews():
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "reviews", "*.jsonl"))):
        for n, line in enumerate(open(p), 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_src"] = f"reviews/{os.path.basename(p)}:{n}"
            out[r["id"]] = r
    return out


def occ_str(o):
    return f"{o['variant']} {o['file']} {o['cls']}.{o['member']} <- {o['parent']}"


def main():
    H = [json.loads(l) for l in open(os.path.join(HERE, "hunks_raw.jsonl"))]
    raw = json.load(open(os.path.join(HERE, "inventory_raw.json")))
    var_models = {v: set(m) for v, m in raw["var_models"].items()}
    R = reviews()
    from graph import model_graph as MG
    snap = json.load(open(os.path.join(ROOT, "graph/hub_10k.json")))
    placed, cls_of = {}, {}
    cfg_of = {m["id"]: m["config"] or {} for m in snap}
    for m in snap:
        try:
            cls_of[m["id"]] = MG.classify(m["config"] or {})
            placed[m["id"]] = cls_of[m["id"]] is not None
        except Exception:
            placed[m["id"]] = False

    out, missing, by, conflicts = [], [], collections.Counter(), []
    findings = collections.defaultdict(lambda: dict(hunks=[], models=set(), types=set(), directions=collections.Counter(),
                                                    reasons=[], suggests=[]))
    for h in H:
        r = rule(h)
        if r and h["id"] in R and R[h["id"]].get("label") == "UNMAPPED":
            conflicts.append((h["id"], r[2][:60]))   # a hand review that found a subtlety outranks a rule
            r = None
        if r:
            c = dict(label=r[0], slot=r[1], reason=r[2].split(" -- ", 1)[-1][:140],
                     by="rule:" + r[2].split("rule:")[1].split(" --")[0])
        elif h["id"] in R:
            x = R[h["id"]]
            c = dict(label=x.get("label"), slot=x.get("slot"), reason=x.get("reason"), by=x["_src"])
            if c["label"] == "UNMAPPED":
                f = ALIASES.get(x.get("finding"), x.get("finding")) or "unnamed"
                c.update(finding=f, direction=x.get("direction"), suggest=x.get("suggest"))
                ms = set()
                for o in h["occ"]:
                    ms |= var_models.get(o["variant"], set())
                F = findings[f]
                F["hunks"].append(h["id"])
                F["models"] |= ms
                F["types"] |= set(h["types"])
                F["directions"][x.get("direction")] += 1
                if x.get("reason") and len(F["reasons"]) < 4 and x["reason"] not in F["reasons"]:
                    F["reasons"].append(x["reason"])
                if x.get("suggest") and len(F["suggests"]) < 2 and x["suggest"] not in F["suggests"]:
                    F["suggests"].append(x["suggest"])
        else:
            c = dict(label=None, reason=None, by=None)
            missing.append(h["id"])
        by[(c["label"], (c["by"] if c["by"].startswith("rule") else "review") if c["by"] else None)] += 1
        ruled = bool(c["by"] and c["by"].startswith("rule"))
        tmax, omax = (160, 2) if ruled else (TEXT_MAX, OCC_MAX)    # ruled hunks: a stub (sweep.py regenerates them)
        text = h["text"] if len(h["text"]) <= tmax else h["text"][:tmax] + "\n... (truncated; sweep.py regenerates it)"
        out.append(dict(id=h["id"], kind=h["kind"], role=h["role"], member=h["member"], models=h["models"],
                        placed_models=h["placed_models"], types=h["types"][:12], n_occ=len(h["occ"]),
                        occ=[occ_str(o) + ("" if o.get("reached", True) else " [unreached]") for o in h["occ"][:omax]],
                        text=text, **c))
    with open(os.path.join(HERE, "hunks.jsonl"), "w") as f:
        for x in out:
            f.write(json.dumps(x, sort_keys=True) + "\n")

    # inventory: type -> code locations -> parent (var_models kept for show.py)
    inv = dict(types=raw["types"], var_models={v: sorted(m) for v, m in var_models.items()},
               note="type -> code (variant: transformers 5.18 module, installed 4.57 module, or Hub remote-code "
                    "variant hub:<sha of its files>) -> parent (how: modular / similarity / root)")
    json.dump(inv, open(os.path.join(HERE, "inventory.json"), "w"), indent=1, sort_keys=True)

    fl = []
    for k, F in findings.items():
        pm = sorted(m for m in F["models"] if placed.get(m))
        pred = LIVE.get(k)
        live = [m for m in pm if pred is None or _live(pred, cfg_of[m], cls_of[m])]
        fl.append(dict(key=k, live_models=len(live), live_rule="config predicate (findings_live.py)" if pred else
                       "every placed carrier", live_note=NOTES.get(k, ""), example_live=live[:5],
                       models=len(F["models"]), placed_models=len(pm), hunks=len(F["hunks"]),
                       types=sorted(F["types"])[:30], direction=F["directions"].most_common(1)[0][0],
                       note=TEXT[k][0] if k in TEXT else (F["reasons"][0] if F["reasons"] else ""),
                       reasons=F["reasons"], suggest=([TEXT[k][1]] if k in TEXT else F["suggests"]),
                       hunk_ids=F["hunks"][:20], example_models=pm[:5]))
    fl.sort(key=lambda x: (-x["live_models"], -x["placed_models"], -x["models"], x["key"]))
    json.dump(fl, open(os.path.join(HERE, "findings.json"), "w"), indent=1)
    write_md(H, out, raw, fl, by, missing)
    if conflicts:
        print(len(conflicts), "hunks a rule calls IRRELEVANT but a review UNMAPPED (review kept):", conflicts[:10])
    if missing:
        print(len(missing), "hunks unclassified, e.g.", missing[:5])
        sys.exit(1)


def write_md(H, out, raw, fl, by, missing):
    types = raw["types"]
    lab = collections.Counter(x["label"] for x in out)
    L = ["# Code-diff sweep: every model type's forward code against its parent", "",
         "Generated by `verify/sweep/build_report.py` (do not edit by hand). Method: `verify/sweep/sweep.py` (inventory, "
         "parent resolution, hunks), `rules.py` (mechanical classifications), `reviews/*.jsonl` (hand classifications, "
         "one reviewer pass per batch), `coverage.py` (the coverage matrix). `tests/test_code_sweep.py` fails when a "
         "snapshot model type has no inventory entry or a hunk has no classification.", "",
         "## Summary", ""]
    n_models = len({m for v in raw["var_models"].values() for m in v})
    codes = collections.Counter()
    for t, e in types.items():
        for c in e["code"]:
            codes["hunt" if c.get("nocode_hunt") else c["variant"].split(":")[0]] += 1
    no_code = {t: e for t, e in types.items() if e["no_code"]}
    L += [f"- model types in graph/hub_10k.json (outer and LM-unwrapped): **{len(types)}**; with code: "
          f"{len(types) - len(no_code)}; no code available: {len(no_code)} ({sum(e['models'] for e in no_code.values())} "
          "models)",
          f"- code locations: transformers 5.18 modules {codes['tf5']}, installed 4.57 modules {codes['tf4']}, Hub "
          f"remote-code variants {codes['hub']} (type x variant pairs; "
          f"{len([v for v in raw['var_models'] if v.startswith('hub:')])} distinct remote variants); found by the "
          f"no-code hunt {codes['hunt']} (type x source pairs for "
          f"{len([t for t, e in types.items() if any(c.get('nocode_hunt') for c in e['code'])])} types: transformers "
          "main, Hub sibling repos, GitHub, inference engines, wrappers' inner LMs; see No-code hunt)",
          f"- models with code: {n_models} of 4384",
          f"- hunks: **{len(out)}** -- MAPPED {lab['MAPPED']}, IRRELEVANT {lab['IRRELEVANT']}, UNMAPPED "
          f"{lab['UNMAPPED']}, unclassified {len(missing)}", "",
          "| label | decided by | hunks |", "|---|---|---:|"]
    for (l, b), n in sorted(by.items(), key=lambda x: (str(x[0][0]), -x[1])):
        L.append(f"| {l} | {b} | {n} |")
    kinds = collections.Counter(x["kind"] for x in out)
    L += ["", "Hunk kinds: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common()), ""]

    L += ["## UNMAPPED findings, ranked by models affected", "",
          "carriers = snapshot models whose code contains a hunk of the finding; placed = of them, the ones classify() places; live = of those, the ones whose config takes the code path (findings_live.py predicate, else every placed carrier). Ranked by live. direction: "
          "over-credit = the graph claims edges the code breaks; over-wall = the graph refuses edges the code allows.",
          "", "| rank | finding | live / placed / all carriers | hunks | direction | what the forward computes | suggested fix | "
              "types |", "|---:|---|---:|---:|---|---|---|---|"]
    for i, f in enumerate(fl, 1):
        L.append(f"| {i} | `{f['key']}` | {f['live_models']} / {f['placed_models']} / {f['models']} | {f['hunks']} | {f['direction']} | "
                 f"{_cell(f['note'])} | {_cell('; '.join(f['suggest']))} | {_cell(', '.join(f['types'][:8]))} |")

    notes_p = os.path.join(HERE, "notes_outside_hunks.json")
    if os.path.exists(notes_p):
        L += ["", "## Observations outside the diff hunks", "",
              "Reviewers' notes on behaviour no hunk carries (inherited unchanged code, classifier paths, version drift); "
              "not counted above.", ""]
        for n in json.load(open(notes_p)):
            L.append(f"- `{n['key']}` ({n['source']}): {n['note']}")

    unplaced = sorted(((t, e) for t, e in types.items() if e["placed"] == 0), key=lambda x: (-x[1]["models"], x[0]))
    L += ["", "## Unplaced types", "",
          f"{len(unplaced)} types ({sum(e['models'] for _, e in unplaced)} models) have no model classify() places "
          "(SKIP_TYPES, drafters / assistants, GGUF stubs without dims, or no heads / norm / positions it can read). Their "
          "hunks are IRRELEVANT by rule:unplaced -- the relation makes no claim about them; whether they should be "
          "placed is a coverage question, not an edge error.", "",
          ", ".join(f"{t} ({e['models']})" for t, e in unplaced), ""]

    L += ["## Conventions and caveats", "",
          "- The reference forward is the eager attention path in fp32 on fixed weights, text-only input. Flash / sdpa "
          "subclasses (rule:backend), float32 upcasts and dtype casts of constants are IRRELEVANT on that basis; "
          "`sliding_window_backend_dependent` records the one case where the default sdpa backend differs from eager.",
          "- rule:unreached is static: a class or method no causal-LM entry class can reach by name (attribute access, "
          "constructor, module function) never runs. Over-inclusive by design (any attribute of the same name "
          "reaches), so it can only under-prune.",
          "- rule:config_default_unused uses the snapshot: a class default every config of the type overrides never "
          "runs. A future config that omits the key needs a re-sweep (the test fails on a new type, not on a new "
          "config of a known type).",
          "- Several IRRELEVANT review labels are snapshot-dependent (code paths no current config switches on: "
          "Nanbeige hyper-connections, MiniCPM InfLLM-v2 sparse attention, MemT memory, ...); the review reason says so.",
          "- The models column counts the carriers of the code (every snapshot model whose type or remote variant "
          "contains the hunk). Where a finding needs a config path (e.g. Gemma-4 enable_moe_block), the models that "
          "take that path are a subset; the reasons in findings.json name them.",
          "- Hunk ids are hashes of (role, member, changed lines): the same edit in many files is one hunk, classified "
          "once. hunks.jsonl keeps a stub of the text for rule-decided hunks; `python verify/sweep/sweep.py` "
          "regenerates the full hunks (hunks_raw.jsonl, not committed).", ""]

    L += ["", "## Coverage matrix", ""]
    cov = os.path.join(HERE, "coverage.md")
    if os.path.exists(cov):
        body = open(cov).read().split("\n", 2)[2] if open(cov).read().startswith("#") else open(cov).read()
        L += ["From `coverage.py` (`coverage.json` has the citations); the last rows are this sweep's UNMAPPED findings "
              "(not read by the classifier).", "", body.strip(), ""]

    L += nocode_hunt_md(types, out)

    L += ["## No code available", "",
          "Types with no transformers module and no `auto_map` in any snapshot config carrying them (GGUF / MLX / ONNX "
          "re-uploads of an unlisted architecture, private code, or a type string no code defines), and whose code "
          "the no-code hunt did not find either.", "",
          "| type | models | placed | examples |", "|---|---:|---:|---|"]
    for t, e in sorted(no_code.items(), key=lambda x: (-x[1]["models"], x[0])):
        L.append(f"| {t} | {e['models']} | {e['placed']} | {', '.join(e['examples'][:2])} |")
    miss = {t: e["remote_missing"] for t, e in types.items() if e["remote_missing"]}
    L += ["", "Remote files an `auto_map` names that the repo does not have (the type's other variants carry its "
          "code): " + "; ".join(f"{t} ({len(m)})" for t, m in sorted(miss.items())), ""]

    L += ["## Inventory", "", "| type | models | placed | code (variant: parent, how) |", "|---|---:|---:|---|"]
    for t, e in sorted(types.items(), key=lambda x: (-x[1]["models"], x[0])):
        cs = []
        for c in e["code"][:4]:
            cs.append(f"{c['variant']}: {c.get('parent') or '-'} ({c.get('how', 'listed')})")
        more = f" + {len(e['code']) - 4} more" if len(e["code"]) > 4 else ""
        L.append(f"| {t} | {e['models']} | {e['placed']} | {_cell('; '.join(cs) + more) or 'NO CODE'} |")
    open(os.path.join(HERE, "SWEEP.md"), "w").write("\n".join(L) + "\n")


def nocode_hunt_md(types, out):
    """The no-code hunt section: per type, where the code was found (nocode_sources.json), the UNMAPPED findings of
    its hunks, and what was tried for the types still without code."""
    p = os.path.join(HERE, "nocode_sources.json")
    if not os.path.exists(p):
        return []
    nc = json.load(open(p))
    res = lambda t: nc[nc[t]["same_as"]] if nc[t].get("same_as") else nc[t]
    hunt = sorted(set(nc) | {t for t, e in types.items() if e["no_code"]},
                  key=lambda t: (-types.get(t, {}).get("models", 0), t))
    hunt = [t for t in hunt if t in types]
    own = {t: {sv["variant"] for sv in res(t).get("sweep", []) if not sv.get("wrapper")} for t in nc if t in types}
    fnd = collections.defaultdict(set)
    for x in out:
        if x["label"] != "UNMAPPED":
            continue
        vs = {o.split(" ", 1)[0] for o in x["occ"]}
        for t, v in own.items():
            if vs & v:
                fnd[t].add(x["finding"])
    st = collections.Counter()
    ms = collections.Counter()
    for t in hunt:
        s = res(t)["status"] if t in nc else "missing"
        st[s] += 1
        ms[s] += types[t]["models"]
    n_models = sum(types[t]["models"] for t in hunt)
    found = [s for s in st if s not in ("missing",)]
    from graph import model_graph as MG                   # distinct models (an outer / text-config pair is 1 model)
    ids = collections.defaultdict(set)
    for m in json.load(open(os.path.join(ROOT, "graph/hub_10k.json"))):
        cfg = m.get("config") or {}
        lm = MG._lm_unwrap(cfg)
        for tt in {str(cfg.get("model_type") or "") or "<none>", str((lm or {}).get("model_type") or "")}:
            ids[tt].add(m["id"])
    st_of = lambda t: res(t)["status"] if t in nc else "missing"
    all_ids = set().union(*(ids[t] for t in hunt))
    miss_ids = set().union(*(ids[t] for t in hunt if st_of(t) == "missing"))
    got_ids = set().union(*(ids[t] for t in hunt if st_of(t) in ("code", "wrapper")))
    L = ["## No-code hunt", "",
         f"The {len(hunt)} types ({n_models} models) that had no transformers module and no `auto_map` at the sweep's "
         "first run. `nocode_sources.json` records, per type, where forward-pass code was found (transformers main "
         "for modules added after 5.18, Hub sibling repos / older revisions / files shipped without an auto_map, "
         "GitHub repos, inference engines read as text: vLLM, SGLang, llama.cpp, MLX-LM) and what was tried; the "
         "files are fetched as text under `$NOCODE_CACHE` (default $TEF_SCRATCH/nocode) and swept like the "
         "rest (`ext:` / `tfmain:` variants; a wrapper or a GGUF tag over a known architecture maps to that "
         "architecture's variant). Engine code is used only where no reference exists, and the entry says whether the "
         "engine fuses or approximates.", "",
         "Status: " + ", ".join(f"{s} {st[s]} types / {ms[s]} models" for s in sorted(st)) + "; with code now "
         f"(code + wrapper): {sum(st[s] for s in found if s != 'paper-only')} types / "
         f"{sum(ms[s] for s in found if s != 'paper-only')} type-model pairs; distinct models: {len(got_ids)} of "
         f"{len(all_ids)} have code, {len(miss_ids - got_ids)} still have none. `new findings` lists the UNMAPPED "
         "keys among the hunks of the type's own (non-wrapper) variants.", "",
         "| type | models | placed | status | where the code was found | parent | new findings | tried (still "
         "missing) |", "|---|---:|---:|---|---|---|---|---|"]
    for t in hunt:
        e = types[t]
        x = res(t) if t in nc else {}
        same = f"same as {nc[t]['same_as']}" if t in nc and nc[t].get("same_as") else ""
        s = x.get("status", "missing")
        tried = "; ".join(x.get("tried", [])) if s in ("missing", "paper-only") else ""
        where = "; ".join(f"`{sv['variant']}` {sv.get('source', '')}" for sv in x.get("sweep", [])) or x.get("found", "")
        L.append(f"| {t} | {e['models']} | {e['placed']} | {s} | {_cell(same or where)} | "
                 f"{_cell('' if same else x.get('parent', ''))} | {_cell(', '.join(sorted(fnd.get(t, ()))) or '-')} | "
                 f"{_cell(tried)} |")
    return L + [""]


def _cell(s):
    return str(s or "").replace("|", "\\|").replace("\n", " ")[:400]


if __name__ == "__main__":
    main()
