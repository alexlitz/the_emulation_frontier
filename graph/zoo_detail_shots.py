"""Screenshots of the zoo page's detail panel (dist/zoo_detail_shots/), in headless Chromium, from file://.

Run:  python -m graph.zoo_detail_shots [--page dist/zoo.html] [--out dist/zoo_detail_shots]

Each card is opened by its permalink, the detail's folds opened, and the panel captured section by section; the time
from navigation to a rendered detail and any console error go to timing.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

DESKTOP = {"width": 1400, "height": 900}
PHONE = {"width": 375, "height": 800}
SHOTS = [  # name, repo, viewport, mobile, settings (weights, kv, hardware, context index, batch index)
    ("01_qwen3_8b_summary", "Qwen/Qwen3-8B", DESKTOP, False, None),
    ("02_gemma3_1b_sliding_curves", "google/gemma-3-1b-it", DESKTOP, False, None),
    ("03_deepseek_r1_q4km_b32", "deepseek-ai/DeepSeek-R1", DESKTOP, False, ("Q4_K_M", "fp8", "b200", None, 5)),
    ("04_qwen3_next_layers", "Qwen/Qwen3-Next-80B-A3B-Instruct", DESKTOP, False, None),
    ("05_phone_qwen3_8b", "Qwen/Qwen3-8B", PHONE, True, None),
]
OPEN = """k => { const d = document.querySelector(`#detail details[data-fold=${k}]`); if (d) d.open = true; }"""
SCROLL = """s => { const e = document.querySelector(s); if (e) e.scrollIntoView({block: 'start'}); }"""


def main():
    from urllib.parse import quote

    from playwright.sync_api import sync_playwright
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="dist/zoo.html")
    ap.add_argument("--out", default="dist/zoo_detail_shots")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    url = "file://" + os.path.abspath(a.page)
    timing = {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, repo, vp, mobile, settings in SHOTS:
            ctx = b.new_context(viewport=vp, is_mobile=mobile, has_touch=mobile, device_scale_factor=2 if mobile else 1)
            pg = ctx.new_page()
            errs = []
            pg.on("console", lambda m, errs=errs: m.type == "error" and errs.append(m.text))
            pg.on("pageerror", lambda e, errs=errs: errs.append(str(e)))
            t0 = time.time()
            pg.goto(url + "#m=" + quote(repo, safe=""))
            pg.wait_for_selector("#detail .detail", timeout=120_000)
            timing[name] = {"detail_s": round(time.time() - t0, 2), "errors": errs}
            if settings:
                w, kv, hw, ci, bi = settings
                for lab, v in (("Weights", w), ("KV cache", kv), ("Hardware", hw)):
                    if v:
                        pg.locator(f"#detail label.dctl:has(span:text-is('{lab}')) select").select_option(v)
                if bi is not None:
                    pg.locator("#detail label.dctl:has(span:text-is('Batch')) input").fill(str(bi))
            pg.evaluate(SCROLL, "#detail")
            pg.screenshot(path=os.path.join(a.out, name + "_a_summary.png"))
            pg.evaluate(SCROLL, "#detail .dcurves")
            pg.screenshot(path=os.path.join(a.out, name + "_b_curves.png"))
            for k in ("params", "flops", "memory", "layers", "roofline"):
                pg.evaluate(OPEN, k)
            for k in ("params", "layers"):
                pg.evaluate(SCROLL, f"#detail details[data-fold={k}]")
                pg.screenshot(path=os.path.join(a.out, f"{name}_c_{k}.png"))
            ctx.close()
        b.close()
    with open(os.path.join(a.out, "timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
