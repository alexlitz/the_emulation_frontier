"""Screenshots of the zoo page for the competitive evaluation (dist/zoo_eval_shots/), in headless Chromium.

Run:  python -m graph.zoo_eval_shots [--page dist/zoo.html] [--out dist/zoo_eval_shots]

Each view is opened from file:// by its permalink; the load time (navigation to the header counts) and any console
error are written to timing.json. Card views also get the panel alone and the panel at full height.
"""
from __future__ import annotations

import argparse
import json
import os
import time

DESKTOP = {"width": 1400, "height": 900}
PHONE = {"width": 375, "height": 800}
C4 = "meta-llama/Llama-3.1-8B,google/gemma-2-9b,Qwen/Qwen3-8B,Qwen/Qwen3-30B-A3B"
SHOTS = [  # name, hash, viewport, mobile, card
    ("01_landing_graph", "", DESKTOP, False, False),
    ("02_models_table", "#v=models", DESKTOP, False, False),
    ("03_card_gemma4_31b", "#m=nvidia%2FGemma-4-31B-IT-NVFP4", DESKTOP, False, True),
    ("04_card_deepseek_v3", "#m=deepseek-ai%2FDeepSeek-V3", DESKTOP, False, True),
    ("05_card_qwen3_next", "#m=Qwen%2FQwen3-Next-80B-A3B-Instruct", DESKTOP, False, True),
    ("06_compare4", "#v=compare&c=" + ",".join(x.replace("/", "%2F") for x in C4.split(",")), DESKTOP, False, False),
    ("07_timeline", "#v=timeline", DESKTOP, False, False),
    ("08_glossary", "#v=glossary", DESKTOP, False, False),
    ("09_phone_models", "#v=models", PHONE, True, False),
    ("10_phone_card_llama", "#m=meta-llama%2FLlama-3.1-8B", PHONE, True, True),
    ("11_lookup_gemma4_31b_it", "#v=lookup&id=google%2Fgemma-4-31B-it", DESKTOP, False, False),
    ("12_phone_landing", "", PHONE, True, False),
]


def main():
    from playwright.sync_api import sync_playwright
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="dist/zoo.html")
    ap.add_argument("--out", default="dist/zoo_eval_shots")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    url = "file://" + os.path.abspath(a.page)
    timing_path = os.path.join(a.out, "timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, h, vp, mobile, card in SHOTS:
            if a.only and a.only not in name:
                continue
            ctx = b.new_context(viewport=vp, is_mobile=mobile, has_touch=mobile, device_scale_factor=2 if mobile else 1)
            pg = ctx.new_page()
            errs = []
            pg.on("console", lambda m, errs=errs: m.type == "error" and errs.append(m.text))
            pg.on("pageerror", lambda e, errs=errs: errs.append(str(e)))
            t0 = time.time()
            pg.goto(url + h)
            pg.wait_for_selector("#stats .stat")
            timing[name] = round(time.time() - t0, 2)
            if "lookup" in h:
                pg.wait_for_function("() => { const s = document.getElementById('lk-status'); return s && s.textContent && !/Fetching/.test(s.textContent); }",
                                     timeout=60_000)
            pg.wait_for_timeout(1200)
            pg.screenshot(path=os.path.join(a.out, name + ".png"))
            if card:
                panel = pg.locator("#panel")
                panel.screenshot(path=os.path.join(a.out, name + "_panel.png"))
                if not mobile:
                    pg.evaluate("() => { const P = document.getElementById('panel'); P.style.maxHeight = 'none'; P.style.position = 'static'; }")
                    panel.screenshot(path=os.path.join(a.out, name + "_full_panel.png"))
            timing[name + "_errors"] = errs
            ctx.close()
        b.close()
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=1)
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
