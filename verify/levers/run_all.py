"""Run every lever script in order; each one's stdout is saved to verify/levers/out/<name>.txt.

Run:  python -m verify.levers.run_all
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ORDER = ("rel", "family_code", "window_audit", "leverage", "relax", "extras", "summary", "eps_lever", "constructions")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in ORDER:
        mod = importlib.import_module(f"verify.levers.{name}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn = getattr(mod, "main", None) or getattr(mod, "run", None) or getattr(mod, "check", None)
            if name == "family_code":
                mod.capabilities()
            else:
                fn()
        with open(os.path.join(OUT, f"{name}.txt"), "w") as f:
            f.write(buf.getvalue())
        print(f"{name}: {len(buf.getvalue().splitlines())} lines")


if __name__ == "__main__":
    main()
