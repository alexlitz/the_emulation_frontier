#!/usr/bin/env python3
"""A `clang` stand-in for tinygrad's CPU backend on a machine without clang: `zig cc` (the ziglang wheel
bundles clang).  pip install ziglang  in the tinygrad venv, then  CC=/path/to/zigcc.py DEV=CPU python recipe.py run
(set TEF_ZIG_PYTHON to that venv's python if this file's shebang finds another interpreter).

tinygrad calls  CC -c -x c -O2 -fPIC ... --target=x86_64-none-unknown-elf -march=native - -o -  (C source on
stdin, the object on stdout). zig cc wants a file in and a file out, a target it knows, and no debug info /
sanitizers / unwind tables (tinygrad's ELF loader does not take their relocations)."""
import os
import subprocess
import sys
import tempfile

PY = os.environ.get("TEF_ZIG_PYTHON", sys.executable)
EXTRA = ["-fPIC", "-g0", "-fno-sanitize=all", "-fno-stack-protector", "-fno-unwind-tables",
         "-fno-asynchronous-unwind-tables"]

args = []
for a in sys.argv[1:]:
    if a.startswith("--target="):
        args += ["-target", a.split("=", 1)[1].split("-")[0] + "-linux-none"] + EXTRA
    else:
        args.append(a)
with tempfile.TemporaryDirectory() as d:
    src, obj = os.path.join(d, "k.c"), os.path.join(d, "k.o")
    if "-" in args:
        with open(src, "wb") as f:
            f.write(sys.stdin.buffer.read())
        args[args.index("-")] = src
    if "-o" in args:
        args[args.index("-o") + 1] = obj
    r = subprocess.run([PY, "-m", "ziglang", "cc"] + args, stderr=subprocess.PIPE)
    if r.returncode:
        sys.stderr.write(r.stderr.decode())
        sys.exit(r.returncode)
    with open(obj, "rb") as f:
        sys.stdout.buffer.write(f.read())
