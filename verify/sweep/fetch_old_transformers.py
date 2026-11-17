"""Download old transformers wheels (4.28 - 4.55) and extract their model sources as TEXT: the reference pool sweep.py
matches remote code against (never imported). Output: ~/.cache/upstream_refs/tf_old/<version>/transformers/models/..."""
import glob, os, subprocess, sys, tempfile, zipfile
VERS = ["4.28.1", "4.31.0", "4.33.3", "4.35.2", "4.36.2", "4.38.2", "4.40.2", "4.42.4", "4.44.2", "4.46.3",
        "4.48.3", "4.50.3", "4.53.3", "4.55.4"]
OUT = os.path.expanduser("~/.cache/upstream_refs/tf_old")
WHEELS = tempfile.mkdtemp(prefix="tfwheels-")
for v in VERS:
    dst = os.path.join(OUT, v)
    if os.path.isdir(dst):
        continue
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-q", f"transformers=={v}", "-d",
                    WHEELS], check=False)
    whl = glob.glob(os.path.join(WHEELS, f"transformers-{v}-*.whl"))
    if not whl:
        print("missing", v)
        continue
    with zipfile.ZipFile(whl[0]) as z:
        for n in z.namelist():
            b = os.path.basename(n)
            if n.startswith("transformers/models/") and n.endswith(".py") and (
                    b.startswith(("modeling_", "configuration_")) and not b.startswith(("modeling_tf_", "modeling_flax_"))):
                z.extract(n, dst)
    print(v, "ok")
