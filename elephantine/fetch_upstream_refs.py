"""Fetch the upstream reference code the universal-host tests run against, into
~/.cache/upstream_refs, without installing anything:

  - mamba_ssm, causal_conv1d, flash-linear-attention: source tarballs (pure-torch `*_ref`
    functions and Triton kernels; layer definitions for fla), used by the plamo2 / Hymba /
    Phi-4-flash / Nemotron-Flash / GLA / RetNet / KDA tests.
  - transformers 5.18 + huggingface_hub 1.33 wheels, unpacked side by side; the Qwen3.5 / Diba /
    Agnes / dQwen3.5 / AliceAI tests run their references with these in a subprocess (the
    installed transformers 4.57 can't share a process with them).
  - granite-switch source at commit 24aa420 (the single-switch layout the HF checkpoint uses).
  - fla-core 0.5.2 wheel (its torch references naive_recurrent_kda / naive_kda_*gate, the KDA
    semantics Youtu-LLM's remote code calls with in-kernel gating).
  - awslabs hybrid-model-factory source at commit b9979c5 (Gated KalmaNet's layer, its Triton
    Chebyshev kernels and HybridQwen3, the GKA-primed HQwen3 test; its HF repo ships no code).


  - `vllm`: an isolated venv ~/.cache/vllm-env with vLLM 0.31.0 + the pb27-vllm plugin from the
    PseudoBrain-Qwen3 repos (the PseudoBrain test runs it in a subprocess on one GPU).

  python -m elephantine.fetch_upstream_refs          # everything but vLLM
  python -m elephantine.fetch_upstream_refs vllm     # the vLLM venv (~8 GB)
"""
import io
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile

ROOT = os.path.expanduser("~/.cache/upstream_refs")
SDISTS = {"mamba-ssm": "2.3.2.post1", "causal-conv1d": "1.7.0", "flash-linear-attention": "0.4.0"}
WHEELS = {"transformers": "5.18.0", "huggingface-hub": "1.33.0"}
TF5 = os.path.join(ROOT, "transformers-5.18.0")
FLA052 = ("fla-core", "0.5.2", os.path.join(ROOT, "fla_core-0.5.2"))
GRANITE = ("https://github.com/generative-computing/granite-switch/archive/24aa420f71b6703f96e40533d83041ce311c30ae.tar.gz",
           os.path.join(ROOT, "granite_switch-24aa420"))
HMF = ("https://github.com/awslabs/hybrid-model-factory/archive/b9979c5ac74915dc529868acea8d35e522e2217c.tar.gz",
       os.path.join(ROOT, "hybrid-model-factory"))


def _pypi(pkg, ver, kind):
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver}/json"))
    return next(u["url"] for u in meta["urls"] if u["packagetype"] == kind)


def main():
    os.makedirs(ROOT, exist_ok=True)
    for pkg, ver in SDISTS.items():
        url = _pypi(pkg, ver, "sdist")
        path = os.path.join(ROOT, os.path.basename(url))
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as tf:
            tf.extractall(ROOT, filter="data")
        print(f"{pkg} {ver} -> {ROOT}")
    os.makedirs(TF5, exist_ok=True)
    for pkg, ver in WHEELS.items():
        data = urllib.request.urlopen(_pypi(pkg, ver, "bdist_wheel")).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(TF5)
        print(f"{pkg} {ver} -> {TF5}")
    url, dest = GRANITE
    with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())) as tf:
        tf.extractall(ROOT + "/_gs", filter="data")
    top = os.path.join(ROOT, "_gs", os.listdir(ROOT + "/_gs")[0], "src")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(top, dest)
    shutil.rmtree(ROOT + "/_gs")
    print(f"granite-switch @24aa420 -> {dest}")
    fetch_fla052()
    fetch_hmf()


def fetch_fla052():
    pkg, ver, dest = FLA052
    data = urllib.request.urlopen(_pypi(pkg, ver, "bdist_wheel")).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    print(f"{pkg} {ver} -> {dest}")


def fetch_hmf():
    url, dest = HMF
    with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())) as tf:
        tf.extractall(ROOT + "/_hmf", filter="data")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(os.path.join(ROOT, "_hmf", os.listdir(ROOT + "/_hmf")[0]), dest)
    shutil.rmtree(ROOT + "/_hmf")
    print(f"hybrid-model-factory @b9979c5 -> {dest}")


def fetch_vllm_env():
    """The PseudoBrain reference: vLLM 0.31.0 (torch 2.13 / cu130, transformers 5.17; ~8 GB) in an
    isolated venv at ~/.cache/vllm-env (never the repo's python), plus the pb27-vllm 0.2.0 plugin
    the PseudoBrain-Qwen3 repos ship (code only, no weights), installed --no-deps."""
    import subprocess
    from huggingface_hub import snapshot_download
    env, py = VLLM_ENV, os.path.join(VLLM_ENV, "bin", "python")
    if not os.path.exists(py):
        subprocess.run(["uv", "venv", env, "--python", "3.12"], check=True)
    subprocess.run(["uv", "pip", "install", "--python", py, f"vllm=={VLLM_VER}"], check=True)
    src = snapshot_download(PB27_REPO, allow_patterns=["pb27_vllm_plugin/*", "config.json"],
                            local_dir=os.path.join(ROOT, "pb27_vllm"))
    subprocess.run(["uv", "pip", "install", "--python", py, "--no-deps",
                    os.path.join(src, "pb27_vllm_plugin")], check=True)
    print(f"vllm {VLLM_VER} + pb27-vllm ({PB27_REPO}) -> {env}")


VLLM_ENV = os.path.expanduser("~/.cache/vllm-env")
VLLM_VER = "0.31.0"
PB27_REPO = "defnoteatin/PseudoBrain-Qwen3-32B-BF16"


if __name__ == "__main__":
    import sys
    fetch_vllm_env() if sys.argv[1:] == ["vllm"] else main()
