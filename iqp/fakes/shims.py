"""Install the fake tools into the dry-run llama.cpp directory."""

from __future__ import annotations

import json
import os
import stat
import sys

from iqp.config import REPO_ROOT

FAKE_BINS = ["llama-perplexity", "llama-quantize", "llama-imatrix", "llama-tokenize", "llama-server"]


def _wrapper(tool: str) -> str:
    return (f"#!{sys.executable}\n"
            "import os, sys\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            f"os.environ['IQP_FAKE_TOOL'] = {tool!r}\n"
            "from iqp.fakes.tool import main\n"
            "main()\n")


def install_fake_llamacpp(cfg) -> None:
    bindir = cfg.llamacpp_dir / "build" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for b in FAKE_BINS:
        p = bindir / b
        p.write_text(_wrapper(b if b != "llama-server" else "llama-tokenize"))
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    conv = cfg.llamacpp_dir / "convert_hf_to_gguf.py"
    conv.write_text(_wrapper("convert_hf_to_gguf.py"))
    os.chmod(conv, 0o755)


def fake_hf_snapshot(cfg) -> None:
    d = cfg.hf_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"_name_or_path": cfg.raw["model"]["hf_id"],
                                               "torch_dtype": "bfloat16", "n_params": 3.88e9}))
