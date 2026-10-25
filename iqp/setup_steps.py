"""Steps: setup (build llama.cpp), download (HF snapshot), convert (HF to base GGUF)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from iqp.util import LOG, StepError, atomic_write_json, atomic_write_text, hf_token, run_cmd

REQUIRED_BINARIES = ["llama-perplexity", "llama-quantize", "llama-imatrix", "llama-server", "llama-tokenize"]


def convert_python(cfg) -> Path:
    if cfg.dry_run:
        return Path(sys.executable)
    return cfg.convert_venv / "bin" / "python"


def step_setup(cfg, force: bool = False) -> None:
    if cfg.dry_run:
        from iqp.fakes.shims import install_fake_llamacpp
        install_fake_llamacpp(cfg)
        LOG.info("dry run: installed fake llama.cpp tools in %s", cfg.llamacpp_dir)
        return

    lc = cfg.raw["llamacpp"]
    d = cfg.llamacpp_dir
    logs = cfg.logs_dir / "setup"
    stamp = d / "build" / "IQP_BUILD_STAMP.json"
    want = {"commit": lc["commit"], "cuda": bool(lc["cuda"]), "cmake_extra_args": list(lc["cmake_extra_args"])}

    if not (d / ".git").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", lc["repo"], str(d)], logs / "git_clone.log")
    run_cmd(["git", "-C", str(d), "fetch", "--tags", "origin"], logs / "git_fetch.log", check=False)
    run_cmd(["git", "-C", str(d), "checkout", "--detach", lc["commit"]], logs / "git_checkout.log")
    head = run_cmd(["git", "-C", str(d), "rev-parse", "HEAD"], logs / "git_head.log").strip().splitlines()[-1]
    LOG.info("llama.cpp at %s", head)

    have_bins = all((d / "build" / "bin" / b).exists() for b in REQUIRED_BINARIES)
    if stamp.exists() and have_bins and not force:
        import json
        if json.loads(stamp.read_text()).get("config") == want:
            LOG.info("llama.cpp build is up to date")
        else:
            force = True
    if force or not have_bins or not stamp.exists():
        if lc["cuda"] and shutil.which("nvcc") is None and not Path("/usr/local/cuda/bin/nvcc").exists():
            raise StepError("llamacpp.cuda is true but nvcc was not found. Install the CUDA toolkit or "
                            "add /usr/local/cuda/bin to PATH.")
        env = {}
        if Path("/usr/local/cuda/bin").exists():
            env["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
        args = ["cmake", "-S", str(d), "-B", str(d / "build"), "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"]
        args.append("-DGGML_CUDA=ON" if lc["cuda"] else "-DGGML_CUDA=OFF")
        args += list(lc["cmake_extra_args"])
        run_cmd(args, logs / "cmake_configure.log", env=env)
        jobs = int(lc.get("build_jobs") or 0) or (os.cpu_count() or 4)
        run_cmd(["cmake", "--build", str(d / "build"), "--config", "Release", "-j", str(jobs), "--target",
                 *REQUIRED_BINARIES], logs / "cmake_build.log", env=env)
        atomic_write_json(stamp, {"config": want, "head": head})

    missing = [b for b in REQUIRED_BINARIES if not (d / "build" / "bin" / b).exists()]
    if missing:
        raise StepError(f"build finished but binaries are missing: {missing}")

    # Converter venv from llama.cpp's own pinned requirements (they pin transformers and protobuf
    # versions that differ from this project's analysis environment).
    venv = cfg.convert_venv
    py = venv / "bin" / "python"
    marker = venv / "IQP_REQS_COMMIT"
    if not py.exists():
        run_cmd([sys.executable, "-m", "venv", str(venv)], logs / "venv_create.log")
    if force or not marker.exists() or marker.read_text().strip() != lc["commit"]:
        run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip"], logs / "venv_pip.log")
        run_cmd([str(py), "-m", "pip", "install", "-r",
                 str(d / "requirements" / "requirements-convert_hf_to_gguf.txt")], logs / "venv_reqs.log")
        atomic_write_text(marker, lc["commit"])
    run_cmd([str(cfg.bin("llama-perplexity")), "--version"], logs / "version.log", check=False)
    LOG.info("setup complete")


def step_download(cfg, force: bool = False) -> None:
    m = cfg.raw["model"]
    out = cfg.hf_dir
    done = out / "IQP_DOWNLOAD_DONE"
    if done.exists() and not force:
        LOG.info("skip (exists): %s", out)
        return
    if cfg.dry_run:
        from iqp.fakes.shims import fake_hf_snapshot
        fake_hf_snapshot(cfg)
        atomic_write_text(done, "dry-run\n")
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    token = hf_token(cfg)
    try:
        snapshot_download(repo_id=m["hf_id"], revision=m.get("revision") or "main", local_dir=str(out),
                          token=token,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
    except GatedRepoError as e:
        raise StepError(f"{m['hf_id']} is gated. Accept the license on huggingface.co and export "
                        f"{m.get('hf_token_env', 'HF_TOKEN')}. ({e})")
    atomic_write_text(done, m["hf_id"] + "\n")
    LOG.info("downloaded %s to %s", m["hf_id"], out)


def step_convert(cfg, force: bool = False) -> None:
    base = cfg.base_variant() if (cfg.gguf_dir / "BASE_DTYPE").exists() else None
    dtype = base.qtype if base else cfg.base_dtype()
    from iqp.config import BASE_ARM, Variant
    v = Variant(dtype, BASE_ARM)
    out = cfg.gguf_path(v)
    if out.exists() and not force:
        LOG.info("skip (exists): %s", out)
        atomic_write_text(cfg.gguf_dir / "BASE_DTYPE", dtype)
        return
    if not (cfg.hf_dir / "config.json").exists():
        raise StepError(f"HF model not found in {cfg.hf_dir}; run the download step first")
    cfg.gguf_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    run_cmd([str(convert_python(cfg)), str(cfg.llamacpp_dir / "convert_hf_to_gguf.py"), str(cfg.hf_dir),
             "--outtype", dtype.lower(), "--outfile", str(tmp)], cfg.logs_dir / "convert" / f"{v.name}.log")
    os.replace(tmp, out)
    atomic_write_text(cfg.gguf_dir / "BASE_DTYPE", dtype)
    atomic_write_json(cfg.gguf_dir / f"{v.name}.quant.json",
                      {"variant": v.name, "file_bytes": out.stat().st_size, "quant_bpw": 16.0})
    if cfg.raw["model"].get("delete_hf_weights_after_convert"):
        for f in cfg.hf_dir.glob("*.safetensors"):
            f.unlink()
        LOG.info("deleted HF safetensors after conversion")
    LOG.info("base GGUF: %s (%s)", out, dtype)
