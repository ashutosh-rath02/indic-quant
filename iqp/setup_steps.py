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


def _build_cache_dir(cfg) -> Path | None:
    if not cfg.raw["llamacpp"].get("cache_build") or cfg.state_dir is None:
        return None
    return cfg.state_dir / "llamacpp-bin"


def _save_build_cache(cfg) -> None:
    """Keep the built binaries and shared libraries with the persisted state (tens of MB), so
    a later session can skip the CUDA build. Binaries use an absolute RUNPATH to build/bin, so
    the cache is only valid when llamacpp.dir is the same path in the next session."""
    cache = _build_cache_dir(cfg)
    if cache is None:
        return
    src = cfg.llamacpp_dir / "build" / "bin"
    tmp = cache.with_name(cache.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in src.iterdir():
        if f.name in REQUIRED_BINARIES or ".so" in f.name or f.name == "IQP_BUILD_STAMP.json":
            shutil.copy2(f, tmp / f.name, follow_symlinks=False)
    shutil.copy2(cfg.llamacpp_dir / "build" / "IQP_BUILD_STAMP.json", tmp / "IQP_BUILD_STAMP.json")
    atomic_write_text(tmp / "BUILD_DIR", str(src.resolve()))
    atomic_write_text(tmp / "CPU_FLAGS", cpu_signature())
    shutil.rmtree(cache, ignore_errors=True)
    os.replace(tmp, cache)
    size = sum(f.stat().st_size for f in cache.iterdir() if f.is_file() and not f.is_symlink())
    LOG.info("cached llama.cpp binaries in %s (%.1f MiB)", cache, size / 2**20)


def _restore_build_cache(cfg, want: dict) -> bool:
    import json
    cache = _build_cache_dir(cfg)
    if cache is None or not (cache / "IQP_BUILD_STAMP.json").exists():
        return False
    stamp = json.loads((cache / "IQP_BUILD_STAMP.json").read_text())
    bindir = cfg.llamacpp_dir / "build" / "bin"
    if stamp.get("config") != want:
        LOG.info("cached llama.cpp build does not match the configured commit or flags; rebuilding")
        return False
    if (cache / "BUILD_DIR").read_text().strip() != str(bindir.resolve()):
        LOG.info("cached llama.cpp build was made for another directory; rebuilding")
        return False
    cpu_file = cache / "CPU_FLAGS"
    portable = "-DGGML_NATIVE=OFF" in [str(a) for a in cfg.raw["llamacpp"].get("cmake_extra_args", [])]
    if not portable and (not cpu_file.exists() or cpu_file.read_text().strip() != cpu_signature()):
        # GGML_NATIVE builds use the build machine's instruction set; another CPU model can
        # crash with an illegal instruction, so only reuse the cache on the same CPU type.
        LOG.info("cached llama.cpp build was made on a different CPU type; rebuilding")
        return False
    if all((bindir / b).exists() for b in REQUIRED_BINARIES):
        return True
    bindir.mkdir(parents=True, exist_ok=True)
    for f in cache.iterdir():
        if f.name in ("BUILD_DIR", "IQP_BUILD_STAMP.json"):
            continue
        dst = bindir / f.name
        if f.is_symlink():
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(os.readlink(f))
        else:
            shutil.copy2(f, dst)
            dst.chmod(dst.stat().st_mode | 0o755)
    shutil.copy2(cache / "IQP_BUILD_STAMP.json", cfg.llamacpp_dir / "build" / "IQP_BUILD_STAMP.json")
    LOG.info("restored llama.cpp binaries from %s; skipping the build", cache)
    return True


def cpu_signature() -> str:
    """CPU model name and instruction set flags, used to decide if a cached build is safe."""
    import hashlib
    try:
        info = Path("/proc/cpuinfo").read_text()
    except OSError:
        import platform
        return platform.machine()
    model = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("model name")), "")
    flags = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("flags")), "")
    return model + " | " + hashlib.sha1(" ".join(sorted(flags.split())).encode()).hexdigest()[:16]


def nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    import re
    import subprocess
    try:
        out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def adjust_cuda_archs(args: list[str], version: tuple[int, int] | None) -> list[str]:
    """CUDA 13 removed support for compute capability below 7.5 (Maxwell, Pascal, Volta).
    Drop those entries from -DCMAKE_CUDA_ARCHITECTURES with a warning instead of failing."""
    if version is None or version[0] < 13:
        return args
    out = []
    for a in args:
        if a.startswith("-DCMAKE_CUDA_ARCHITECTURES="):
            archs = a.split("=", 1)[1].split(";")
            keep = [x for x in archs if not x[:2].isdigit() or int(x[:2]) >= 75]
            if keep != archs:
                LOG.warning("nvcc %d.%d cannot build for %s (dropped in CUDA 13); building for %s only. "
                            "A P100 will not work with this build.", version[0], version[1],
                            ";".join(sorted(set(archs) - set(keep))), ";".join(keep) or "native")
            a = "-DCMAKE_CUDA_ARCHITECTURES=" + (";".join(keep) or "native")
        out.append(a)
    return out


def find_nvcc() -> Path | None:
    w = shutil.which("nvcc")
    if w:
        return Path(w)
    cands = []
    if os.environ.get("CUDA_HOME"):
        cands.append(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc")
    cands.append(Path("/usr/local/cuda/bin/nvcc"))
    cands += sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True)
    for c in cands:
        if c.exists():
            return c
    return None


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

    if not force:
        _restore_build_cache(cfg, want)
    have_bins = all((d / "build" / "bin" / b).exists() for b in REQUIRED_BINARIES)
    if stamp.exists() and have_bins and not force:
        import json
        if json.loads(stamp.read_text()).get("config") == want:
            LOG.info("llama.cpp build is up to date")
        else:
            force = True
    if force or not have_bins or not stamp.exists():
        env = {}
        if lc["cuda"]:
            nvcc = find_nvcc()
            if nvcc is None:
                raise StepError("llamacpp.cuda is true but nvcc was not found on PATH, in $CUDA_HOME/bin or "
                                "in /usr/local/cuda*/bin. Install the CUDA toolkit (on Kaggle, choose a GPU "
                                "accelerator; the GPU images ship nvcc), or set llamacpp.cuda: false for a "
                                "CPU build.")
            LOG.info("nvcc: %s", nvcc)
            env["PATH"] = str(nvcc.parent) + ":" + os.environ.get("PATH", "")
            env["CUDACXX"] = str(nvcc)
        args = ["cmake", "-S", str(d), "-B", str(d / "build"), "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"]
        args.append("-DGGML_CUDA=ON" if lc["cuda"] else "-DGGML_CUDA=OFF")
        extra = list(lc["cmake_extra_args"])
        if lc["cuda"]:
            extra = adjust_cuda_archs(extra, nvcc_version(Path(env["CUDACXX"])))
        args += extra
        run_cmd(args, logs / "cmake_configure.log", env=env)
        jobs = int(lc.get("build_jobs") or 0) or (os.cpu_count() or 4)
        run_cmd(["cmake", "--build", str(d / "build"), "--config", "Release", "-j", str(jobs), "--target",
                 *REQUIRED_BINARIES], logs / "cmake_build.log", env=env)
        atomic_write_json(stamp, {"config": want, "head": head})
        _save_build_cache(cfg)

    missing = [b for b in REQUIRED_BINARIES if not (d / "build" / "bin" / b).exists()]
    if missing:
        raise StepError(f"build finished but binaries are missing: {missing}")
    cache = _build_cache_dir(cfg)
    if cache is not None and not (cache / "IQP_BUILD_STAMP.json").exists():
        _save_build_cache(cfg)

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
    if not force and (cfg.meta_dir / "BASE_DTYPE").exists() and cfg.gguf_path(cfg.base_variant()).exists() \
            and (cfg.hf_dir / "config.json").exists():
        LOG.info("skip: base GGUF already exists")
        return
    if cfg.dry_run:
        from iqp.fakes.shims import fake_hf_snapshot
        fake_hf_snapshot(cfg)
        atomic_write_text(done, "dry-run\n")
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    token = hf_token(cfg)
    need = _remote_weight_bytes(m["hf_id"], m.get("revision") or "main", token)
    if need:
        from iqp.diskcheck import require_free
        require_free(out, need, f"download of {m['hf_id']}")
    try:
        snapshot_download(repo_id=m["hf_id"], revision=m.get("revision") or "main", local_dir=str(out),
                          token=token,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
    except GatedRepoError as e:
        raise StepError(f"{m['hf_id']} is gated. Accept the license on huggingface.co and export "
                        f"{m.get('hf_token_env', 'HF_TOKEN')}. ({e})")
    atomic_write_text(done, m["hf_id"] + "\n")
    LOG.info("downloaded %s to %s", m["hf_id"], out)


def _remote_weight_bytes(repo: str, revision: str, token) -> int:
    """Total size of the safetensors files in a model repo, or 0 if it cannot be read."""
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
        return sum((s.size or 0) for s in (info.siblings or []) if s.rfilename.endswith(".safetensors"))
    except Exception as e:  # gated or offline: the download itself reports the real error
        LOG.info("could not read file sizes for %s (%s); skipping the disk pre-check", repo, e)
        return 0


def step_convert(cfg, force: bool = False) -> None:
    base = cfg.base_variant() if (cfg.meta_dir / "BASE_DTYPE").exists() else None
    dtype = base.qtype if base else cfg.base_dtype()
    from iqp.config import BASE_ARM, Variant
    v = Variant(dtype, BASE_ARM)
    out = cfg.gguf_path(v)
    if out.exists() and not force:
        LOG.info("skip (exists): %s", out)
        atomic_write_text(cfg.meta_dir / "BASE_DTYPE", dtype)
        _maybe_delete_hf(cfg)
        return
    if not (cfg.hf_dir / "config.json").exists():
        raise StepError(f"HF model not found in {cfg.hf_dir}; run the download step first")
    if not any(cfg.hf_dir.glob("*.safetensors")) and not cfg.dry_run:
        raise StepError(f"no safetensors in {cfg.hf_dir} (deleted after an earlier conversion?); "
                        "rerun the download step with --force")
    cfg.gguf_dir.mkdir(parents=True, exist_ok=True)
    hf_bytes = sum(f.stat().st_size for f in cfg.hf_dir.glob("*.safetensors"))
    from iqp.diskcheck import require_free
    # A BF16/F16 text GGUF is at most the size of the HF weights (the vision tower is dropped).
    require_free(cfg.gguf_dir, int(hf_bytes * 1.05), "convert (base GGUF)")
    tmp = out.with_name(out.name + ".part")
    run_cmd([str(convert_python(cfg)), str(cfg.llamacpp_dir / "convert_hf_to_gguf.py"), str(cfg.hf_dir),
             "--outtype", dtype.lower(), "--outfile", str(tmp)], cfg.logs_dir / "convert" / f"{v.name}.log")
    os.replace(tmp, out)
    atomic_write_text(cfg.meta_dir / "BASE_DTYPE", dtype)
    atomic_write_json(cfg.quant_meta_path(v),
                      {"variant": v.name, "qtype": v.qtype, "arm": v.arm, "file_bytes": out.stat().st_size,
                       "quant_bpw": 16.0})
    _maybe_delete_hf(cfg)
    LOG.info("base GGUF: %s (%s, %.2f GiB)", out, dtype, out.stat().st_size / 2**30)


def _maybe_delete_hf(cfg) -> None:
    """Free the HF weights once the base GGUF exists. The tokenizer and config files are kept,
    since prepare_data counts tokens with tokenizer.json and base_dtype reads config.json."""
    if not cfg.raw["model"].get("delete_hf_weights_after_convert"):
        return
    freed = 0
    for f in list(cfg.hf_dir.glob("*.safetensors")):
        try:
            freed += f.stat().st_size
            f.unlink()
        except OSError as e:  # e.g. a read-only symlink target
            LOG.warning("could not delete %s: %s", f, e)
    if freed:
        LOG.info("deleted HF safetensors after conversion (%.2f GiB freed)", freed / 2**30)
