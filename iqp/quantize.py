"""Steps build_imatrix and quantize."""

from __future__ import annotations

import os

from iqp.config import STATIC_ARM
from iqp.parsers import parse_imatrix_output, parse_quantize_output
from iqp.util import LOG, StepError, atomic_write_json, read_json, run_cmd


def _gpu_args(cfg) -> list[str]:
    lc = cfg.raw["llamacpp"]
    a = ["-ngl", str(lc["n_gpu_layers"])]
    if int(lc.get("threads") or 0) > 0:
        a += ["-t", str(lc["threads"])]
    return a


def step_build_imatrix(cfg, force: bool = False) -> None:
    c = cfg.raw["calibration"]
    base = cfg.gguf_path(cfg.base_variant())
    if not base.exists():
        raise StepError(f"base GGUF missing: {base}")
    chunks = int(c["token_budget"]) // int(c["ctx"])
    for arm in cfg.calib_arms:
        out = cfg.imatrix_path(arm)
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists() and not force:
            LOG.info("skip (exists): %s", out)
            continue
        text = cfg.calib_text_path(arm)
        if not text.exists():
            raise StepError(f"calibration text missing for arm {arm}: {text}; run prepare_data")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part.gguf")
        log = run_cmd([str(cfg.bin("llama-imatrix")), "-m", str(base), "-f", str(text), "-o", str(tmp),
                       "-c", str(c["ctx"]), "-b", str(c["ctx"]), "--chunks", str(chunks),
                       "--output-frequency", "1000", *_gpu_args(cfg)],
                      cfg.logs_dir / "build_imatrix" / f"{arm}.log")
        info = parse_imatrix_output(log)
        if info["n_chunks"] != chunks:
            raise StepError(f"imatrix arm {arm} processed {info['n_chunks']} chunks, expected {chunks}; "
                            "the calibration file is too short")
        os.replace(tmp, out)
        atomic_write_json(meta, {"arm": arm, "chunks": chunks, "ctx": c["ctx"],
                                 "tokens": chunks * int(c["ctx"]), **info})
        LOG.info("imatrix %s: %d chunks x %d ctx", arm, chunks, c["ctx"])


def step_quantize(cfg, force: bool = False) -> None:
    base_v = cfg.base_variant()
    base = cfg.gguf_path(base_v)
    if not base.exists():
        raise StepError(f"base GGUF missing: {base}")
    threads = int(cfg.raw["quant"].get("threads") or 0) or (os.cpu_count() or 4)
    for v in cfg.quant_variants():
        out = cfg.gguf_path(v)
        meta = cfg.gguf_dir / f"{v.name}.quant.json"
        if out.exists() and meta.exists() and not force:
            LOG.info("skip (exists): %s", out.name)
            continue
        cmd = [str(cfg.bin("llama-quantize"))]
        if v.arm != STATIC_ARM:
            im = cfg.imatrix_path(v.arm)
            if not im.exists():
                raise StepError(f"imatrix missing for arm {v.arm}: {im}; run build_imatrix")
            cmd += ["--imatrix", str(im)]
        tmp = out.with_name(out.name + ".part")
        cmd += [str(base), str(tmp), v.qtype, str(threads)]
        log = run_cmd(cmd, cfg.logs_dir / "quantize" / f"{v.name}.log")
        info = parse_quantize_output(log)
        os.replace(tmp, out)
        info.update({"variant": v.name, "qtype": v.qtype, "arm": v.arm, "file_bytes": out.stat().st_size})
        atomic_write_json(meta, info)
        LOG.info("quantized %s: %.1f MiB, %.3f bpw", v.name, info["quant_mib"], info["quant_bpw"])
    bm = cfg.gguf_dir / f"{base_v.name}.quant.json"
    if not bm.exists():
        atomic_write_json(bm, {"variant": base_v.name, "qtype": base_v.qtype, "arm": base_v.arm,
                               "file_bytes": base.stat().st_size, "quant_bpw": 16.0})


def load_sizes(cfg) -> list[dict]:
    rows = []
    for v in cfg.all_variants():
        meta = cfg.gguf_dir / f"{v.name}.quant.json"
        if meta.exists():
            m = read_json(meta)
            rows.append({"variant": v.name, "qtype": v.qtype, "arm": v.arm,
                         "file_bytes": m.get("file_bytes"), "file_gib": (m.get("file_bytes") or 0) / 2**30,
                         "bpw": m.get("quant_bpw")})
    return rows
