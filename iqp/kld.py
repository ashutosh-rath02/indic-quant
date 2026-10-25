"""Step run_kld: base logits per language, then KL divergence for every quantized variant.

Flags verified in llama.cpp common/arg.cpp and tools/perplexity/perplexity.cpp:
  base run:  llama-perplexity -m BASE -f TEXT -c CTX -b BATCH --chunks N --kl-divergence-base FILE
  quant run: llama-perplexity -m QUANT -c CTX -b BATCH --kl-divergence-base FILE --kl-divergence
The quant run reads the tokens from FILE, so -f is not needed and every variant scores the
exact same token sequence. Statistics cover the second half of each chunk (n_ctx/2 - 1
positions per chunk), as in perplexity.cpp. Fewer than 100 scored tokens prints no summary.
"""

from __future__ import annotations

import math
import struct

from iqp.parsers import ParseError, parse_base_ppl_output, parse_kld_output
from iqp.prepare import TokenCounter
from iqp.quantize import _gpu_args
from iqp.util import LOG, StepError, atomic_write_json, read_json, run_cmd


def plan_chunks(token_counts: dict[str, int], ctx: int, max_chunks: int, equalize: bool) -> dict[str, int]:
    """Number of chunks per language. llama-perplexity needs at least 2*ctx tokens."""
    avail = {c: n // ctx for c, n in token_counts.items()}
    short = {c: n for c, n in token_counts.items() if n < 2 * ctx}
    if short:
        raise StepError(f"eval text too short for ctx={ctx} (need >= {2 * ctx} tokens): {short}")
    if equalize:
        n = min(min(avail.values()), max_chunks)
        return {c: n for c in avail}
    return {c: min(a, max_chunks) for c, a in avail.items()}


def read_logits_header(path) -> dict:
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic != b"_logits_":
            raise StepError(f"{path} is not a llama.cpp logits file")
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", fh.read(12))
    return {"n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk}


def logits_file_bytes(n_ctx: int, n_vocab: int, n_chunk: int) -> int:
    nv = 2 * ((n_vocab + 1) // 2) + 4
    return 20 + 4 * n_ctx * n_chunk + n_chunk * (n_ctx - 1 - n_ctx // 2) * nv * 2


def step_run_kld(cfg, force: bool = False) -> None:
    k = cfg.raw["kld"]
    ctx, batch = int(k["ctx"]), int(k["batch"])
    base_v = cfg.base_variant()
    base = cfg.gguf_path(base_v)
    if not base.exists():
        raise StepError(f"base GGUF missing: {base}")
    plan_path = cfg.kld_base_dir / "plan.json"
    if plan_path.exists() and not force:
        plan = read_json(plan_path)
    else:
        counter = TokenCounter(cfg)
        counts = {c: counter.llama_count(cfg.eval_kld_path(c).read_text(encoding="utf-8"), f"kld_{c}")
                  for c in cfg.langs}
        chunks = plan_chunks(counts, ctx, int(k["max_chunks"]), bool(k["equalize_chunks"]))
        plan = {"ctx": ctx, "token_counts": counts, "chunks": chunks,
                "scored_tokens_per_lang": {c: n * (ctx - 1 - ctx // 2) for c, n in chunks.items()}}
        atomic_write_json(plan_path, plan)
    LOG.info("KLD plan: ctx=%d chunks=%s", ctx, plan["chunks"])

    raw = cfg.raw_results_dir / "kld"
    for code in cfg.langs:
        logits = cfg.kld_base_dir / f"{code}.logits"
        base_json = cfg.kld_base_dir / f"{code}.base.json"
        pending = [v for v in cfg.quant_variants() if force or not (raw / v.name / f"{code}.json").exists()]
        if not pending and base_json.exists():
            continue
        if not (logits.exists() and base_json.exists()) or force:
            logits.parent.mkdir(parents=True, exist_ok=True)
            out = run_cmd([str(cfg.bin("llama-perplexity")), "-m", str(base), "-f", str(cfg.eval_kld_path(code)),
                           "-c", str(ctx), "-b", str(batch), "--chunks", str(plan["chunks"][code]),
                           "--kl-divergence-base", str(logits), *_gpu_args(cfg)],
                          cfg.logs_dir / "run_kld" / f"base_{code}.log")
            info = parse_base_ppl_output(out)
            info["header"] = read_logits_header(logits)
            atomic_write_json(base_json, info)
            LOG.info("base logits %s: PPL %.3f over %d chunks (%.2f GiB)", code, info["ppl"],
                     info["header"]["n_chunk"], logits.stat().st_size / 2**30)
        for v in cfg.quant_variants():
            res = raw / v.name / f"{code}.json"
            if res.exists() and not force:
                continue
            q = cfg.gguf_path(v)
            if not q.exists():
                LOG.warning("missing quantized model %s; skipping", q.name)
                continue
            out = run_cmd([str(cfg.bin("llama-perplexity")), "-m", str(q), "-c", str(ctx), "-b", str(batch),
                           "--kl-divergence-base", str(logits), "--kl-divergence", *_gpu_args(cfg)],
                          cfg.logs_dir / "run_kld" / v.name / f"{code}.log")
            try:
                stats = parse_kld_output(out)
            except ParseError as e:
                raise StepError(f"KLD parse failed for {v.name}/{code}: {e}")
            stats.update({"variant": v.name, "language": code, "ctx": ctx})
            atomic_write_json(res, stats)
            LOG.info("KLD %-22s %s mean=%.5f p99=%.4f same_top=%.2f%%", v.name, code, stats["kld_mean"],
                     stats["kld_p99"], stats["same_top_p_pct"])
        if k.get("delete_base_logits_when_done") and logits.exists() and all(
                (raw / v.name / f"{code}.json").exists() for v in cfg.quant_variants()):
            logits.unlink()
            LOG.info("deleted base logits for %s (all variants scored)", code)


def estimate_logits_disk(n_vocab: int, ctx: int, chunks: int, n_langs: int) -> float:
    return n_langs * logits_file_bytes(ctx, n_vocab, chunks) / 2**30


def kld_combined_unc(a_unc: float, b_unc: float) -> float:
    """Conservative uncertainty of a difference (ignores the positive correlation between arms)."""
    return math.sqrt(a_unc ** 2 + b_unc ** 2)
