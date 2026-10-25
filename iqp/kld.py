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


def gpu_name() -> str:
    """Name of the first visible GPU, or "cpu". Recorded with every base run."""
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip().splitlines()
        return out[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def check_reference_gpu(cfg, codes) -> None:
    """Warn when base logits were made on a different GPU type than an earlier session's.

    Variants scored in different sessions are compared to base logits computed in that
    session. Numerics differ slightly between GPU types, so mixing a T4 session with a P100
    session adds a small noise floor to the KLD of the near-lossless variants.
    """
    hist_path = cfg.kld_meta_dir / "reference_gpus.json"
    hist = read_json(hist_path) if hist_path.exists() else {}
    now = gpu_name()
    for code in codes:
        seen = hist.setdefault(code, [])
        if seen and now not in seen:
            LOG.warning("base logits for %s were computed on %s in an earlier session and on %s now. Choose "
                        "the same accelerator in every session; results stay usable but variants scored in "
                        "different sessions have slightly different references.", code, ", ".join(seen), now)
        if now not in seen:
            seen.append(now)
    atomic_write_json(hist_path, hist)


def kld_plan(cfg, force: bool = False) -> dict:
    """Chunks per language, computed once and kept with the persisted state so a later session
    scores exactly the same token sequences."""
    k = cfg.raw["kld"]
    ctx = int(k["ctx"])
    plan_path = cfg.kld_meta_dir / "plan.json"
    if plan_path.exists() and not force:
        plan = read_json(plan_path)
        missing = [c for c in cfg.langs if c not in plan["chunks"]]
        if not missing and int(plan["ctx"]) == ctx:
            return plan
        LOG.info("KLD plan is stale (languages %s or ctx changed); recomputing", missing)
    counter = TokenCounter(cfg)
    counts = {c: counter.llama_count(cfg.eval_kld_path(c).read_text(encoding="utf-8"), f"kld_{c}")
              for c in cfg.langs}
    chunks = plan_chunks(counts, ctx, int(k["max_chunks"]), bool(k["equalize_chunks"]))
    plan = {"ctx": ctx, "token_counts": counts, "chunks": chunks,
            "scored_tokens_per_lang": {c: n * (ctx - 1 - ctx // 2) for c, n in chunks.items()}}
    atomic_write_json(plan_path, plan)
    LOG.info("KLD plan: ctx=%d chunks=%s", ctx, plan["chunks"])
    return plan


def logits_path(cfg, code: str):
    return cfg.kld_base_dir / f"{code}.logits"


def base_json_path(cfg, code: str):
    return cfg.kld_meta_dir / f"{code}.base.json"


def kld_result_path(cfg, v, code: str):
    return cfg.raw_results_dir / "kld" / v.name / f"{code}.json"


def ensure_base_logits(cfg, code: str, plan: dict, force: bool = False) -> None:
    """Run the base model once over a language's KLD text and save its logits."""
    k = cfg.raw["kld"]
    logits = logits_path(cfg, code)
    base_json = base_json_path(cfg, code)
    if logits.exists() and base_json.exists() and not force:
        return
    base = cfg.gguf_path(cfg.base_variant())
    if not base.exists():
        raise StepError(f"base GGUF missing: {base}")
    logits.parent.mkdir(parents=True, exist_ok=True)
    vocab = cfg.vocab_size()
    if vocab:
        from iqp.diskcheck import require_free
        require_free(logits.parent, logits_file_bytes(int(plan["ctx"]), vocab, int(plan["chunks"][code])),
                     f"base logits for {code}")
    out = run_cmd([str(cfg.bin("llama-perplexity")), "-m", str(base), "-f", str(cfg.eval_kld_path(code)),
                   "-c", str(plan["ctx"]), "-b", str(k["batch"]), "--chunks", str(plan["chunks"][code]),
                   "--kl-divergence-base", str(logits), *_gpu_args(cfg)],
                  cfg.logs_dir / "run_kld" / f"base_{code}.log")
    info = parse_base_ppl_output(out)
    info["header"] = read_logits_header(logits)
    info["gpu"] = gpu_name()
    atomic_write_json(base_json, info)
    LOG.info("base logits %s: PPL %.3f over %d chunks (%.2f GiB)", code, info["ppl"],
             info["header"]["n_chunk"], logits.stat().st_size / 2**30)


def kld_variant_lang(cfg, v, code: str, plan: dict, force: bool = False) -> dict | None:
    """Score one quantized variant on one language against the saved base logits."""
    res = kld_result_path(cfg, v, code)
    if res.exists() and not force:
        return read_json(res)
    q = cfg.gguf_path(v)
    if not q.exists():
        LOG.warning("missing quantized model %s; skipping", q.name)
        return None
    k = cfg.raw["kld"]
    out = run_cmd([str(cfg.bin("llama-perplexity")), "-m", str(q), "-c", str(plan["ctx"]), "-b", str(k["batch"]),
                   "--kl-divergence-base", str(logits_path(cfg, code)), "--kl-divergence", *_gpu_args(cfg)],
                  cfg.logs_dir / "run_kld" / v.name / f"{code}.log")
    try:
        stats = parse_kld_output(out)
    except ParseError as e:
        raise StepError(f"KLD parse failed for {v.name}/{code}: {e}")
    stats.update({"variant": v.name, "language": code, "ctx": plan["ctx"],
                  "reference_gpu": read_json(base_json_path(cfg, code)).get("gpu")
                  if base_json_path(cfg, code).exists() else None})
    atomic_write_json(res, stats)
    LOG.info("KLD %-22s %s mean=%.5f p99=%.4f same_top=%.2f%%", v.name, code, stats["kld_mean"],
             stats["kld_p99"], stats["same_top_p_pct"])
    return stats


def maybe_delete_logits(cfg, code: str) -> bool:
    """Delete a language's base logits once every configured variant has been scored on it."""
    logits = logits_path(cfg, code)
    if not cfg.raw["kld"].get("delete_base_logits_when_done") or not logits.exists():
        return False
    if all(kld_result_path(cfg, v, code).exists() for v in cfg.configured_quant_variants()):
        logits.unlink()
        LOG.info("deleted base logits for %s (all variants scored)", code)
        return True
    return False


def step_run_kld(cfg, force: bool = False) -> None:
    base = cfg.gguf_path(cfg.base_variant())
    if not base.exists():
        raise StepError(f"base GGUF missing: {base}")
    plan = kld_plan(cfg, force)
    variants = cfg.run_quant_variants()
    for code in cfg.langs:
        pending = [v for v in variants if force or not kld_result_path(cfg, v, code).exists()]
        if not pending and base_json_path(cfg, code).exists():
            continue
        ensure_base_logits(cfg, code, plan, force)
        for v in variants:
            kld_variant_lang(cfg, v, code, plan, force)
        maybe_delete_logits(cfg, code)


def estimate_logits_disk(n_vocab: int, ctx: int, chunks: int, n_langs: int) -> float:
    return n_langs * logits_file_bytes(ctx, n_vocab, chunks) / 2**30


def kld_combined_unc(a_unc: float, b_unc: float) -> float:
    """Conservative uncertainty of a difference (ignores the positive correlation between arms)."""
    return math.sqrt(a_unc ** 2 + b_unc ** 2)
