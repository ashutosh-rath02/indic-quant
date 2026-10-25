"""Step stream: process one quantized variant at a time so disk use stays small.

For each variant: quantize, KLD on every language, MILU and translation if the variant is
selected for them, record timings, delete the variant GGUF. Base logits are computed once per
language before the first variant and kept until every configured variant has been scored on
that language. Peak disk is the base GGUF, the base logits of all languages, one quantized
variant and small files.

Resumable at variant granularity and below: every per-language result file is written
atomically as soon as it exists, so an interrupted session loses at most the variant that was
running. A time budget (--max-hours) stops cleanly before starting a variant that would
probably not finish, using the per-variant timings measured so far.
"""

from __future__ import annotations

import os
import time

from iqp import evals, kld
from iqp.diskcheck import GIB, require_free
from iqp.quantize import quantize_variant, write_base_meta
from iqp.util import LOG, StepError, atomic_write_json, read_json

STATUS_FILE = "stream_status.json"
TIMINGS_FILE = "stream_timings.json"

# Bits per weight used only to estimate the disk a quantized variant needs before it exists.
APPROX_BPW = {"Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q4_K_M": 4.9, "Q4_K_S": 4.6,
              "Q4_0": 4.5, "Q3_K_M": 4.0, "Q3_K_S": 3.6, "Q3_K_L": 4.3, "Q2_K": 3.2, "IQ4_XS": 4.3,
              "IQ3_M": 3.7, "IQ3_XXS": 3.1, "IQ2_M": 2.9, "IQ2_S": 2.6, "IQ2_XS": 2.4, "IQ2_XXS": 2.2}


def _scfg(cfg) -> dict:
    s = {"delete_variant_gguf": True, "safety_factor": 1.15, "aggregate_each_variant": True,
         "fallback_minutes": {}, "priority": [], "bundle_each_variant": False}
    s.update(cfg.raw.get("stream") or {})
    return s


def _state_root(cfg):
    return (cfg.state_dir or cfg.work_dir) / "stream" / cfg.model_name


def status_path(cfg):
    return cfg.results_dir / STATUS_FILE


def timings_path(cfg):
    return _state_root(cfg) / TIMINGS_FILE


def load_timings(cfg) -> dict:
    p = timings_path(cfg)
    if p.exists():
        try:
            return read_json(p)
        except Exception:  # a torn write cannot happen (atomic), but be lenient with old files
            LOG.warning("could not read %s; starting with empty timings", p)
    return {"variants": {}}


# ------------------------------------------------------------------ work per variant

def pending_tasks(cfg, v, milu_names: set[str], mt_names: set[str], force: bool = False) -> dict:
    kld_langs = [] if v.is_base else [c for c in cfg.langs
                                      if force or not kld.kld_result_path(cfg, v, c).exists()]
    milu = v.name in milu_names and bool(evals.milu_todo(cfg, v, force))
    mt = v.name in mt_names and bool(evals.mt_todo(cfg, v, force))
    return {"kld_langs": kld_langs, "milu": milu, "mt": mt}


def is_done(tasks: dict) -> bool:
    return not tasks["kld_langs"] and not tasks["milu"] and not tasks["mt"]


def _mean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return sum(xs) / len(xs) if xs else None


def estimate_seconds(cfg, v, tasks: dict, timings: dict) -> float | None:
    """Expected wall time for the pending work of one variant, from measured timings.

    Returns None when nothing has been measured and no fallback is configured, in which case
    the variant is started (there is no basis for refusing it).
    """
    recs = list(timings.get("variants", {}).values())
    fb = {k: float(m) * 60 for k, m in (_scfg(cfg).get("fallback_minutes") or {}).items()}
    total, known = 0.0, False

    def part(key, measured, n=1.0):
        nonlocal total, known
        val = measured if measured is not None else fb.get(key)
        if val is not None:
            total += val * n
            known = True

    if not v.is_base:
        is_iq = v.qtype.startswith("IQ")
        same_kind = [r.get("quantize_s") for r in recs if r.get("qtype", "").startswith("IQ") == is_iq]
        part("quantize", _mean(same_kind) or _mean([r.get("quantize_s") for r in recs]))
        if tasks["kld_langs"]:
            per_lang = _mean([r["kld_s"] / r["kld_langs"] for r in recs if r.get("kld_langs")])
            part("kld_per_lang", per_lang, len(tasks["kld_langs"]))
    if tasks["milu"]:
        part("milu", _mean([r.get("milu_s") for r in recs]))
    if tasks["mt"]:
        part("mt", _mean([r.get("mt_s") for r in recs]))
    return total if known else None


def order_variants(variants: list, priority: list[str]) -> list:
    """Stable sort: variants matching an earlier priority pattern come first."""
    import fnmatch

    def rank(v):
        for i, pat in enumerate(priority or []):
            if (pat.lower() == "base" and v.is_base) or fnmatch.fnmatchcase(v.name, pat):
                return i
        return len(priority or [])
    return sorted(variants, key=rank)


def _variant_disk_bytes(cfg, v) -> int:
    base = cfg.gguf_path(cfg.base_variant())
    if not base.exists():
        return 0
    bpw = APPROX_BPW.get(v.qtype, 8.5)
    return int(base.stat().st_size * bpw / 16 * 1.08)


def process_variant(cfg, v, tasks: dict, plan: dict, force: bool = False) -> dict:
    """Run every pending task for one variant and return its timing record."""
    rec = {"variant": v.name, "qtype": v.qtype, "arm": v.arm, "started": time.time(),
           "quantize_s": 0.0, "kld_s": 0.0, "kld_langs": 0, "milu_s": 0.0, "mt_s": 0.0}
    gguf = cfg.gguf_path(v)
    if not v.is_base and not (gguf.exists() and cfg.quant_meta_path(v).exists() and not force):
        require_free(gguf.parent, _variant_disk_bytes(cfg, v), f"quantized variant {v.name}")
        t0 = time.time()
        quantize_variant(cfg, v, force=True)
        rec["quantize_s"] = time.time() - t0
    if tasks["kld_langs"]:
        t0 = time.time()
        for code in tasks["kld_langs"]:
            kld.kld_variant_lang(cfg, v, code, plan, force)
        rec["kld_s"] = time.time() - t0
        rec["kld_langs"] = len(tasks["kld_langs"])
    if tasks["milu"]:
        t0 = time.time()
        evals.milu_variant(cfg, v, force)
        rec["milu_s"] = time.time() - t0
    if tasks["mt"]:
        t0 = time.time()
        evals.mt_variant(cfg, v, force)
        rec["mt_s"] = time.time() - t0
    rec["total_s"] = time.time() - rec["started"]
    if gguf.exists():
        rec["gguf_bytes"] = gguf.stat().st_size
    return rec


def _delete_variant(cfg, v) -> None:
    if v.is_base or not _scfg(cfg)["delete_variant_gguf"]:
        return
    g = cfg.gguf_path(v)
    if g.exists():
        size = g.stat().st_size
        g.unlink()
        LOG.info("deleted %s (%.2f GiB freed)", g.name, size / GIB)


def _clean_partials(cfg) -> None:
    for p in cfg.gguf_dir.glob("*.part"):
        LOG.info("removing partial file from an interrupted run: %s", p.name)
        p.unlink()


def _write_status(cfg, **kw) -> None:
    atomic_write_json(status_path(cfg), {"model": cfg.model_name, "updated": time.time(), **kw})


# ------------------------------------------------------------------ step

def step_stream(cfg, force: bool = False) -> None:
    from iqp.fertility import step_fertility

    # IQP_SESSION_START (unix time, set by the Kaggle notebook in its first cell) makes the budget
    # count from the start of the session instead of the start of this step.
    t_start = float(os.environ.get("IQP_SESSION_START") or time.time())
    max_hours = cfg.runtime.get("max_hours")
    budget_s = float(max_hours) * 3600 if max_hours else None
    s = _scfg(cfg)
    base_v = cfg.base_variant()
    base = cfg.gguf_path(base_v)
    if not base.exists():
        raise StepError(f"base GGUF missing: {base}; run download and convert first")
    if not cfg.eval_kld_path(cfg.langs[0]).exists():
        raise StepError("prepared data missing; run prepare_data (or restore) first")
    run_vs = order_variants(cfg.run_variants(), s.get("priority") or [])
    quant_vs = [v for v in run_vs if not v.is_base]
    for arm in sorted({v.arm for v in quant_vs if v.arm not in ("static", "base")}):
        if not cfg.imatrix_path(arm).exists():
            raise StepError(f"imatrix for arm {arm} missing: {cfg.imatrix_path(arm)}; run build_imatrix")
    cfg.gguf_dir.mkdir(parents=True, exist_ok=True)
    _clean_partials(cfg)
    from iqp.session import write_marker
    write_marker(cfg)  # so even a session that is cut off leaves a restorable output
    write_base_meta(cfg)
    step_fertility(cfg, force=False)

    milu_names = {v.name for v in evals.milu_variants(cfg)}
    mt_names = {v.name for v in evals.mt_variants(cfg)}
    work = []
    for v in run_vs:
        tasks = pending_tasks(cfg, v, milu_names, mt_names, force)
        if is_done(tasks):
            LOG.info("skip (done): %s", v.name)
            _delete_variant(cfg, v)  # a GGUF left behind by an interrupted run
            continue
        work.append((v, tasks))
    LOG.info("stream: %d of %d variants pending%s", len(work), len(run_vs),
             f", time budget {max_hours} h" if budget_s else "")

    plan = kld.kld_plan(cfg, force=False)
    langs_needed = sorted({c for _, t in work for c in t["kld_langs"]}, key=cfg.langs.index)
    vocab = cfg.vocab_size()
    if langs_needed and vocab:
        need = sum(kld.logits_file_bytes(int(plan["ctx"]), vocab, int(plan["chunks"][c]))
                   for c in langs_needed if not kld.logits_path(cfg, c).exists())
        biggest = max((_variant_disk_bytes(cfg, v) for v, t in work if not v.is_base), default=0)
        require_free(cfg.kld_base_dir, need + biggest,
                     f"base logits for {len(langs_needed)} languages plus the largest variant")
    for code in langs_needed:
        kld.ensure_base_logits(cfg, code, plan, force=False)
    kld.check_reference_gpu(cfg, langs_needed)

    timings = load_timings(cfg)
    done_now, stopped_before = [], None
    for i, (v, tasks) in enumerate(work):
        elapsed = time.time() - t_start
        if budget_s is not None:
            est = estimate_seconds(cfg, v, tasks, timings)
            if est is not None and elapsed + est * float(s["safety_factor"]) > budget_s:
                stopped_before = v.name
                remaining = [w.name for w, _ in work[i:]]
                LOG.warning("time budget: %.2f h used, %s needs about %.0f min (x%.2f safety) and would pass "
                            "the %.2f h budget. Stopping cleanly; %d variants remain. Resume next session: "
                            "restore this session's output and rerun the stream step.",
                            elapsed / 3600, v.name, est / 60, float(s["safety_factor"]), budget_s / 3600,
                            len(remaining))
                _write_status(cfg, complete=False, reason="time budget", stopped_before=v.name,
                              remaining=remaining, done_this_session=done_now, elapsed_s=elapsed)
                break
            LOG.info("variant %d/%d %s: %.2f h used, estimate %s", i + 1, len(work), v.name, elapsed / 3600,
                     f"{est / 60:.1f} min" if est is not None else "unknown")
        else:
            LOG.info("variant %d/%d %s", i + 1, len(work), v.name)
        _write_status(cfg, complete=False, reason="running", running=v.name,
                      remaining=[w.name for w, _ in work[i:]], done_this_session=done_now,
                      elapsed_s=elapsed)
        rec = process_variant(cfg, v, tasks, plan, force)
        timings.setdefault("variants", {})[v.name] = rec
        atomic_write_json(timings_path(cfg), timings)
        _delete_variant(cfg, v)
        for code in cfg.langs:
            kld.maybe_delete_logits(cfg, code)
        done_now.append(v.name)
        LOG.info("variant %s done in %.1f min (quantize %.1f, kld %.1f, milu %.1f, mt %.1f)", v.name,
                 rec["total_s"] / 60, rec["quantize_s"] / 60, rec["kld_s"] / 60, rec["milu_s"] / 60,
                 rec["mt_s"] / 60)
        if s["aggregate_each_variant"]:
            _quiet_aggregate(cfg)
        if s["bundle_each_variant"]:
            _quiet_bundle(cfg)

    if stopped_before is None:
        # Check the whole configured set, not only this run's --variants selection.
        saved = cfg.runtime.get("variants")
        cfg.runtime["variants"] = None
        try:
            all_milu = {v.name for v in evals.milu_variants(cfg)}
            all_mt = {v.name for v in evals.mt_variants(cfg)}
            left = [v.name for v in cfg.run_variants()
                    if not is_done(pending_tasks(cfg, v, all_milu, all_mt, False))]
        finally:
            cfg.runtime["variants"] = saved
        _write_status(cfg, complete=not left, reason="finished" if not left else "variants left",
                      remaining=left, done_this_session=done_now, elapsed_s=time.time() - t_start)
        LOG.info("stream finished: %d variants this session, %d remaining", len(done_now), len(left))


def _quiet_aggregate(cfg) -> None:
    """Refresh results_tidy.csv after each variant so partial results are always usable."""
    try:
        from iqp.aggregate import step_aggregate
        step_aggregate(cfg)
    except Exception as e:  # analysis must never stop the measurement loop
        LOG.warning("aggregate after variant failed (ignored): %s", e)


def _quiet_bundle(cfg) -> None:
    try:
        from iqp.session import step_bundle
        step_bundle(cfg, push=False)
    except Exception as e:
        LOG.warning("bundle after variant failed (ignored): %s", e)


def stream_complete(cfg) -> bool:
    p = status_path(cfg)
    return p.exists() and bool(read_json(p).get("complete"))


