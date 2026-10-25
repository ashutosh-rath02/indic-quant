"""Command line entry point: python -m iqp <step> [--config config.yaml] [--dry-run] [--force].

Two ways to run the variant loop:
  classic: quantize every variant, then KLD, MILU and MT over all of them (needs about 80 GB).
  stream:  per variant quantize, KLD, MILU, MT, then delete the GGUF (peak about 25 GB for
           Gemma 3 4B). Resumable, with a --max-hours budget. Used by the Kaggle configs.
`all` follows the config's `pipeline:` setting.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from iqp.config import Config
from iqp.util import LOG, StepError, setup_logging


def _steps():
    from iqp import aggregate, evals, fertility, kld, plots, prepare, quantize, report, session, setup_steps, stream
    return {
        "restore": session.step_restore,
        "setup": setup_steps.step_setup,
        "download": setup_steps.step_download,
        "convert": setup_steps.step_convert,
        "validate_data": prepare.step_validate,
        "prepare_data": prepare.step_prepare_data,
        "build_imatrix": quantize.step_build_imatrix,
        "quantize": quantize.step_quantize,
        "fertility": fertility.step_fertility,
        "run_kld": kld.step_run_kld,
        "run_milu": evals.step_run_milu,
        "run_mt": evals.step_run_mt,
        "aggregate": aggregate.step_aggregate,
        "plot": plots.step_plot,
        "report": report.step_report,
        "stream": stream.step_stream,
        "bundle": session.step_bundle,
    }


PIPELINE = ["setup", "download", "convert", "validate_data", "prepare_data", "build_imatrix", "quantize",
            "fertility", "run_kld", "run_milu", "run_mt", "aggregate", "plot", "report"]
ANALYSIS = ["aggregate", "plot", "report"]
STREAM_PIPELINE = ["restore", "setup", "download", "convert", "validate_data", "prepare_data", "build_imatrix",
                   "stream", "aggregate", "plot", "report", "bundle"]


def pipeline_for(cfg) -> list[str]:
    return STREAM_PIPELINE if cfg.raw.get("pipeline", "classic") == "stream" else PIPELINE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m iqp", description=__doc__)
    p.add_argument("step", help="one of: " + ", ".join(
        PIPELINE + ["stream", "restore", "bundle", "all", "analysis", "plan", "status"]))
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dry-run", action="store_true", help="fake tools and data; writes to *_dryrun dirs")
    p.add_argument("--force", action="store_true", help="recompute outputs even if they exist")
    p.add_argument("--from", dest="from_step", default=None, help="with 'all': start at this step")
    p.add_argument("--variants", default=None,
                   help="comma separated variant names or patterns (Q4_K_M__*, *__indic, base) to restrict "
                        "quantize, run_kld, run_milu, run_mt and stream; applied within the config's 'variants'")
    p.add_argument("--max-hours", type=float, default=None,
                   help="stream: stop cleanly before a variant that would likely end past this many hours "
                        "after the step started")
    p.add_argument("--restore-from", default=None,
                   help="restore: directory (or tar.gz, or 'hf[:owner/repo]') holding a previous session's output; "
                        "overrides paths.restore_from")
    a = p.parse_args(argv)

    cfg = Config.load(a.config, dry_run=a.dry_run)
    cfg.runtime.update({"variants": a.variants, "max_hours": a.max_hours, "restore_from": a.restore_from})
    try:
        cfg.run_variants()
    except ValueError as e:
        p.error(str(e))
    steps = _steps()
    if a.step == "plan":
        return _plan(cfg)
    if a.step == "status":
        return _status(cfg)
    if a.step == "all":
        pipe = pipeline_for(cfg)
        if a.from_step and a.from_step not in pipe:
            p.error(f"--from {a.from_step!r} is not a step of the {cfg.raw.get('pipeline', 'classic')} pipeline: "
                    + ", ".join(pipe))
        names = pipe[pipe.index(a.from_step):] if a.from_step else pipe
    elif a.step == "analysis":
        names = ANALYSIS
    elif a.step in steps:
        names = [a.step]
    else:
        p.error(f"unknown step {a.step!r}")
        return 2
    for name in names:
        setup_logging(cfg.logs_dir, name)
        LOG.info("=== step %s%s ===", name, " (dry run)" if cfg.dry_run else "")
        t0 = time.time()
        try:
            steps[name](cfg, force=a.force)
        except StepError as e:
            LOG.error("step %s failed: %s", name, e)
            return 1
        except Exception:
            LOG.error("step %s crashed:\n%s", name, traceback.format_exc())
            return 1
        LOG.info("=== step %s done in %.1fs ===", name, time.time() - t0)
    return 0


def _plan(cfg) -> int:
    from iqp import evals
    vs = cfg.run_variants()
    print(f"model: {cfg.raw['model']['hf_id']}  base: {cfg.base_variant().qtype}")
    print(f"languages: {', '.join(cfg.langs)}")
    print(f"calibration arms: {', '.join(cfg.calib_arms)} (token budget {cfg.raw['calibration']['token_budget']})")
    print(f"{len(vs)} variants:")
    for v in vs:
        print("  ", v.name)
    n_kld = len([v for v in vs if not v.is_base]) * len(cfg.langs)
    milu = evals.milu_variants(cfg)
    mt = evals.mt_variants(cfg)
    print(f"KLD runs: {n_kld}; MILU server sessions: {len(milu)}; MT server sessions: {len(mt)}")
    if milu and len(milu) < len(vs):
        print("MILU variants: " + ", ".join(v.name for v in milu))
    if mt and len(mt) < len(vs):
        print("MT variants:   " + ", ".join(v.name for v in mt))
    print(f"pipeline: {cfg.raw.get('pipeline', 'classic')}: " + " ".join(pipeline_for(cfg)))
    return 0


def _status(cfg) -> int:
    def mark(ok):
        return "done" if ok else "todo"
    print(f"{mark(cfg.bin('llama-perplexity').exists())}  llama.cpp build")
    print(f"{mark((cfg.hf_dir / 'config.json').exists())}  HF model")
    print(f"{mark(cfg.gguf_path(cfg.base_variant()).exists())}  base GGUF")
    print(f"{mark((cfg.data_dir / 'calib' / 'manifest.json').exists())}  prepared data")
    for arm in cfg.calib_arms:
        print(f"{mark(cfg.imatrix_path(arm).exists())}  imatrix {arm}")
    raw = cfg.raw_results_dir
    for v in cfg.run_variants():
        g = cfg.gguf_path(v).exists() or cfg.quant_meta_path(v).exists()
        k = sum((raw / "kld" / v.name / f"{c}.json").exists() for c in cfg.langs)
        m = sum((raw / "milu" / v.name / f"{c}.json").exists() for c in cfg.langs)
        t = sum((raw / "mt" / v.name / f"{c}.json").exists() for c in cfg.langs if c != "eng_Latn")
        print(f"{mark(g)}  {v.name:24s} kld {k}/{len(cfg.langs)}  milu {m}/{len(cfg.langs)}  "
              f"mt {t}/{len(cfg.langs) - 1}")
    print(f"{mark((cfg.results_dir / 'REPORT.md').exists())}  REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
