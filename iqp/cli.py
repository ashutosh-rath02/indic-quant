"""Command line entry point: python -m iqp <step> [--config config.yaml] [--dry-run] [--force]."""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from iqp.config import Config
from iqp.util import LOG, StepError, setup_logging


def _steps():
    from iqp import aggregate, evals, fertility, kld, plots, prepare, quantize, report, setup_steps
    return {
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
    }


PIPELINE = ["setup", "download", "convert", "validate_data", "prepare_data", "build_imatrix", "quantize",
            "fertility", "run_kld", "run_milu", "run_mt", "aggregate", "plot", "report"]
ANALYSIS = ["aggregate", "plot", "report"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m iqp", description=__doc__)
    p.add_argument("step", help="one of: " + ", ".join(PIPELINE + ["all", "analysis", "plan", "status"]))
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dry-run", action="store_true", help="fake tools and data; writes to *_dryrun dirs")
    p.add_argument("--force", action="store_true", help="recompute outputs even if they exist")
    p.add_argument("--from", dest="from_step", default=None, help="with 'all': start at this step")
    a = p.parse_args(argv)

    cfg = Config.load(a.config, dry_run=a.dry_run)
    steps = _steps()
    if a.step == "plan":
        return _plan(cfg)
    if a.step == "status":
        return _status(cfg)
    if a.step == "all":
        names = PIPELINE[PIPELINE.index(a.from_step):] if a.from_step else PIPELINE
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
    vs = cfg.all_variants()
    print(f"model: {cfg.raw['model']['hf_id']}  base: {cfg.base_variant().qtype}")
    print(f"languages: {', '.join(cfg.langs)}")
    print(f"calibration arms: {', '.join(cfg.calib_arms)} (token budget {cfg.raw['calibration']['token_budget']})")
    print(f"{len(vs)} variants:")
    for v in vs:
        print("  ", v.name)
    n_kld = len(cfg.quant_variants()) * len(cfg.langs)
    print(f"KLD runs: {n_kld}; MILU/MT server sessions: {len(vs)} each")
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
    for v in cfg.all_variants():
        g = cfg.gguf_path(v).exists()
        k = sum((raw / "kld" / v.name / f"{c}.json").exists() for c in cfg.langs)
        m = sum((raw / "milu" / v.name / f"{c}.json").exists() for c in cfg.langs)
        t = sum((raw / "mt" / v.name / f"{c}.json").exists() for c in cfg.langs if c != "eng_Latn")
        print(f"{mark(g)}  {v.name:24s} kld {k}/{len(cfg.langs)}  milu {m}/{len(cfg.langs)}  "
              f"mt {t}/{len(cfg.langs) - 1}")
    print(f"{mark((cfg.results_dir / 'REPORT.md').exists())}  REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
