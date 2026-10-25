"""Step report: write results/REPORT.md from the aggregated CSVs and decision.json."""

from __future__ import annotations

import datetime as _dt
import math

import pandas as pd

from iqp.util import LOG, atomic_write_text, read_json


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float):
        if abs(x) >= 100:
            return f"{x:.1f}"
        return f"{x:.{nd}f}"
    return str(x)


def _md_table(df: pd.DataFrame, nd=4) -> str:
    if df.empty:
        return "_no data_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [_fmt(v, nd) for v in r.values]) + " |")
    return "\n".join(lines) + "\n"


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}%"


def build_report(cfg) -> str:
    rd = cfg.results_dir
    df = pd.read_csv(rd / "results_tidy.csv")
    decision = read_json(rd / "decision.json")
    try:
        deltas = pd.read_csv(rd / "calibration_deltas.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        deltas = pd.DataFrame()
    names = {c: cfg.lang_name(c) for c in cfg.langs}
    out: list[str] = []
    dry = " (DRY RUN: synthetic numbers, not real measurements)" if cfg.dry_run else ""
    out.append(f"# Indic quantization pilot: {cfg.model_name}{dry}\n")
    out.append(f"Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from `{rd.name}/results_tidy.csv`. "
               f"Model `{cfg.raw['model']['hf_id']}`, base precision {cfg.base_variant().qtype}, "
               f"llama.cpp commit `{cfg.raw['llamacpp']['commit'][:12]}`.\n")
    if cfg.dry_run:
        out.append("> This report was produced by the dry-run pipeline with fake llama.cpp tools. "
                   "It exercises the code path only. Do not interpret any number below.\n")

    # ---- decision
    out.append("## Decision\n")
    v = decision.get("verdict", "insufficient data")
    crit = decision["criteria"]
    out.append(f"**Verdict: {v}.**\n")
    out.append(f"Criteria: Indic-calibrated imatrix must cut mean KLD over Indic languages by at least "
               f"{100 * crit['min_relative_kld_reduction']:.0f}% relative to the English imatrix at the same "
               f"quant type in at least {crit['min_quants_with_reduction']} of "
               f"{len(cfg.raw['quant']['calibrated_types'])} calibrated types, and the worst Indic language "
               f"must show at least {crit['min_language_gap_ratio']:.1f}x the English mean KLD at "
               f"`{crit['gap_reference_variant']}`.\n")
    out.append(f"- Calibration condition: {'met' if decision.get('calibration_condition') else 'not met'} "
               f"({decision.get('n_qtypes_passing', 0)} of {len(decision.get('per_qtype', {}))} calibrated types pass).")
    for q, r in decision.get("per_qtype", {}).items():
        out.append(f"  - {q}: mean relative KLD reduction over Indic languages {_pct(r['mean_rel_reduction'])}; "
                   f"{r['langs_improved']}/{r['n_langs']} languages improved, {r['langs_beyond_2sigma']} beyond "
                   f"2 combined standard errors.")
    g = decision.get("gap") or {}
    if g:
        out.append(f"- Language gap condition: {'met' if decision.get('gap_condition') else 'not met'}. "
                   f"At {g['variant']}, English mean KLD {_fmt(g['english_kld'])}, worst Indic "
                   f"{names.get(g['worst_indic_language'], g['worst_indic_language'])} "
                   f"{_fmt(g['worst_indic_kld'])} ({g['ratio']:.2f}x); mean Indic/English ratio "
                   f"{g['mean_indic_ratio']:.2f}x.")
    else:
        out.append("- Language gap condition: no data.")
    ec = decision.get("english_cost") or {}
    if ec:
        out.append("- Cost to English of the Indic imatrix (positive means English also improved): " +
                   ", ".join(f"{q} {_pct(x)}" for q, x in ec.items()) + ".")
    ht = decision.get("heldout_tamil") or {}
    if ht:
        out.append("- Held-out Tamil (Indic mix without Tamil vs English imatrix): " + "; ".join(
            f"{q}: full mix {_pct(r['indic_rel_reduction'])}, without Tamil {_pct(r['indic_no_tam_rel_reduction'])}"
            for q, r in ht.items()) + ".")
    out.append("")
    out.append("Interpretation guide: a strong signal justifies the full benchmark (more models, all 22 "
               "scheduled languages, larger calibration budgets). A partial signal means only one of the two "
               "effects is present; check whether the per-language gap alone is worth publishing. Differences "
               "smaller than the reported uncertainties should be treated as noise.\n")

    # ---- KLD table
    kld = df[df.metric == "kld_mean"].pivot_table(index="variant", columns="language", values="value", aggfunc="first")
    if not kld.empty:
        order = [v.name for v in cfg.quant_variants() if v.name in kld.index]
        kld = kld.loc[order, [c for c in cfg.langs if c in kld.columns]].rename(columns=names)
        kld.index.name = "variant"
        out.append("## Mean KL divergence per language\n")
        out.append(f"Lower is better. Each language is scored on the same number of chunks "
                   f"(ctx {cfg.raw['kld']['ctx']}); token counts differ by language because the tokenizer "
                   f"splits scripts differently, so one chunk covers less text in high-fertility languages.\n")
        out.append(_md_table(kld, 4))
    for metric, title, nd in (("same_top_p_pct", "Top-token agreement with the base model (%)", 2),
                              ("kld_p99", "99th percentile KLD", 3), ("ppl_ratio", "PPL(Q) / PPL(base)", 4),
                              ("milu_accuracy", "MILU accuracy", 3), ("mt_chrfpp", "English to X chrF++", 2)):
        t = df[df.metric == metric].pivot_table(index="variant", columns="language", values="value", aggfunc="first")
        if t.empty:
            continue
        order = [v.name for v in cfg.all_variants() if v.name in t.index]
        t = t.loc[order, [c for c in cfg.langs if c in t.columns]].rename(columns=names)
        t.index.name = "variant"
        out.append(f"## {title}\n")
        out.append(_md_table(t, nd))

    # ---- calibration deltas
    if not deltas.empty:
        out.append("## Calibration arms at equal size\n")
        out.append("Relative KLD reduction versus the English imatrix for the same quant type "
                   "(positive means the arm is better than English calibration).\n")
        t = deltas.pivot_table(index=["qtype", "arm"], columns="language", values="rel_reduction_vs_en",
                               aggfunc="first")
        t = t[[c for c in cfg.langs if c in t.columns]].rename(columns=names)
        t.index = [f"{q} {a}" for q, a in t.index]
        t.index.name = "qtype arm"
        out.append(_md_table(t * 100, 1).replace("| qtype arm |", "| qtype arm (%) |"))

    # ---- sizes
    sz = df[df.metric.isin(["file_gib", "bpw"])].pivot_table(index="variant", columns="metric", values="value",
                                                             aggfunc="first")
    if not sz.empty:
        order = [v.name for v in cfg.all_variants() if v.name in sz.index]
        sz = sz.loc[order]
        sz.index.name = "variant"
        out.append("## Variant sizes\n")
        out.append("Bits per weight as reported by llama-quantize (all tensors, including embeddings).\n")
        out.append(_md_table(sz, 3))

    # ---- fertility
    fert = df[df.metric.str.startswith("fertility_")]
    if not fert.empty:
        t = fert.pivot_table(index="language", columns="metric", values="value", aggfunc="first")
        t = t.rename(index=names)
        t.columns = [c.replace("fertility_", "") for c in t.columns]
        t.index.name = "language"
        out.append("## Tokenizer fertility\n")
        out.append(_md_table(t, 3))

    # ---- plots
    out.append("## Figures\n")
    for f, cap in (("kld_heatmap.png", "Mean KLD, language by variant"),
                   ("retention_vs_bpw.png", "Damage and retention vs bits per weight"),
                   ("calibration_delta.png", "Indic vs English calibration at equal size"),
                   ("fertility_vs_kld.png", "Tokenizer fertility vs KLD")):
        if (cfg.plots_dir / f).exists():
            out.append(f"![{cap}](plots/{f})\n")
    out.append("## Notes and limitations\n")
    out.append("- KLD is measured against the base GGUF, not the original HF weights.\n"
               "- MILU uses the official prompt format with the first k validation examples as shots; "
               "scores are next-token letter log-probabilities, not generated answers.\n"
               "- chrF++ uses greedy decoding through the model's chat template; it measures the combined "
               "effect of translation quality and instruction following.\n"
               "- One model and one calibration seed: treat all findings as a pilot.\n")
    return "\n".join(out)


def step_report(cfg, force: bool = False) -> None:
    text = build_report(cfg)
    atomic_write_text(cfg.results_dir / "REPORT.md", text)
    LOG.info("wrote %s", cfg.results_dir / "REPORT.md")
