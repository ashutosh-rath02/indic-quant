"""Step aggregate: collect raw JSON results into tidy CSVs and compute the decision metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from iqp.config import STATIC_ARM, Variant
from iqp.quantize import load_sizes
from iqp.util import LOG, atomic_write_json, read_json

TIDY_COLUMNS = ["model", "variant", "qtype", "arm", "language", "language_name", "role", "metric", "value",
                "uncertainty", "n"]

KLD_METRICS = {
    "kld_mean": ("kld_mean", "kld_mean_unc"),
    "kld_median": ("kld_median", None),
    "kld_p99": ("kld_p99", None),
    "kld_p99_9": ("kld_p99_9", None),
    "kld_max": ("kld_max", None),
    "same_top_p_pct": ("same_top_p_pct", "same_top_p_pct_unc"),
    "ppl_ratio": ("ppl_ratio", "ppl_ratio_unc"),
    "ln_ppl_ratio": ("ln_ppl_ratio", "ln_ppl_ratio_unc"),
    "ppl_q": ("ppl_q", "ppl_q_unc"),
    "ppl_base": ("ppl_base", "ppl_base_unc"),
    "dp_rms_pct": ("dp_rms_pct", "dp_rms_pct_unc"),
}


def _row(cfg, v: Variant, code: str | None, metric: str, value, unc=None, n=None) -> dict:
    return {"model": cfg.model_name, "variant": v.name, "qtype": v.qtype, "arm": v.arm,
            "language": code or "", "language_name": cfg.lang_name(code) if code else "",
            "role": cfg.lang(code).get("role", "") if code else "", "metric": metric,
            "value": value, "uncertainty": unc, "n": n}


def collect_tidy(cfg) -> pd.DataFrame:
    raw = cfg.raw_results_dir
    rows: list[dict] = []
    variants = cfg.all_variants()
    for v in variants:
        for code in cfg.langs:
            f = raw / "kld" / v.name / f"{code}.json"
            if f.exists():
                d = read_json(f)
                n = d.get("n_chunks")
                for metric, (k, ku) in KLD_METRICS.items():
                    if k in d:
                        rows.append(_row(cfg, v, code, metric, d[k], d.get(ku) if ku else None, n))
            for task, key in (("milu", "accuracy"), ("mt", "chrfpp")):
                f = raw / task / v.name / f"{code}.json"
                if f.exists():
                    d = read_json(f)
                    unc = None
                    if task == "milu" and "ci_low" in d:
                        unc = (d["ci_high"] - d["ci_low"]) / 2
                    rows.append(_row(cfg, v, code, f"{task}_{'accuracy' if task == 'milu' else 'chrfpp'}",
                                     d[key], unc, d.get("n")))
        if v.is_base:
            for code in cfg.langs:
                f = cfg.kld_base_dir / f"{code}.base.json"
                if f.exists():
                    d = read_json(f)
                    rows.append(_row(cfg, v, code, "ppl_base_run", d["ppl"], d.get("ppl_unc"), d.get("n_chunks")))
    sizes = {s["variant"]: s for s in load_sizes(cfg)}
    for v in variants:
        s = sizes.get(v.name)
        if s:
            rows.append(_row(cfg, v, None, "file_gib", s["file_gib"]))
            if s.get("bpw") is not None:
                rows.append(_row(cfg, v, None, "bpw", s["bpw"]))
    base_v = cfg.base_variant()
    for code in cfg.langs:
        f = raw / "fertility" / f"{code}.json"
        if f.exists():
            d = read_json(f)
            for k in ("tokens_per_word", "tokens_per_char", "tokens_per_grapheme", "n_tokens", "n_words", "n_chars"):
                rows.append(_row(cfg, base_v, code, f"fertility_{k}", d[k]))
    df = pd.DataFrame(rows, columns=TIDY_COLUMNS)
    df["uncertainty"] = pd.to_numeric(df["uncertainty"], errors="coerce").astype(float)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    return df


def _metric_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df.metric == metric]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="variant", columns="language", values="value", aggfunc="first")


def add_retention(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and chrF++ retention relative to the base model, per language."""
    out = []
    base = cfg.base_variant().name
    for metric in ("milu_accuracy", "mt_chrfpp"):
        sub = df[df.metric == metric]
        if sub.empty:
            continue
        b = sub[sub.variant == base].set_index("language")["value"]
        for _, r in sub.iterrows():
            if r.language in b.index and b[r.language] and not math.isnan(b[r.language]):
                rr = r.copy()
                rr["metric"] = metric + "_retention"
                rr["value"] = r["value"] / b[r.language]
                rr["uncertainty"] = float("nan")
                out.append(rr)
    if out:
        extra = pd.DataFrame(out).astype(df.dtypes.to_dict(), errors="ignore")
        df = pd.concat([df, extra], ignore_index=True)
    return df


def calibration_deltas(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Per calibrated qtype and language: KLD under each arm and relative change vs the English arm."""
    kld = df[(df.metric == "kld_mean")]
    if kld.empty:
        return pd.DataFrame()
    rows = []
    for q in cfg.raw["quant"]["calibrated_types"]:
        for code in cfg.langs:
            vals = {}
            uncs = {}
            for arm in [STATIC_ARM] + cfg.calib_arms:
                s = kld[(kld.qtype == q) & (kld.arm == arm) & (kld.language == code)]
                if not s.empty:
                    vals[arm] = float(s.value.iloc[0])
                    uncs[arm] = float(s.uncertainty.iloc[0]) if pd.notna(s.uncertainty.iloc[0]) else float("nan")
            if "en" not in vals:
                continue
            for arm, val in vals.items():
                if arm == "en":
                    continue
                rel = (vals["en"] - val) / vals["en"] if vals["en"] > 0 else float("nan")
                rows.append({"qtype": q, "language": code, "language_name": cfg.lang_name(code),
                             "role": cfg.lang(code).get("role"), "arm": arm, "kld_arm": val,
                             "kld_en": vals["en"], "rel_reduction_vs_en": rel,
                             "abs_reduction_vs_en": vals["en"] - val,
                             "unc_diff": math.sqrt(uncs.get(arm, 0) ** 2 + uncs.get("en", 0) ** 2)})
    return pd.DataFrame(rows)


def decide(cfg, df: pd.DataFrame, deltas: pd.DataFrame) -> dict:
    d = cfg.raw["decision"]
    indic = cfg.indic_langs
    res: dict = {"criteria": dict(d), "per_qtype": {}, "gap": {}, "heldout_tamil": {}}
    n_pass = 0
    if not deltas.empty:
        sub = deltas[(deltas.arm == "indic") & (deltas.language.isin(indic))]
        for q, g in sub.groupby("qtype"):
            mean_rel = float(g.rel_reduction_vs_en.mean())
            n_lang_improved = int((g.rel_reduction_vs_en > 0).sum())
            sig = int((g.abs_reduction_vs_en > 2 * g.unc_diff).sum())
            ok = mean_rel >= d["min_relative_kld_reduction"]
            n_pass += int(ok)
            res["per_qtype"][q] = {"mean_rel_reduction": mean_rel, "langs_improved": n_lang_improved,
                                   "langs_beyond_2sigma": sig, "n_langs": int(len(g)), "passes": bool(ok)}
        eng = deltas[(deltas.arm == "indic") & (deltas.language.isin(cfg.control_langs))]
        res["english_cost"] = {q: float(g.rel_reduction_vs_en.mean()) for q, g in eng.groupby("qtype")}
        if "indic_no_tam" in cfg.calib_arms and "tam_Taml" in cfg.langs:
            for q in cfg.raw["quant"]["calibrated_types"]:
                t = deltas[(deltas.qtype == q) & (deltas.language == "tam_Taml")].set_index("arm")
                if "indic" in t.index and "indic_no_tam" in t.index:
                    full = float(t.loc["indic", "rel_reduction_vs_en"])
                    held = float(t.loc["indic_no_tam", "rel_reduction_vs_en"])
                    res["heldout_tamil"][q] = {"indic_rel_reduction": full, "indic_no_tam_rel_reduction": held,
                                               "transfer_fraction": (held / full) if full > 0 else float("nan")}
    kld = df[df.metric == "kld_mean"]
    ref = d["gap_reference_variant"]
    r = kld[kld.variant == ref].set_index("language")["value"]
    if not r.empty and any(c in r.index for c in cfg.control_langs):
        en = float(r[[c for c in cfg.control_langs if c in r.index]].mean())
        ind = r[[c for c in indic if c in r.index]]
        if len(ind) and en > 0:
            worst = ind.idxmax()
            res["gap"] = {"variant": ref, "english_kld": en, "worst_indic_language": worst,
                          "worst_indic_kld": float(ind.max()), "ratio": float(ind.max() / en),
                          "mean_indic_ratio": float(ind.mean() / en)}
    calib_ok = n_pass >= d["min_quants_with_reduction"]
    gap_ok = bool(res["gap"]) and res["gap"]["ratio"] >= d["min_language_gap_ratio"]
    res["n_qtypes_passing"] = n_pass
    res["calibration_condition"] = calib_ok
    res["gap_condition"] = gap_ok
    if not res["per_qtype"] or not res["gap"]:
        res["verdict"] = "insufficient data"
    elif calib_ok and gap_ok:
        res["verdict"] = "strong signal"
    elif calib_ok or gap_ok:
        res["verdict"] = "partial signal"
    else:
        res["verdict"] = "weak signal"
    return res


def step_aggregate(cfg, force: bool = False) -> None:
    out = cfg.results_dir
    out.mkdir(parents=True, exist_ok=True)
    df = collect_tidy(cfg)
    if df.empty:
        LOG.warning("no results found to aggregate")
    df = add_retention(cfg, df)
    df.to_csv(out / "results_tidy.csv", index=False)
    deltas = calibration_deltas(cfg, df)
    deltas.to_csv(out / "calibration_deltas.csv", index=False)
    pd.DataFrame(load_sizes(cfg)).to_csv(out / "variants.csv", index=False)
    for metric in ("kld_mean", "kld_median", "kld_p99", "same_top_p_pct", "ppl_ratio", "milu_accuracy",
                   "mt_chrfpp"):
        t = _metric_table(df, metric)
        if not t.empty:
            order = [v.name for v in cfg.all_variants() if v.name in t.index]
            t.loc[order, [c for c in cfg.langs if c in t.columns]].to_csv(out / f"wide_{metric}.csv")
    decision = decide(cfg, df, deltas)
    atomic_write_json(out / "decision.json", decision)
    LOG.info("aggregate: %d tidy rows; verdict: %s", len(df), decision["verdict"])
