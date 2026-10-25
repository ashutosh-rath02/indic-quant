"""Step plot: static PNG figures for the report."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from iqp.util import LOG  # noqa: E402

# Categorical slots in fixed order (validated palette, light mode), sequential blue ramp,
# neutral chrome. Series identity never relies on color alone: lines get markers and
# direct end labels, scatter points get text labels.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ARM_COLORS = {"static": "#898781", "en": SERIES[0], "indic": SERIES[1], "indic_no_tam": SERIES[2]}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    })


def lang_colors(langs: list[str]) -> dict[str, str]:
    return {c: SERIES[i % len(SERIES)] for i, c in enumerate(langs)}


def plot_kld_heatmap(cfg, df: pd.DataFrame, path) -> bool:
    sub = df[df.metric == "kld_mean"]
    if sub.empty:
        return False
    t = sub.pivot_table(index="variant", columns="language", values="value", aggfunc="first")
    order = [v.name for v in cfg.quant_variants() if v.name in t.index]
    cols = [c for c in cfg.langs if c in t.columns]
    t = t.loc[order, cols]
    vals = t.values.astype(float)
    logv = np.log10(np.clip(vals, 1e-5, None))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 3, 0.42 * len(order) + 1.8))
    im = ax.imshow(logv, aspect="auto", cmap=cmap)
    ax.grid(False)
    ax.set_xticks(range(len(cols)), [cfg.lang_name(c) for c in cols])
    ax.set_yticks(range(len(order)), [o.replace("__", "  ") for o in order])
    lo, hi = np.nanmin(logv), np.nanmax(logv)
    for i in range(len(order)):
        for j in range(len(cols)):
            v = vals[i, j]
            if math.isnan(v):
                continue
            frac = (logv[i, j] - lo) / (hi - lo + 1e-12)
            ax.text(j, i, f"{v:.3f}" if v < 1 else f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if frac > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log10 mean KLD", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"Mean KL divergence vs {cfg.base_variant().qtype} ({cfg.model_name})", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _bpw_map(df: pd.DataFrame) -> dict[str, float]:
    b = df[df.metric == "bpw"]
    return dict(zip(b.variant, b.value.astype(float)))


def plot_retention_curves(cfg, df: pd.DataFrame, path) -> bool:
    """Per-language curves vs bits per weight: KLD, top-token agreement, and task retention.

    Uses the static arm (plus IQ2_M with the English imatrix, since IQ2_M has no static arm)
    so each language has one point per size.
    """
    bpw = _bpw_map(df)
    if not bpw:
        return False
    need_im = set(cfg.raw["quant"]["requires_imatrix"])
    panels = [("kld_mean", "Mean KLD (log scale)", True), ("same_top_p_pct", "Same top token (%)", False)]
    # Raw task scores (not ratios): a ratio to the base is unstable when the base is near chance.
    if not df[df.metric == "milu_accuracy"].empty:
        panels.append(("milu_accuracy", "MILU accuracy (chance 0.25)", False))
    if not df[df.metric == "mt_chrfpp"].empty:
        panels.append(("mt_chrfpp", "English to X chrF++", False))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    colors = lang_colors(cfg.langs)

    def pick(r):
        return r.arm == "static" or (r.qtype in need_im and r.arm == "en") or r.arm == "base"
    for ax, (metric, label, logy) in zip(axes[0], panels):
        sub = df[df.metric == metric]
        sub = sub[sub.apply(pick, axis=1)] if not sub.empty else sub
        for i, code in enumerate(cfg.langs):
            s = sub[sub.language == code].copy()
            s["bpw"] = s.variant.map(bpw)
            s = s.dropna(subset=["bpw"]).sort_values("bpw")
            if s.empty:
                continue
            ax.plot(s.bpw, s.value, color=colors[code], marker=MARKERS[i % len(MARKERS)], markersize=6,
                    label=cfg.lang_name(code))
        if metric == "milu_accuracy":
            ax.axhline(0.25, color=MUTED, linewidth=1, linestyle="--")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("bits per weight")
        ax.set_title(label, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Damage vs size per language (static quants; IQ2_M uses the English imatrix)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_calibration_delta(cfg, deltas: pd.DataFrame, path) -> bool:
    """Relative change in mean KLD vs the English imatrix, same qtype (hence same size)."""
    if deltas.empty:
        return False
    qtypes = [q for q in cfg.raw["quant"]["calibrated_types"] if q in set(deltas.qtype)]
    arms = [a for a in ["indic", "indic_no_tam", "static"] if a in set(deltas.arm)]
    fig, axes = plt.subplots(1, len(qtypes), figsize=(4.3 * len(qtypes), 4.4), squeeze=False, sharey=True)
    langs = [c for c in cfg.langs if c in set(deltas.language)]
    width = 0.8 / max(len(arms), 1)
    for ax, q in zip(axes[0], qtypes):
        s = deltas[deltas.qtype == q]
        x = np.arange(len(langs))
        for k, arm in enumerate(arms):
            vals = [100 * float(s[(s.language == c) & (s.arm == arm)].rel_reduction_vs_en.iloc[0])
                    if not s[(s.language == c) & (s.arm == arm)].empty else np.nan for c in langs]
            ax.bar(x + (k - (len(arms) - 1) / 2) * width, vals, width * 0.9, color=ARM_COLORS.get(arm, SERIES[3]),
                   label={"indic": "Indic imatrix", "indic_no_tam": "Indic without Tamil",
                          "static": "no imatrix"}.get(arm, arm), edgecolor=SURFACE, linewidth=1)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xticks(x, [cfg.lang_name(c) for c in langs], rotation=40, ha="right")
        ax.set_title(q, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("KLD reduction vs English imatrix (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Calibration effect at equal file size (positive = less damage than English calibration)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_fertility_vs_kld(cfg, df: pd.DataFrame, path) -> bool:
    fert = df[df.metric == "fertility_tokens_per_word"].set_index("language")["value"]
    kld = df[(df.metric == "kld_mean")]
    ref = cfg.raw["decision"]["gap_reference_variant"]
    alts = [ref, "Q3_K_M__static", "Q2_K__static"]
    variants = [v for v in alts if v in set(kld.variant)]
    if fert.empty or not variants:
        return False
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 4.2), squeeze=False)
    colors = lang_colors(cfg.langs)
    for ax, v in zip(axes[0], variants):
        k = kld[kld.variant == v].set_index("language")["value"]
        for code in cfg.langs:
            if code in fert.index and code in k.index:
                ax.scatter(fert[code], k[code], s=70, color=colors[code], edgecolor=SURFACE, linewidth=2, zorder=3)
                ax.annotate(cfg.lang_name(code), (fert[code], k[code]), textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=INK2)
        ax.set_xlabel("tokens per word")
        ax.set_title(v.replace("__", " "), loc="left")
    axes[0][0].set_ylabel("mean KLD")
    fig.suptitle("Tokenizer fertility vs quantization damage", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def step_plot(cfg, force: bool = False) -> dict:
    _style()
    out = cfg.plots_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.results_dir / "results_tidy.csv")
    dpath = cfg.results_dir / "calibration_deltas.csv"
    try:
        deltas = pd.read_csv(dpath)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        deltas = pd.DataFrame()
    made = {
        "kld_heatmap": plot_kld_heatmap(cfg, df, out / "kld_heatmap.png"),
        "retention_curves": plot_retention_curves(cfg, df, out / "retention_vs_bpw.png"),
        "calibration_delta": plot_calibration_delta(cfg, deltas, out / "calibration_delta.png"),
        "fertility_vs_kld": plot_fertility_vs_kld(cfg, df, out / "fertility_vs_kld.png"),
    }
    LOG.info("plots: %s", ", ".join(k for k, v in made.items() if v) or "none")
    return made
