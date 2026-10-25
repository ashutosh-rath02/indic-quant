"""Parsers for llama.cpp tool output.

Formats were checked against tools/perplexity/perplexity.cpp, src/llama-quant.cpp and
tools/tokenize/tokenize.cpp at the pinned commit, and against output captured from real
CPU runs (see tests/fixtures).
"""

from __future__ import annotations

import re

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
_PM = r"(?:±|\+/-)"


class ParseError(ValueError):
    pass


def _f(s: str) -> float:
    return float(s)


# Lines in the "====== Perplexity statistics ======" and "KL divergence statistics" blocks.
_KLD_PATTERNS: dict[str, re.Pattern] = {
    "ppl_q": re.compile(rf"^Mean PPL\(Q\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})", re.M),
    "ppl_base": re.compile(rf"^Mean PPL\(base\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})", re.M),
    "ln_ppl_ratio": re.compile(rf"^Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})", re.M),
    "ppl_ratio": re.compile(rf"^Mean PPL\(Q\)/PPL\(base\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})", re.M),
    "ppl_diff": re.compile(rf"^Mean PPL\(Q\)-PPL\(base\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})", re.M),
    "ppl_corr_pct": re.compile(rf"^Cor\(ln\(PPL\(Q\)\), ln\(PPL\(base\)\)\)\s*:\s*({_NUM})\s*%", re.M),
    "kld_mean": re.compile(rf"^Mean\s+KLD\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})", re.M),
    "kld_max": re.compile(rf"^Maximum KLD\s*:\s*({_NUM})", re.M),
    "kld_p99_9": re.compile(rf"^99\.9%\s+KLD\s*:\s*({_NUM})", re.M),
    "kld_p99": re.compile(rf"^99\.0%\s+KLD\s*:\s*({_NUM})", re.M),
    "kld_p95": re.compile(rf"^95\.0%\s+KLD\s*:\s*({_NUM})", re.M),
    "kld_p90": re.compile(rf"^90\.0%\s+KLD\s*:\s*({_NUM})", re.M),
    "kld_median": re.compile(rf"^Median\s+KLD\s*:\s*({_NUM})", re.M),
    "kld_p10": re.compile(rf"^10\.0%\s+KLD\s*:\s*({_NUM})", re.M),
    "kld_min": re.compile(rf"^Minimum KLD\s*:\s*({_NUM})", re.M),
    "dp_mean_pct": re.compile(rf"^Mean\s+Δp\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})\s*%", re.M),
    "dp_rms_pct": re.compile(rf"^RMS Δp\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})\s*%", re.M),
    "same_top_p_pct": re.compile(rf"^Same top p\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})\s*%", re.M),
}

_WITH_UNC = {"ppl_q", "ppl_base", "ln_ppl_ratio", "ppl_ratio", "ppl_diff", "kld_mean", "dp_mean_pct",
             "dp_rms_pct", "same_top_p_pct"}
REQUIRED_KLD_KEYS = ("kld_mean", "kld_median", "kld_p99", "same_top_p_pct", "ppl_ratio", "ppl_q", "ppl_base")

_CHUNK_HDR = re.compile(r"^chunk\s+PPL\s+ln\(PPL\(Q\)/PPL\(base\)\)\s+KL Divergence", re.M)
_CHUNK_ROW = re.compile(rf"^\s*(\d+)\s+({_NUM})\s*{_PM}\s*({_NUM})\s+({_NUM})\s*{_PM}\s*({_NUM})\s+"
                        rf"({_NUM})\s*{_PM}\s*({_NUM})\s+({_NUM})\s*{_PM}\s*({_NUM})\s*%\s+"
                        rf"({_NUM})\s*{_PM}\s*({_NUM})\s*%\s*$", re.M)
_N_CHUNKS = re.compile(r"kl_divergence: computing over (\d+) chunks, n_ctx=(\d+)")


def _strip_log_prefix(text: str) -> str:
    """llama.cpp can prefix log lines with a timestamp and level ("0.01.692 I ").

    LOG() lines (the statistics) have no prefix, but colored or prefixed builds may differ.
    """
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"^\d+\.\d+\.\d+\.\d+ [IWED] ", "", text, flags=re.M)


def parse_kld_output(text: str) -> dict:
    """Parse the final statistics of `llama-perplexity --kl-divergence`.

    Returns a flat dict of floats. Percent values keep the _pct suffix (0..100).
    Raises ParseError if any required statistic is missing (for example when the
    tool bailed out because fewer than 100 tokens were scored).
    """
    text = _strip_log_prefix(text)
    out: dict[str, float] = {}
    for key, pat in _KLD_PATTERNS.items():
        matches = pat.findall(text)
        if not matches:
            continue
        m = matches[-1]
        if isinstance(m, tuple):
            out[key] = _f(m[0])
            if key in _WITH_UNC and len(m) > 1:
                out[key + "_unc"] = _f(m[1])
        else:
            out[key] = _f(m)
    missing = [k for k in REQUIRED_KLD_KEYS if k not in out]
    if missing:
        hint = ""
        if "you need at least" in text or "tokenizes to only" in text:
            hint = " (input text too short for the requested context)"
        elif "inconsistent vocabulary" in text:
            hint = " (base logits were produced by a model with a different vocabulary)"
        raise ParseError(f"KLD output is missing {missing}{hint}")
    m = _N_CHUNKS.search(text)
    if m:
        out["n_chunks"] = float(m.group(1))
        out["n_ctx"] = float(m.group(2))
    return out


def parse_kld_chunks(text: str) -> list[dict]:
    """Per-chunk running statistics table (cumulative values, as llama.cpp prints them)."""
    text = _strip_log_prefix(text)
    if not _CHUNK_HDR.search(text):
        return []
    rows = []
    for m in _CHUNK_ROW.finditer(text):
        g = m.groups()
        rows.append({
            "chunk": int(g[0]), "ppl_q": _f(g[1]), "ln_ppl_ratio": _f(g[3]), "kld_mean": _f(g[5]),
            "dp_rms_pct": _f(g[7]), "same_top_p_pct": _f(g[9]),
        })
    return rows


_BASE_FINAL = re.compile(rf"Final estimate: PPL = ({_NUM}) \+/- ({_NUM})")
_BASE_CHUNKS = re.compile(r"perplexity: calculating perplexity over (\d+) chunks, n_ctx=(\d+)")
_HAVE_TOKENS = re.compile(r"tokenizes to only (\d+) tokens|have (\d+) tokens")


def parse_base_ppl_output(text: str) -> dict:
    """Parse the output of the base-logits run (`--kl-divergence-base FILE` without --kl-divergence)."""
    text = _strip_log_prefix(text)
    out: dict[str, float] = {}
    m = _BASE_FINAL.findall(text)
    if not m:
        raise ParseError("base perplexity run has no 'Final estimate' line")
    out["ppl"] = _f(m[-1][0])
    out["ppl_unc"] = _f(m[-1][1])
    c = _BASE_CHUNKS.search(text)
    if c:
        out["n_chunks"] = float(c.group(1))
        out["n_ctx"] = float(c.group(2))
    return out


_QSIZE = re.compile(rf"quant size\s*=\s*({_NUM}) MiB \(({_NUM}) BPW\)")
_MSIZE = re.compile(rf"model size\s*=\s*({_NUM}) MiB \(({_NUM}) BPW\)")


def parse_quantize_output(text: str) -> dict:
    """Sizes and bits per weight reported by llama-quantize (computed over all tensors)."""
    q = _QSIZE.findall(text)
    m = _MSIZE.findall(text)
    if not q:
        raise ParseError("llama-quantize output has no 'quant size' line")
    out = {"quant_mib": _f(q[-1][0]), "quant_bpw": _f(q[-1][1])}
    if m:
        out["orig_mib"] = _f(m[-1][0])
        out["orig_bpw"] = _f(m[-1][1])
    return out


_TOK_COUNT = re.compile(r"^Total number of tokens:\s*(\d+)\s*$", re.M)
_TOK_IDS = re.compile(r"^\[([\d,\s]*)\]\s*$", re.M)


def parse_tokenize_output(text: str) -> dict:
    """Parse `llama-tokenize --ids --show-count` output."""
    out: dict = {}
    c = _TOK_COUNT.findall(text)
    if c:
        out["n_tokens"] = int(c[-1])
    ids = _TOK_IDS.findall(text)
    if ids:
        body = ids[-1].strip()
        out["ids"] = [int(x) for x in body.split(",") if x.strip()] if body else []
        out.setdefault("n_tokens", len(out["ids"]))
    if "n_tokens" not in out:
        raise ParseError("llama-tokenize output has neither an id list nor a token count")
    return out


_IM_CHUNKS = re.compile(r"compute_imatrix: computing over (\d+) chunks, n_ctx=(\d+)")
_IM_FINAL = re.compile(rf"Final estimate: PPL = ({_NUM}) \+/- ({_NUM})")


def parse_imatrix_output(text: str) -> dict:
    text = _strip_log_prefix(text)
    out: dict = {}
    m = _IM_CHUNKS.search(text)
    if m:
        out["n_chunks"] = int(m.group(1))
        out["n_ctx"] = int(m.group(2))
    f = _IM_FINAL.findall(text)
    if f:
        out["calib_ppl"] = _f(f[-1][0])
    if "n_chunks" not in out:
        raise ParseError("llama-imatrix output has no 'computing over N chunks' line")
    return out
