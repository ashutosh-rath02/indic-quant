"""Parsers against output captured from real llama.cpp runs (commit 74bca39, CPU build)."""

import json
import math

import pytest

from iqp.parsers import (ParseError, parse_base_ppl_output, parse_imatrix_output, parse_kld_chunks,
                         parse_kld_output, parse_quantize_output, parse_tokenize_output)


def test_kld_real_output(fixture_text):
    d = parse_kld_output(fixture_text("llama_perplexity_kld_qwen3-0.6b_q4km_cpu.txt"))
    assert d["kld_mean"] == pytest.approx(0.093945)
    assert d["kld_mean_unc"] == pytest.approx(0.005537)
    assert d["kld_median"] == pytest.approx(0.041239)
    assert d["kld_p99"] == pytest.approx(0.715960)
    assert d["kld_p99_9"] == pytest.approx(1.322077)
    assert d["kld_max"] == pytest.approx(1.338138)
    assert d["same_top_p_pct"] == pytest.approx(84.383)
    assert d["same_top_p_pct_unc"] == pytest.approx(1.316)
    assert d["ppl_ratio"] == pytest.approx(1.061253)
    assert d["ppl_q"] == pytest.approx(7.588079)
    assert d["ppl_base"] == pytest.approx(7.150115)
    assert d["ln_ppl_ratio"] == pytest.approx(0.059450)
    assert d["ppl_corr_pct"] == pytest.approx(96.84)
    assert d["dp_rms_pct"] == pytest.approx(9.974)
    assert d["n_chunks"] == 6 and d["n_ctx"] == 256


def test_kld_percentile_lines_not_confused(fixture_text):
    # "99.0%   KLD" must not match the "99.9%" or " 1.0%" lines.
    d = parse_kld_output(fixture_text("llama_perplexity_kld_qwen3-0.6b_q4km_cpu.txt"))
    assert d["kld_p99"] < d["kld_p99_9"] < d["kld_max"]
    assert d["kld_p10"] == pytest.approx(0.003663)
    assert d["kld_min"] == pytest.approx(0.000068)


def test_kld_chunk_table(fixture_text):
    rows = parse_kld_chunks(fixture_text("llama_perplexity_kld_qwen3-0.6b_q4km_cpu.txt"))
    assert [r["chunk"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert rows[-1]["kld_mean"] == pytest.approx(0.09395)
    assert rows[0]["same_top_p_pct"] == pytest.approx(81.102)


def test_kld_with_log_prefixes_and_colors(fixture_text):
    text = fixture_text("llama_perplexity_kld_qwen3-0.6b_q4km_cpu.txt")
    prefixed = "\n".join(("0.02.123.456 I " + ln) if ln.startswith(("Mean", "Median", "99.0", "Same")) else ln
                         for ln in text.splitlines())
    prefixed = prefixed.replace("Same top p", "\x1b[0mSame top p")
    d = parse_kld_output(prefixed)
    assert d["kld_mean"] == pytest.approx(0.093945)
    assert d["same_top_p_pct"] == pytest.approx(84.383)


def test_kld_too_few_tokens_raises():
    # perplexity.cpp returns before printing statistics when fewer than 100 tokens were scored.
    text = ("kl_divergence: computing over 1 chunks, n_ctx=64, batch_size=64, n_seq=1\n\n"
            "chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              "
            "Δp RMS            Same top p\n"
            "   1       8.9909 ±    1.5987       0.11613 ±    0.04671       0.08799 ±    0.01183     "
            "7.418 ±  0.767 %    81.102 ±  3.488 %\n")
    with pytest.raises(ParseError):
        parse_kld_output(text)
    assert len(parse_kld_chunks(text)) == 1


def test_kld_error_hint():
    with pytest.raises(ParseError, match="vocabulary"):
        parse_kld_output("kl_divergence: inconsistent vocabulary (262144 vs 151936)\n")


def test_kld_nan_values():
    text = ("Mean PPL(Q)                   :        nan ±        nan\n"
            "Mean PPL(base)                :   7.150115 ±   0.580422\n"
            "Mean PPL(Q)/PPL(base)         :        nan ±        nan\n"
            "Mean    KLD:        nan ±        nan\n"
            "99.0%   KLD:        nan\n"
            "Median  KLD:        nan\n"
            "Same top p:  0.000 ± 0.000 %\n")
    d = parse_kld_output(text)
    assert math.isnan(d["kld_mean"])


def test_base_ppl(fixture_text):
    d = parse_base_ppl_output(fixture_text("llama_perplexity_base_qwen3-0.6b_cpu.txt"))
    assert d["ppl"] == pytest.approx(7.1619)
    assert d["ppl_unc"] == pytest.approx(0.58447)
    assert d["n_chunks"] == 6


def test_quantize_sizes(fixture_text):
    d = parse_quantize_output(fixture_text("llama_quantize_q4km_tail.txt"))
    assert d["quant_mib"] == pytest.approx(456.11)
    assert d["quant_bpw"] == pytest.approx(5.09)
    assert d["orig_bpw"] == pytest.approx(16.0)


def test_quantize_missing():
    with pytest.raises(ParseError):
        parse_quantize_output("ERROR: this quantization requires an importance matrix!")


def test_imatrix(fixture_text):
    d = parse_imatrix_output(fixture_text("llama_imatrix_qwen3-0.6b_cpu.txt"))
    assert d["n_chunks"] == 8 and d["n_ctx"] == 256
    assert d["calib_ppl"] > 0


def test_tokenize_ids_and_count():
    out = "[151643, 9707, 1879]\nTotal number of tokens: 3\n"
    d = parse_tokenize_output(out)
    assert d["n_tokens"] == 3 and d["ids"] == [151643, 9707, 1879]
    assert parse_tokenize_output("Total number of tokens: 1068\n")["n_tokens"] == 1068
    assert parse_tokenize_output("[]\n")["n_tokens"] == 0
    with pytest.raises(ParseError):
        parse_tokenize_output("garbage")


def test_server_fixture_shape(fixture_text):
    j = json.loads(fixture_text("llama_server_completion_nprobs20.json"))
    top = j["completion_probabilities"][0]["top_logprobs"]
    assert len(top) == 20 and {"token", "logprob", "id"} <= set(top[0])
