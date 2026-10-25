"""Step fertility: tokens per word and per character with the model's own GGUF tokenizer."""

from __future__ import annotations

from iqp import textproc as tp
from iqp.parsers import parse_tokenize_output
from iqp.util import LOG, StepError, atomic_write_json, run_cmd


def fertility_stats(text: str, n_tokens: int) -> dict:
    words = tp.count_words(text)
    chars = tp.count_chars(text)
    graphemes = tp.count_graphemes_approx(text)
    return {"n_tokens": n_tokens, "n_words": words, "n_chars": chars, "n_graphemes": graphemes,
            "tokens_per_word": n_tokens / max(words, 1), "tokens_per_char": n_tokens / max(chars, 1),
            "tokens_per_grapheme": n_tokens / max(graphemes, 1), "chars_per_token": chars / max(n_tokens, 1)}


def step_fertility(cfg, force: bool = False) -> None:
    base = cfg.gguf_path(cfg.base_variant())
    if not base.exists():
        raise StepError(f"base GGUF missing: {base}")
    raw = cfg.raw_results_dir / "fertility"
    for code in cfg.langs:
        out = raw / f"{code}.json"
        if out.exists() and not force:
            LOG.info("skip (exists): %s", out)
            continue
        f = cfg.eval_kld_path(code)
        text = f.read_text(encoding="utf-8")
        res = run_cmd([str(cfg.bin("llama-tokenize")), "-m", str(base), "-f", str(f), "--show-count", "--ids",
                       "--no-bos", "--log-disable"], cfg.logs_dir / "fertility" / f"{code}.log")
        n = parse_tokenize_output(res)["n_tokens"]
        st = fertility_stats(text, n)
        st.update({"language": code, "source": "llama-tokenize on base GGUF, KLD eval text"})
        atomic_write_json(out, st)
        LOG.info("fertility %s: %.2f tokens/word, %.3f tokens/char", code, st["tokens_per_word"],
                 st["tokens_per_char"])
