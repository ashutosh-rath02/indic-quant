"""Step prepare_data: evaluation text, translation set, MILU prompts, calibration sets.

Disjointness guarantees enforced here (the step fails if any is violated):
  1. KLD text and translation sentences come from different FLORES+ splits (devtest vs dev),
     or from disjoint id sets when a single-split source is used (IN22-Gen, local).
  2. Calibration paragraphs come from a different corpus (Wikipedia or Sangraha) and are
     filtered so none contains an evaluation sentence or shares a word n-gram with one.
     After assembly the final calibration files are re-checked for verbatim containment.
  3. MILU test questions and few-shot examples come from different splits (test vs validation).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from iqp import textproc as tp
from iqp.parsers import parse_tokenize_output
from iqp.util import LOG, StepError, atomic_write_json, atomic_write_text, hf_token, read_json, run_cmd, write_jsonl


def _src(cfg):
    if cfg.dry_run:
        from iqp.fakes import data as fake
        return fake
    from iqp import sources
    return sources


# --------------------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------------------

class TokenCounter:
    """Counts tokens with the model tokenizer.

    Uses the HF tokenizer.json (fast, per paragraph) when present, and llama-tokenize on the
    base GGUF for authoritative totals. Without tokenizer.json, per-paragraph counts are
    estimated from a characters-per-token ratio measured with llama-tokenize.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._hf = None
        tj = cfg.hf_dir / "tokenizer.json"
        if tj.exists():
            try:
                from tokenizers import Tokenizer
                self._hf = Tokenizer.from_file(str(tj))
            except Exception as e:  # pragma: no cover
                LOG.warning("could not load tokenizer.json (%s); falling back to llama-tokenize", e)
        self._ratio: dict[str, float] = {}

    def llama_count(self, text: str, tag: str) -> int:
        base = self.cfg.gguf_path(self.cfg.base_variant())
        if not base.exists():
            raise StepError(f"base GGUF missing ({base}); run convert before prepare_data")
        f = self.cfg.data_dir / "tmp" / f"tok_{tag}.txt"
        atomic_write_text(f, text)
        out = run_cmd([str(self.cfg.bin("llama-tokenize")), "-m", str(base), "-f", str(f), "--show-count",
                       "--ids", "--no-bos", "--log-disable"],
                      self.cfg.logs_dir / "prepare_data" / f"tokenize_{tag}.log")
        return parse_tokenize_output(out)["n_tokens"]

    def calibrate(self, code: str, sample: str) -> None:
        if self._hf is None and sample:
            n = self.llama_count(sample, f"ratio_{code}")
            self._ratio[code] = max(len(sample) / max(n, 1), 0.1)

    def count(self, text: str, code: str) -> int:
        if self._hf is not None:
            return len(self._hf.encode(text, add_special_tokens=False).ids)
        return int(math.ceil(len(text) / self._ratio.get(code, 3.0)))


# --------------------------------------------------------------------------------------
# Evaluation data
# --------------------------------------------------------------------------------------

def _load_eval(cfg, token):
    """Return (kld: {code: {id: text}}, mt: {code: {id: text}}) with disjoint ids."""
    e = cfg.raw["eval"]
    src = _src(cfg)
    codes = sorted(set(cfg.langs) | {"eng_Latn"})
    if e["source"] == "flores_plus":
        kld = {c: src.load_flores(cfg, c, e["flores_kld_split"], token) for c in codes}
        mt = {c: src.load_flores(cfg, c, e["flores_mt_split"], token) for c in codes}
        if e["flores_kld_split"] == e["flores_mt_split"]:
            raise StepError("eval.flores_kld_split and eval.flores_mt_split must differ")
        return kld, mt
    if e["source"] == "in22_gen":
        allt = src.load_in22(cfg, codes, token)
    elif e["source"] == "local":
        allt = {c: {**{f"k{i}": t for i, t in src.load_local_eval(cfg, c, "kld").items()}} for c in codes}
        mt_local = {c: {f"m{i}": t for i, t in src.load_local_eval(cfg, c, "mt").items()} for c in codes}
        return allt, mt_local
    else:
        raise StepError(f"unknown eval.source {e['source']}")
    ids = sorted(set.intersection(*[set(d) for d in allt.values()]))
    rng = random.Random(e["seed"])
    rng.shuffle(ids)
    n_mt = cfg.raw["mt"]["n_sentences"]
    mt_ids, kld_ids = set(ids[:n_mt]), set(ids[n_mt:])
    kld = {c: {i: t for i, t in d.items() if i in kld_ids} for c, d in allt.items()}
    mt = {c: {i: t for i, t in d.items() if i in mt_ids} for c, d in allt.items()}
    return kld, mt


def _parallel_ids(per_lang: dict[str, dict[str, str]]) -> list[str]:
    common = set.intersection(*[set(d) for d in per_lang.values()])
    common = {i for i in common if all(per_lang[c][i] for c in per_lang)}

    def key(i):
        return (0, int(i)) if str(i).isdigit() else (1, str(i))
    return sorted(common, key=key)


def prepare_eval(cfg, token, force: bool) -> dict:
    manifest_path = cfg.data_dir / "eval" / "manifest.json"
    if manifest_path.exists() and not force:
        LOG.info("skip (exists): %s", manifest_path)
        return read_json(manifest_path)
    kld, mt = _load_eval(cfg, token)
    kld_ids = _parallel_ids(kld)
    mt_ids_all = _parallel_ids(mt)
    if cfg.raw["eval"]["source"] != "flores_plus":
        # FLORES+ uses different splits; single-split sources must have disjoint ids.
        tp.assert_disjoint_ids(kld_ids, mt_ids_all, "KLD vs MT sentence ids")
    # Even across FLORES+ splits, no MT source sentence may also appear in the KLD text.
    kld_norm = {tp.normalize(kld["eng_Latn"][i]) for i in kld_ids} if "eng_Latn" in kld else set()
    mt_ids_all = [i for i in mt_ids_all if tp.normalize(mt["eng_Latn"][i]) not in kld_norm]
    if not kld_ids:
        raise StepError("no parallel KLD sentences found across the enabled languages")

    for c in cfg.langs:
        text = "\n".join(kld[c][i] for i in kld_ids)
        atomic_write_text(cfg.eval_kld_path(c), text + "\n")

    rng = random.Random(cfg.raw["eval"]["seed"])
    mt_ids = list(mt_ids_all)
    rng.shuffle(mt_ids)
    mt_ids = sorted(mt_ids[: cfg.raw["mt"]["n_sentences"]], key=lambda i: (len(str(i)), str(i)))
    rows = [{"id": i, "eng_Latn": mt["eng_Latn"][i], **{c: mt[c][i] for c in cfg.langs if c != "eng_Latn"}}
            for i in mt_ids]
    write_jsonl(cfg.eval_mt_path(), rows)

    # Every eval sentence, all languages, for the calibration overlap filter.
    all_eval = {c: [kld[c][i] for i in kld_ids] + [mt[c][i] for i in mt_ids] for c in kld}
    atomic_write_json(cfg.data_dir / "eval" / "all_eval_sentences.json", all_eval)
    manifest = {
        "source": cfg.raw["eval"]["source"],
        "kld_split": cfg.raw["eval"].get("flores_kld_split"),
        "mt_split": cfg.raw["eval"].get("flores_mt_split"),
        "n_kld_sentences": len(kld_ids),
        "n_mt_sentences": len(mt_ids),
        "languages": cfg.langs,
    }
    atomic_write_json(manifest_path, manifest)
    LOG.info("eval data: %d KLD sentences, %d MT sentences per language", len(kld_ids), len(mt_ids))
    return manifest


# --------------------------------------------------------------------------------------
# MILU
# --------------------------------------------------------------------------------------

LETTERS = "ABCD"


def format_milu_question(row: dict, mcfg: dict, with_answer: bool) -> str:
    """Prompt format of the official MILU lm-eval task (utils_milu.doc_to_text)."""
    from iqp.sources import target_to_index

    opts = [str(row[f]) for f in mcfg["option_fields"]]
    s = "Question: " + str(row[mcfg["question_field"]]).strip() + "\nChoices:\n"
    for letter, o in zip(LETTERS, opts):
        s += f"{letter}. {o.strip()}\n"
    s += "Answer:"
    if with_answer:
        s += " " + LETTERS[target_to_index(row[mcfg["target_field"]], len(opts))]
    return s


def build_milu_prompt(shots: list[dict], row: dict, mcfg: dict) -> str:
    parts = [format_milu_question(s, mcfg, True) for s in shots]
    parts.append(format_milu_question(row, mcfg, False))
    return "\n\n".join(parts)


def prepare_milu(cfg, token, counter: TokenCounter | None, force: bool) -> None:
    from iqp.sources import target_to_index

    m = cfg.raw["milu"]
    if not m.get("enabled", True):
        return
    src = _src(cfg)
    max_prompt_tokens = cfg.raw["server"]["ctx_per_slot"] - 64
    for c in cfg.langs:
        out = cfg.milu_path(c, "test")
        if out.exists() and not force:
            LOG.info("skip (exists): %s", out)
            continue
        name = cfg.lang(c)["milu"]
        test = src.load_milu(cfg, name, "test", token)
        val = src.load_milu(cfg, name, "validation", token)
        test_keys = {tp.text_hash(str(r[m["question_field"]])) for r in test}
        val = [r for r in val if tp.text_hash(str(r[m["question_field"]])) not in test_keys]
        rng = random.Random(f"{m['seed']}:{c}")
        idx = list(range(len(test)))
        rng.shuffle(idx)
        idx = sorted(idx[: m["sample_cap"]])
        # Few-shot: the first k validation examples (same sampler as the official lm-eval task),
        # identical for every question of a language, so prompts share a cacheable prefix.
        shots = val[: m["k_shot"]]
        rows = []
        for i in idx:
            r = test[i]
            k = len(shots)
            prompt = build_milu_prompt(shots[:k], r, m)
            if counter is not None:
                while k > 0 and counter.count(prompt, c) > max_prompt_tokens:
                    k -= 1
                    prompt = build_milu_prompt(shots[:k], r, m)
            rows.append({
                "idx": i,
                "prompt": prompt,
                "answer": LETTERS[target_to_index(r[m["target_field"]], len(m["option_fields"]))],
                "k_shot": k,
                "subject": r.get(m.get("subject_field") or "subject"),
            })
        write_jsonl(out, rows)
        LOG.info("MILU %s: %d questions (%d-shot), %d validation rows available", c, len(rows), len(shots), len(val))


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------

def _arm_quotas(cfg, arm: str) -> dict[str, int]:
    c = cfg.raw["calibration"]
    spec = c["arms"][arm]
    budget = int(c["token_budget"])
    langs = list(spec["langs"])
    en_frac = float(spec.get("english_fraction", 0.0) or 0.0)
    quotas: dict[str, int] = {}
    if en_frac > 0 and "eng_Latn" not in langs:
        quotas["eng_Latn"] = int(round(budget * en_frac))
    rest = budget - sum(quotas.values())
    for code in langs:
        quotas[code] = quotas.get(code, 0) + rest // len(langs)
    return quotas


def collect_calibration_pool(cfg, code: str, token, overlap: tp.OverlapFilter, counter: TokenCounter,
                             need_tokens: int) -> list[str]:
    """Paragraphs for one language, deduplicated and filtered against eval text."""
    c = cfg.raw["calibration"]
    pool_path = cfg.data_dir / "calib" / "pool" / f"{code}.jsonl"
    if pool_path.exists():
        rows = [json.loads(x) for x in pool_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if sum(r["n_tokens"] for r in rows) >= need_tokens * 1.15:
            return [r["text"] for r in rows]
    src = _src(cfg)
    paras: list[str] = []
    seen: set[str] = set()
    n_chars = n_rejected = 0
    target_chars = int(c["candidate_chars_per_lang"])
    for doc in src.iter_calibration_docs(cfg, code, token, c["seed"]):
        for p in tp.split_paragraphs(doc, c["min_paragraph_chars"], c["max_paragraph_chars"]):
            h = tp.text_hash(p)
            if h in seen:
                continue
            seen.add(h)
            if not overlap.ok(p):
                n_rejected += 1
                continue
            paras.append(p)
            n_chars += len(p)
        if n_chars >= target_chars:
            break
    rng = random.Random(f"{c['seed']}:{code}:pool")
    rng.shuffle(paras)
    counter.calibrate(code, "\n".join(paras[:200]))
    rows = [{"text": p, "n_tokens": counter.count(p, code)} for p in paras]
    total = sum(r["n_tokens"] for r in rows)
    write_jsonl(pool_path, rows)
    LOG.info("calibration pool %s: %d paragraphs, ~%d tokens, %d rejected for eval overlap",
             code, len(rows), total, n_rejected)
    if total < need_tokens * 1.15:
        raise StepError(f"calibration pool for {code} has only ~{total} tokens, need {need_tokens}; "
                        "raise calibration.candidate_chars_per_lang")
    return [r["text"] for r in rows]


def assemble_arm(per_lang_paras: dict[str, list[str]], quotas: dict[str, int], counts: dict[str, list[int]],
                 seed: int, margin: float = 0.10) -> tuple[str, dict[str, int]]:
    """Select paragraphs per language until quota*(1+margin) tokens, then interleave."""
    chosen: dict[str, list[str]] = {}
    got: dict[str, int] = {}
    for code, q in quotas.items():
        target = int(q * (1 + margin))
        sel, n = [], 0
        for p, k in zip(per_lang_paras[code], counts[code]):
            if n >= target:
                break
            sel.append(p)
            n += k
        if n < q:
            raise StepError(f"not enough calibration text for {code}: {n} < {q} tokens")
        chosen[code], got[code] = sel, n
    if len(chosen) == 1:
        text = "\n\n".join(next(iter(chosen.values())))
    else:
        text = "\n\n".join(p for _, p in tp.interleave(chosen, seed))
    return text, got


def prepare_calibration(cfg, token, counter: TokenCounter, force: bool) -> None:
    c = cfg.raw["calibration"]
    eval_path = cfg.data_dir / "eval" / "all_eval_sentences.json"
    all_eval = read_json(eval_path)
    flat_eval = [s for v in all_eval.values() for s in v]
    need: dict[str, int] = {}
    for arm in cfg.calib_arms:
        for code, q in _arm_quotas(cfg, arm).items():
            need[code] = max(need.get(code, 0), q)
    arms_todo = [a for a in cfg.calib_arms if force or not cfg.calib_text_path(a).exists()]
    if not arms_todo:
        LOG.info("calibration sets exist; skipping")
        return
    overlap = tp.OverlapFilter(flat_eval, int(c["ngram_filter"]))
    pools = {code: collect_calibration_pool(cfg, code, token, overlap, counter, n) for code, n in need.items()}
    counts = {code: [counter.count(p, code) for p in paras] for code, paras in pools.items()}
    manifest: dict = {"token_budget": c["token_budget"], "ctx": c["ctx"], "source": c["source"], "arms": {}}
    mpath = cfg.data_dir / "calib" / "manifest.json"
    if mpath.exists():
        manifest = read_json(mpath)
    for arm in arms_todo:
        quotas = _arm_quotas(cfg, arm)
        text, est = assemble_arm(pools, quotas, counts, c["seed"])
        tp.assert_no_text_overlap(text, flat_eval, f"calibration arm {arm}")
        atomic_write_text(cfg.calib_text_path(arm), text + "\n")
        total = counter.llama_count(text, f"calib_{arm}")
        if total < c["token_budget"]:
            raise StepError(f"calibration arm {arm}: {total} tokens < budget {c['token_budget']}")
        manifest["arms"][arm] = {"quotas": quotas, "estimated_tokens": est, "llama_tokens_total": total,
                                 "chunks": c["token_budget"] // c["ctx"], "chars": len(text)}
        LOG.info("calibration arm %s: %d tokens in file, %d will be used (%s)", arm, total, c["token_budget"],
                 ", ".join(f"{k}={v}" for k, v in quotas.items()))
    atomic_write_json(mpath, manifest)


def step_prepare_data(cfg, force: bool = False) -> None:
    token = hf_token(cfg)
    prepare_eval(cfg, token, force)
    counter = TokenCounter(cfg)
    prepare_milu(cfg, token, counter, force)
    prepare_calibration(cfg, token, counter, force)
    LOG.info("prepare_data complete")


def step_validate(cfg, force: bool = False) -> None:
    """Open one file of every configured dataset and check its schema. Fails loudly."""
    token = hf_token(cfg)
    src = _src(cfg)
    probs = []
    e = cfg.raw["eval"]
    lang = "hin_Deva" if "hin_Deva" in cfg.langs else cfg.langs[-1]
    checks = []
    if e["source"] == "flores_plus":
        checks.append(("FLORES+ " + e["flores_kld_split"], lambda: src.load_flores(cfg, lang, e["flores_kld_split"], token)))
        checks.append(("FLORES+ " + e["flores_mt_split"], lambda: src.load_flores(cfg, "eng_Latn", e["flores_mt_split"], token)))
    elif e["source"] == "in22_gen":
        checks.append(("IN22-Gen", lambda: src.load_in22(cfg, ["eng_Latn", lang], token)))
    if cfg.raw["milu"].get("enabled", True):
        checks.append(("MILU test", lambda: src.load_milu(cfg, cfg.lang(lang)["milu"], "test", token)))
        checks.append(("MILU validation", lambda: src.load_milu(cfg, cfg.lang(lang)["milu"], "validation", token)))

    def first_doc():
        for d in src.iter_calibration_docs(cfg, lang, token, 0):
            return d
        raise StepError("calibration source returned no documents")
    checks.append((f"calibration ({cfg.raw['calibration']['source']})", first_doc))
    for name, fn in checks:
        try:
            r = fn()
            if isinstance(r, dict):
                sample = next(iter(r.values()))
                sample = next(iter(sample.values())) if isinstance(sample, dict) else sample
            elif isinstance(r, list):
                sample = r[0]
            else:
                sample = r
            LOG.info("OK  %s: sample=%s", name, str(sample)[:160].replace("\n", " "))
        except Exception as ex:
            LOG.error("FAIL %s: %s", name, ex)
            probs.append(f"{name}: {ex}")
    if probs:
        raise StepError("dataset validation failed:\n  " + "\n  ".join(probs))
    LOG.info("all dataset checks passed")
