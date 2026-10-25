"""Fake versions of iqp.sources loaders (same signatures, synthetic content)."""

from __future__ import annotations

import random

from iqp.fakes.common import fake_sentence, h01

N_PER_SPLIT = 60


def load_flores(cfg, code: str, split: str, token) -> dict[str, str]:
    off = 0 if split == "devtest" else 10000
    return {str(off + i): fake_sentence(code, f"{split}:{i}") for i in range(N_PER_SPLIT)}


def load_in22(cfg, codes, token):
    return {c: {str(i): fake_sentence(c, f"in22:{i}") for i in range(2 * N_PER_SPLIT)} for c in codes}


def load_local_eval(cfg, code, part):
    return {str(i): fake_sentence(code, f"local:{part}:{i}") for i in range(N_PER_SPLIT)}


def _code_for_milu(cfg, lang_name: str) -> str:
    for c, spec in cfg.raw["languages"]["registry"].items():
        if spec["milu"] == lang_name:
            return c
    return "eng_Latn"


def fake_answer_index(question: str) -> int:
    return int(h01("milu-answer", question) * 4)


def load_milu(cfg, lang_name: str, split: str, token) -> list[dict]:
    code = _code_for_milu(cfg, lang_name)
    n = 80 if split == "test" else 12
    rows = []
    for i in range(n):
        q = fake_sentence(code, f"milu:{split}:{i}", n_words=8).rstrip(".।") + "?"
        ans = fake_answer_index(q)
        rows.append({"question": q, "option1": fake_sentence(code, f"o1:{split}:{i}", n_words=2),
                     "option2": fake_sentence(code, f"o2:{split}:{i}", n_words=2),
                     "option3": fake_sentence(code, f"o3:{split}:{i}", n_words=2),
                     "option4": fake_sentence(code, f"o4:{split}:{i}", n_words=2),
                     "target": f"option{ans + 1}", "subject": ["History", "Physics", "Law"][i % 3]})
    return rows


def iter_calibration_docs(cfg, code: str, token, seed: int):
    """Synthetic 'Wikipedia' documents. Every 7th document deliberately embeds an evaluation
    sentence so the overlap filter is exercised."""
    rng = random.Random(f"{seed}:{code}")
    leak = load_flores(cfg, code, "devtest", None)
    leak_ids = list(leak)
    for d in range(100000):
        paras = []
        for p in range(rng.randint(2, 5)):
            paras.append(" ".join(fake_sentence(code, f"calib:{d}:{p}:{s}", "calib") for s in range(rng.randint(3, 6))))
        if d % 7 == 3:
            paras.append(paras[0] + " " + leak[leak_ids[d % len(leak_ids)]])
        yield "\n\n".join(paras)
