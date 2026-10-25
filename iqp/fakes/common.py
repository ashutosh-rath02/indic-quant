"""Deterministic synthetic text and a toy damage model shared by the fake tools."""

from __future__ import annotations

import hashlib
import math
import random

# (consonant range start, count, vowel-sign list, chars per token for the toy tokenizer)
SCRIPTS = {
    "Latn": (None, None, None, 4.2),
    "Deva": (0x0915, 33, [0x093E, 0x093F, 0x0940, 0x0941, 0x0947, 0x094B], 2.1),
    "Beng": (0x0995, 33, [0x09BE, 0x09BF, 0x09C0, 0x09C1, 0x09C7, 0x09CB], 1.9),
    "Taml": (0x0B95, 18, [0x0BBE, 0x0BBF, 0x0BC0, 0x0BC1, 0x0BC6, 0x0BCA], 1.4),
    "Telu": (0x0C15, 30, [0x0C3E, 0x0C3F, 0x0C40, 0x0C41, 0x0C46, 0x0C4A], 1.5),
    "Knda": (0x0C95, 30, [0x0CBE, 0x0CBF, 0x0CC0, 0x0CC1, 0x0CC6, 0x0CCA], 1.5),
    "Mlym": (0x0D15, 30, [0x0D3E, 0x0D3F, 0x0D40, 0x0D41, 0x0D46, 0x0D4A], 1.3),
    "Gujr": (0x0A95, 30, [0x0ABE, 0x0ABF, 0x0AC0, 0x0AC1, 0x0AC7, 0x0ACB], 1.8),
    "Orya": (0x0B15, 30, [0x0B3E, 0x0B3F, 0x0B40, 0x0B41, 0x0B47, 0x0B4B], 1.4),
    "Guru": (0x0A15, 30, [0x0A3E, 0x0A3F, 0x0A40, 0x0A41, 0x0A47, 0x0A4B], 1.8),
}
EN_WORDS = ("the of and to in is was for on that with as by at from his her an were are which this be "
            "river city people year state government school music science water history village language "
            "festival temple market road train farmer family court election film book teacher").split()

QTYPE_KLD = {"Q8_0": 0.0009, "Q6_K": 0.0032, "Q5_K_M": 0.0075, "Q4_K_M": 0.021, "Q3_K_M": 0.062,
             "Q2_K": 0.24, "IQ2_M": 0.33}
QTYPE_BPW = {"BF16": 16.0, "F16": 16.0, "Q8_0": 8.50, "Q6_K": 6.56, "Q5_K_M": 5.71, "Q4_K_M": 4.92,
             "Q3_K_M": 3.95, "Q2_K": 3.18, "IQ2_M": 2.93}
LANG_MULT = {"eng": 1.0, "hin": 1.6, "mar": 1.9, "ben": 2.0, "tel": 2.6, "tam": 2.9, "kan": 2.5,
             "mal": 3.0, "guj": 2.1, "ory": 2.7, "pan": 2.2}


def h01(*parts) -> float:
    d = hashlib.sha1("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64


def script_of(code: str) -> str:
    return code.split("_")[1]


def fake_word(rng: random.Random, code: str) -> str:
    sc = script_of(code)
    start, n, signs, _ = SCRIPTS[sc]
    if start is None:
        return rng.choice(EN_WORDS)
    out = []
    for _ in range(rng.randint(1, 3)):
        out.append(chr(start + rng.randrange(n)))
        if rng.random() < 0.6:
            out.append(chr(rng.choice(signs)))
    return "".join(out)


def fake_sentence(code: str, sid, salt: str = "eval", n_words: int | None = None) -> str:
    rng = random.Random(f"{salt}:{code}:{sid}")
    n = n_words or rng.randint(9, 18)
    words = [fake_word(rng, code) for _ in range(n)] + [str(rng.randint(10, 99))]
    end = "." if script_of(code) in ("Latn", "Taml", "Telu", "Knda", "Mlym") else "।"
    return " ".join(words) + end


def toy_token_count(text: str) -> int:
    """Toy tokenizer: chars-per-token ratio by script, spaces free."""
    total = 0.0
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        ratio = 4.2
        for sc, (start, n, _signs, r) in SCRIPTS.items():
            if start is not None and (start & 0xFF80) <= cp < (start & 0xFF80) + 0x80:
                ratio = r
                break
        total += 1.0 / ratio
    return int(math.ceil(total))


def toy_kld(qtype: str, arm: str, lang3: str, seed: str = "") -> float:
    base = QTYPE_KLD.get(qtype, 0.05)
    m = LANG_MULT.get(lang3, 2.0)
    if arm == "static":
        f = 1.18
    elif arm == "en":
        f = 1.0 if lang3 == "eng" else 1.02
    elif arm == "indic":
        f = 1.04 if lang3 == "eng" else 0.84
    elif arm == "indic_no_tam":
        f = 1.04 if lang3 == "eng" else (0.93 if lang3 == "tam" else 0.85)
    else:
        f = 1.0
    noise = 1.0 + 0.06 * (h01(qtype, arm, lang3, seed) - 0.5)
    return base * m * f * noise
