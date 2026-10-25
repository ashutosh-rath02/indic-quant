"""Pure text utilities: normalization, deduplication, overlap filtering, disjointness checks."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from typing import Iterable

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[।॥\.,;:!?\"'“”‘’()\[\]{}<>«»\-–—_/\\|*#@%&+=~`^]")


def normalize(text: str) -> str:
    """NFC, lowercase, punctuation stripped, whitespace collapsed. Used only for matching."""
    t = unicodedata.normalize("NFC", text).lower()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = normalize(text).split()
    if len(w) < n:
        return set()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def split_paragraphs(doc: str, min_chars: int, max_chars: int) -> list[str]:
    """Split a document on blank lines / newlines, drop short or list-like fragments."""
    out = []
    for p in re.split(r"\n\s*\n|\n", doc):
        p = _WS.sub(" ", p).strip()
        if len(p) < min_chars:
            continue
        if len(p) > max_chars:
            p = p[:max_chars].rsplit(" ", 1)[0]
        out.append(p)
    return out


def dedup(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for it in items:
        h = text_hash(it)
        if h in seen:
            continue
        seen.add(h)
        out.append(it)
    return out


class OverlapFilter:
    """Rejects calibration paragraphs that contain an eval sentence or share an n-gram with one."""

    def __init__(self, eval_sentences: Iterable[str], n: int):
        self.n = n
        self.sentences = {normalize(s) for s in eval_sentences if normalize(s)}
        self.ngrams: set[tuple[str, ...]] = set()
        if n > 0:
            for s in self.sentences:
                self.ngrams |= word_ngrams(s, n)

    def ok(self, paragraph: str) -> bool:
        np_ = normalize(paragraph)
        if np_ in self.sentences:
            return False
        if self.n > 0:
            w = np_.split()
            for i in range(len(w) - self.n + 1):
                if tuple(w[i:i + self.n]) in self.ngrams:
                    return False
        else:
            for s in self.sentences:
                if len(s) > 20 and s in np_:
                    return False
        return True


def find_contained_sentences(corpus: str, sentences: Iterable[str], min_len: int = 20) -> list[str]:
    """Return eval sentences (normalized) that appear verbatim inside the corpus (normalized)."""
    nc = " " + normalize(corpus) + " "
    hits = []
    for s in sentences:
        ns = normalize(s)
        if len(ns) >= min_len and (" " + ns + " ") in nc:
            hits.append(s)
    return hits


def assert_disjoint_ids(a: Iterable[str], b: Iterable[str], what: str) -> None:
    inter = set(a) & set(b)
    if inter:
        raise AssertionError(f"{what}: {len(inter)} shared ids, e.g. {sorted(inter)[:5]}")


def assert_no_text_overlap(calib_text: str, eval_sentences: Iterable[str], what: str) -> None:
    hits = find_contained_sentences(calib_text, eval_sentences)
    if hits:
        raise AssertionError(f"{what}: {len(hits)} evaluation sentences occur in calibration text, "
                             f"e.g. {hits[:2]}")


def interleave(per_lang: dict[str, list[str]], seed: int) -> list[tuple[str, str]]:
    """Round-robin paragraphs across languages so any prefix of the result is roughly balanced.

    llama-imatrix consumes the first N*ctx tokens of its input, so interleaving keeps the
    language mix even if the tail is cut off.
    """
    rng = random.Random(seed)
    queues = {k: list(v) for k, v in per_lang.items()}
    order = sorted(queues)
    out: list[tuple[str, str]] = []
    while any(queues.values()):
        rng.shuffle(order)
        for k in order:
            if queues[k]:
                out.append((k, queues[k].pop(0)))
    return out


def take_until(paragraphs: list[str], char_target: int) -> list[str]:
    out, n = [], 0
    for p in paragraphs:
        if n >= char_target:
            break
        out.append(p)
        n += len(p) + 1
    return out


def count_words(text: str) -> int:
    return len(text.split())


def count_chars(text: str) -> int:
    """Unicode code points excluding whitespace (combining marks count as characters)."""
    return sum(1 for ch in text if not ch.isspace())


def count_graphemes_approx(text: str) -> int:
    """Approximate extended grapheme clusters: base characters, with combining marks
    (Unicode categories Mn, Mc, Me) and virama/ZWJ joins attached to the previous cluster."""
    n = 0
    join_next = False
    for ch in text:
        if ch.isspace():
            join_next = False
            continue
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Mc", "Me") or ch in ("‌", "‍"):
            if unicodedata.name(ch, "").endswith("VIRAMA") or ch == "‍":
                join_next = True
            continue
        if join_next:
            join_next = False
            continue
        n += 1
    return n


def clean_translation(text: str) -> str:
    """Light cleanup of chat-model translations: drop think blocks, labels, wrapping quotes."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"^\s*(translation|अनुवाद|output)\s*[:：]\s*", "", t, flags=re.I)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = lines[0] if lines else ""
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    if len(t) >= 2 and t[0] in "“\"" and t[-1] in "”\"":
        t = t[1:-1].strip()
    return t
