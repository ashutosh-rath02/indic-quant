"""Pure scoring helpers for MILU (letter log-probabilities) and MT (chrF++)."""

from __future__ import annotations

import math

LETTERS = "ABCD"


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def letter_logprobs(top_logprobs: list[dict], letters: str = LETTERS) -> dict[str, float | None]:
    """Collapse top-N next-token candidates onto answer letters.

    Tokens such as "A", " A", "A." or "(A" all map to letter A; probabilities of all
    variants are summed (log-sum-exp). Letters absent from the top-N list get None.
    """
    buckets: dict[str, list[float]] = {L: [] for L in letters}
    for t in top_logprobs:
        tok = str(t.get("token", "")).strip().strip("(").rstrip(".):")
        if len(tok) == 1 and tok.upper() in buckets and tok == tok.upper():
            lp = t.get("logprob")
            if lp is None and t.get("prob") is not None:
                lp = math.log(max(t["prob"], 1e-30))
            if lp is not None:
                buckets[tok.upper()].append(float(lp))
    return {L: (_logsumexp(v) if v else None) for L, v in buckets.items()}


def predict_letter(lps: dict[str, float | None], floor: float) -> tuple[str, int]:
    """Argmax letter; missing letters get `floor` (below the smallest returned logprob).

    Returns (letter, number of letters that were present in the top-N list).
    Ties break toward the earlier letter, so behaviour is deterministic.
    """
    present = sum(1 for v in lps.values() if v is not None)
    best, best_v = None, -math.inf
    for L, v in lps.items():
        val = v if v is not None else floor
        if val > best_v:
            best, best_v = L, val
    return best or LETTERS[0], present


def milu_score_response(resp: dict, letters: str = LETTERS) -> dict:
    probs = resp.get("completion_probabilities") or []
    if not probs:
        raise ValueError("response has no completion_probabilities; was n_probs > 0 sent?")
    top = probs[0].get("top_logprobs") or probs[0].get("top_probs") or []
    lps = letter_logprobs(top, letters)
    vals = [t.get("logprob") for t in top if t.get("logprob") is not None]
    floor = (min(vals) - 1.0) if vals else -100.0
    pred, present = predict_letter(lps, floor)
    return {"pred": pred, "letters_present": present, "letter_logprobs": lps}


def chrf_pp(hyps: list[str], refs: list[str]) -> float:
    """Corpus chrF++ (sacrebleu CHRF with word_order=2)."""
    from sacrebleu.metrics import CHRF

    if len(hyps) != len(refs):
        raise ValueError("hyps and refs differ in length")
    return float(CHRF(word_order=2).corpus_score(hyps, [refs]).score)


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    import random

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2)) - 1]
