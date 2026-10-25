import json
import math

import pytest

from iqp import textproc as tp
from iqp.scoring import bootstrap_ci, chrf_pp, letter_logprobs, milu_score_response, predict_letter


def test_normalize_and_hash():
    assert tp.normalize("  Hello,   World!  ") == "hello world"
    assert tp.normalize("भारत की राजधानी।") == "भारत की राजधानी"
    assert tp.text_hash("A b.") == tp.text_hash("a   B")


def test_dedup_keeps_first():
    assert tp.dedup(["One two.", "one TWO", "three"]) == ["One two.", "three"]


def test_overlap_filter_exact_and_ngram():
    ev = ["The quick brown fox jumps over the lazy dog near the river bank today."]
    f = tp.OverlapFilter(ev, n=8)
    assert not f.ok("Intro text. The quick brown fox jumps over the lazy dog near the river bank today. More.")
    assert not f.ok("x y z quick brown fox jumps over the lazy dog near w")  # shares an 8-gram
    assert f.ok("A completely different paragraph about monsoon rainfall in Kerala and its farms.")
    f0 = tp.OverlapFilter(ev, n=0)
    assert not f0.ok("prefix " + ev[0])


def test_no_text_overlap_assertion():
    ev = ["सेवा संबंधी लोगों के लिए भेष कई गुणों का संयोजन है, जैसे कि उनके जूते।"]
    with pytest.raises(AssertionError):
        tp.assert_no_text_overlap("कुछ पाठ। " + ev[0] + " और पाठ।", ev, "test")
    tp.assert_no_text_overlap("बिलकुल अलग पाठ जो मूल्यांकन में नहीं है और लंबा भी है।", ev, "test")


def test_disjoint_ids():
    tp.assert_disjoint_ids(["1", "2"], ["3"], "x")
    with pytest.raises(AssertionError):
        tp.assert_disjoint_ids(["1", "2"], ["2"], "x")


def test_interleave_balanced_prefix():
    per = {"a": [f"a{i}" for i in range(10)], "b": [f"b{i}" for i in range(10)], "c": [f"c{i}" for i in range(10)]}
    out = tp.interleave(per, seed=1)
    assert len(out) == 30
    first9 = [k for k, _ in out[:9]]
    assert all(first9.count(k) == 3 for k in "abc")
    # order within a language is preserved
    assert [p for k, p in out if k == "a"] == per["a"]


def test_split_paragraphs():
    doc = "short\n\n" + "x" * 100 + "\n" + "y " * 3000
    ps = tp.split_paragraphs(doc, 80, 500)
    assert len(ps) == 2 and len(ps[1]) <= 500


def test_counts():
    assert tp.count_words("नमस्ते दुनिया") == 2
    assert tp.count_chars("ab c") == 3
    # न + म + स्ते (स + ् + त + े) -> 3 approximate graphemes
    assert tp.count_graphemes_approx("नमस्ते") == 3


def test_clean_translation():
    assert tp.clean_translation("<think>\n</think>\n\n\"नमस्ते दुनिया\"") == "नमस्ते दुनिया"
    assert tp.clean_translation("Translation: হ্যালো\n\nNote: literal") == "হ্যালো"


def test_letter_logprobs_merges_variants():
    top = [{"token": " A", "logprob": -1.0}, {"token": "A", "logprob": -2.0}, {"token": " B", "logprob": -0.5},
           {"token": " The", "logprob": -3.0}, {"token": " a", "logprob": -0.1}]
    lp = letter_logprobs(top)
    assert lp["A"] == pytest.approx(math.log(math.exp(-1) + math.exp(-2)))
    assert lp["B"] == -0.5 and lp["C"] is None
    pred, present = predict_letter(lp, floor=-10)
    assert pred == "B" and present == 2


def test_milu_score_real_server_response(fixture_text):
    resp = json.loads(fixture_text("llama_server_completion_nprobs20.json"))
    r = milu_score_response(resp)
    assert r["letters_present"] == 4
    assert r["pred"] in "ABCD"
    top = resp["completion_probabilities"][0]["top_logprobs"]
    best = max((t for t in top if t["token"].strip() in "ABCD" and t["token"].strip()), key=lambda t: t["logprob"])
    assert r["pred"] == best["token"].strip()


def test_milu_score_requires_probs():
    with pytest.raises(ValueError):
        milu_score_response({"content": " A"})


def test_chrf_pp():
    assert chrf_pp(["नमस्ते दुनिया"], ["नमस्ते दुनिया"]) == pytest.approx(100.0)
    s = chrf_pp(["the cat sat"], ["the cat sat on the mat"])
    assert 0 < s < 100


def test_bootstrap_ci():
    lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 0, 1], seed=0)
    assert 0 <= lo <= 0.625 <= hi <= 1
