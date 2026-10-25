"""Config, variants, data disjointness, aggregation, decision logic, plotting and the dry run."""

import json

import pandas as pd
import pytest

from iqp.config import Config, Variant
from iqp.kld import logits_file_bytes, plan_chunks
from iqp.prepare import assemble_arm, build_milu_prompt, format_milu_question, _arm_quotas
from iqp.sources import target_to_index
from iqp.util import StepError, read_json, read_jsonl


def test_variants_from_default_config(dry_cfg):
    names = [v.name for v in dry_cfg.all_variants()]
    assert names[0] == "BF16__base"
    assert "IQ2_M__static" not in names           # IQ quants need an imatrix
    for q in ("Q4_K_M", "Q3_K_M", "Q2_K"):
        for arm in ("static", "en", "indic", "indic_no_tam"):
            assert f"{q}__{arm}" in names
    assert {"IQ2_M__en", "IQ2_M__indic", "IQ2_M__indic_no_tam"} <= set(names)
    assert "Q8_0__en" not in names
    assert len(names) == 1 + 3 + 3 * 4 + 3 == 19
    assert Variant.parse("Q2_K__indic_no_tam") == Variant("Q2_K", "indic_no_tam")


def test_config_validation_rejects_bad_arm(tmp_path):
    import yaml
    from iqp.config import REPO_ROOT
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    raw["calibration"]["arms"]["static"] = {"langs": ["eng_Latn"]}
    with pytest.raises(ValueError):
        Config(raw)
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    raw["languages"]["enabled"].append("xxx_Latn")
    with pytest.raises(ValueError):
        Config(raw)


def test_arm_quotas_equal_budget(dry_cfg):
    budget = dry_cfg.raw["calibration"]["token_budget"]
    for arm in dry_cfg.calib_arms:
        q = _arm_quotas(dry_cfg, arm)
        assert budget - len(q) < sum(q.values()) <= budget
    q = _arm_quotas(dry_cfg, "indic")
    assert len(set(q.values())) == 1 and "eng_Latn" not in q
    dry_cfg.raw["calibration"]["arms"]["indic"]["english_fraction"] = 0.2
    q = _arm_quotas(dry_cfg, "indic")
    assert q["eng_Latn"] == round(0.2 * budget)


def test_assemble_arm_meets_quota():
    paras = {"a": ["x" * 10] * 20, "b": ["y" * 10] * 20}
    counts = {"a": [5] * 20, "b": [5] * 20}
    text, got = assemble_arm(paras, {"a": 30, "b": 30}, counts, seed=0)
    assert got["a"] >= 30 and got["b"] >= 30
    with pytest.raises(StepError):
        assemble_arm(paras, {"a": 1000}, counts, seed=0)


def test_plan_chunks():
    assert plan_chunks({"en": 5000, "hi": 9000}, 512, 24, True) == {"en": 9, "hi": 9}
    assert plan_chunks({"en": 5000, "hi": 9000}, 512, 24, False) == {"en": 9, "hi": 17}
    assert plan_chunks({"en": 50000}, 512, 24, True) == {"en": 24}
    with pytest.raises(StepError):
        plan_chunks({"en": 900}, 512, 24, True)


def test_logits_file_size_matches_real_file():
    # Real base logits file from the CPU smoke run: Qwen3-0.6B, ctx 256, 6 chunks -> 231562724 bytes.
    assert logits_file_bytes(256, 151936, 6) == 231562724


def test_milu_prompt_matches_official_format():
    mcfg = {"question_field": "question", "option_fields": ["option1", "option2", "option3", "option4"],
            "target_field": "target"}
    row = {"question": "Q?", "option1": "a", "option2": "b", "option3": "c", "option4": "d", "target": "option3"}
    assert format_milu_question(row, mcfg, False) == "Question: Q?\nChoices:\nA. a\nB. b\nC. c\nD. d\nAnswer:"
    assert format_milu_question(row, mcfg, True).endswith("Answer: C")
    p = build_milu_prompt([row], row, mcfg)
    assert p.count("Question:") == 2 and p.endswith("Answer:")
    assert target_to_index("option1") == 0 and target_to_index("4") == 3 and target_to_index("B") == 1
    with pytest.raises(ValueError):
        target_to_index("option5")


@pytest.fixture
def dry_run_done(dry_cfg):
    from iqp.cli import PIPELINE, _steps
    steps = _steps()
    for name in PIPELINE:
        steps[name](dry_cfg, force=False)
    return dry_cfg


def test_dry_run_end_to_end(dry_run_done):
    cfg = dry_run_done
    rd = cfg.results_dir
    df = pd.read_csv(rd / "results_tidy.csv")
    assert list(df.columns)[:5] == ["model", "variant", "qtype", "arm", "language"]
    kld = df[df.metric == "kld_mean"]
    assert len(kld) == len(cfg.quant_variants()) * len(cfg.langs)
    # one row per model x variant x language x metric
    assert not df.duplicated(subset=["model", "variant", "language", "metric"]).any()
    sizes = df[df.metric == "bpw"]
    assert set(sizes.variant) == {v.name for v in cfg.all_variants()}
    for f in ("kld_heatmap.png", "retention_vs_bpw.png", "calibration_delta.png", "fertility_vs_kld.png"):
        assert (cfg.plots_dir / f).stat().st_size > 10000
    rep = (rd / "REPORT.md").read_text()
    assert "DRY RUN" in rep and "## Decision" in rep and "Verdict:" in rep
    assert "--" not in rep.replace("|---", "")
    d = read_json(rd / "decision.json")
    assert d["verdict"] in ("strong signal", "partial signal", "weak signal")


def test_dry_run_data_disjointness(dry_run_done):
    from iqp import textproc as tp
    cfg = dry_run_done
    man = read_json(cfg.data_dir / "eval" / "manifest.json")
    assert man["kld_split"] != man["mt_split"]
    all_eval = read_json(cfg.data_dir / "eval" / "all_eval_sentences.json")
    flat = [s for v in all_eval.values() for s in v]
    for arm in cfg.calib_arms:
        text = cfg.calib_text_path(arm).read_text()
        assert tp.find_contained_sentences(text, flat) == []
    # MT sentences are not in the KLD text
    for code in cfg.langs:
        kld_text = cfg.eval_kld_path(code).read_text()
        mt = read_jsonl(cfg.eval_mt_path())
        assert tp.find_contained_sentences(kld_text, [r[code] for r in mt]) == []
    # the fake calibration source plants eval sentences; the filter must have rejected them
    log = (cfg.logs_dir / "prepare_data.log").read_text() if (cfg.logs_dir / "prepare_data.log").exists() else ""
    cm = read_json(cfg.data_dir / "calib" / "manifest.json")
    assert set(cm["arms"]) == set(cfg.calib_arms)
    assert all(a["llama_tokens_total"] >= cm["token_budget"] for a in cm["arms"].values())
    assert "indic_no_tam" not in cm["arms"] or "tam_Taml" not in cm["arms"]["indic_no_tam"]["quotas"]
    del log


def test_dry_run_is_resumable(dry_run_done):
    cfg = dry_run_done
    target = cfg.raw_results_dir / "kld" / "Q4_K_M__indic" / "hin_Deva.json"
    mtime = target.stat().st_mtime_ns
    from iqp.kld import step_run_kld
    step_run_kld(cfg)
    assert target.stat().st_mtime_ns == mtime
    target.unlink()
    step_run_kld(cfg)
    assert target.exists()


def test_decision_logic(dry_cfg):
    from iqp.aggregate import decide

    def frame(rel_by_q, gap):
        rows, drows = [], []
        for q, rel in rel_by_q.items():
            for code in dry_cfg.indic_langs:
                drows.append({"qtype": q, "language": code, "arm": "indic", "rel_reduction_vs_en": rel,
                              "abs_reduction_vs_en": rel * 0.1, "unc_diff": 0.001})
        rows.append({"variant": "Q4_K_M__static", "language": "eng_Latn", "metric": "kld_mean", "value": 0.02})
        for code in dry_cfg.indic_langs:
            rows.append({"variant": "Q4_K_M__static", "language": code, "metric": "kld_mean", "value": 0.02 * gap})
        return pd.DataFrame(rows), pd.DataFrame(drows)

    df, de = frame({"Q4_K_M": 0.15, "Q3_K_M": 0.12, "Q2_K": 0.02, "IQ2_M": 0.2}, 2.5)
    assert decide(dry_cfg, df, de)["verdict"] == "strong signal"
    df, de = frame({"Q4_K_M": 0.15, "Q3_K_M": 0.12}, 1.3)
    assert decide(dry_cfg, df, de)["verdict"] == "partial signal"
    df, de = frame({"Q4_K_M": 0.01, "Q3_K_M": 0.02}, 1.3)
    assert decide(dry_cfg, df, de)["verdict"] == "weak signal"
    assert decide(dry_cfg, pd.DataFrame(columns=["variant", "language", "metric", "value"]),
                  pd.DataFrame())["verdict"] == "insufficient data"


def test_cli_plan_and_status(dry_cfg, capsys):
    from iqp import cli
    assert cli._plan(dry_cfg) == 0
    assert "19 variants" in capsys.readouterr().out
    assert cli._status(dry_cfg) == 0
