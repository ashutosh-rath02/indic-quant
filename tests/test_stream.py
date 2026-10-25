"""Streaming variant mode, time budget, restore/bundle, variant filters, disk checks, Kaggle configs."""

import json
from collections import namedtuple

import pytest

from iqp.config import REPO_ROOT, Config, expand_env
from iqp.util import StepError, read_json

PRE_STREAM = ["setup", "download", "convert", "prepare_data", "build_imatrix"]


def kaggle_cfg(tmp_path, name="kaggle.yaml", sub="s1", **extra):
    root = tmp_path / sub
    overrides = {"paths": {"work_dir": str(root / "scratch"), "output_root": str(root / "out"),
                           "state_dir": str(root / "out" / "state"), "results_dir": str(root / "out" / "results"),
                           "logs_dir": str(root / "out" / "logs"), "restore_from": ""},
                 # per-variant aggregate and bundle are slow in a 19 variant loop; one test enables them
                 "stream": {"aggregate_each_variant": False, "bundle_each_variant": False}}
    for k, v in extra.items():
        overrides.setdefault(k, {}).update(v)
    return Config.load(REPO_ROOT / "configs" / name, dry_run=True, overrides=overrides)


def run_steps(cfg, names):
    from iqp.cli import _steps
    steps = _steps()
    for n in names:
        steps[n](cfg, force=False)


@pytest.fixture
def prepared(tmp_path):
    cfg = kaggle_cfg(tmp_path)
    run_steps(cfg, PRE_STREAM)
    return cfg


def kld_done(cfg, v):
    return all((cfg.raw_results_dir / "kld" / v.name / f"{c}.json").exists() for c in cfg.langs)


# ------------------------------------------------------------------ config

@pytest.mark.parametrize("name,model,vocab_note", [("kaggle.yaml", "gemma-3-4b-it", "262144"),
                                                     ("kaggle-qwen3-4b.yaml", "qwen3-4b", "151936")])
def test_kaggle_configs_load(name, model, vocab_note, monkeypatch):
    monkeypatch.delenv("IQP_WORK_DIR", raising=False)
    monkeypatch.delenv("IQP_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("IQP_CUDA_ARCHS", raising=False)
    cfg = Config.load(REPO_ROOT / "configs" / name)
    assert cfg.model_name == model
    assert cfg.raw["pipeline"] == "stream"
    assert cfg.base_dtype() == "F16" and cfg.base_variant().qtype == "F16"
    assert str(cfg.work_dir) == "/tmp/iqp-work"
    for d in (cfg.results_dir, cfg.logs_dir, cfg.state_dir):
        assert str(d).startswith("/kaggle/working/")
    assert str(cfg.imatrix_dir).startswith("/kaggle/working/") and str(cfg.data_dir).startswith("/kaggle/working/")
    assert str(cfg.kld_base_dir).startswith("/tmp/iqp-work") and str(cfg.gguf_dir).startswith("/tmp/iqp-work")
    assert cfg.raw["llamacpp"]["cmake_extra_args"] == ["-DCMAKE_CUDA_ARCHITECTURES=60;75"]
    assert cfg.raw["kld"]["max_chunks"] == 12 and cfg.raw["kld"]["delete_base_logits_when_done"]
    assert cfg.raw["milu"]["sample_cap"] == 200 and cfg.raw["mt"]["n_sentences"] == 100
    assert cfg.raw["model"]["delete_hf_weights_after_convert"]
    assert vocab_note in (REPO_ROOT / "configs" / name).read_text()
    from iqp import evals
    milu = [v.name for v in evals.milu_variants(cfg)]
    assert milu == [v.name for v in evals.mt_variants(cfg)]
    assert milu[0] == "F16__base" and "Q8_0__static" in milu and len(milu) == 10
    assert {"Q4_K_M__indic", "Q2_K__indic_no_tam"} <= set(milu) and "Q3_K_M__en" not in milu
    assert len(cfg.run_variants()) == 19


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("IQP_WORK_DIR", "/scratch/x")
    monkeypatch.setenv("IQP_CUDA_ARCHS", "75")
    cfg = Config.load(REPO_ROOT / "configs" / "kaggle.yaml")
    assert str(cfg.work_dir) == "/scratch/x" and str(cfg.llamacpp_dir) == "/scratch/x/llama.cpp"
    assert cfg.raw["llamacpp"]["cmake_extra_args"] == ["-DCMAKE_CUDA_ARCHITECTURES=75"]
    assert expand_env("${NOPE_UNSET_VAR:-d}/a") == "d/a"
    with pytest.raises(ValueError):
        expand_env("${NOPE_UNSET_VAR}")


def test_cuda13_drops_pascal():
    from iqp.setup_steps import adjust_cuda_archs
    a = ["-DCMAKE_CUDA_ARCHITECTURES=60;75", "-DX=1"]
    assert adjust_cuda_archs(a, (12, 4)) == a
    assert adjust_cuda_archs(a, (13, 0)) == ["-DCMAKE_CUDA_ARCHITECTURES=75", "-DX=1"]
    assert adjust_cuda_archs(a, None) == a


# ------------------------------------------------------------------ variant selection

def test_variant_filters(dry_cfg):
    cfg = dry_cfg
    assert len(cfg.run_variants()) == 19
    cfg.runtime["variants"] = "Q4_K_M__*,base"
    names = [v.name for v in cfg.run_variants()]
    assert names == ["BF16__base", "Q4_K_M__static", "Q4_K_M__en", "Q4_K_M__indic", "Q4_K_M__indic_no_tam"]
    cfg.runtime["variants"] = "*__indic"
    assert [v.name for v in cfg.run_quant_variants()] == ["Q4_K_M__indic", "Q3_K_M__indic", "Q2_K__indic",
                                                          "IQ2_M__indic"]
    cfg.runtime["variants"] = "Q9_K__static"
    with pytest.raises(ValueError):
        cfg.run_variants()
    cfg.runtime["variants"] = None
    cfg.raw["variants"] = ["Q8_0__static", "Q2_K__*"]
    assert [v.name for v in cfg.run_variants()] == ["Q8_0__static", "Q2_K__static", "Q2_K__en", "Q2_K__indic",
                                                    "Q2_K__indic_no_tam"]
    # per-task lists stay inside the top-level selection
    assert [v.name for v in cfg.select_variants(["Q8_0__static", "Q4_K_M__*"])] == ["Q8_0__static"]
    # but the delete-logits rule uses the configured set, not the CLI filter
    cfg.runtime["variants"] = "Q8_0__static"
    assert len(cfg.configured_quant_variants()) == 5


def test_cli_variants_flag(tmp_path, capsys):
    from iqp import cli
    rc = cli.main(["plan", "--config", str(REPO_ROOT / "configs" / "kaggle.yaml"), "--variants", "Q2_K__*"])
    assert rc == 0 and "4 variants" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["plan", "--config", str(REPO_ROOT / "configs" / "kaggle.yaml"), "--variants", "bogus"])


def test_classic_quantize_respects_filter(dry_cfg):
    run_steps(dry_cfg, ["setup", "download", "convert", "prepare_data", "build_imatrix"])
    dry_cfg.runtime["variants"] = "Q2_K__en"
    from iqp.quantize import step_quantize
    step_quantize(dry_cfg)
    made = sorted(p.name for p in dry_cfg.gguf_dir.glob("*.gguf"))
    assert made == ["gemma-3-4b-it-BF16__base.gguf", "gemma-3-4b-it-Q2_K__en.gguf"]


# ------------------------------------------------------------------ streaming

def test_stream_processes_and_deletes(prepared):
    cfg = prepared
    cfg.raw["stream"]["aggregate_each_variant"] = True
    cfg.raw["stream"]["bundle_each_variant"] = True
    from iqp.stream import load_timings, status_path, step_stream
    hf_weights = list(cfg.hf_dir.glob("*.safetensors"))
    assert hf_weights == []  # dry snapshot has none; deletion path is covered by test_convert_deletes_hf
    step_stream(cfg)
    quant = [v for v in cfg.run_variants() if not v.is_base]
    for v in quant:
        assert kld_done(cfg, v), v.name
        assert not cfg.gguf_path(v).exists(), f"{v.name} GGUF was not deleted"
        assert cfg.quant_meta_path(v).exists()
    assert cfg.gguf_path(cfg.base_variant()).exists()
    from iqp import evals
    for v in evals.milu_variants(cfg):
        assert all((cfg.raw_results_dir / "milu" / v.name / f"{c}.json").exists() for c in cfg.langs)
    q3 = cfg.raw_results_dir / "milu" / "Q3_K_M__en"
    assert not q3.exists()  # not selected for MILU
    assert not list(cfg.kld_base_dir.glob("*.logits")), "base logits should be deleted when all variants are done"
    st = read_json(status_path(cfg))
    assert st["complete"] and st["remaining"] == []
    t = load_timings(cfg)
    assert set(t["variants"]) == {v.name for v in cfg.run_variants()}
    # priority: the Q4_K_M arms run first
    assert st["done_this_session"][:4] == ["Q4_K_M__static", "Q4_K_M__en", "Q4_K_M__indic", "Q4_K_M__indic_no_tam"]
    assert (cfg.results_dir / "results_tidy.csv").exists()
    from iqp.session import BUNDLE_NAME, MARKER, output_root
    assert (output_root(cfg) / MARKER).exists() and (output_root(cfg) / BUNDLE_NAME).exists()


def test_stream_resume_skips_done(prepared, monkeypatch):
    cfg = prepared
    from iqp import stream
    calls = []
    real = stream.process_variant

    def spy(cfg_, v, tasks, plan, force=False):
        calls.append(v.name)
        return real(cfg_, v, tasks, plan, force)

    monkeypatch.setattr(stream, "process_variant", spy)
    cfg.runtime["variants"] = "Q4_K_M__*"
    stream.step_stream(cfg)
    assert calls == ["Q4_K_M__static", "Q4_K_M__en", "Q4_K_M__indic", "Q4_K_M__indic_no_tam"]
    # logits are kept: other configured variants still need them
    assert len(list(cfg.kld_base_dir.glob("*.logits"))) == len(cfg.langs)
    st = read_json(stream.status_path(cfg))
    assert not st["complete"] and "Q8_0__static" in st["remaining"]
    calls.clear()
    cfg.runtime["variants"] = None
    # an interrupted variant: one result missing, its GGUF and a partial file left behind
    res = cfg.raw_results_dir / "kld" / "Q4_K_M__en" / "tam_Taml.json"
    res.unlink()
    (cfg.gguf_dir / "junk.gguf.part").write_text("x")
    stream.step_stream(cfg)
    assert calls[0] == "Q4_K_M__en"
    assert "Q4_K_M__static" not in calls and len(calls) == 19 - 4 + 1
    assert res.exists() and not (cfg.gguf_dir / "junk.gguf.part").exists()
    calls.clear()
    stream.step_stream(cfg)
    assert calls == []


def test_stream_time_budget_stops_cleanly(prepared, monkeypatch, caplog):
    cfg = prepared
    from iqp import stream
    from iqp.util import atomic_write_json
    # Measured in an "earlier session": every variant takes 30 minutes.
    atomic_write_json(stream.timings_path(cfg), {"variants": {
        "Q4_K_M__static": {"qtype": "Q4_K_M", "quantize_s": 600, "kld_s": 600, "kld_langs": 6, "milu_s": 300,
                           "mt_s": 300}}})
    cfg.runtime["max_hours"] = 0.25
    stream.step_stream(cfg)
    st = read_json(stream.status_path(cfg))
    assert st["complete"] is False and st["reason"] == "time budget"
    assert st["stopped_before"] == "Q4_K_M__static" and st["done_this_session"] == []
    assert len(st["remaining"]) == 19
    assert "Resume next session" in caplog.text
    assert not list(cfg.gguf_dir.glob("*Q4_K_M__static*"))
    # a large budget runs everything
    cfg.runtime["max_hours"] = 1000
    stream.step_stream(cfg)
    assert read_json(stream.status_path(cfg))["complete"]


def test_stream_budget_uses_measured_times(prepared, monkeypatch):
    cfg = prepared
    from iqp import stream
    clock = {"t": 1000.0}
    monkeypatch.setattr(stream.time, "time", lambda: clock["t"])
    real = stream.process_variant

    def slow(cfg_, v, tasks, plan, force=False):
        rec = real(cfg_, v, tasks, plan, force)
        clock["t"] += 600  # each variant "takes" 10 minutes
        rec.update({"quantize_s": 200, "kld_s": 400 if tasks["kld_langs"] else 0,
                    "kld_langs": len(tasks["kld_langs"]), "milu_s": 0, "mt_s": 0, "total_s": 600})
        return rec

    monkeypatch.setattr(stream, "process_variant", slow)
    monkeypatch.delenv("IQP_SESSION_START", raising=False)
    cfg.raw["stream"]["fallback_minutes"] = {}
    cfg.raw["stream"]["safety_factor"] = 1.0
    cfg.raw["milu"]["enabled"] = False
    cfg.raw["mt"]["enabled"] = False
    cfg.runtime["max_hours"] = 0.5  # 30 min: three 10 minute variants fit, the fourth would end at 40
    stream.step_stream(cfg)
    st = read_json(stream.status_path(cfg))
    assert st["reason"] == "time budget" and len(st["done_this_session"]) == 3


def test_estimate_seconds(dry_cfg):
    from iqp.config import Variant
    from iqp.stream import estimate_seconds
    tasks = {"kld_langs": ["a", "b"], "milu": True, "mt": False}
    v = Variant("Q4_K_M", "en")
    assert estimate_seconds(dry_cfg, v, tasks, {"variants": {}}) is None
    dry_cfg.raw["stream"] = {"fallback_minutes": {"quantize": 1, "kld_per_lang": 1, "milu": 2}}
    assert estimate_seconds(dry_cfg, v, tasks, {"variants": {}}) == 60 + 120 + 120
    t = {"variants": {"x": {"qtype": "Q8_0", "quantize_s": 10, "kld_s": 60, "kld_langs": 6, "milu_s": 30}}}
    assert estimate_seconds(dry_cfg, v, tasks, t) == 10 + 2 * 10 + 30


def test_stream_requires_prepared_inputs(dry_cfg):
    from iqp.stream import step_stream
    with pytest.raises(StepError):
        step_stream(dry_cfg)


# ------------------------------------------------------------------ disk checks

def test_disk_check(tmp_path, monkeypatch):
    from iqp import diskcheck
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(diskcheck.shutil, "disk_usage", lambda p: Usage(100 * 2**30, 99 * 2**30, 1 * 2**30))
    diskcheck.require_free(tmp_path / "not" / "yet", 100 * 2**20, "small thing")
    with pytest.raises(StepError, match="not enough free disk for base logits"):
        diskcheck.require_free(tmp_path / "not" / "yet", 5 * 2**30, "base logits")


def test_stream_fails_early_without_disk(prepared, monkeypatch):
    cfg = prepared
    from iqp import diskcheck, stream
    (cfg.hf_dir / "config.json").write_text(json.dumps({"vocab_size": 262144}))
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(diskcheck.shutil, "disk_usage", lambda p: Usage(20 * 2**30, 19 * 2**30, 2**30))
    with pytest.raises(StepError, match="not enough free disk"):
        stream.step_stream(cfg)
    assert not list(cfg.kld_base_dir.glob("*.logits"))


def test_convert_deletes_hf_weights(tmp_path):
    cfg = kaggle_cfg(tmp_path)
    run_steps(cfg, ["setup", "download"])
    w = cfg.hf_dir / "model-00001-of-00002.safetensors"
    w.write_bytes(b"\0" * 1024)
    (cfg.hf_dir / "tokenizer.json").write_text("{}")
    run_steps(cfg, ["convert"])
    assert not w.exists() and (cfg.hf_dir / "tokenizer.json").exists() and (cfg.hf_dir / "config.json").exists()
    assert cfg.gguf_path(cfg.base_variant()).exists()
    assert (cfg.meta_dir / "BASE_DTYPE").read_text() == "F16"


# ------------------------------------------------------------------ restore / bundle

def test_restore_resumes_second_session(tmp_path, monkeypatch):
    from iqp import stream
    from iqp.session import step_bundle, step_restore
    s1 = kaggle_cfg(tmp_path, sub="s1")
    run_steps(s1, PRE_STREAM)
    s1.runtime["variants"] = "Q4_K_M__*,Q2_K__static"
    stream.step_stream(s1)
    step_bundle(s1)
    out1 = s1.state_dir.parent

    # Session 2: fresh disk; session 1's output attached read-only under an input directory.
    inp = tmp_path / "kaggle_input" / "iqp-session-1"
    import shutil
    shutil.copytree(out1, inp / "iqp")
    for p in inp.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    s2 = kaggle_cfg(tmp_path, sub="s2", paths={"restore_from": str(tmp_path / "kaggle_input")})
    step_restore(s2)
    assert s2.imatrix_path("indic").exists() and s2.eval_kld_path("hin_Deva").exists()
    assert (s2.kld_meta_dir / "plan.json").exists()
    assert (s2.raw_results_dir / "kld" / "Q4_K_M__en" / "hin_Deva.json").exists()
    assert not s2.gguf_dir.exists()  # big files are not carried over
    # imatrix and prepare_data must not recompute; only the base model is rebuilt
    run_steps(s2, ["setup", "download", "convert", "prepare_data", "build_imatrix"])
    calls = []
    real = stream.process_variant
    monkeypatch.setattr(stream, "process_variant",
                        lambda c, v, t, p, force=False: calls.append(v.name) or real(c, v, t, p, force))
    stream.step_stream(s2)
    assert set(calls).isdisjoint({"Q4_K_M__static", "Q4_K_M__en", "Q4_K_M__indic", "Q4_K_M__indic_no_tam",
                                  "Q2_K__static"})
    assert len(calls) == 19 - 5
    assert read_json(stream.status_path(s2))["complete"]


def test_restore_from_bundle_file(tmp_path):
    from iqp.session import BUNDLE_NAME, output_root, step_bundle, step_restore
    s1 = kaggle_cfg(tmp_path, sub="s1")
    run_steps(s1, ["setup", "download", "convert", "prepare_data"])
    step_bundle(s1)
    bundle = output_root(s1) / BUNDLE_NAME
    only = tmp_path / "dl"
    only.mkdir()
    (only / BUNDLE_NAME).write_bytes(bundle.read_bytes())
    s2 = kaggle_cfg(tmp_path, sub="s2")
    s2.runtime["restore_from"] = str(only)
    step_restore(s2)
    assert s2.eval_kld_path("eng_Latn").exists()


def test_restore_missing_source(tmp_path):
    from iqp.session import step_restore
    s = kaggle_cfg(tmp_path, paths={"restore_from": str(tmp_path / "nothing"), "restore_required": False})
    step_restore(s)  # first session: nothing to restore is fine
    s.runtime["restore_from"] = str(tmp_path / "nothing")
    with pytest.raises(StepError, match="does not exist"):
        step_restore(s)


def test_restore_keeps_newer_local_files(tmp_path):
    from iqp.session import copy_tree_no_clobber
    src, dst = tmp_path / "a", tmp_path / "b"
    (src / "x").mkdir(parents=True)
    (dst / "x").mkdir(parents=True)
    (src / "x" / "f.json").write_text("old")
    (src / "x" / "g.json.tmp").write_text("partial")
    import os
    import time
    os.utime(src / "x" / "f.json", (time.time() - 100, time.time() - 100))
    (dst / "x" / "f.json").write_text("new")
    c, s = copy_tree_no_clobber(src, dst)
    assert (dst / "x" / "f.json").read_text() == "new" and (c, s) == (0, 1)
    assert not (dst / "x" / "g.json.tmp").exists()


def test_stream_all_pipeline_dry(tmp_path):
    from iqp import cli
    cfg = kaggle_cfg(tmp_path)
    assert cli.pipeline_for(cfg) == cli.STREAM_PIPELINE
    run_steps(cfg, [s for s in cli.STREAM_PIPELINE if s != "validate_data"])
    assert (cfg.results_dir / "REPORT.md").exists()
    assert read_json(cfg.results_dir / "stream_status.json")["complete"]


def test_kaggle_notebook_is_valid():
    nb_path = REPO_ROOT / "kaggle" / "indic_quant_pilot.ipynb"
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4 and nb["cells"]
    try:
        import nbformat
        nbformat.validate(nbformat.reads(nb_path.read_text(), as_version=4))
    except ImportError:
        pass
    import ast
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "UserSecretsClient().get_secret(\"HF_TOKEN\")" in src and "--max-hours" in src
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(ln for ln in "".join(c["source"]).splitlines() if not ln.strip().startswith("!")))
    for step in ("restore", "setup", "download", "convert", "validate_data", "prepare_data", "build_imatrix",
                 "stream", "bundle"):
        assert f'"{step}"' in src
    assert (REPO_ROOT / "kaggle" / "run_kaggle.sh").stat().st_mode & 0o111
