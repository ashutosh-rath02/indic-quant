"""Converter venv creation falls back to virtualenv when stdlib venv cannot bootstrap pip."""
import subprocess
import sys
from pathlib import Path

from iqp import setup_steps
from iqp.util import StepError


def _fake_run(calls, venv_ok):
    def run(cmd, log_path, **kw):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if cmd[1:3] == ["-m", "venv"]:
            if not venv_ok:
                raise StepError("command failed with exit code 1: python -m venv (ensurepip)")
            (Path(cmd[3]) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(cmd[3]) / "bin" / "python").write_text("")
        if cmd[1:3] == ["-m", "virtualenv"]:
            (Path(cmd[3]) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(cmd[3]) / "bin" / "python").write_text("")
        return ""
    return run


def test_stdlib_venv_used_when_it_works(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_steps, "run_cmd", _fake_run(calls, venv_ok=True))
    setup_steps.create_venv(tmp_path / "v", tmp_path / "logs")
    assert [c[1:3] for c in calls] == [["-m", "venv"]]


def test_falls_back_to_virtualenv(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_steps, "run_cmd", _fake_run(calls, venv_ok=False))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    setup_steps.create_venv(tmp_path / "v", tmp_path / "logs")
    steps = [c[1:4] for c in calls]
    assert steps[0][:2] == ["-m", "venv"]
    assert ["-m", "pip", "install"] in steps
    assert steps[-1][:2] == ["-m", "virtualenv"]
    assert (tmp_path / "v" / "bin" / "python").exists()
    assert all(c[0] == sys.executable for c in calls)
