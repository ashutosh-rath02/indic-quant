"""Logging, subprocess, and file helpers shared by all steps."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

LOG = logging.getLogger("iqp")


class StepError(RuntimeError):
    """A pipeline step failed in a way the user must fix."""


def setup_logging(logs_dir: Path, step: str) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{step}.log"
    for h in list(LOG.handlers):
        if getattr(h, "_iqp_step", False):
            LOG.removeHandler(h)
            h.close()
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in LOG.handlers):
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        LOG.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    fh._iqp_step = True  # type: ignore[attr-defined]
    LOG.addHandler(fh)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    return log_path


def run_cmd(cmd: Sequence[str], log_path: Path, env: dict | None = None, cwd: Path | None = None,
            check: bool = True, stdin_text: str | None = None) -> str:
    """Run a command, tee combined stdout/stderr to log_path, return the captured text."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    printable = " ".join(shlex.quote(str(c)) for c in cmd)
    LOG.info("run: %s", printable)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"$ {printable}\n\n")
        fh.flush()
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                                env=full_env, cwd=str(cwd) if cwd else None)
        if stdin_text is not None:
            assert proc.stdin is not None
            proc.stdin.write(stdin_text.encode("utf-8"))
            proc.stdin.close()
        chunks: list[str] = []
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            chunks.append(line)
            fh.write(line)
        proc.wait()
        fh.write(f"\n[exit code {proc.returncode}, {time.time() - t0:.1f}s]\n")
    out = "".join(chunks)
    if check and proc.returncode != 0:
        tail = "".join(chunks[-25:])
        raise StepError(f"command failed with exit code {proc.returncode}: {printable}\n"
                        f"see {log_path}\n--- last lines ---\n{tail}")
    return out


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def should_skip(path: Path, force: bool) -> bool:
    if force:
        return False
    if path.exists():
        LOG.info("skip (exists): %s", path)
        return True
    return False


def hf_token(cfg) -> str | None:
    env_name = cfg.raw["model"].get("hf_token_env", "HF_TOKEN")
    tok = os.environ.get(env_name) or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return tok or None
