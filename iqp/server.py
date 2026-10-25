"""Managed llama-server process and a small HTTP client.

Endpoints and fields verified in tools/server/README.md at the pinned commit and exercised
against a real CPU build: GET /health, POST /completion (n_predict, n_probs, temperature,
cache_prompt, completion_probabilities[].top_logprobs[].{token,logprob}),
POST /v1/chat/completions (chat_template_kwargs), POST /tokenize.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from iqp.util import LOG, StepError


class LlamaServer:
    def __init__(self, cfg, model_path: Path, log_path: Path):
        self.cfg = cfg
        self.s = cfg.raw["server"]
        self.model_path = model_path
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://{self.s['host']}:{self.s['port']}"

    def command(self) -> list[str]:
        lc = self.cfg.raw["llamacpp"]
        s = self.s
        cmd = [str(self.cfg.bin("llama-server")), "-m", str(self.model_path), "--host", s["host"],
               "--port", str(s["port"]), "-np", str(s["parallel"]),
               "-c", str(int(s["parallel"]) * int(s["ctx_per_slot"])), "-ngl", str(lc["n_gpu_layers"]),
               "--jinja", "--no-webui", "-s", str(s["seed"])]
        if int(lc.get("threads") or 0) > 0:
            cmd += ["-t", str(lc["threads"])]
        return cmd + [str(a) for a in s.get("extra_args", [])]

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._healthy():
            raise StepError(f"something is already listening on {self.base_url}; stop it or change server.port")
        self._fh = open(self.log_path, "w", encoding="utf-8")
        cmd = self.command()
        self._fh.write("$ " + " ".join(cmd) + "\n\n")
        self._fh.flush()
        LOG.info("starting llama-server for %s", self.model_path.name)
        self.proc = subprocess.Popen(cmd, stdout=self._fh, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < float(self.s["startup_timeout_s"]):
            if self.proc.poll() is not None:
                raise StepError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            if self._healthy():
                LOG.info("llama-server ready after %.1fs", time.time() - t0)
                return self
            time.sleep(0.5)
        self.__exit__(None, None, None)
        raise StepError(f"llama-server did not become healthy within {self.s['startup_timeout_s']}s")

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if getattr(self, "_fh", None):
            self._fh.close()

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=2) as r:
                return r.status == 200 and json.loads(r.read() or b"{}").get("status") == "ok"
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            return False

    def post(self, path: str, body: dict, retries: int = 3) -> dict:
        data = json.dumps(body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(self.base_url + path, data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=float(self.s["request_timeout_s"])) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:500]
                if e.code in (400, 404):
                    raise StepError(f"llama-server {path} returned {e.code}: {msg}")
                last = RuntimeError(f"HTTP {e.code}: {msg}")
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last = e
            time.sleep(1.0 + attempt)
        raise StepError(f"llama-server {path} failed after {retries} attempts: {last}")
