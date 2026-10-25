"""In-process stand-in for iqp.server.LlamaServer used by --dry-run."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from iqp.fakes.common import QTYPE_KLD, h01, toy_kld
from iqp.fakes.data import fake_answer_index
from iqp.util import read_jsonl

LETTERS = "ABCD"


class FakeLlamaServer:
    def __init__(self, cfg, model_path: Path, log_path: Path):
        self.cfg = cfg
        self.m = json.loads(Path(model_path).read_text())
        self.log_path = log_path
        self._refs = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(f"fake llama-server for {self.m}\n")
        return self

    def __exit__(self, *exc):
        return False

    def _damage(self, lang3: str) -> float:
        if self.m["arm"] == "base":
            return 0.0
        return toy_kld(self.m["qtype"], self.m["arm"], lang3)

    def _lang_of_text(self, text: str) -> str:
        for ch in text:
            cp = ord(ch)
            for lo, l3 in ((0x0900, "hin"), (0x0980, "ben"), (0x0B80, "tam"), (0x0C00, "tel"), (0x0C80, "kan"),
                           (0x0D00, "mal"), (0x0A80, "guj"), (0x0B00, "ory"), (0x0A00, "pan")):
                if lo <= cp < lo + 0x80:
                    return l3
        return "eng"

    def post(self, path: str, body: dict) -> dict:
        if path == "/completion":
            prompt = body["prompt"]
            last_q = prompt.rsplit("Question: ", 1)[-1].split("\nChoices:")[0]
            ans = fake_answer_index(last_q)
            lang3 = self._lang_of_text(last_q)
            base_acc = {"eng": 0.62, "hin": 0.50}.get(lang3, 0.42)
            acc = 0.25 + (base_acc - 0.25) * math.exp(-3.0 * self._damage(lang3))
            correct = h01("milu", last_q, self.m["qtype"], self.m["arm"]) < acc
            pred = ans if correct else (ans + 1 + int(h01("w", last_q) * 3)) % 4
            top = [{"token": " " + L, "logprob": (-0.3 if i == pred else -2.0 - i * 0.3)} for i, L in enumerate(LETTERS)]
            top.append({"token": " The", "logprob": -4.0})
            return {"content": " " + LETTERS[pred],
                    "completion_probabilities": [{"token": " " + LETTERS[pred], "logprob": -0.3, "top_logprobs": top}]}
        if path == "/v1/chat/completions":
            msg = body["messages"][-1]["content"]
            src = msg.rsplit("\n\n", 1)[-1]
            if self._refs is None:
                self._refs = {r["eng_Latn"]: r for r in read_jsonl(self.cfg.eval_mt_path())}
            target = re.search(r"into (\w+)\.", msg).group(1)
            code = next(c for c in self.cfg.langs if self.cfg.lang_name(c) == target)
            ref = self._refs[src][code]
            words = ref.split()
            dmg = self._damage(code.split("_")[0])
            keep = [w for i, w in enumerate(words) if h01("mt", src, i) > 0.25 + 0.6 * min(dmg, 1.0)]
            return {"choices": [{"message": {"role": "assistant", "content": " ".join(keep) or words[0]},
                                 "finish_reason": "stop"}]}
        raise ValueError(f"fake server: unsupported path {path}")


assert QTYPE_KLD  # keep import used
