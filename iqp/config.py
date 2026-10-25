"""Configuration loading, variant enumeration, and canonical file layout."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

STATIC_ARM = "static"
BASE_ARM = "base"

DRY_RUN_OVERRIDES = {
    "calibration": {"token_budget": 2048, "ctx": 256, "candidate_chars_per_lang": 30000},
    "kld": {"ctx": 128, "batch": 128, "max_chunks": 4},
    "milu": {"sample_cap": 24, "k_shot": 2},
    "mt": {"n_sentences": 12},
    "server": {"client_threads": 2},
}


@dataclass(frozen=True)
class Variant:
    qtype: str   # BF16 / F16 for the base, otherwise a llama-quantize type name
    arm: str     # base | static | <calibration arm name>

    @property
    def name(self) -> str:
        return f"{self.qtype}__{self.arm}"

    @property
    def is_base(self) -> bool:
        return self.arm == BASE_ARM

    @classmethod
    def parse(cls, name: str) -> "Variant":
        if "__" not in name:
            raise ValueError(f"variant name must look like QTYPE__ARM, got {name!r}")
        q, a = name.split("__", 1)
        return cls(q, a)


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


class Config:
    def __init__(self, raw: dict, path: Path | None = None, dry_run: bool = False):
        self.raw = raw
        self.path = path
        self.dry_run = dry_run
        self._validate()

    # ---- loading -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, dry_run: bool = False, overrides: dict | None = None) -> "Config":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if overrides:
            raw = _deep_update(copy.deepcopy(raw), overrides)
        if dry_run:
            # Dry runs never touch real outputs and use tiny budgets so they finish in seconds.
            for key in ("work_dir", "results_dir", "logs_dir"):
                raw["paths"][key] = str(Path(raw["paths"][key] + "_dryrun"))
            raw = _deep_update(raw, DRY_RUN_OVERRIDES)
        return cls(raw, path, dry_run)

    def _validate(self) -> None:
        r = self.raw
        for section in ("paths", "model", "llamacpp", "languages", "quant", "calibration", "eval", "kld"):
            if section not in r:
                raise ValueError(f"config is missing section '{section}'")
        reg = r["languages"]["registry"]
        for code in r["languages"]["enabled"]:
            if code not in reg:
                raise ValueError(f"enabled language {code} has no entry in languages.registry")
        for arm, spec in r["calibration"]["arms"].items():
            if arm in (STATIC_ARM, BASE_ARM) or "__" in arm:
                raise ValueError(f"calibration arm name {arm!r} is reserved or contains '__'")
            for code in spec["langs"]:
                if code not in reg:
                    raise ValueError(f"calibration arm {arm} uses unknown language {code}")
        for q in r["quant"]["calibrated_types"]:
            if q not in r["quant"]["ladder"]:
                raise ValueError(f"calibrated type {q} is not in quant.ladder")
        if r["calibration"]["token_budget"] % r["calibration"]["ctx"] != 0:
            raise ValueError("calibration.token_budget must be a multiple of calibration.ctx")

    # ---- paths -----------------------------------------------------------
    def _p(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def work_dir(self) -> Path:
        return self._p(self.raw["paths"]["work_dir"])

    @property
    def results_dir(self) -> Path:
        return self._p(self.raw["paths"]["results_dir"])

    @property
    def logs_dir(self) -> Path:
        return self._p(self.raw["paths"]["logs_dir"])

    @property
    def llamacpp_dir(self) -> Path:
        if self.dry_run:
            return self.work_dir / "llama.cpp"
        return self._p(self.raw["llamacpp"]["dir"])

    @property
    def convert_venv(self) -> Path:
        if self.dry_run:
            return self.work_dir / "venv-convert"
        return self._p(self.raw["llamacpp"]["convert_venv"])

    def bin(self, name: str) -> Path:
        return self.llamacpp_dir / "build" / "bin" / name

    @property
    def model_name(self) -> str:
        return self.raw["model"]["short_name"]

    @property
    def hf_dir(self) -> Path:
        return self.work_dir / "hf" / self.model_name

    @property
    def gguf_dir(self) -> Path:
        return self.work_dir / "gguf" / self.model_name

    @property
    def data_dir(self) -> Path:
        # Per model: calibration budgets and MILU prompt truncation use the model's tokenizer.
        return self.work_dir / "data" / self.model_name

    @property
    def imatrix_dir(self) -> Path:
        return self.work_dir / "imatrix" / self.model_name

    @property
    def kld_base_dir(self) -> Path:
        return self.work_dir / "kld_base" / self.model_name

    @property
    def raw_results_dir(self) -> Path:
        return self.results_dir / "raw" / self.model_name

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    def gguf_path(self, v: Variant) -> Path:
        return self.gguf_dir / f"{self.model_name}-{v.name}.gguf"

    def imatrix_path(self, arm: str) -> Path:
        return self.imatrix_dir / f"{arm}.imatrix.gguf"

    def calib_text_path(self, arm: str) -> Path:
        return self.data_dir / "calib" / f"{arm}.txt"

    def eval_kld_path(self, code: str) -> Path:
        return self.data_dir / "eval" / "kld" / f"{code}.txt"

    def eval_mt_path(self) -> Path:
        return self.data_dir / "eval" / "mt.jsonl"

    def milu_path(self, code: str, split: str) -> Path:
        return self.data_dir / "milu" / code / f"{split}.jsonl"

    # ---- languages ---------------------------------------------------------
    @property
    def langs(self) -> list[str]:
        return list(self.raw["languages"]["enabled"])

    def lang(self, code: str) -> dict:
        return self.raw["languages"]["registry"][code]

    def lang_name(self, code: str) -> str:
        return self.lang(code)["name"]

    @property
    def indic_langs(self) -> list[str]:
        return [c for c in self.langs if self.lang(c).get("role") == "indic"]

    @property
    def control_langs(self) -> list[str]:
        return [c for c in self.langs if self.lang(c).get("role") == "control"]

    # ---- variants ----------------------------------------------------------
    def base_dtype(self) -> str:
        """BF16 or F16. Resolves "auto" from the HF config when it is available."""
        want = str(self.raw["model"].get("base_dtype", "auto")).lower()
        if want in ("bf16", "f16"):
            return want.upper()
        cfg_path = self.hf_dir / "config.json"
        if cfg_path.exists():
            import json
            c = json.loads(cfg_path.read_text())
            dt = c.get("torch_dtype") or c.get("dtype") or (c.get("text_config") or {}).get("torch_dtype") \
                or (c.get("text_config") or {}).get("dtype")
            if dt and "bfloat16" in str(dt):
                return "BF16"
            return "F16"
        return "BF16"

    def base_variant(self) -> Variant:
        # Resolve once and persist so later steps agree even if hf weights are deleted.
        marker = self.gguf_dir / "BASE_DTYPE"
        if marker.exists():
            return Variant(marker.read_text().strip(), BASE_ARM)
        return Variant(self.base_dtype(), BASE_ARM)

    @property
    def calib_arms(self) -> list[str]:
        return list(self.raw["calibration"]["arms"].keys())

    def quant_variants(self) -> list[Variant]:
        q = self.raw["quant"]
        need_im = set(q["requires_imatrix"])
        out: list[Variant] = []
        for qt in q["ladder"]:
            if qt not in need_im:
                out.append(Variant(qt, STATIC_ARM))
            if qt in q["calibrated_types"]:
                for arm in self.calib_arms:
                    out.append(Variant(qt, arm))
        return out

    def all_variants(self) -> list[Variant]:
        return [self.base_variant()] + self.quant_variants()

    def select_variants(self, spec) -> list[Variant]:
        """Resolve a variants spec ("all" or list of names) to Variant objects."""
        allv = self.all_variants()
        if spec in (None, "all"):
            return allv
        names = {v.name: v for v in allv}
        out = []
        for s in spec:
            if s in ("base", "BASE"):
                out.append(self.base_variant())
            elif s in names:
                out.append(names[s])
            else:
                raise ValueError(f"unknown variant {s!r}; known: {sorted(names)}")
        return out
