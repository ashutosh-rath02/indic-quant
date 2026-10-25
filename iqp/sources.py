"""Loaders for Hugging Face datasets, with explicit schema validation.

Dataset layouts (checked on the Hub on 2026-10-25; the gated files marked "confirmed" were
opened with an authorized token and their columns read):

* openlanguagedata/flores_plus (gated, confirmed): files {dev,devtest}/{iso639_3}_{script}.jsonl,
  one JSON object per line with id (int, from 0, restarting in each split), text, iso_639_3,
  iso_15924, glottocode, variant, url, domain, topic, split. Only id and text are used.
* ai4bharat/MILU (gated, confirmed): {Language}/{test,validation}-00000-of-00001.parquet, columns
  question, option1..option4, target ("option1".."option4"), is_translated, language, domain,
  subject. Hindi test has about 14.8k rows.
* ai4bharat/IN22-Gen (gated, fallback only): data/train-00000-of-00001.parquet exists (the card
  maps it to split "test"). The per-language column names (sentence_{code}) are not confirmed.
* wikimedia/wikipedia: {snapshot}.{wiki}/train-*.parquet, columns id, url, title, text.
* ai4bharat/sangraha: verified/{lang3}/data-*.parquet, columns doc_id, text.

Every loader still validates the columns it needs and raises SchemaError with the observed
schema when they differ.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from iqp.util import LOG


class SchemaError(RuntimeError):
    pass


def _require_columns(name: str, columns: list[str], required: list[str]) -> None:
    missing = [c for c in required if c not in columns]
    LOG.info("schema %s: %s", name, columns)
    if missing:
        raise SchemaError(
            f"{name}: expected columns {required} but {missing} are missing.\n"
            f"Observed columns: {columns}\n"
            "Update the field names in config.yaml (eval/milu sections) to match.")


def _download(repo: str, filename: str, token: str | None, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, EntryNotFoundError

    try:
        p = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", token=token,
                            cache_dir=str(cache_dir))
    except GatedRepoError as e:
        raise SchemaError(f"{repo} is gated. Accept its terms on huggingface.co and export HF_TOKEN. ({e})")
    except RepositoryNotFoundError as e:
        raise SchemaError(f"{repo} not found or not accessible with the current token. ({e})")
    except EntryNotFoundError as e:
        raise SchemaError(f"{repo} has no file {filename}. Check the *_template fields in config.yaml. ({e})")
    return Path(p)


# --------------------------------------------------------------------------------------
# Evaluation text
# --------------------------------------------------------------------------------------

def load_flores(cfg, code: str, split: str, token: str | None) -> dict[str, str]:
    """Return {sentence_id: text} for one FLORES+ language and split."""
    e = cfg.raw["eval"]
    fname = e["flores_file_template"].format(split=split, code=code)
    path = _download(e["flores_repo"], fname, token, cfg.work_dir / "hf_cache")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SchemaError(f"FLORES+ file {fname} is empty")
    _require_columns(f"flores_plus:{fname}", list(rows[0].keys()), [e["flores_id_field"], e["flores_text_field"]])
    out = {str(r[e["flores_id_field"]]): r[e["flores_text_field"]].strip() for r in rows}
    LOG.info("flores_plus %s %s: %d sentences", split, code, len(out))
    return out


def load_in22(cfg, codes: list[str], token: str | None) -> dict[str, dict[str, str]]:
    """Return {code: {id: text}} from IN22-Gen (single split, n-way parallel)."""
    import pyarrow.parquet as pq

    e = cfg.raw["eval"]
    path = _download(e["in22_repo"], e["in22_file"], token, cfg.work_dir / "hf_cache")
    table = pq.read_table(path)
    cols = table.column_names
    need = [e["in22_column_template"].format(code=c) for c in codes]
    _require_columns("IN22-Gen", cols, need + (["id"] if "id" in cols else []))
    ids = [str(i) for i in table.column("id").to_pylist()] if "id" in cols else [str(i) for i in range(table.num_rows)]
    out = {}
    for c, col in zip(codes, need):
        out[c] = {i: (t or "").strip() for i, t in zip(ids, table.column(col).to_pylist())}
    return out


def load_local_eval(cfg, code: str, part: str) -> dict[str, str]:
    d = Path(cfg.raw["eval"]["local_dir"]) / part / f"{code}.txt"
    if not d.exists():
        raise SchemaError(f"local eval file missing: {d}")
    lines = [ln.strip() for ln in d.read_text(encoding="utf-8").splitlines()]
    return {str(i): t for i, t in enumerate(lines) if t}


# --------------------------------------------------------------------------------------
# MILU
# --------------------------------------------------------------------------------------

def load_milu(cfg, lang_name: str, split: str, token: str | None) -> list[dict]:
    import pyarrow.parquet as pq

    m = cfg.raw["milu"]
    if m.get("local_dir"):
        path = Path(m["local_dir"]) / lang_name / f"{split}.parquet"
    else:
        path = _download(m["repo"], m["file_template"].format(lang=lang_name, split=split), token,
                         cfg.work_dir / "hf_cache")
    table = pq.read_table(path)
    required = [m["question_field"], *m["option_fields"], m["target_field"]]
    _require_columns(f"MILU:{lang_name}/{split}", table.column_names, required)
    rows = table.to_pylist()
    targets = {str(r[m["target_field"]]) for r in rows[:200]}
    bad = [t for t in targets if not _target_index_ok(t, len(m["option_fields"]))]
    if bad:
        raise SchemaError(f"MILU:{lang_name}/{split}: unexpected target values {bad[:5]}; "
                          "expected 'option1'..'option4' or 1..4")
    return rows


def _target_index_ok(t: str, n: int) -> bool:
    try:
        target_to_index(t, n)
        return True
    except ValueError:
        return False


def target_to_index(t, n_options: int = 4) -> int:
    """MILU targets look like 'option2'. Also accept '2', 2 or 'B'."""
    s = str(t).strip()
    if s.lower().startswith("option"):
        s = s[6:]
    if s.isdigit():
        i = int(s) - 1
    elif len(s) == 1 and s.upper() in "ABCDEFGH":
        i = "ABCDEFGH".index(s.upper())
    else:
        raise ValueError(f"cannot interpret MILU target {t!r}")
    if not 0 <= i < n_options:
        raise ValueError(f"MILU target {t!r} out of range")
    return i


# --------------------------------------------------------------------------------------
# Calibration text
# --------------------------------------------------------------------------------------

def iter_calibration_docs(cfg, code: str, token: str | None, seed: int):
    """Yield raw documents (strings) for one language from the configured calibration source."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    c = cfg.raw["calibration"]
    lang = cfg.lang(code)
    api = HfApi(token=token)
    if c["source"] == "wikipedia":
        repo = c["wikipedia_repo"]
        prefix = f"{c['wikipedia_snapshot']}.{lang['wiki']}/"
    elif c["source"] == "sangraha":
        repo = c["sangraha_repo"]
        prefix = f"{c['sangraha_config']}/{lang['sangraha']}/"
    else:
        raise ValueError(f"unknown calibration source {c['source']}")
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(prefix)
                   and f.endswith(".parquet"))
    if not files:
        raise SchemaError(f"no parquet files under {repo}/{prefix}; check calibration config")
    rng = random.Random(f"{seed}:{code}")
    fs = HfFileSystem(token=token)
    fname = files[0] if len(files) == 1 else rng.choice(files[: min(len(files), 4)])
    LOG.info("calibration source %s: %s/%s", code, repo, fname)
    with fs.open(f"datasets/{repo}/{fname}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        _require_columns(f"{repo}:{fname}", pf.schema_arrow.names, ["text"])
        groups = list(range(pf.metadata.num_row_groups))
        rng.shuffle(groups)
        for g in groups:
            for t in pf.read_row_group(g, columns=["text"]).column(0).to_pylist():
                if t:
                    yield t
