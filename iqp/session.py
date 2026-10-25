"""Steps restore and bundle: carry small artifacts from one session to the next.

On Kaggle only /kaggle/working survives a session ("Save Version"), and a later session sees
an earlier one's output only if it is attached as an input (it then appears read-only under
/kaggle/input/...). The persisted layout is

    <output_root>/IQP_OUTPUT.json     marker with the relative paths below
    <output_root>/state/              prepared data, imatrices, KLD plan, size records, timings,
                                      cached llama.cpp binaries
    <output_root>/results/            raw per-variant results, CSVs, plots, report
    <output_root>/logs/
    <output_root>/iqp-output.tar.gz   optional single-file copy of the three directories

`restore` finds the marker (or a bundle) under restore_from, copies everything into the
configured directories without overwriting newer local files, and the stream step then skips
completed variants. The F16 GGUF and base logits are large and are recomputed instead.

Optionally the bundle can also be pushed to and pulled from a private Hugging Face dataset repo
(paths.hf_state_repo); restore_from "hf" or "hf:<owner/repo>" pulls it.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from iqp.util import LOG, StepError, atomic_write_json, hf_token

MARKER = "IQP_OUTPUT.json"
BUNDLE_NAME = "iqp-output.tar.gz"
SKIP_PARTS = {"tmp", "__pycache__"}


def output_root(cfg) -> Path | None:
    v = cfg.raw["paths"].get("output_root")
    if not v:
        return None
    return cfg._p(v)  # Config.load already appends _dryrun in dry runs


def _rel(root: Path, p: Path | None) -> str | None:
    if p is None:
        return None
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def write_marker(cfg) -> Path | None:
    """Record where state, results and logs live relative to output_root."""
    root = output_root(cfg)
    if root is None:
        return None
    m = {"format": 1, "model": cfg.model_name, "hf_id": cfg.raw["model"]["hf_id"],
         "run_name": cfg.raw.get("run_name"), "updated": time.time(),
         "dirs": {"state": _rel(root, cfg.state_dir), "results": _rel(root, cfg.results_dir),
                  "logs": _rel(root, cfg.logs_dir)}}
    atomic_write_json(root / MARKER, m)
    return root / MARKER


def _find(base: Path, name: str, max_depth: int = 5) -> list[Path]:
    hits = []
    base = Path(base)
    if not base.exists():
        return hits
    for dirpath, dirnames, filenames in os.walk(base):
        depth = len(Path(dirpath).relative_to(base).parts)
        if depth >= max_depth:
            dirnames[:] = []
        if name in filenames:
            hits.append(Path(dirpath) / name)
    return sorted(hits, key=lambda p: len(p.parts))


def copy_tree_no_clobber(src: Path, dst: Path, force: bool = False) -> tuple[int, int]:
    """Copy files from src to dst. Existing destination files are kept unless force is set or
    the source copy is strictly newer. Returns (copied, skipped)."""
    copied = skipped = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS]
        rel = Path(dirpath).relative_to(src)
        for fn in filenames:
            if fn.endswith(".tmp") or fn.endswith(".part"):
                continue
            s = Path(dirpath) / fn
            d = dst / rel / fn
            if d.exists() and not force and d.stat().st_mtime >= s.stat().st_mtime:
                skipped += 1
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            # Kaggle inputs are read-only; make restored copies writable.
            d.chmod(d.stat().st_mode | 0o200)
            copied += 1
    return copied, skipped


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise StepError(f"refusing to extract {m.name!r} outside {dest}")
        if m.issym() or m.islnk():
            raise StepError(f"refusing to extract link {m.name!r} from the bundle")
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
    else:  # Python before 3.10.12; members were checked above
        tar.extractall(dest)


def _pull_hf(cfg, spec: str, tmp: Path) -> Path:
    repo = spec.split(":", 1)[1] if ":" in spec else cfg.raw["paths"].get("hf_state_repo")
    if not repo:
        raise StepError("restore_from 'hf' needs paths.hf_state_repo or 'hf:<owner/repo>'")
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename=f"{cfg.model_name}/{BUNDLE_NAME}", repo_type="dataset",
                        token=hf_token(cfg), cache_dir=str(tmp / "hf"))
    LOG.info("pulled bundle from hf dataset %s", repo)
    return Path(p)


def _matching_markers(cfg, base: Path) -> list[Path]:
    """Output roots under base for this model, oldest first (by the marker's 'updated')."""
    found = []
    for marker in _find(base, MARKER):
        try:
            m = json.loads(marker.read_text())
        except Exception:
            continue
        if m.get("model") == cfg.model_name:
            found.append((float(m.get("updated") or 0), marker.parent))
    return [p for _, p in sorted(found, key=lambda t: t[0])]


def locate_sources(cfg, spec: str, tmp: Path) -> list[Path]:
    """Output roots (directories holding IQP_OUTPUT.json for this model), oldest first.

    Several may be found when outputs of more than one earlier session are attached (for
    example /kaggle/input with session 1 and session 2 outputs); they are restored in order
    and newer files win.
    """
    if spec == "hf" or spec.startswith("hf:"):
        bundle = _pull_hf(cfg, spec, tmp)
    else:
        src = Path(spec)
        if not src.exists():
            raise StepError(f"restore_from path does not exist: {src}. On Kaggle, attach the previous "
                            "session's output with Add Input and check the path under /kaggle/input.")
        bundle = src if src.is_file() else None
        if bundle is None:
            roots = _matching_markers(cfg, src)
            if roots:
                return roots
            found = _find(src, BUNDLE_NAME)
            if not found:
                raise StepError(f"no {MARKER} for model {cfg.model_name} and no {BUNDLE_NAME} under {src}")
            bundle = found[0]
    ex = tmp / "bundle"
    ex.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        _safe_extract(tar, ex)
    roots = _matching_markers(cfg, ex)
    if roots:
        return roots
    raise StepError(f"bundle {bundle} has no {MARKER} for model {cfg.model_name}")


def restore(cfg, spec: str, force: bool = False) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="iqp-restore-", dir=str(cfg.work_dir.parent)
                                if cfg.work_dir.parent.exists() else None))
    try:
        stats: dict = {}
        for root in locate_sources(cfg, spec, tmp):
            LOG.info("restore: using previous output at %s", root)
            m = json.loads((root / MARKER).read_text())
            for key, dst in (("state", cfg.state_dir or cfg.work_dir), ("results", cfg.results_dir),
                             ("logs", cfg.logs_dir)):
                rel = m["dirs"].get(key)
                if rel is None:
                    LOG.warning("previous output did not record a %s directory; skipping it", key)
                    continue
                src = root / rel
                if not src.exists():
                    continue
                if src.resolve() == Path(dst).resolve():
                    LOG.info("restore %s: source and destination are the same directory", key)
                    continue
                c, s = copy_tree_no_clobber(src, Path(dst), force)
                agg = stats.setdefault(key, {"copied": 0, "kept_existing": 0})
                agg["copied"] += c
                agg["kept_existing"] += s
                LOG.info("restore %s: %d files copied to %s, %d existing files kept", key, c, dst, s)
        return stats
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def step_restore(cfg, force: bool = False) -> None:
    """Copy a previous session's output into place. With paths.restore_required false (the
    Kaggle default), finding nothing is normal for a first session and only logged; an
    explicit --restore-from always has to succeed."""
    spec = cfg.restore_from
    required = bool(cfg.runtime.get("restore_from")) or bool(cfg.raw["paths"].get("restore_required", True))
    if not spec:
        LOG.info("restore: no paths.restore_from or --restore-from given; nothing to do")
        return
    try:
        stats = restore(cfg, spec, force)
    except StepError as e:
        if required:
            raise
        LOG.info("restore: nothing to restore (%s). This is expected in a first session.", e)
        return
    if not stats and required:
        raise StepError(f"restore from {spec} copied nothing")
    write_marker(cfg)


def step_bundle(cfg, force: bool = False, push: bool = True) -> None:
    """Write the marker and a single tar.gz of state, results and logs (and optionally push it)."""
    root = output_root(cfg)
    marker = write_marker(cfg)
    if root is None or marker is None:
        LOG.info("bundle: paths.output_root is not set; nothing to do")
        return
    bundle = root / BUNDLE_NAME
    tmp = bundle.with_name(bundle.name + ".part")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(marker, arcname=MARKER)
        for key, d in (("state", cfg.state_dir), ("results", cfg.results_dir), ("logs", cfg.logs_dir)):
            rel = _rel(root, d)
            if d is None or rel is None or not d.exists():
                continue
            tar.add(d, arcname=rel, filter=lambda ti: None if (set(Path(ti.name).parts) & SKIP_PARTS
                                                                or ti.name.endswith((".part", ".tmp")))
                    else ti)
    os.replace(tmp, bundle)
    LOG.info("bundle: %s (%.1f MiB)", bundle, bundle.stat().st_size / 2**20)
    repo = cfg.raw["paths"].get("hf_state_repo")
    if repo and push and not cfg.dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token(cfg))
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(bundle), path_in_repo=f"{cfg.model_name}/{BUNDLE_NAME}",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"iqp state for {cfg.model_name}")
        LOG.info("bundle pushed to private hf dataset %s", repo)
