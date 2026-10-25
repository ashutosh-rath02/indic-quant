"""Free space checks. Kaggle scratch and output space is not guaranteed, so steps that write
large files check first and fail early with a message that says what to change."""

from __future__ import annotations

import shutil
from pathlib import Path

from iqp.util import LOG, StepError

GIB = 2**30


def _existing_parent(path: Path) -> Path:
    p = Path(path).absolute()
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return p


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(_existing_parent(Path(path))).free


def require_free(path: Path, need_bytes: int, what: str, margin_bytes: int = GIB // 2) -> None:
    """Raise StepError unless the filesystem holding path has need_bytes plus a margin free."""
    if need_bytes <= 0:
        return
    free = free_bytes(path)
    if free < need_bytes + margin_bytes:
        raise StepError(
            f"not enough free disk for {what}: need about {need_bytes / GIB:.2f} GiB "
            f"(+{margin_bytes / GIB:.1f} GiB margin) under {_existing_parent(Path(path))}, "
            f"but only {free / GIB:.2f} GiB is free. Free space, point paths.work_dir at a larger "
            f"disk, or lower kld.max_chunks (base logits) / the number of languages.")
    LOG.info("disk ok for %s: need %.2f GiB, %.2f GiB free under %s", what, need_bytes / GIB, free / GIB,
             _existing_parent(Path(path)))


def disk_report(paths: dict[str, Path]) -> str:
    seen = {}
    for name, p in paths.items():
        root = _existing_parent(Path(p))
        u = shutil.disk_usage(root)
        seen.setdefault((u.total, u.free), []).append(f"{name}={p}")
    return "; ".join(f"{', '.join(v)}: {k[1] / GIB:.1f} of {k[0] / GIB:.1f} GiB free" for k, v in seen.items())
