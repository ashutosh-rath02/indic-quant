import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_text():
    def _read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")
    return _read


@pytest.fixture
def dry_cfg(tmp_path):
    """A dry-run Config whose outputs live in a temporary directory."""
    from iqp.config import Config

    overrides = {"paths": {"work_dir": str(tmp_path / "work"), "results_dir": str(tmp_path / "results"),
                           "logs_dir": str(tmp_path / "logs")}}
    return Config.load(ROOT / "config.yaml", dry_run=True, overrides=overrides)
