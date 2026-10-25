"""CUDA configure hints: explicit toolkit root and libcuda, with a no-VMM fallback."""
import subprocess
from pathlib import Path

from iqp import setup_steps


def _fake_toolkit(tmp_path: Path, stub: bool) -> Path:
    root = tmp_path / "cuda"
    (root / "bin").mkdir(parents=True)
    nvcc = root / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    if stub:
        (root / "lib64" / "stubs").mkdir(parents=True)
        (root / "lib64" / "stubs" / "libcuda.so").write_text("")
    return nvcc


def test_stub_is_preferred(tmp_path):
    nvcc = _fake_toolkit(tmp_path, stub=True)
    args = setup_steps.cuda_cmake_args(nvcc)
    root = nvcc.resolve().parent.parent
    assert f"-DCUDAToolkit_ROOT={root}" in args
    assert f"-DCMAKE_CUDA_COMPILER={nvcc}" in args
    assert f"-DCUDA_cuda_driver_LIBRARY={root / 'lib64' / 'stubs' / 'libcuda.so'}" in args
    assert "-DGGML_CUDA_NO_VMM=ON" not in args


def test_no_libcuda_falls_back_to_no_vmm(tmp_path, monkeypatch):
    nvcc = _fake_toolkit(tmp_path, stub=False)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if "libcuda" in p.name else real_exists(p))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=""))
    args = setup_steps.cuda_cmake_args(nvcc)
    assert not any(a.startswith("-DCUDA_cuda_driver_LIBRARY=") for a in args)
    assert "-DGGML_CUDA_NO_VMM=ON" in args


def test_ldconfig_fallback(tmp_path, monkeypatch):
    nvcc = _fake_toolkit(tmp_path, stub=False)
    drv = tmp_path / "drv" / "libcuda.so.1"
    drv.parent.mkdir()
    drv.write_text("")
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists",
                        lambda p: real_exists(p) if p == drv else (False if "libcuda" in p.name else real_exists(p)))
    out = f"\tlibcuda.so.1 (libc6,x86-64) => {drv}\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))
    args = setup_steps.cuda_cmake_args(nvcc)
    assert f"-DCUDA_cuda_driver_LIBRARY={drv}" in args
