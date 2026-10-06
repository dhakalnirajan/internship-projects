"""Fail-fast smoke test: syntax + import check of every project module.

Run this BEFORE pytest (see .github/workflows/tests.yml). A file with a
syntax error, a bad import, or a missing dependency aborts the build here
with a one-line error instead of an opaque pytest collection traceback.

Usage:
    python -m cnn_benchmark.smoke_test
"""
import compileall
import importlib
import pkgutil
import re
import sys


_SKIP = re.compile(r"\.(git|freebuff|venv)|node_modules|__pycache__|build|dist")


def compile_all():
    """Byte-compile every .py file under the repo root, skipping dot-dirs."""
    ok = compileall.compile_file("cnn_benchmark/smoke_test.py", quiet=2, force=True)
    for root in ("nn", "cnn_benchmark", "src"):
        ok = compileall.compile_dir(root, quiet=2, force=True, rx=_SKIP) and ok
        if not ok:
            print(f"SMOKE FAIL: syntax error under {root}/", file=sys.stderr)
            return False
    return True


def import_all():
    """Import every module of the in-repo packages, catching real errors.

    Optional heavy dependencies (torch, tensorflow, jax, mlx) are tolerated
    as missing: the modules that need them must raise ImportError only for
    that reason.
    """
    import cnn_benchmark
    import nn

    optional = {"torch", "torchvision", "tensorflow", "jax", "mlx", "mlx.core"}
    failures = []
    for pkg in (cnn_benchmark, nn):
        for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
            name = m.name
            try:
                importlib.import_module(name)
            except ImportError as e:
                root = (e.msg.split("'") or [""])[1] if "'" in e.msg else ""
                if any(opt in e.msg for opt in optional):
                    continue  # missing optional dependency: fine in numpy-core job
                failures.append(f"{name}: {type(e).__name__}: {e}")
            except Exception as e:
                failures.append(f"{name}: {type(e).__name__}: {e}")
    if failures:
        print("SMOKE FAIL: module import errors:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    ok = compile_all()
    if ok:
        ok = import_all()
    if not ok:
        sys.exit(1)
    print("SMOKE OK: all modules compile and import cleanly")
