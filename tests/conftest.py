"""Fixture helpers: generate a small dataset, build state and a result set."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_cli(*args, cwd=None, check=True):
    proc = subprocess.run([PY, "-m", "ringdetect.cli", *args],
                          cwd=cwd or ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


class Fixture:
    """A generated dataset plus optional state/result directories."""

    def __init__(self, root: Path, *, accounts=800, days=8, rings=3,
                 households=12, seed=5, gzip=False):
        self.root = Path(root)
        self.data = self.root / "data"
        self.state = self.root / "state" / "state.db"
        self.out = self.root / "results"
        from ringdetect.generate import GenConfig, generate
        from datetime import date

        cfg = GenConfig(accounts=accounts, days=days, rings=rings,
                        households=households, seed=seed,
                        start_day=date(2026, 9, 1))
        self.manifest = generate(cfg, self.data)
        self.gzip = gzip

    @property
    def files(self):
        if self.gzip:
            return sorted(str(p) for p in (self.data / "daily").glob("*.csv.gz"))
        return sorted(str(p) for p in (self.data / "daily").glob("*.csv"))

    def import_all(self, *extra):
        patterns = [str(self.data / "daily" / ("*.csv.gz" if self.gzip else "*.csv"))]
        run_cli("import", "--state", str(self.state), "--quiet", *extra, *patterns)
        return self

    def build(self, *extra):
        run_cli("build", "--state", str(self.state), "--out", str(self.out),
                "--quiet", *extra)
        return self

    def summary(self):
        return json.loads((self.out / "summary.json").read_text())

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)
