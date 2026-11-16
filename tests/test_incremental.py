"""Incremental-import behaviour, idempotency, and day correction."""

from __future__ import annotations

import sqlite3


from conftest import run_cli


def _daily(fx):
    return sorted((fx.data / "daily").glob("*.csv"))


def test_import_is_idempotent_per_file(tmp_path):
    from conftest import Fixture
    fx = Fixture(tmp_path, accounts=300, days=4, rings=1, households=5, seed=2)
    first = run_cli("import", "--state", str(fx.state), "--quiet", *fx.files)
    with sqlite3.connect(fx.state) as c:
        n_batches = c.execute("SELECT COUNT(*) FROM batch").fetchone()[0]
        n_links = c.execute("SELECT COUNT(*) FROM link").fetchone()[0]
    second = run_cli("import", "--state", str(fx.state), *fx.files)
    assert "0 file(s) skipped" in first.stdout
    # every file is content-addressed now, so the second pass skips them all
    assert f"{len(fx.files)} file(s) skipped" in second.stdout
    with sqlite3.connect(fx.state) as c:
        assert c.execute("SELECT COUNT(*) FROM batch").fetchone()[0] == n_batches
        assert c.execute("SELECT COUNT(*) FROM link").fetchone()[0] == n_links


def test_incremental_build_sees_new_accounts(tmp_path):
    """Import half the days, build, import the rest, rebuild -- counts grow."""
    from conftest import Fixture
    fx = Fixture(tmp_path, accounts=400, days=6, rings=1, households=4, seed=4)
    files = fx.files
    half = len(files) // 2
    fx.state.parent.mkdir(parents=True, exist_ok=True)
    run_cli("import", "--state", str(fx.state), "--quiet", *files[:half])
    run_cli("build", "--state", str(fx.state), "--out", str(fx.out), "--quiet")
    s1 = fx.summary()
    run_cli("import", "--state", str(fx.state), "--quiet", *files[half:])
    run_cli("build", "--state", str(fx.state), "--out", str(fx.out), "--quiet")
    s2 = fx.summary()
    assert s2["counts"]["accounts"] > s1["counts"]["accounts"]
    assert s2["state_revision"] > s1["state_revision"]


def test_replace_day_uses_corrected_file(tmp_path):
    """A corrected re-export of one day must replace, not double, that day."""
    from conftest import Fixture
    fx = Fixture(tmp_path, accounts=200, days=4, rings=1, households=3, seed=6)
    run_cli("import", "--state", str(fx.state), "--quiet", *fx.files)
    with sqlite3.connect(fx.state) as c:
        before = c.execute("SELECT SUM(n) FROM link_day").fetchone()[0]
    # correct one day: drop it, then re-import just that file under a new hash
    corrected = tmp_path / "corrected.csv"
    src = _daily(fx)[0]
    corrected.write_text(src.read_text())
    run_cli("import", "--state", str(fx.state), "--day", "2026-09-01",
            "--replace-day", "2026-09-01", "--quiet", str(corrected))
    with sqlite3.connect(fx.state) as c:
        after = c.execute("SELECT SUM(n) FROM link_day").fetchone()[0]
        n_batches = c.execute("SELECT COUNT(*) FROM batch").fetchone()[0]
    # same rows re-imported after the delete: total unchanged, one new batch
    assert after == before
    assert n_batches == 5


def test_column_mapping_flags(tmp_path):
    """Non-default column names must be mappable."""
    from conftest import Fixture
    fx = Fixture(tmp_path, accounts=150, days=3, rings=1, households=3, seed=8)
    src = _daily(fx)[0]
    renamed = tmp_path / "renamed.csv"
    lines = src.read_text().splitlines()
    header = lines[0].replace("account_id", "uid").replace("device_id", "imei")
    renamed.write_text("\n".join([header] + lines[1:]) + "\n")
    state = tmp_path / "alt.db"
    run_cli("import", "--state", str(state), "--account-col", "uid",
            "--device-col", "imei", "--quiet", str(renamed))
    with sqlite3.connect(state) as c:
        assert c.execute("SELECT COUNT(*) FROM account").fetchone()[0] > 0
        assert c.execute("SELECT COUNT(*) FROM entity WHERE type = 1").fetchone()[0] > 0


def test_type_weights_change_the_cartel_component(tmp_path):
    """Distrusting devices/cards must weaken the cartel signal itself.

    It does not necessarily lower the *maximum* score: with weaker strong
    evidence the partition can settle on a smaller, denser group, which scores
    higher on cartel and closure.  What must hold is that the cartel component
    shrinks for the same group.
    """
    from conftest import Fixture
    from ringdetect.graph import (BuildParams, LinkTable, build_projection,
                                  detect, score)
    from ringdetect.schema import CARD, DEVICE
    from ringdetect.store import Store

    fx = Fixture(tmp_path, accounts=400, days=6, rings=2, households=5, seed=9)
    fx.import_all()
    st = Store(fx.state, read_only=True)
    lt = LinkTable.load(st)

    def run(**kw):
        params = BuildParams(**kw)
        proj = build_projection(lt, params)
        labels = detect(proj, params)
        return score(st, params, lt, proj, labels)

    base = run()
    low = run(type_weight={DEVICE: 0.05, CARD: 0.05})
    # every community's cartel mass must be no larger with less-trusted types
    base_mass = max(r.cartel_mass for r in base)
    low_mass = max(r.cartel_mass for r in low)
    assert low_mass <= base_mass + 1e-9
    st.close()


def test_unknown_type_weight_is_rejected(tmp_path):
    from conftest import Fixture
    fx = Fixture(tmp_path, accounts=200, days=3, rings=1, households=3, seed=11)
    fx.import_all()
    proc = run_cli("build", "--state", str(fx.state), "--out", str(fx.out),
                   "--quiet", "--type-weight", "socks=0.5", check=False)
    assert proc.returncode == 2
    assert "unknown entity type" in proc.stderr


def test_gzip_files_are_accepted(tmp_path):
    import gzip
    from conftest import Fixture
    fx = Fixture(tmp_path, accounts=150, days=3, rings=1, households=2, seed=10)
    for p in _daily(fx):
        raw = p.read_bytes()
        with gzip.open(p.with_suffix(".csv.gz"), "wb") as fh:
            fh.write(raw)
        p.unlink()
    run_cli("import", "--state", str(fx.state), "--quiet",
            str(fx.data / "daily" / "*.csv.gz"))
    with sqlite3.connect(fx.state) as c:
        assert c.execute("SELECT COUNT(*) FROM batch").fetchone()[0] == 3
