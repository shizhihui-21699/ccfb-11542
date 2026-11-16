"""End-to-end tests: generate -> import -> build -> evaluate -> serve."""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request

import pytest

from ringdetect.webapp import make_server


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from conftest import Fixture
    root = tmp_path_factory.mktemp("e2e")
    fx = Fixture(root, accounts=900, days=10, rings=3, households=15, seed=3)
    fx.import_all()
    fx.build()
    return fx


# ------------------------------------------------------------------ build

def test_build_writes_result_files(built):
    assert (built.out / "results.sqlite").exists()
    summary = built.summary()
    assert summary["n_rings"] >= 1
    assert summary["counts"]["accounts"] == built.manifest["config"]["accounts"]
    assert summary["projection"]["accounts_in_graph"] > 0


def test_result_schema_is_self_contained(built):
    """A result directory must be portable: no paths into the state DB."""
    conn = sqlite3.connect(built.out / "results.sqlite")
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("ring", "ring_member", "ring_entity", "ring_edge", "ring_link", "meta"):
        assert t in tables
    conn.close()
    # copy the result dir alone and confirm it opens
    import shutil
    dest = built.root / "portable"
    shutil.copytree(built.out, dest)
    conn = sqlite3.connect(dest / "results.sqlite")
    assert conn.execute("SELECT COUNT(*) FROM ring").fetchone()[0] > 0
    conn.close()


def test_planted_rings_are_recovered(built):
    """Recall: every planted ring should appear as one reported ring."""
    conn = sqlite3.connect(built.out / "results.sqlite")
    members: dict[int, set[str]] = {}
    for rid, key in conn.execute("SELECT ring_id, account_key FROM ring_member"):
        members.setdefault(rid, set()).add(key)
    conn.close()
    for i, ring in enumerate(built.manifest["rings"]):
        gt = set(ring["accounts"])
        best = max((len(m & gt) / len(gt) for m in members.values()), default=0)
        assert best >= 0.6, f"planted ring {i} (size {len(gt)}) recovered only {best:.0%}"


def test_households_are_not_flagged_at_high_score(built):
    """Precision: legitimate co-tenants must not appear in a high-score ring."""
    conn = sqlite3.connect(built.out / "results.sqlite")
    flagged = {r[0] for r in conn.execute("SELECT id FROM ring WHERE score >= 40")}
    members: dict[int, set[str]] = {}
    for rid, key in conn.execute("SELECT ring_id, account_key FROM ring_member"):
        members.setdefault(rid, set()).add(key)
    conn.close()
    hits = []
    for h in built.manifest["households"]:
        hh = set(h["accounts"])
        for rid in flagged:
            if len(members.get(rid, set()) & hh) >= 2:
                hits.append((h["household"], rid))
    assert not hits, f"households flagged as rings: {hits}"


def test_evaluate_command_reports_recovery(built):
    from conftest import run_cli
    proc = run_cli("evaluate", "--out", str(built.out),
                   "--manifest", str(built.data / "manifest.json"), "--json")
    data = json.loads(proc.stdout)
    assert data["planted_rings"] == 3
    assert data["mean_recovered_fraction"] >= 0.6
    assert "thresholds" in data


def test_incremental_import_matches_bulk(built, tmp_path):
    """Importing a second copy of the same files must change nothing."""
    from conftest import run_cli
    before = built.summary()["counts"]
    run_cli("import", "--state", str(built.state), "--quiet", *built.files)
    after = sqlite3.connect(built.state)
    n = after.execute("SELECT COUNT(*) FROM batch").fetchone()[0]
    accounts = after.execute("SELECT COUNT(*) FROM account").fetchone()[0]
    after.close()
    assert accounts == before["accounts"]
    assert n == before["batches"], "re-import must not create new batches"


# -------------------------------------------------------------- web layer

def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server(built):
    port = _free_port()
    httpd = make_server(built.out, built.state, "127.0.0.1", port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def test_api_summary(server, built):
    d = _get(server + "/api/summary")
    assert d["counts"]["accounts"] == built.manifest["config"]["accounts"]
    assert d["has_state"] is True
    assert isinstance(d["histogram"], list)


def test_api_ring_list_filters(server):
    allr = _get(server + "/api/rings?min_score=0&limit=500")
    hot = _get(server + "/api/rings?min_score=40&limit=500")
    assert hot["total"] <= allr["total"]
    for r in hot["items"]:
        assert r["score"] >= 40


def test_api_ring_detail_has_evidence(server):
    lst = _get(server + "/api/rings?min_score=40&limit=1")
    assert lst["items"], "no high-score ring to inspect"
    rid = lst["items"][0]["id"]
    d = _get(f"{server}/api/rings/{rid}")
    assert d["members"]
    assert d["evidence_entities"] or d["weak_only"]
    assert isinstance(d["parts"], dict)
    assert d["n_accounts"] >= len(d["members"])


def test_api_ring_graph_is_bipartite(server):
    lst = _get(server + "/api/rings?min_score=40&limit=1")
    rid = lst["items"][0]["id"]
    g = _get(f"{server}/api/rings/{rid}/graph")
    assert g["mode"] == "bipartite"
    kinds = {n["kind"] for n in g["nodes"]}
    assert "account" in kinds
    assert kinds & {"device", "card", "address", "ip"}
    ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids


def test_api_ring_graph_omits_unshared_entities(server):
    """Only entities shared by >=2 members of the ring should be drawn."""
    lst = _get(server + "/api/rings?min_score=40&limit=1")
    rid = lst["items"][0]["id"]
    g = _get(f"{server}/api/rings/{rid}/graph")
    # every non-account node must touch at least two account nodes
    touch: dict[str, int] = {}
    for e in g["edges"]:
        for end in (e["source"], e["target"]):
            if end.startswith("e:"):
                touch[end] = touch.get(end, 0) + 1
    for eid, n in touch.items():
        assert n >= 2, f"entity {eid} drawn with only one account"


def test_api_account_roundtrip(server):
    lst = _get(server + "/api/rings?min_score=40&limit=1")
    d = _get(f"{server}/api/rings/{lst['items'][0]['id']}")
    key = d["members"][0]["account_key"]
    back = _get(f"{server}/api/accounts/{key}")
    assert back["account"] == key
    assert back["rings"]


def test_api_batches(server):
    d = _get(server + "/api/batches")
    assert len(d["items"]) == 10  # 10 daily files


def test_api_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(server + "/api/rings/999999", timeout=5)
    assert exc.value.code == 404


def test_index_page_served(server):
    with urllib.request.urlopen(server + "/", timeout=5) as r:
        body = r.read().decode()
    assert "Ring Review Console" in body
    assert "/vendor/cytoscape.min.js" in body


def test_path_traversal_blocked(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(server + "/../ringdetect/store.py", timeout=5)
    assert exc.value.code in (403, 404)


def test_summary_flags_stale_results(built, tmp_path):
    """After a new day is imported the served summary must say it is behind."""
    import shutil

    from conftest import run_cli
    state = tmp_path / "state.db"
    shutil.copy(built.state, state)
    # add one further day so the revision moves past the built one
    from ringdetect.generate import GenConfig, generate
    from datetime import date
    extra = tmp_path / "extra"
    generate(GenConfig(accounts=120, days=1, rings=1, households=2, seed=42,
                       start_day=date(2026, 10, 20)), extra)
    day_file = sorted((extra / "daily").glob("*.csv"))[0]
    run_cli("import", "--state", str(state), "--quiet", str(day_file))

    port = _free_port()
    httpd = make_server(built.out, state, "127.0.0.1", port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        d = _get(f"http://127.0.0.1:{port}/api/summary")
        assert d["stale"] is True
        assert d["state_revision_now"] > d["state_revision"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_summary_not_stale_when_current(built, server):
    assert _get(server + "/api/summary")["stale"] is False
