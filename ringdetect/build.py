"""Orchestration: state -> projection -> communities -> scored result set.

Produces a small self-contained ``results.sqlite`` plus ``summary.json``.  The
web server reads only those two files; it opens the state database read-only
when it needs the full bipartite detail of a single ring.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import graph as G
from .schema import TYPE_NAMES
from .store import Store

RESULT_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS ring (
    id              INTEGER PRIMARY KEY,
    score           REAL NOT NULL,
    n_accounts      INTEGER NOT NULL,
    n_members_shown INTEGER NOT NULL,
    strong_sum      REAL NOT NULL,
    weak_sum        REAL NOT NULL,
    internal_weight REAL NOT NULL,
    closure         REAL NOT NULL,
    density         REAL NOT NULL,
    parts           TEXT NOT NULL,
    weak_only       INTEGER NOT NULL,
    cartel_size     INTEGER NOT NULL,
    cartel_mass     REAL NOT NULL,
    burst_raw       REAL NOT NULL,
    n_evidence      INTEGER NOT NULL,
    type_counts     TEXT NOT NULL,
    first_day       INTEGER,
    last_day        INTEGER,
    signup_med_day  INTEGER
);

CREATE TABLE IF NOT EXISTS ring_member (
    ring_id     INTEGER NOT NULL,
    account_id  INTEGER NOT NULL,
    account_key TEXT NOT NULL,
    n_records   INTEGER,
    first_day   INTEGER,
    last_day    INTEGER,
    signup_day  INTEGER,
    n_strong    INTEGER NOT NULL DEFAULT 0,
    n_weak      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ring_id, account_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ring_member_key ON ring_member (account_key);
CREATE INDEX IF NOT EXISTS ring_member_acct ON ring_member (account_id);

CREATE TABLE IF NOT EXISTS ring_entity (
    ring_id          INTEGER NOT NULL,
    rank             INTEGER NOT NULL,
    type             TEXT NOT NULL,
    value            TEXT NOT NULL,
    accounts_in_ring INTEGER NOT NULL,
    accounts_total   INTEGER NOT NULL,
    rarity           REAL NOT NULL,
    type_weight      REAL NOT NULL,
    PRIMARY KEY (ring_id, rank)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ring_entity_value ON ring_entity (value);

CREATE TABLE IF NOT EXISTS ring_edge (
    ring_id INTEGER NOT NULL,
    rank    INTEGER NOT NULL,
    a_key   TEXT NOT NULL,
    b_key   TEXT NOT NULL,
    weight  REAL NOT NULL,
    PRIMARY KEY (ring_id, rank)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS ring_link (
    ring_a INTEGER NOT NULL,
    ring_b INTEGER NOT NULL,
    type   TEXT NOT NULL,
    value  TEXT NOT NULL,
    n_a    INTEGER NOT NULL,
    n_b    INTEGER NOT NULL,
    rarity REAL NOT NULL,
    PRIMARY KEY (ring_a, ring_b, type, value)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ring_link_a ON ring_link (ring_a);
CREATE INDEX IF NOT EXISTS ring_link_b ON ring_link (ring_b);
"""


@dataclass
class BuildOutcome:
    out_dir: Path
    seconds: float
    n_accounts: int
    n_entities: int
    n_links: int
    n_pairs: int
    n_vertices_in_graph: int
    n_communities: int
    n_rings: int
    timings: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def run_build(store_path: str | os.PathLike, out_dir: str | os.PathLike,
              params: G.BuildParams | None = None, *, min_score: float = 0.0,
              verbose: bool = True) -> BuildOutcome:
    params = params or G.BuildParams()
    t_all = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "results.sqlite.tmp"
    if tmp.exists():
        tmp.unlink()

    timings: dict[str, float] = {}
    store = Store(store_path, read_only=True)
    try:
        counts = store.counts()
        et_counts = store.entity_type_counts()
        revision = store.revision

        t = time.perf_counter()
        lt = G.LinkTable.load(store)
        proj = G.build_projection(lt, params)
        timings["project"] = _secs(t)
        _log(verbose, f"projection   {len(proj.w):>12,} pairs   "
                      f"{len(proj.accounts):>10,} accounts   {_secs(t):6.1f}s")

        t = time.perf_counter()
        labels = G.detect(proj, params)
        n_comm = int(labels.max()) + 1 if len(labels) else 0
        timings["leiden"] = _secs(t)
        _log(verbose, f"leiden       {n_comm:>12,} communities            {_secs(t):6.1f}s")

        t = time.perf_counter()
        rings = G.score(store, params, lt, proj, labels)
        if min_score > 0:
            rings = [r for r in rings if r.score >= min_score]
        timings["score"] = _secs(t)

        account_keys = G._account_keys(store, proj.accounts)
        t = time.perf_counter()
        G.enrich(store, params, lt, proj, labels, rings, account_keys)
        timings["evidence"] = _secs(t)
        _log(verbose, f"evidence     {len(rings):>12,} rings                 {_secs(t):6.1f}s")

        t = time.perf_counter()
        bridges = G.bridging_entities(store, params, lt, proj, labels, rings)
        timings["bridges"] = _secs(t)

        t = time.perf_counter()
        _write(tmp, params, store, proj, labels, rings, account_keys, bridges,
               revision, min_score, timings)
        timings["write"] = _secs(t)
        _log(verbose, f"write        results.sqlite                          {_secs(t):6.1f}s")
    finally:
        store.close()

    os.replace(tmp, out / "results.sqlite")
    total = round(time.perf_counter() - t_all, 2)
    timings["total"] = total

    summary = _summary(store_path, total, timings, counts, et_counts, revision,
                       params, proj, n_comm, rings, min_score)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return BuildOutcome(out_dir=out, seconds=total, n_accounts=counts["accounts"],
                        n_entities=counts["entities"], n_links=counts["links"],
                        n_pairs=int(proj.stats.get("pairs", 0)),
                        n_vertices_in_graph=len(proj.accounts),
                        n_communities=n_comm, n_rings=len(rings),
                        timings=timings, summary=summary)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _write(path: Path, params: G.BuildParams, store: Store, proj: G.Projection,
           labels: np.ndarray, rings: list[G.Ring], account_keys: np.ndarray,
           bridges: list[tuple], revision: int, min_score: float,
           timings: dict) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        conn.execute("BEGIN")

        member_ids = [a for r in rings for a in r.members]
        meta = store.account_meta_by_id(member_ids)

        # per-member strong/weak entity counts within its ring
        strong_counts, weak_counts = _member_strength(store, proj, labels, rings,
                                                      account_keys)

        ring_rows, member_rows, ent_rows, edge_rows, link_rows = [], [], [], [], []
        for rid, ring in enumerate(rings):
            days_f, days_l, days_s = [], [], []
            tc = {name: 0 for name in TYPE_NAMES}
            for ent in ring.evidence_entities:
                tc[ent["type"]] = tc.get(ent["type"], 0) + 1
            for a in ring.members:
                m = meta.get(a, {})
                if m.get("first_day") is not None:
                    days_f.append(m["first_day"])
                if m.get("last_day") is not None:
                    days_l.append(m["last_day"])
                if m.get("signup_day") is not None:
                    days_s.append(m["signup_day"])
                member_rows.append((
                    rid, a, account_keys[_vertex_of(proj, a)],
                    m.get("n_records"), m.get("first_day"), m.get("last_day"),
                    m.get("signup_day"),
                    strong_counts.get((ring.label, a), 0),
                    weak_counts.get((ring.label, a), 0),
                ))
            ring_rows.append((
                rid, ring.score, ring.n_accounts, len(ring.members),
                ring.strong_sum, ring.weak_sum, ring.internal_weight, ring.closure,
                ring.density, json.dumps(ring.parts, ensure_ascii=False),
                int(ring.weak_only), ring.cartel_size, ring.cartel_mass, ring.burst_raw,
                len(ring.evidence_entities),
                json.dumps(tc, ensure_ascii=False),
                min(days_f) if days_f else None,
                max(days_l) if days_l else None,
                int(np.median(days_s)) if days_s else None,
            ))
            for rank, ent in enumerate(ring.evidence_entities):
                ent_rows.append((rid, rank, ent["type"], ent["value"],
                                 ent["accounts_in_ring"], ent["accounts_total"],
                                 ent["rarity"], ent["type_weight"]))
            for rank, edge in enumerate(ring.evidence_edges):
                edge_rows.append((rid, rank, edge["a"], edge["b"], edge["weight"]))

        for a, b, etype, value, na, nb, rar in bridges[:200_000]:
            link_rows.append((a, b, etype, value, na, nb, rar))

        conn.executemany("INSERT INTO ring VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ring_rows)
        conn.executemany("INSERT INTO ring_member VALUES (?,?,?,?,?,?,?,?,?)", member_rows)
        conn.executemany("INSERT INTO ring_entity VALUES (?,?,?,?,?,?,?,?)", ent_rows)
        if edge_rows:
            conn.executemany("INSERT INTO ring_edge VALUES (?,?,?,?,?)", edge_rows)
        if link_rows:
            conn.executemany("INSERT OR IGNORE INTO ring_link VALUES (?,?,?,?,?,?,?)", link_rows)

        meta_rows = {
            "params": json.dumps(params.as_dict(), ensure_ascii=False),
            "n_rings": str(len(rings)),
            "min_score": str(min_score),
            "state_revision": str(revision),
            "result_schema_version": str(RESULT_SCHEMA_VERSION),
            "timings": json.dumps(timings),
        }
        conn.executemany("INSERT INTO meta (k, v) VALUES (?,?)", list(meta_rows.items()))
        conn.execute("COMMIT")
        conn.execute("ANALYZE")
    finally:
        conn.close()


def _vertex_of(proj: G.Projection, aid: int) -> int:
    return int(np.searchsorted(proj.accounts, aid))


def _member_strength(store: Store, proj: G.Projection, labels: np.ndarray,
                     rings: list[G.Ring], account_keys: np.ndarray
                     ) -> tuple[dict, dict]:
    """(ring label, account id) -> count of strong / weak entities shared in ring."""
    from .schema import CARD, DEVICE

    if not rings:
        return {}, {}
    lt = store.link_arrays()
    a, e = lt[0], lt[1]
    vmap = np.full(int(a.max()) + 1 if len(a) else 1, -1, dtype=np.int64)
    vmap[proj.accounts] = np.arange(len(proj.accounts))
    vert = vmap[a]
    ok = vert >= 0
    lab = labels[vert[ok]]
    acc = a[ok]
    ent = e[ok]

    ent_type = np.zeros(int(e.max()) + 1 if len(e) else 1, dtype=np.int8)
    rows = store.conn.execute("SELECT id, type FROM entity").fetchall()
    for r in rows:
        if r[0] < len(ent_type):
            ent_type[r[0]] = r[1]

    # entity -> number of accounts in this ring
    cell = lab.astype(np.int64) * (int(e.max()) + 2) + ent
    uniq, cnt = np.unique(cell, return_counts=True)
    shared = cnt >= 2

    # map each (account, entity) link to its ring's entity count
    pos = np.searchsorted(uniq, lab.astype(np.int64) * (int(e.max()) + 2) + ent)
    pos = np.clip(pos, 0, len(uniq) - 1)
    et = ent_type[ent]
    is_strong = (et == DEVICE) | (et == CARD)

    strong: dict[tuple[int, int], int] = {}
    weak: dict[tuple[int, int], int] = {}
    sel = shared[pos]
    for aid, l, s in zip(acc[sel], lab[sel], is_strong[sel]):
        key = (int(l), int(aid))
        if s:
            strong[key] = strong.get(key, 0) + 1
        else:
            weak[key] = weak.get(key, 0) + 1
    return strong, weak


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def _summary(store_path, total, timings, counts, et_counts, revision, params,
             proj, n_comm, rings, min_score) -> dict:
    scores = np.array([r.score for r in rings]) if rings else np.zeros(0)
    sizes = np.array([r.n_accounts for r in rings]) if rings else np.zeros(0)
    weak_only = sum(1 for r in rings if r.weak_only)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state": str(store_path),
        "state_revision": revision,
        "seconds": total,
        "timings": timings,
        "params": params.as_dict(),
        "min_score": min_score,
        "counts": counts,
        "entities_by_type": et_counts,
        "projection": proj.stats,
        "n_communities": n_comm,
        "n_rings": len(rings),
        "score": {
            "max": float(scores.max()) if len(scores) else None,
            "p50": float(np.percentile(scores, 50)) if len(scores) else None,
            "p90": float(np.percentile(scores, 90)) if len(scores) else None,
            "n_ge_40": int((scores >= 40).sum()),
            "n_ge_60": int((scores >= 60).sum()),
            "n_ge_75": int((scores >= 75).sum()),
        },
        "size": {
            "max": int(sizes.max()) if len(sizes) else 0,
            "p50": int(np.percentile(sizes, 50)) if len(sizes) else 0,
            "total_accounts": int(sizes.sum()),
        },
        "weak_only_rings": weak_only,
    }


def _secs(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)
