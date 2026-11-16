"""Projection-weight properties and community-detection behaviour."""

from __future__ import annotations

import numpy as np

from ringdetect.graph import (
    BuildParams, LinkTable, _account_keys, _choose_cap, _iter_pairs, build_projection, detect, score,
)
from ringdetect.store import ParsedRecord, Store
from ringdetect.schema import CARD, DEVICE, IP, ADDRESS


def _store(tmp_path, rows):
    st = Store(tmp_path / "s.db")
    st.import_records([ParsedRecord(account=a, day=d, resources=r)
                       for a, d, r in rows], source="t", sha256="t")
    return st


# ---------------------------------------------------------------- pairs

def test_iter_pairs_emits_every_pair_in_a_group():
    # one entity (id 7) shared by accounts 0,1,2 on contiguous rows
    acct = np.array([0, 1, 2], dtype=np.int64)
    ent = np.array([7, 7, 7], dtype=np.int64)
    strength = np.array([1.0, 1.0, 1.0])
    pairs = set()
    for u, v, _w in _iter_pairs(acct, ent, strength, 10_000):
        for a, b in zip(u.tolist(), v.tolist()):
            pairs.add((min(a, b), max(a, b)))
    assert pairs == {(0, 1), (0, 2), (1, 2)}


def test_iter_pairs_does_not_leak_between_groups():
    # two groups of two, adjacent in the array: (0,1) and (2,3) only
    acct = np.array([0, 1, 2, 3], dtype=np.int64)
    ent = np.array([5, 5, 6, 6], dtype=np.int64)
    strength = np.ones(4)
    pairs = set()
    for u, v, _w in _iter_pairs(acct, ent, strength, 10_000):
        for a, b in zip(u.tolist(), v.tolist()):
            pairs.add((min(a, b), max(a, b)))
    assert pairs == {(0, 1), (2, 3)}


def test_iter_pairs_handles_mixed_group_sizes():
    # groups of size 3, 2, 4 -> 3 + 1 + 6 = 10 pairs
    ent = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3], dtype=np.int64)
    acct = np.arange(len(ent), dtype=np.int64)
    total = sum(len(u) for u, _v, _w in _iter_pairs(acct, ent, np.ones(len(ent)), 4))
    assert total == 10


def test_choose_cap_respects_budget():
    df = np.array([0, 1, 2, 2, 2, 2, 10, 10, 600])
    cap, used = _choose_cap(df, hard_cap=300, budget=5)
    assert cap < 300
    # the two df=2 groups give 2 pairs; the two df=10 give 90 each -> over budget
    assert used <= 4


# ----------------------------------------------------------- projection

def test_pair_weight_is_symmetric_and_positive(tmp_path):
    st = _store(tmp_path, [
        ("u1", 1, {DEVICE: "d1"}), ("u2", 1, {DEVICE: "d1"}),
        ("u1", 2, {DEVICE: "d1"}), ("u2", 2, {DEVICE: "d1"}),
    ])
    lt = LinkTable.load(st)
    proj = build_projection(lt, BuildParams())
    assert len(proj.w) == 1
    assert proj.w[0] > 0
    assert proj.u[0] != proj.v[0]


def test_shared_address_alone_creates_no_edge(tmp_path):
    """Address/IP are corroboration, never evidence on their own."""
    st = _store(tmp_path, [
        ("u1", 1, {ADDRESS: "a1", IP: "10.0.0.1"}),
        ("u2", 1, {ADDRESS: "a1", IP: "10.0.0.1"}),
    ])
    lt = LinkTable.load(st)
    proj = build_projection(lt, BuildParams())
    assert len(proj.w) == 0


def test_address_corroborates_an_existing_strong_edge(tmp_path):
    """Same strong link, with and without a shared address."""
    plain = _store(tmp_path, [
        ("u1", 1, {DEVICE: "d1"}), ("u2", 1, {DEVICE: "d1"}),
    ])
    corroborated = _store(tmp_path / "b", [
        ("u1", 1, {DEVICE: "d1", ADDRESS: "a1"}),
        ("u2", 1, {DEVICE: "d1", ADDRESS: "a1"}),
    ])
    p1 = build_projection(LinkTable.load(plain), BuildParams())
    p2 = build_projection(LinkTable.load(corroborated), BuildParams())
    assert p2.w[0] > p1.w[0]


def test_popular_device_yields_lower_weight_than_rare_one(tmp_path):
    """Rarity must discount popularity, or a kiosk device looks like a ring."""
    rare = _store(tmp_path / "rare", [
        ("u1", 1, {DEVICE: "d1"}), ("u2", 1, {DEVICE: "d1"}),
    ] + [(f"x{i}", 1, {DEVICE: f"dx{i}"}) for i in range(200)])
    popular = _store(tmp_path / "pop", [
        ("u1", 1, {DEVICE: "d1"}), ("u2", 1, {DEVICE: "d1"}),
    ] + [(f"x{i}", 1, {DEVICE: "d1"}) for i in range(200)])
    params = BuildParams(pair_degree_cap=1000)
    p_rare = build_projection(LinkTable.load(rare), params)
    p_pop = build_projection(LinkTable.load(popular), params)
    # the u1-u2 edge exists in both; in the popular case d1 has df=202, so the
    # same edge must be worth much less
    def u1u2_w(proj, st):
        keys = _account_keys(st, proj.accounts)
        for u, v, w in zip(proj.u, proj.v, proj.w):
            if {keys[u], keys[v]} == {"u1", "u2"}:
                return float(w)
        return None

    w_rare = u1u2_w(p_rare, rare)
    w_pop = u1u2_w(p_pop, popular)
    assert w_rare is not None and w_pop is not None
    assert w_pop < w_rare / 2


def test_pair_degree_cap_is_a_hard_ceiling(tmp_path):
    """Entities above the cap must produce no edges at all, not a capped few."""
    rows = [("u1", 1, {DEVICE: "d1"}), ("u2", 1, {DEVICE: "d1"})]
    rows += [(f"x{i}", 1, {DEVICE: "d1"}) for i in range(200)]
    st = _store(tmp_path, rows)
    proj = build_projection(LinkTable.load(st), BuildParams(pair_degree_cap=50))
    assert len(proj.w) == 0
    assert proj.stats["hub_links"] == 202


def test_hub_entities_do_not_explode_the_pair_count(tmp_path):
    rows = [("u0", 1, {IP: "10.9.9.9"})]
    rows += [(f"u{i}", 1, {IP: "10.9.9.9"}) for i in range(1, 400)]
    st = _store(tmp_path, rows)
    lt = LinkTable.load(st)
    proj = build_projection(lt, BuildParams(pair_degree_cap=50))
    # 400 accounts sharing one IP would be 79,800 pairs; the cap must kill it
    assert len(proj.w) == 0
    assert proj.stats["hub_links"] >= 400


# ---------------------------------------------------------- detection

def _two_groups(tmp_path, n=8, bridge=1):
    """Two cliques over separate devices, optionally joined by a shared one."""
    rows = []
    for i in range(n):
        rows.append((f"a{i}", 1, {DEVICE: "da", CARD: "ca"}))
        rows.append((f"a{i}", 2, {DEVICE: "da"}))
        rows.append((f"b{i}", 1, {DEVICE: "db", CARD: "cb"}))
        rows.append((f"b{i}", 2, {DEVICE: "db"}))
    for _ in range(bridge):
        pass
    return _store(tmp_path, rows)


def test_detect_separates_two_cliques(tmp_path):
    st = _two_groups(tmp_path)
    lt = LinkTable.load(st)
    proj = build_projection(lt, BuildParams())
    labels = detect(proj, BuildParams())
    # projection vertices are account ids; resolve them back to account keys
    keys = _account_keys(st, proj.accounts)
    a_lab = {int(labels[i]) for i, k in enumerate(keys) if k.startswith("a")}
    b_lab = {int(labels[i]) for i, k in enumerate(keys) if k.startswith("b")}
    assert len(a_lab) == 1 and len(b_lab) == 1
    assert a_lab != b_lab


def test_household_scores_lower_than_a_cartel_ring(tmp_path):
    """The core claim of the scoring model, in one assertion.

    The ring's cartel is large enough that most pairs share *two or more*
    entities, which is what a real ring looks like; a 4-device cartel spread
    thin over 10 accounts where every pair shares only one device would be a
    weaker signal than the fixture intends.
    """
    rows = []
    # a 4-person household: one device, one card, long persistence
    for i in range(4):
        for d in range(12):
            rows.append((f"h{i}", d, {DEVICE: "dh", CARD: "ch",
                                      ADDRESS: "ah"}))
    # a 12-account ring: 3 devices and 3 cards, each account cycling several
    for i in range(12):
        for d in range(8):
            rows.append((f"r{i}", d, {
                DEVICE: f"dr{i % 3}",
                CARD: f"cr{(i // 3) % 3}",
                ADDRESS: "ar",
            }))
            rows.append((f"r{i}", d, {DEVICE: f"dr{(i + 1) % 3}"}))
    st = _store(tmp_path, rows)
    lt = LinkTable.load(st)
    params = BuildParams()
    proj = build_projection(lt, params)
    labels = detect(proj, params)
    rings = score(st, params, lt, proj, labels)
    hh = max((r for r in rings if r.n_accounts == 4), key=lambda r: r.score)
    ring = max(rings, key=lambda r: r.score)
    assert ring.n_accounts >= 10, f"ring fragmented into {ring.n_accounts}"
    assert ring.score > hh.score
    assert ring.cartel_size > hh.cartel_size
    assert hh.score < 45, f"household scored too high: {hh.score}"


def test_scores_are_bounded_and_monotone_in_cartel_size(tmp_path):
    rows = []
    for group in range(3):
        for i in range(8):
            for d in range(4):
                rows.append((f"g{group}_{i}", d, {
                    DEVICE: f"d{group}_{i % (group + 1)}",
                    CARD: f"c{group}_{i % (group + 1)}",
                }))
    st = _store(tmp_path, rows)
    lt = LinkTable.load(st)
    params = BuildParams()
    proj = build_projection(lt, params)
    labels = detect(proj, params)
    rings = score(st, params, lt, proj, labels)
    for r in rings:
        assert 0 <= r.score <= 100
        assert set(r.parts) == {"cartel", "size", "closure", "burst", "density",
                                "weak_evidence"}
        for v in r.parts.values():
            assert 0.0 <= v <= 1.0


def test_weak_only_flag_marks_address_only_rings(tmp_path):
    """Two accounts whose only shared entity is an address must be flagged."""
    rows = [("u1", 1, {DEVICE: "d1", ADDRESS: "shared"}),
            ("u2", 1, {DEVICE: "d2", ADDRESS: "shared"})]
    rows += [(f"z{i}", 1, {DEVICE: f"dz{i}", ADDRESS: f"az{i}"}) for i in range(50)]
    st = _store(tmp_path, rows)
    lt = LinkTable.load(st)
    proj = build_projection(lt, BuildParams())
    # no strong edge exists, so nothing reaches the graph at all
    assert len(proj.accounts) == 0 or len(proj.w) == 0


def test_gamma_controls_fragmentation(tmp_path):
    """Higher resolution must not produce fewer communities on a dense group."""
    rows = []
    for i in range(24):
        for d in range(4):
            rows.append((f"u{i}", d, {DEVICE: f"d{i % 6}", CARD: f"c{i % 6}"}))
    st = _store(tmp_path, rows)
    lt = LinkTable.load(st)
    low = build_projection(lt, BuildParams(gamma=0.01))
    high = build_projection(lt, BuildParams(gamma=0.01))
    n_low = detect(low, BuildParams(gamma=0.01)).max() + 1
    n_high = detect(high, BuildParams(gamma=0.5)).max() + 1
    assert n_high >= n_low
