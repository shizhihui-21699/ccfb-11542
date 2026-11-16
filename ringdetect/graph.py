"""Weighted account-account projection graph, community detection, scoring.

Why this shape
--------------
The data is bipartite: accounts on one side, resources (device / card /
address / ip) on the other.  Rings are clusters of accounts sharing several
*rare* resources, so we project onto the account side and cut that.

Projection weight
-----------------
For an entity ``e`` shared by ``d`` accounts::

    strength(a, e) = sqrt( type_w[e] * rarity(e) * persistence(a, e) / (d - 1) )
    pair(a, b)     = sum over shared e of strength(a, e) * strength(b, e)

``rarity(e) = 1 - ln(d) / ln(A + 1)`` is ~1 for a device two accounts share and
~0 for an IP half the user base shares.  Dividing by ``d - 1`` holds an
entity's total donated weight constant however popular it is, so a NAT gateway
cannot manufacture millions of strong pairs.  The sqrt splits that weight
evenly between the two endpoints.

Community detection
-------------------
Leiden with the **Constant Potts Model** objective, not modularity.  Modularity
has a resolution limit: it cannot separate a 20-account ring from the giant
component it sits in unless the whole graph is tiny, which makes it unusable
at production size.  CPM maximises ``sum_c (m_c - gamma * C(n_c, 2))``: it keeps
a group merged while its average internal pair weight exceeds ``gamma``.  That
is a *density* threshold, so one tuned ``gamma`` means the same thing at 10
accounts and at 10 million.  Leiden rather than Louvain additionally guarantees
each community is internally connected -- which matters because every community
is presented to an analyst as a coherent piece of evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import igraph as ig
import numpy as np

from .schema import CARD, DEFAULT_TYPE_WEIGHT, DEVICE, TYPE_NAMES
from .store import Store

#: Pairs materialised per batch.  Bounds peak memory on hub-heavy data.
PAIR_CHUNK = 6_000_000


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass
class BuildParams:
    #: entities shared by more accounts than this do not generate pair edges
    #: (they still appear in evidence).  Prevents quadratic blow-up on NAT IPs.
    pair_degree_cap: int = 300
    #: ceiling on materialised pairs; the effective cap is lowered to fit.
    pair_budget: int = 20_000_000
    #: CPM resolution: the minimum average internal pair weight for a group to
    #: stay merged.  This is the one knob that trades ring completeness against
    #: fragmentation, and it is calibrated on the 1M fixture: 0.03 recovers
    #: 99.5% of every planted ring, while 0.12 splits the larger ones
    #: (their cartel spreads over more accounts, so each pair is individually
    #: weaker).  Raising it yields tighter, smaller, cleaner groups -- useful
    #: when the analyst would rather triage than sweep.
    gamma: float = 0.03
    #: ring-to-ring bridging entities are reported up to this many accounts.
    bridge_degree_cap: int = 500
    #: communities smaller than this are not reported.
    min_size: int = 2
    #: days of persistence at which account->entity strength is halved.
    persist_tau: float = 3.0
    type_weight: dict = field(default_factory=lambda: dict(DEFAULT_TYPE_WEIGHT))
    #: window (days) for the registration-burst signal.
    burst_window: int = 7
    #: multiplier applied to the strong-evidence aggregate before the noisy-OR.
    strong_scale: float = 1.0
    #: final multiplier on pair weight; keeps the scale gamma is calibrated to.
    pair_scale: float = 1.0
    n_iterations: int = 2
    #: cap on accounts embedded in one ring's detail view.
    max_detail_accounts: int = 500

    def resolved_type_weight(self) -> np.ndarray:
        """Per-type trust as a dense array indexed by entity type.

        Types absent from ``type_weight`` keep their schema default rather than
        silently becoming zero, so ``BuildParams(type_weight={DEVICE: .2})``
        means "distrust devices" and not "ignore addresses and IPs".
        """
        tw = np.zeros(5, dtype=np.float64)
        for etype, w in DEFAULT_TYPE_WEIGHT.items():
            tw[int(etype)] = float(w)
        for etype, w in self.type_weight.items():
            tw[int(etype)] = float(w)
        return tw

    def as_dict(self) -> dict:
        return {
            "pair_degree_cap": self.pair_degree_cap,
            "pair_budget": self.pair_budget,
            "gamma": self.gamma,
            "bridge_degree_cap": self.bridge_degree_cap,
            "min_size": self.min_size,
            "persist_tau": self.persist_tau,
            "type_weight": {TYPE_NAMES[int(k)]: float(v)
                            for k, v in self.type_weight.items()},
            "burst_window": self.burst_window,
            "n_iterations": self.n_iterations,
        }


@dataclass
class Projection:
    accounts: np.ndarray      # vertex -> account id
    u: np.ndarray             # int32, vertex index
    v: np.ndarray
    w: np.ndarray             # float32 pair weight (what Leiden optimises)
    evidence: np.ndarray      # float32 noisy-OR probability in [0, 1]
    stats: dict = field(default_factory=dict)


@dataclass
class Ring:
    """One community, fully realised for output."""

    label: int
    score: float
    parts: dict
    n_accounts: int
    strong_sum: float
    weak_sum: float
    internal_weight: float
    closure: float
    density: float
    weak_only: bool
    cartel_size: int = 0
    cartel_mass: float = 0.0
    burst_raw: float = 0.0
    members: list[int] = field(default_factory=list)   # account ids, sorted
    accounts_truncated: bool = False
    evidence_entities: list[dict] = field(default_factory=list)
    evidence_edges: list[dict] = field(default_factory=list)


@dataclass
class Analysis:
    proj: Projection
    labels: np.ndarray
    rings: list[Ring]
    account_keys: np.ndarray           # vertex -> account key (str)
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# account <-> entity load
# --------------------------------------------------------------------------

@dataclass
class LinkTable:
    """All links as dense-by-entity-id numpy arrays."""

    a: np.ndarray        # int64 account id
    e: np.ndarray        # int64 entity id
    n: np.ndarray
    n_days: np.ndarray
    first: np.ndarray
    last: np.ndarray
    n_entities: int
    n_accounts: int
    df: np.ndarray       # entity id -> number of accounts
    ent_type: np.ndarray  # entity id -> type (int8)

    @classmethod
    def load(cls, store: Store) -> "LinkTable":
        a, e, n, n_days, first, last = store.link_arrays()
        n_acc = store.conn.execute("SELECT COUNT(*) FROM account").fetchone()[0]
        n_ent = int(e.max()) + 1 if len(e) else 1
        df = np.bincount(e, minlength=n_ent).astype(np.int64) if len(e) else np.zeros(n_ent, np.int64)
        ent_type = np.zeros(n_ent, dtype=np.int8)
        rows = store.conn.execute("SELECT id, type FROM entity").fetchall()
        if rows:
            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            tys = np.fromiter((r[1] for r in rows), dtype=np.int8, count=len(rows))
            ids = ids[ids < n_ent]
            ent_type[ids] = tys[:len(ids)]
        return cls(a=a, e=e, n=n, n_days=n_days, first=first, last=last,
                   n_entities=n_ent, n_accounts=int(n_acc), df=df, ent_type=ent_type)

    def rarity(self, decay: float = 1.0) -> np.ndarray:
        log_a = math.log(self.n_accounts + 1) if self.n_accounts > 1 else 1.0
        r = 1.0 - np.log(np.maximum(self.df.astype(np.float64), 1.0)) / log_a
        return np.clip(r, 0.0, 1.0) ** decay


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def _choose_cap(df: np.ndarray, hard_cap: int,
                budget: int) -> tuple[int, int]:
    """Lower the popularity cap so the pair count fits ``budget``.

    Returns ``(cap, n_entities_used)``.
    """
    relevant = df[(df >= 2) & (df <= hard_cap)]
    if len(relevant) == 0:
        return hard_cap, 0
    relevant = np.sort(relevant)
    pairs = (relevant * (relevant - 1)) // 2
    cum = np.cumsum(pairs, dtype=np.int64)
    if cum[-1] <= budget:
        return hard_cap, int(len(relevant))
    k = int(np.searchsorted(cum, budget, side="right"))
    if k >= len(relevant):
        return hard_cap, int(len(relevant))
    cap = int(relevant[max(k - 1, 0)])
    return max(cap, 2), k


def build_projection(lt: LinkTable, params: BuildParams) -> Projection:
    if len(lt.a) == 0 or lt.n_accounts < 2:
        z = np.zeros(0, dtype=np.int64)
        return Projection(z, np.zeros(0, np.int32), np.zeros(0, np.int32),
                          np.zeros(0, np.float32), np.zeros(0, np.float32),
                          {"links": int(len(lt.a)), "pairs": 0, "hub_links": 0,
                           "entities_used": 0, "effective_cap": params.pair_degree_cap,
                           "accounts_in_graph": 0})

    tw = params.resolved_type_weight()
    rarity = lt.rarity()
    type_w = tw[lt.ent_type]
    persist = lt.n_days.astype(np.float64) / (lt.n_days.astype(np.float64)
                                              + params.persist_tau)

    # --- two channels: strong corroboration vs weak-only -------------------
    #
    # A shared *device* or *card* means the same human controls both accounts.
    # A shared *address* or *IP* does not: two unrelated households on the same
    # street block share an address, and a whole ISP pool shares an IP.  If
    # those were allowed to form edges on their own, co-tenants would be welded
    # into fake rings of 10+ accounts.
    #
    # So weak entities never create an edge, and never add weight on their own;
    # they only *corroborate* an edge that strong evidence already links.  That
    # is a noisy-OR: ``1 - prod(1 - p_e)`` over the edge's evidence, which is
    # the right arithmetic for independent pieces of evidence rather than
    # summing them, which would let one very popular address dominate.
    etype = lt.ent_type
    strong_type = (etype == DEVICE) | (etype == CARD)
    strong_w = np.where(strong_type, type_w, 0.0)
    weak_w = np.where(strong_type, 0.0, type_w)

    cap, n_used = _choose_cap(lt.df, params.pair_degree_cap, params.pair_budget)
    usable_s = (lt.df[lt.e] >= 2) & (lt.df[lt.e] <= cap) & (strong_w[lt.e] > 0)
    usable_w = (lt.df[lt.e] >= 2) & (lt.df[lt.e] <= cap) & (weak_w[lt.e] > 0)
    hub_links = int((lt.df[lt.e] > params.pair_degree_cap).sum())

    # Per-pair evidence mass.  Deliberately *not* divided by the entity's
    # degree: dividing would make a device 14 accounts share weaker per pair
    # than one that two strangers both happened to touch once, which is
    # backwards -- a ring's whole signal is many accounts reusing one cartel.
    # Rarity already discounts popularity smoothly, so this keeps only rarity
    # and the account's own persistence.
    strength_s = np.sqrt(np.maximum(strong_w[lt.e] * rarity[lt.e] * persist, 0.0))
    strength_w = np.sqrt(np.maximum(weak_w[lt.e] * rarity[lt.e] * persist, 0.0))

    base_stats = {"links": int(len(lt.a)), "hub_links": hub_links,
                  "entities_used": n_used, "effective_cap": int(cap)}
    sel_s = np.flatnonzero(usable_s)
    if len(sel_s) == 0:
        z = np.zeros(0, dtype=np.int64)
        return Projection(z, np.zeros(0, np.int32), np.zeros(0, np.int32),
                          np.zeros(0, np.float32), np.zeros(0, np.float32),
                          {**base_stats, "pairs": 0, "accounts_in_graph": 0})

    sel_s = sel_s[np.argsort(lt.e[sel_s], kind="stable")]
    up, vp, sp = [], [], []
    for lu, lv, lw in _iter_pairs(lt.a[sel_s], lt.e[sel_s], strength_s[sel_s], PAIR_CHUNK):
        up.append(lu)
        vp.append(lv)
        sp.append(lw)
    u = np.concatenate(up)
    v = np.concatenate(vp)
    strong_val = np.concatenate(sp).astype(np.float64)
    del up, vp, sp
    if len(u) == 0:
        z = np.zeros(0, dtype=np.int64)
        return Projection(z, np.zeros(0, np.int32), np.zeros(0, np.int32),
                          np.zeros(0, np.float32), np.zeros(0, np.float32),
                          {**base_stats, "pairs": 0, "accounts_in_graph": 0})

    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    key = lo * (lt.n_accounts + 1) + hi
    order = np.argsort(key, kind="stable")
    key = key[order]
    strong_val = strong_val[order]
    del order, u, v
    uniq, start = np.unique(key, return_index=True)
    # strong channel: sqrt of the sum of squared per-entity strengths, i.e. a
    # Euclidean combination.  Many weak-but-rare devices shared by two accounts
    # should count, but four of them should not count 4x one of them.
    strong_agg = np.sqrt(np.add.reduceat(strong_val ** 2, start))
    strong_agg *= float(params.strong_scale)

    # weak channel: corroboration only, and only for edges strong evidence
    # already links.
    weak_val = np.zeros(len(uniq), dtype=np.float64)
    sel_w = np.flatnonzero(usable_w)
    if len(sel_w):
        sel_w = sel_w[np.argsort(lt.e[sel_w], kind="stable")]
        wp, wv, ww = [], [], []
        for lu, lv, lw in _iter_pairs(lt.a[sel_w], lt.e[sel_w], strength_w[sel_w],
                                      PAIR_CHUNK):
            wp.append(lu)
            wv.append(lv)
            ww.append(lw)
        if wp:
            wu = np.concatenate(wp)
            wv = np.concatenate(wv)
            wval = np.concatenate(ww).astype(np.float64)
            wlo = np.minimum(wu, wv)
            whi = np.maximum(wu, wv)
            wkey = wlo * (lt.n_accounts + 1) + whi
            worder = np.argsort(wkey, kind="stable")
            wkey = wkey[worder]
            wval = wval[worder]
            wuniq, wstart = np.unique(wkey, return_index=True)
            wagg = np.sqrt(np.add.reduceat(wval ** 2, wstart))
            # align weak aggregates onto the strong pair list
            pos = np.searchsorted(wuniq, uniq)
            pos_c = np.clip(pos, 0, len(wuniq) - 1) if len(wuniq) else pos
            hit = (pos < len(wuniq)) & (wuniq[pos_c] == uniq)
            weak_val = np.where(hit, wagg[pos_c], 0.0)

    # Noisy-OR over the two channels: the probability that *at least one*
    # piece of independent evidence holds.  ``1 - prod(1 - p)``, not the
    # complement of the product of p's -- the latter would score an edge with
    # no evidence at all as maximally suspicious.
    p = 1.0 - (1.0 - np.clip(strong_agg, 0.0, 1.0)) * (
        1.0 - np.clip(weak_val, 0.0, 1.0))
    wsum = (float(params.pair_scale) * p).astype(np.float32)
    uu, vv = np.divmod(uniq, lt.n_accounts + 1)

    accounts = np.unique(np.concatenate([uu, vv]))
    remap = np.full(lt.n_accounts + 1, -1, dtype=np.int64)
    remap[accounts] = np.arange(len(accounts))
    stats = {**base_stats, "pairs": int(len(uniq)),
             "accounts_in_graph": int(len(accounts))}
    return Projection(accounts, remap[uu].astype(np.int32),
                      remap[vv].astype(np.int32), wsum,
                      (p).astype(np.float32), stats)


def _iter_pairs(s_acct: np.ndarray, s_ent: np.ndarray, s_str: np.ndarray,
                chunk_pairs: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (u, v, weight) for account pairs sharing an entity.

    Rows are sorted by entity, so entities form contiguous runs.  Runs are
    bucketed by exact size; within a size bucket all runs are laid out as a
    (groups x k) matrix and the k-1 off-diagonals are extracted with one
    boolean mask each.  No per-entity Python iteration.
    """
    n = len(s_ent)
    if n == 0:
        return
    new_grp = np.ones(n, dtype=bool)
    new_grp[1:] = s_ent[1:] != s_ent[:-1]
    gstart = np.flatnonzero(new_grp)
    gsize = np.diff(np.append(gstart, n))

    by_size = np.argsort(gsize, kind="stable")
    sizes_sorted = gsize[by_size]
    splits = np.flatnonzero(np.diff(sizes_sorted)) + 1
    for seg in np.split(by_size, splits):
        if len(seg) == 0:
            continue
        k = int(gsize[seg[0]])
        if k < 2:
            continue
        rows = (gstart[seg][:, None] + np.arange(k)[None, :]).ravel()
        base_acct = s_acct[rows]
        base_str = s_str[rows]
        per_grp = k * (k - 1) // 2
        per_chunk = max(1, chunk_pairs // per_grp)
        for g0 in range(0, len(seg), per_chunk):
            g1 = min(g0 + per_chunk, len(seg))
            m = g1 - g0
            acct = base_acct[g0 * k:g1 * k]
            st = base_str[g0 * k:g1 * k]
            within = np.arange(m * k) % k
            for d in range(1, k):
                # rows p and p+d are in the same run iff (p % k) + d < k, which
                # is exactly the `ok` mask -- no cross-run pairs leak through.
                ok = within[:m * k - d] < (k - d)
                idx = np.flatnonzero(ok)
                if len(idx) == 0:
                    continue
                yield acct[idx], acct[idx + d], (st[idx] * st[idx + d]).astype(np.float32)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def detect(proj: Projection, params: BuildParams) -> np.ndarray:
    n = len(proj.accounts)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    edges = np.column_stack([proj.u, proj.v])
    g = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    g.es["weight"] = proj.w.tolist()
    part = g.community_leiden(objective_function="CPM", weights="weight",
                              resolution=params.gamma,
                              n_iterations=params.n_iterations)
    return np.asarray(part.membership, dtype=np.int32)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _vertex_labels(lt: LinkTable, proj: Projection, labels: np.ndarray) -> np.ndarray:
    """account id -> vertex index, with -1 for accounts outside the graph."""
    m = np.full(lt.n_accounts + 1, -1, dtype=np.int64)
    m[proj.accounts] = np.arange(len(proj.accounts))
    return m


def _cluster_entity_cells(lab: np.ndarray, ent: np.ndarray,
                          n_entities: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = lab.astype(np.int64) * n_entities + ent
    uniq, counts = np.unique(cell, return_counts=True)
    return uniq // n_entities, uniq % n_entities, counts


def _burst_fraction(store: Store, proj: Projection, labels: np.ndarray,
                    n_lab: int, window: int) -> np.ndarray:
    """Fraction of each community's accounts registered inside one window.

    Sort by (label, signup day), then search ``key + window`` in the combined
    array and clamp to the community's end.  Keys are monotone across the
    sorted array, so one global ``searchsorted`` answers every community --
    no Python loop over communities, which at 1M accounts would dominate.
    """
    if len(proj.accounts) == 0:
        return np.zeros(n_lab)
    signup = np.full(int(proj.accounts.max()) + 1, -1, dtype=np.int64)
    for r in store.conn.execute(
            "SELECT id, COALESCE(signup_day, first_day) FROM account"):
        if r[1] is not None and r[0] < len(signup):
            signup[r[0]] = r[1]
    day = signup[proj.accounts]
    valid = day >= 0
    if not valid.any():
        return np.zeros(n_lab)

    base = int(day[valid].min())
    off = np.where(valid, day - base, 0)
    span = int(off.max()) + window + 2
    lab = labels.astype(np.int64)
    key = lab * span + off
    order = np.lexsort((off, lab))
    key_s = key[order]
    valid_s = valid[order]
    lab_s = lab[order]

    ends = np.searchsorted(key_s, np.arange(n_lab) * span, side="right")

    hi = np.searchsorted(key_s, key_s + window, side="right")
    hi = np.minimum(hi, ends[lab_s])
    counts = hi - np.arange(len(key_s))
    counts[~valid_s] = 0
    best = np.zeros(n_lab, dtype=np.int64)
    np.maximum.at(best, lab_s[valid_s], counts[valid_s])
    sizes = np.bincount(labels, minlength=n_lab)
    return np.divide(best, sizes, out=np.zeros(n_lab), where=sizes > 0)


@dataclass
class ScoreWeights:
    """Weights sum to 1; each component is already in [0, 1].

    The design is driven by the one structural difference between a fraud ring
    and legitimate co-tenancy: a ring recycles a *cartel* of several rare
    devices and cards, so many of its accounts each share several of them.  A
    household shares exactly one device and one card no matter how many people
    live there.  ``cartel`` therefore carries the most weight.
    """

    cartel: float = 0.55
    size: float = 0.10
    closure: float = 0.08
    burst: float = 0.10
    density: float = 0.07
    weak: float = 0.10


#: An entity counts toward the cartel only if at least two accounts in the ring
#: share it -- one is not shared at all.
CARTEL_MIN_ACCOUNTS = 2
#: Cartel *diversity* at which the component saturates: the number of distinct
#: rare devices/cards the ring recycles.  This is the sharpest signal we have.
#: A household, however large, shares exactly one device and one card, so it
#: sits at 2; a planted ring of 6 accounts already cycles 4-8, and a 20-account
#: ring reaches 8.  On the 1M fixture the two distributions do not overlap at
#: all (households 2, rings 4-8), so the ramp runs 2 -> 5.
CARTEL_FLOOR = 2.0
CARTEL_RAMP = 5.0


def score(store: Store, params: BuildParams, lt: LinkTable, proj: Projection,
          labels: np.ndarray) -> list[Ring]:
    n_lab = int(labels.max()) + 1 if len(labels) else 0
    if n_lab == 0:
        return []

    sizes = np.bincount(labels, minlength=n_lab).astype(np.int64)
    w64 = proj.w.astype(np.float64)
    lu = labels[proj.u].astype(np.int64)
    lv = labels[proj.v].astype(np.int64)
    inside = lu == lv
    internal_w = np.bincount(lu[inside], weights=w64[inside], minlength=n_lab)
    total_w = (np.bincount(lu, weights=w64, minlength=n_lab)
               + np.bincount(lv, weights=w64, minlength=n_lab) - internal_w)

    # ---- entity evidence per (ring, entity) ------------------------------
    vmap = _vertex_labels(lt, proj, labels)
    vert = vmap[lt.a]
    in_g = vert >= 0
    cl, ee, cnt = _cluster_entity_cells(labels[vert[in_g]], lt.e[in_g], lt.n_entities)

    tw = params.resolved_type_weight()
    rarity = lt.rarity()
    multi = cnt >= 2
    c_cl = cl[multi]
    c_e = ee[multi]
    c_k = cnt[multi]
    c_t = lt.ent_type[c_e]
    ev = tw[c_t] * rarity[c_e] * np.minimum(c_k - 1.0, 5.0) / 5.0
    strong = (c_t == DEVICE) | (c_t == CARD)
    strong_sum = np.bincount(c_cl[strong], weights=ev[strong], minlength=n_lab)
    weak_sum = np.bincount(c_cl[~strong], weights=ev[~strong], minlength=n_lab)

    # ---- cartel: distinct rare strong entities shared widely in the ring --
    cartel_mask = strong & (c_k >= CARTEL_MIN_ACCOUNTS) & (rarity[c_e] >= 0.35)
    cartel_count = np.bincount(c_cl[cartel_mask], minlength=n_lab).astype(np.float64)
    # Weight each distinct cartel entity by how strongly it is shared, so four
    # devices reaching three accounts each count more than four devices reaching
    # two, but not without bound.
    cartel_mass = np.bincount(
        c_cl[cartel_mask],
        weights=(np.minimum(c_k[cartel_mask] - 1.0, 4.0) / 4.0
                 * rarity[c_e[cartel_mask]]),
        minlength=n_lab)

    # ---- core membership: how much of the group actually belongs ---------
    #
    # A ring is not just "shares a cartel" -- its members each re-use several
    # of the shared devices/cards.  A legitimate household contributes one
    # device and one card, so most of its members touch only those two; a
    # stranger's account that happens to reuse one shared entity gets welded on
    # but is peripheral.  Core fraction = the share of members sharing at least
    # two *distinct* strong entities with another member.  At 1M accounts the
    # rings sit at ~0.9 and households ~0.5, which is the cleanest split we
    # found; multiplying it into the cartel term is what stops a household
    # plus a couple of coincidental accounts from scoring like a ring.
    core_fraction = _core_fraction(lt, proj, labels, n_lab)

    # ---- burst, ranked against the population ----------------------------
    burst_raw = _burst_fraction(store, proj, labels, n_lab, params.burst_window)
    burst = _percentile_rank(burst_raw, sizes)

    sw = ScoreWeights()
    density = np.divide(2.0 * internal_w, sizes * (sizes - 1.0),
                        out=np.zeros(n_lab), where=sizes > 1)
    closure = np.divide(internal_w, internal_w + total_w,
                        out=np.zeros(n_lab), where=(internal_w + total_w) > 0)

    # Ramp over the *diversity* of the cartel.  Two distinct shared rare
    # entities is exactly what a household looks like and scores 0; five or
    # more is a ring no matter how many accounts it has.  Sandwiching the
    # evidence this way is what makes the threshold mean the same thing at 6
    # accounts and at 20.
    cartel_f = (np.clip((cartel_count - CARTEL_FLOOR) / (CARTEL_RAMP - CARTEL_FLOOR),
                        0.0, 1.0)
                * np.clip(cartel_mass / 3.0, 0.0, 1.0)
                * core_fraction)
    size_f = np.clip(np.log2(np.maximum(sizes, 1) + 1) / 4.0, 0.0, 1.0)
    density_f = np.clip(density / 0.6, 0.0, 1.0)
    weak_f = 1.0 - np.exp(-weak_sum / 4.0)

    raw = (sw.cartel * cartel_f + sw.size * size_f + sw.closure * closure
           + sw.burst * burst + sw.density * density_f + sw.weak * weak_f)
    scores = np.round(100.0 * np.clip(raw, 0.0, 1.0), 1)

    keep = np.flatnonzero(sizes >= params.min_size)
    order = keep[np.lexsort((sizes[keep], -scores[keep]))]

    out: list[Ring] = []
    for c in order:
        c = int(c)
        # A ring whose evidence is address/IP only is a different, much weaker
        # claim than one sharing devices and cards.  Surface that explicitly.
        weak_only = bool(strong_sum[c] < 1e-9)
        out.append(Ring(
            label=c, score=float(scores[c]),
            parts={
                "cartel": round(float(cartel_f[c]), 3),
                "size": round(float(size_f[c]), 3),
                "closure": round(float(closure[c]), 3),
                "burst": round(float(burst[c]), 3),
                "density": round(float(density_f[c]), 3),
                "weak_evidence": round(float(weak_f[c]), 3),
            },
            n_accounts=int(sizes[c]),
            strong_sum=round(float(strong_sum[c]), 4),
            weak_sum=round(float(weak_sum[c]), 4),
            internal_weight=round(float(internal_w[c]), 5),
            closure=round(float(closure[c]), 5),
            density=round(float(density[c]), 6),
            weak_only=weak_only,
            cartel_size=int(cartel_count[c]),
            cartel_mass=round(float(cartel_mass[c]), 3),
            burst_raw=round(float(burst_raw[c]), 3),
            members=[],
        ))
    return out


def _core_fraction(lt: LinkTable, proj: Projection, labels: np.ndarray,
                   n_lab: int) -> np.ndarray:
    """Share of each community's members sharing >=2 strong entities inside it.

    Vectorised: count, for every (community, account) pair, the distinct strong
    entities of that community the account touches, then divide the members
    reaching >=2 by the community size.
    """
    if n_lab == 0 or len(proj.accounts) == 0:
        return np.zeros(n_lab)
    etype = lt.ent_type
    strong = (etype == DEVICE) | (etype == CARD)
    keep = strong[lt.e]
    a = lt.a[keep]
    e = lt.e[keep]
    if len(a) == 0:
        return np.zeros(n_lab)

    vmap = np.full(lt.n_accounts + 1, -1, dtype=np.int64)
    vmap[proj.accounts] = np.arange(len(proj.accounts))
    vert = vmap[a]
    ok = vert >= 0
    vert = vert[ok]
    e = e[ok]
    if len(vert) == 0:
        return np.zeros(n_lab)

    # distinct (account, entity) pairs with the entity shared in that community
    lab = labels[vert].astype(np.int64)
    n_ent = max(lt.n_entities, int(e.max()) + 1 if len(e) else 1)
    ent_key = lab * n_ent + e
    uniq, cnt = np.unique(ent_key, return_counts=True)
    shared = cnt >= 2
    if not shared.any():
        return np.zeros(n_lab)
    u_key = uniq[shared]

    # accounts that hold each shared (community, entity)
    acc_cell = np.searchsorted(u_key, ent_key).astype(np.int64)
    inside = acc_cell < len(u_key)
    hit = np.zeros(len(ent_key), dtype=bool)
    hit[inside] = u_key[acc_cell[inside]] == ent_key[inside]
    if not hit.any():
        return np.zeros(n_lab)
    acct_global = proj.accounts[vert[hit]]
    pair_key = acct_global.astype(np.int64) * (n_lab + 1) + lab[hit]
    p_uniq, p_cnt = np.unique(pair_key, return_counts=True)
    p_lab = p_uniq % (n_lab + 1)
    core = np.bincount(p_lab[p_cnt >= 2], minlength=n_lab).astype(np.float64)
    sizes = np.bincount(labels, minlength=n_lab).astype(np.float64)
    return np.divide(core, sizes, out=np.zeros(n_lab), where=sizes > 0)


def _percentile_rank(values: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Rank each value among communities of comparable size.

    A raw burst of 1.0 is unremarkable when thousands of 2-account groups also
    have it, so the signal is only useful once it is expressed as "unusually
    concentrated *for a group this size*".  Communities are bucketed by size
    (exact size for small groups, powers of two above) and ranked within the
    bucket.
    """
    n = len(values)
    if n == 0:
        return values
    bucket = np.where(sizes <= 8, sizes, 8 + np.floor(np.log2(np.maximum(sizes, 1))).astype(np.int64))
    out = np.zeros(n)
    for b in np.unique(bucket):
        sel = np.flatnonzero(bucket == b)
        vals = values[sel]
        if len(sel) == 1:
            out[sel] = 0.5
            continue
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(len(sel), dtype=np.float64)
        ranks[order] = np.arange(len(sel), dtype=np.float64)
        # average ties so identical values get identical credit
        uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
        avg = np.zeros(len(uniq))
        start = 0
        for i, c in enumerate(counts):
            avg[i] = (start + (c - 1) / 2.0) / (len(sel) - 1)
            start += int(c)
        out[sel] = avg[inv]
    return out


# --------------------------------------------------------------------------
# evidence + members
# --------------------------------------------------------------------------

def _values_for(store: Store, ids: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    if not ids:
        return out
    q = store.conn
    for i in range(0, len(ids), 800):
        part = ids[i:i + 800]
        marks = ",".join("?" * len(part))
        for r in q.execute(f"SELECT id, value FROM entity WHERE id IN ({marks})", part):
            out[r[0]] = r[1]
    return out


def enrich(store: Store, params: BuildParams, lt: LinkTable, proj: Projection,
           labels: np.ndarray, rings: list[Ring],
           account_keys: np.ndarray) -> None:
    """Fill members, evidence entities and evidence edges on each ring."""
    n_lab = int(labels.max()) + 1 if len(labels) else 0
    if not rings or n_lab == 0:
        return

    # -- members ----------------------------------------------------------
    acc_order = np.lexsort((account_keys, labels))
    lab_sorted = labels[acc_order]
    starts = np.searchsorted(lab_sorted, np.arange(n_lab), side="left")
    ends = np.searchsorted(lab_sorted, np.arange(n_lab), side="right")
    for ring in rings:
        lo, hi = int(starts[ring.label]), int(ends[ring.label])
        members = acc_order[lo:hi]
        keep = min(len(members), params.max_detail_accounts)
        ring.members = [int(proj.accounts[i]) for i in members[:keep]]
        ring.accounts_truncated = len(members) > keep

    # -- per-ring entity evidence ----------------------------------------
    vmap = _vertex_labels(lt, proj, labels)
    vert = vmap[lt.a]
    in_g = vert >= 0
    cl, ee, cnt = _cluster_entity_cells(labels[vert[in_g]], lt.e[in_g], lt.n_entities)
    tw = params.resolved_type_weight()
    rarity = lt.rarity()

    multi = cnt >= 2
    c_cl = cl[multi]
    c_e = ee[multi]
    c_k = cnt[multi].astype(np.float64)
    c_t = lt.ent_type[c_e]
    ent_score = tw[c_t] * rarity[c_e] * np.minimum(c_k - 1.0, 5.0) / 5.0

    order = np.lexsort((-ent_score, c_cl))
    oc_cl = c_cl[order]
    oc_e = c_e[order]
    oc_k = c_k[order]
    keep_mask = _first_n_per_run(oc_cl, 12)
    picked = np.flatnonzero(keep_mask)

    wanted = [int(oc_e[i]) for i in picked]
    values = _values_for(store, sorted(set(wanted)))

    buckets: dict[int, list[int]] = {}
    for i in picked:
        buckets.setdefault(int(oc_cl[i]), []).append(int(i))

    # -- per-ring evidence edges ------------------------------------------
    lu = labels[proj.u].astype(np.int64)
    lv = labels[proj.v].astype(np.int64)
    inside = lu == lv
    e_cl = lu[inside]
    e_u = proj.u[inside]
    e_v = proj.v[inside]
    e_w = proj.w[inside]
    eorder = np.lexsort((-e_w, e_cl))
    e_cl = e_cl[eorder]
    e_u, e_v, e_w = e_u[eorder], e_v[eorder], e_w[eorder]
    top_e = np.flatnonzero(_first_n_per_run(e_cl, 15))
    ebuckets: dict[int, list[int]] = {}
    for i in top_e:
        ebuckets.setdefault(int(e_cl[i]), []).append(int(i))

    for ring in rings:
        ents = []
        for i in buckets.get(ring.label, ()):
            eid = int(oc_e[i])
            ents.append({
                "type": TYPE_NAMES[int(lt.ent_type[eid])],
                "value": values.get(eid, ""),
                "accounts_in_ring": int(oc_k[i]),
                "accounts_total": int(lt.df[eid]),
                "rarity": round(float(rarity[eid]), 3),
                "type_weight": round(float(tw[lt.ent_type[eid]]), 2),
            })
        ring.evidence_entities = ents
        edges = []
        for i in ebuckets.get(ring.label, ()):
            edges.append({
                "a": str(account_keys[e_u[i]]),
                "b": str(account_keys[e_v[i]]),
                "weight": round(float(e_w[i]), 4),
            })
        ring.evidence_edges = edges


def _run_start(sorted_vals: np.ndarray) -> np.ndarray:
    if len(sorted_vals) == 0:
        return np.zeros(0, dtype=np.int64)
    new = np.ones(len(sorted_vals), dtype=bool)
    new[1:] = sorted_vals[1:] != sorted_vals[:-1]
    starts = np.flatnonzero(new)
    return np.repeat(starts, np.diff(np.append(starts, len(sorted_vals))))


def _first_n_per_run(sorted_vals: np.ndarray, n: int) -> np.ndarray:
    return (np.arange(len(sorted_vals)) - _run_start(sorted_vals)) < n


# --------------------------------------------------------------------------
# ring <-> ring bridges
# --------------------------------------------------------------------------

def bridging_entities(store: Store, params: BuildParams, lt: LinkTable,
                      proj: Projection, labels: np.ndarray,
                      rings: list[Ring]) -> list[tuple[int, int, str, str, int, int, float]]:
    """Entities shared by two different rings.

    Returns ``(ring_a, ring_b, type, value, n_a, n_b, rarity)`` sorted strongest
    first.  Rings are account-disjoint, so a shared entity is either too weak
    to merge them or a hub; both are worth an analyst's attention.
    """
    if not rings or len(labels) == 0:
        return []
    n_lab = int(labels.max()) + 1
    label_to_ring = {r.label: i for i, r in enumerate(rings)}
    vmap = _vertex_labels(lt, proj, labels)
    vert = vmap[lt.a]
    in_g = vert >= 0
    lab = labels[vert[in_g]].astype(np.int64)
    ent = lt.e[in_g]
    if len(ent) == 0:
        return []

    uniq, cnt = np.unique(ent * (n_lab + 1) + lab, return_counts=True)
    ue = uniq // (n_lab + 1)
    ul = uniq % (n_lab + 1)

    new = np.ones(len(ue), dtype=bool)
    new[1:] = ue[1:] != ue[:-1]
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], len(ue))
    span = ends - starts                       # runs (one per entity) -> cells
    ent_of_run = ue[starts]                    # run index -> entity id
    cand = np.flatnonzero((span >= 2) & (lt.df[ent_of_run] <= params.bridge_degree_cap))
    if len(cand) == 0:
        return []

    rarity = lt.rarity()
    values = _values_for(store, sorted({int(ent_of_run[c]) for c in cand}))
    out: list[tuple] = []
    for ci in cand:
        s, e = int(starts[ci]), int(ends[ci])
        eid = int(ent_of_run[ci])
        rr = [(label_to_ring[int(ul[j])], int(cnt[j])) for j in range(s, e)
              if int(ul[j]) in label_to_ring]
        for x in range(len(rr)):
            for y in range(x + 1, len(rr)):
                a, na = rr[x]
                b, nb = rr[y]
                if a == b:
                    continue
                if a > b:
                    a, b, na, nb = b, a, nb, na
                out.append((a, b, TYPE_NAMES[int(lt.ent_type[eid])],
                            values.get(eid, ""), na, nb, round(float(rarity[eid]), 3)))
    out.sort(key=lambda t: (-(t[4] + t[5]), -t[6]))
    return out


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def _account_keys(store: Store, accounts: np.ndarray) -> np.ndarray:
    """Account key per vertex, in vertex order."""
    n = len(accounts)
    out = np.empty(n, dtype=object)
    if n == 0:
        return out
    for i in range(0, n, 5000):
        part = [int(x) for x in accounts[i:i + 5000]]
        marks = ",".join("?" * len(part))
        got = {r[0]: r[1].split("\x1f", 1)[-1] for r in store.conn.execute(
            f"SELECT id, key FROM account WHERE id IN ({marks})", part)}
        for j, aid in enumerate(part):
            out[i + j] = got.get(aid, str(aid))
    return out
