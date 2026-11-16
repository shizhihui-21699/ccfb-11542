"""Score a result set against a planted-ground-truth manifest.

Two questions, computed separately:

*Recovery* -- for each planted ring, the largest fraction of its accounts that
land in a single reported ring.  Reported per ring and averaged.

*Contamination* -- legitimate sharers (households) that end up inside a ring
reported at or above a score threshold.  Rings are account-disjoint, so a
household's accounts can be split across rings; we count any household with at
least two accounts inside the same reported ring as contaminated.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def evaluate(out_dir: str | Path, manifest_path: str | Path, *,
             as_json: bool = False, thresholds=(30, 40, 50, 60, 70, 80)) -> int:
    out_dir = Path(out_dir)
    res = out_dir / "results.sqlite"
    if not res.exists():
        raise FileNotFoundError(f"{res} not found; run the build first")
    manifest = json.loads(Path(manifest_path).read_text())

    conn = sqlite3.connect(f"file:{res}?mode=ro", uri=True)
    try:
        rings = conn.execute(
            "SELECT id, score, n_accounts FROM ring ORDER BY score DESC"
        ).fetchall()
        member = {}
        ring_of: dict[str, list[int]] = {}
        for rid, key in conn.execute("SELECT ring_id, account_key FROM ring_member"):
            member.setdefault(rid, set()).add(key)
            ring_of.setdefault(key, []).append(rid)
        ring_score = {r[0]: r[1] for r in rings}
        ring_size = {r[0]: r[2] for r in rings}
    finally:
        conn.close()

    # ---- tag each reported ring by how many planted rings it touches -----
    planted = {i: set(r["accounts"]) for i, r in enumerate(manifest["rings"])}
    household = {i: set(h["accounts"]) for i, h in enumerate(manifest["households"])}
    known = set().union(*planted.values()) if planted else set()
    known |= set().union(*household.values()) if household else set()

    ring_tags: dict[int, set[int]] = {}
    for rid, keys in member.items():
        for k in keys:
            for pi, pset in planted.items():
                if k in pset:
                    ring_tags.setdefault(rid, set()).add(pi)

    # ---- recovery --------------------------------------------------------
    recovery = []
    for pi, pset in planted.items():
        best = 0.0
        best_ring = None
        for rid, keys in member.items():
            inter = len(keys & pset)
            if inter > best * len(pset):
                best = inter / len(pset) if pset else 0.0
                best_ring = rid
        recovery.append({
            "ring": pi, "size": len(pset),
            "recovered_fraction": round(best, 3),
            "best_ring": best_ring,
            "best_ring_score": round(ring_score.get(best_ring, 0.0), 1) if best_ring is not None else None,
            "best_ring_size": ring_size.get(best_ring) if best_ring is not None else None,
        })

    # ---- contamination of legitimate sharers -----------------------------
    def contamination(threshold: float) -> dict:
        flagged = {rid for rid, sc in ring_score.items() if sc >= threshold}
        flagged |= {rid for rid, sc in ring_score.items() if sc >= threshold}
        bad = []
        for hi, hset in household.items():
            for rid in flagged:
                inter = len(member.get(rid, set()) & hset)
                if inter >= 2:
                    bad.append({"household": hi, "ring": rid,
                                "score": round(ring_score.get(rid, 0), 1),
                                "accounts_in_ring": inter,
                                "household_size": len(hset)})
                    break
        return {
            "threshold": threshold,
            "n_rings_flagged": len(flagged),
            "households_contaminated": len(bad),
            "household_contamination_rate": round(len(bad) / max(1, len(household)), 4),
            "details": bad[:20],
            "n_flagged_accounts": int(sum(ring_size[r] for r in flagged)),
        }

    # ---- label purity of flagged rings -----------------------------------
    def purity(threshold: float) -> dict:
        flagged = [rid for rid, sc in ring_score.items() if sc >= threshold]
        n_single = n_mixed = 0
        families = 0
        for rid in flagged:
            tags = ring_tags.get(rid, set())
            if len(tags) == 1:
                n_single += 1
            elif len(tags) > 1:
                n_mixed += 1
            if any(len(member[rid] & set(manifest["rings"][t]["accounts"])) >= 3
                   for t in tags):
                families += 1
        return {
            "threshold": threshold, "n_flagged": len(flagged),
            "rings_matching_one_planted": n_single,
            "rings_spanning_multiple": n_mixed,
            "rings_recovering_a_planted_ring": families,
        }

    summary = {
        "reported_rings": len(rings),
        "planted_rings": len(planted),
        "planted_households": len(household),
        "recovery": recovery,
        "mean_recovered_fraction": round(
            sum(r["recovered_fraction"] for r in recovery) / max(1, len(recovery)), 3),
        "full_recovery": sum(1 for r in recovery if r["recovered_fraction"] >= 0.8),
        "thresholds": {
            str(t): {"contamination": contamination(t), "purity": purity(t)}
            for t in thresholds
        },
        "coverage_of_known_accounts": _coverage(member, known),
    }

    if as_json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"reported rings {summary['reported_rings']:,}  "
          f"planted {summary['planted_rings']}  households {summary['planted_households']}")
    print(f"mean ring recovery {summary['mean_recovered_fraction']:.2f}   "
          f"fully recovered {summary['full_recovery']}/{summary['planted_rings']}")
    print("\nper-ring recovery:")
    for r in recovery:
        print(f"  ring {r['ring']:2d}  size {r['size']:3d}  "
              f"recovered {r['recovered_fraction']:.2f}  "
              f"in ring {r['best_ring']} (score {r['best_ring_score']}, "
              f"size {r['best_ring_size']})")
    print("\nby threshold:")
    print(f"{'score>=':>8} {'rings':>7} {'accounts':>9} {'households hit':>15} {'rate':>6}")
    for t in thresholds:
        c = summary["thresholds"][str(t)]["contamination"]
        print(f"{t:>8} {c['n_rings_flagged']:>7} {c['n_flagged_accounts']:>9} "
              f"{c['households_contaminated']:>15} "
              f"{c['household_contamination_rate']:>6.1%}")
    return 0


def _coverage(member: dict, known: set) -> dict:
    covered = sum(1 for k in known if k in member)
    return {"accounts_in_any_reported_ring": covered,
            "known_accounts": len(known),
            "fraction": round(covered / max(1, len(known)), 4)}
