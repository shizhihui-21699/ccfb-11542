"""Synthetic daily data with planted rings and planted *legitimate* sharing.

The point of the generator is the ground truth: every row carries a
``scenario`` label, and alongside the data files we write a manifest naming the
planted rings and the legitimate shared households.  That lets the test-suite
(and a human) ask the two questions that matter:

* recall -- what fraction of each planted ring is recovered together?
* precision -- are the shared households left alone?

Population model
----------------
``normal``
    Independent shoppers.  Each has a stable device and (usually) a stable
    address; IPs are drawn from a small ISP pool per region, so many unrelated
    accounts share an IP -- the classic false positive.
``household`` (legitimate co-tenancy, 2-4 accounts)
    Family / roommates.  They *do* share one device, one address and often one
    card, and they order on overlapping days.  They must not be reported.
``ring`` (fraud, 6-30 accounts)
    A small pool of devices and cards reused across fresh accounts, one or two
    shipping addresses, a burst of registrations and orders, and a few decoy
    entities borrowed from the normal population to blur the picture.
``solo``
    Accounts with almost nothing shared -- the bulk, to make the graph sparse.

A ring's accounts share a *cartel* of 2-4 devices and 2-4 cards, which is what
makes the projected pair weight high: many accounts x several rare entities.
A household shares exactly one device and one card, so its pair weight is
lower even though the accounts look just as "co-located".
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

FIELDS = ["account_id", "order_id", "event_date", "device_id", "card_id",
          "address_id", "ip", "amount", "signup_date", "scenario"]


@dataclass
class GenConfig:
    accounts: int = 20_000
    days: int = 30
    start_day: date = date(2026, 9, 1)
    rings: int = 8
    ring_size: tuple[int, int] = (6, 22)
    households: int = 60
    household_size: tuple[int, int] = (2, 4)
    #: accounts that appear on many devices/IPs (people who upgrade phones,
    #: travel) -- noise that should not look like a ring
    drifters: int = 150
    ip_pool: int = 2000
    seed: int = 7
    rows_per_account: tuple[int, int] = (1, 6)


def generate(cfg: GenConfig, out_dir: Path) -> dict:
    rng = random.Random(cfg.seed)
    out_dir = Path(out_dir)
    raw_dir = out_dir / "daily"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)

    devices = [f"dev-{i:07x}" for i in range(cfg.accounts * 3)]
    cards = [f"card-{i:07x}" for i in range(cfg.accounts * 2)]
    ips = [f"10.{i // 256}.{i % 256}.{(i * 37) % 251}" for i in range(cfg.ip_pool)]
    addrs = [f"addr-{i:06x}" for i in range(cfg.accounts // 2 + 10)]

    # Entities are *drawn without replacement* where uniqueness matters.  If two
    # unrelated households were handed the same device, the generator would be
    # planting a fraud signal and then calling it benign.  The pools are large
    # enough for the requested account count; if a caller overshoots they are
    # extended rather than allowed to raise StopIteration mid-generation.
    dev_pool = list(rng.sample(devices, len(devices)))
    card_pool = list(rng.sample(cards, len(cards)))
    dev_pos = card_pos = 0

    def fresh_dev() -> str:
        nonlocal dev_pos
        if dev_pos >= len(dev_pool):
            extra = f"dev-extra-{dev_pos:08x}"
            dev_pos += 1
            return extra
        v = dev_pool[dev_pos]
        dev_pos += 1
        return v

    def fresh_card() -> str:
        nonlocal card_pos
        if card_pos >= len(card_pool):
            extra = f"card-extra-{card_pos:08x}"
            card_pos += 1
            return extra
        v = card_pool[card_pos]
        card_pos += 1
        return v

    # --- allocate accounts ------------------------------------------------
    plan: list[dict] = []
    used = 0

    def take(n: int) -> list[int]:
        nonlocal used
        ids = list(range(used, used + n))
        used += n
        return ids

    rings_truth = []
    for r in range(cfg.rings):
        size = rng.randint(*cfg.ring_size)
        ids = take(size)
        # A cartel larger than the ring would leave accounts sharing nothing,
        # which is a generator artefact rather than a detection failure.
        max_cartel = max(2, min(4, size // 2))
        n_dev = rng.randint(2, max_cartel)
        n_card = rng.randint(2, max_cartel)
        devices_r = rng.sample(devices, n_dev)
        cards_r = rng.sample(cards, n_card)
        addrs_r = rng.sample(addrs, rng.randint(1, 2))
        start = rng.randrange(0, max(1, cfg.days - 8))
        rings_truth.append({
            "ring": r, "accounts": [f"u{ i:07d}" for i in ids],
            "devices": devices_r, "cards": cards_r, "addresses": addrs_r,
            "start_offset": start, "size": size,
        })
        plan.append({"kind": "ring", "ids": ids, "ring": r,
                     "devices": devices_r, "cards": cards_r, "addrs": addrs_r,
                     "start": start})

    households_truth = []
    for h in range(cfg.households):
        size = rng.randint(*cfg.household_size)
        ids = take(size)
        dev = fresh_dev()
        card = fresh_card()
        addr = rng.choice(addrs)   # addresses are genuinely shared by blocks
        households_truth.append({"household": h,
                                 "accounts": [f"u{i:07d}" for i in ids]})
        plan.append({"kind": "household", "ids": ids, "devices": [dev],
                     "cards": [card], "addrs": [addr],
                     "start": rng.randrange(cfg.days)})

    for _ in range(cfg.drifters):
        ids = take(1)
        plan.append({"kind": "drifter", "ids": ids,
                     "devices": [fresh_dev() for _ in range(rng.randint(2, 3))],
                     "cards": [fresh_card() for _ in range(rng.randint(1, 2))],
                     "addrs": rng.sample(addrs, rng.randint(1, 3)),
                     "start": rng.randrange(cfg.days)})

    # ``accounts`` is a floor, not a hard cap: the planted populations are
    # allocated first, so a small budget is topped up with normal shoppers
    # instead of crashing.  A test that asks for 150 accounts and plants 166
    # worth of rings is a test-data problem, not a reason to refuse.
    remaining = max(0, cfg.accounts - used)
    total_accounts = max(used, cfg.accounts)
    for i in take(remaining):
        plan.append({"kind": "normal", "ids": [i],
                     "devices": [fresh_dev()],
                     "cards": [fresh_card()],
                     "addrs": [rng.choice(addrs)],
                     "start": rng.randrange(cfg.days)})

    # --- emit rows, split per day ----------------------------------------
    rows_by_day: dict[int, list[dict]] = {d: [] for d in range(cfg.days)}
    order_seq = 0
    for p in plan:
        kind = p["kind"]
        if kind == "ring":
            # Burst: all accounts active soon after ring start, many rows each.
            # Each *account* gets its own primary device and card drawn from the
            # cartel.  Drawing per-row instead would let a 7-account ring whose
            # cartel has 4 devices end up with almost no two accounts sharing
            # anything -- that is not a ring, and an analyst would rightly say
            # so.  Accounts are spread round-robin over the cartel so each
            # shared entity reaches several accounts.
            assignments = []
            for i, aid in enumerate(p["ids"]):
                assignments.append((
                    aid,
                    p["devices"][i % len(p["devices"])],
                    p["cards"][i % len(p["cards"])],
                ))
            for aid, primary_dev, primary_card in assignments:
                signup = p["start"] + rng.randint(0, 5)
                n_rows = rng.randint(2, 5)
                for _ in range(n_rows):
                    day = min(cfg.days - 1, signup + rng.randint(0, 6))
                    device = primary_dev if rng.random() < 0.88 else rng.choice(p["devices"])
                    card = primary_card if rng.random() < 0.85 else rng.choice(p["cards"])
                    # a little entity noise from the wider population
                    if rng.random() < 0.03:
                        device = rng.choice(devices)
                    if rng.random() < 0.03:
                        card = rng.choice(cards)
                    rows_by_day[day].append(_row(
                        rng, aid, order_seq, cfg, day, signup,
                        device=device, card=card,
                        addr=rng.choice(p["addrs"]),
                        ip=rng.choice(ips),
                        scenario="ring",
                    ))
                    order_seq += 1
        elif kind == "household":
            # co-tenants: overlapping days, shared device/card/address
            for aid in p["ids"]:
                signup = max(0, p["start"] - rng.randint(30, 400))
                for _ in range(rng.randint(2, 8)):
                    day = min(cfg.days - 1, p["start"] + rng.randint(0, cfg.days // 2))
                    rows_by_day[day].append(_row(
                        rng, aid, order_seq, cfg, day, signup,
                        device=p["devices"][0],
                        card=p["cards"][0] if rng.random() < 0.8 else fresh_card(),
                        addr=p["addrs"][0],
                        ip=rng.choice(ips), scenario="household",
                    ))
                    order_seq += 1
        elif kind == "drifter":
            aid = p["ids"][0]
            signup = max(0, p["start"] - rng.randint(30, 900))
            for _ in range(rng.randint(2, 7)):
                day = min(cfg.days - 1, p["start"] + rng.randint(0, cfg.days // 2))
                rows_by_day[day].append(_row(
                    rng, aid, order_seq, cfg, day, signup,
                    device=rng.choice(p["devices"]), card=rng.choice(p["cards"]),
                    addr=rng.choice(p["addrs"]), ip=rng.choice(ips),
                    scenario="drifter",
                ))
                order_seq += 1
        else:
            aid = p["ids"][0]
            signup = max(0, p["start"] - rng.randint(5, 500))
            for _ in range(rng.randint(*cfg.rows_per_account)):
                day = min(cfg.days - 1, p["start"] + rng.randint(0, cfg.days // 2))
                rows_by_day[day].append(_row(
                    rng, aid, order_seq, cfg, day, signup,
                    device=p["devices"][0], card=p["cards"][0],
                    addr=p["addrs"][0], ip=rng.choice(ips),
                    scenario="normal",
                ))
                order_seq += 1

    for d in range(cfg.days):
        rows = rows_by_day[d]
        rng.shuffle(rows)
        day = cfg.start_day + timedelta(days=d)
        path = raw_dir / f"events-{day.isoformat()}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    manifest = {
        "config": {
            "accounts": total_accounts, "requested_accounts": cfg.accounts,
            "days": cfg.days, "seed": cfg.seed,
            "start_day": cfg.start_day.isoformat(),
            "rings": cfg.rings, "households": cfg.households,
            "drifters": cfg.drifters, "ip_pool": cfg.ip_pool,
        },
        "rings": rings_truth,
        "households": households_truth,
        "rows_per_day": {str(d): len(rows_by_day[d]) for d in range(cfg.days)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _row(rng, aid, order_seq, cfg, day, signup, *, device, card, addr, ip, scenario):
    return {
        "account_id": f"u{aid:07d}",
        "order_id": f"o{order_seq:09d}",
        "event_date": (cfg.start_day + timedelta(days=day)).isoformat(),
        "device_id": device,
        "card_id": card,
        "address_id": addr,
        "ip": ip,
        "amount": f"{rng.uniform(5, 900):.2f}",
        "signup_date": (cfg.start_day - timedelta(days=max(0, day - signup))).isoformat(),
        "scenario": scenario,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="generate synthetic daily ring data")
    ap.add_argument("--out", default="data/sim", help="output directory")
    ap.add_argument("--accounts", type=int, default=20_000)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rings", type=int, default=8)
    ap.add_argument("--households", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--gzip", action="store_true", help="gzip each daily file")
    args = ap.parse_args(argv)

    cfg = GenConfig(accounts=args.accounts, days=args.days, rings=args.rings,
                    households=args.households, seed=args.seed,
                    start_day=date.fromisoformat(args.start))
    manifest = generate(cfg, Path(args.out))
    if args.gzip:
        for path in (Path(args.out) / "daily").glob("*.csv"):
            raw = path.read_bytes()
            with gzip.open(path.with_suffix(".csv.gz"), "wb") as fh:
                fh.write(raw)
            path.unlink()
    print(json.dumps({"out": str(args.out), "rings": len(manifest["rings"]),
                      "households": len(manifest["households"]),
                      "accounts": manifest["config"]["accounts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
