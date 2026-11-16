"""CLI: ``ringdetect <command>``.

    ringdetect init-state   --state state.db
    ringdetect import       --state state.db data/daily/*.csv
    ringdetect build        --state state.db --out results/
    ringdetect serve        --out results/ [--state state.db]
    ringdetect evaluate     --out results/ --manifest data/sim/manifest.json
    ringdetect info         --state state.db
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

from .ingest import FieldMap, ParseError, ParseStats, iter_records
from .schema import ConfigError
from .store import Store, StoreError, file_sha256, to_day_int


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ringdetect",
                                 description="offline coupon-fraud ring detection")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init-state", help="create an empty state database")
    p.add_argument("--state", default="state/state.db",
                   help="path of the state database to create")

    p = sub.add_parser("import", help="import one or more daily files")
    p.add_argument("--state", default="state/state.db",
                   help="state database (created if absent)")
    p.add_argument("--day", type=_date_arg, default=None,
                   help="day for files with no date column (YYYY-MM-DD)")
    p.add_argument("--replace-day", type=_date_arg, action="append", default=None,
                   help="drop this day's ledger before importing (corrected file)")
    p.add_argument("--account-col", default=None, help="account id column")
    p.add_argument("--day-col", default=None, help="event date column (auto-detected)")
    p.add_argument("--signup-col", default=None,
                   help="registration date column (auto-detected)")
    p.add_argument("--device-col", default=None, help="device column")
    p.add_argument("--card-col", default=None, help="card column")
    p.add_argument("--address-col", default=None, help="address column")
    p.add_argument("--ip-col", default=None, help="IP column")
    p.add_argument("--ip-prefix", type=int, default=0,
                   help="collapse IPv4 to /N (e.g. 24) to reduce NAT noise")
    p.add_argument("--card-salt-env", default="RINGDETECT_CARD_SALT",
                   help="env var holding the HMAC salt for raw card numbers")
    p.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("files", nargs="+",
                   help="CSV/TSV/JSONL files or globs; .gz is read transparently")

    p = sub.add_parser("build", help="run detection and write a result set")
    p.add_argument("--state", default="state/state.db",
                   help="state database written by `import`")
    p.add_argument("--out", default="results",
                   help="result directory (results.sqlite + summary.json)")
    p.add_argument("--gamma", type=float, default=0.03,
                   help="CPM resolution; raise for smaller, tighter rings")
    p.add_argument("--min-size", type=int, default=2,
                   help="do not report rings smaller than this")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="do not store rings scoring below this")
    p.add_argument("--pair-degree-cap", type=int, default=300,
                   help="entities used by more accounts than this create no edges")
    p.add_argument("--pair-budget", type=int, default=20_000_000,
                   help="ceiling on materialised account pairs")
    p.add_argument("--type-weight", action="append", default=None,
                   metavar="TYPE=W", help="e.g. address=0.4; repeatable")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("serve", help="serve the web viewer")
    p.add_argument("--out", default="results", help="result directory to serve")
    p.add_argument("--state", default=None,
                   help="state database; enables the full bipartite ring view")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765,
                   help="first port to try; the next free one is used if taken")
    p.add_argument("--open", action="store_true", help="open a browser")

    p = sub.add_parser("evaluate", help="recall/precision against a manifest")
    p.add_argument("--out", default="results", help="result directory to evaluate")
    p.add_argument("--manifest", required=True,
                   help="manifest.json written by ringdetect.generate")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    p = sub.add_parser("generate", help="generate synthetic data with known rings")
    p.add_argument("--out", default="data/sim", help="output directory")
    p.add_argument("--accounts", type=int, default=20_000,
                   help="minimum account count (raised if planted groups need more)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--rings", type=int, default=8, help="planted fraud rings")
    p.add_argument("--households", type=int, default=60,
                   help="planted legitimate co-tenant groups")
    p.add_argument("--drifters", type=int, default=150,
                   help="accounts that legitimately change devices")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start", default="2026-09-01", help="first day (YYYY-MM-DD)")
    p.add_argument("--gzip", action="store_true", help="gzip each daily file")

    p = sub.add_parser("info", help="show state / result statistics")
    p.add_argument("--state", default="state/state.db")
    p.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    except (StoreError, ParseError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args) -> int:
    if args.cmd == "init-state":
        with Store(args.state):
            pass
        print(f"initialised {args.state}")
        return 0

    if args.cmd == "generate":
        from datetime import date

        from .generate import GenConfig, generate
        cfg = GenConfig(accounts=args.accounts, days=args.days, rings=args.rings,
                        households=args.households, drifters=args.drifters,
                        seed=args.seed, start_day=date.fromisoformat(args.start))
        manifest = generate(cfg, Path(args.out))
        if args.gzip:
            import gzip
            for path in sorted((Path(args.out) / "daily").glob("*.csv")):
                raw = path.read_bytes()
                with gzip.open(path.with_suffix(".csv.gz"), "wb") as fh:
                    fh.write(raw)
                path.unlink()
        print(f"{manifest['config']['accounts']:,} accounts, "
              f"{len(manifest['rings'])} rings, "
              f"{len(manifest['households'])} households -> {args.out}")
        return 0

    if args.cmd == "import":
        return _cmd_import(args)
    if args.cmd == "build":
        return _cmd_build(args)
    if args.cmd == "serve":
        from .webapp import serve
        return serve(args)
    if args.cmd == "evaluate":
        from .evaluate import evaluate
        return evaluate(args.out, args.manifest, as_json=args.json)
    if args.cmd == "info":
        return _cmd_info(args)
    raise SystemExit(f"unknown command {args.cmd}")


def _cmd_import(args) -> int:
    salt = None
    if args.card_salt_env:
        val = os.environ.get(args.card_salt_env)
        if val:
            salt = val.encode()
    # FieldMap.from_headers treats None as "auto-detect"; the CLI defaults to
    # the generator's column names, so translate absent flags to defaults.
    fm = FieldMap(
        account=args.account_col,
        day=args.day_col, signup=args.signup_col,
        device=args.device_col if args.device_col is not None else "device_id",
        card=args.card_col if args.card_col is not None else "card_id",
        address=args.address_col if args.address_col is not None else "address_id",
        ip=args.ip_col if args.ip_col is not None else "ip",
    )

    files: list[str] = []
    for pattern in args.files:
        matches = sorted(glob.glob(pattern))
        files.extend(matches or [pattern])
    if not files:
        raise StoreError("no input files matched")

    replace_days = [to_day_int(d) for d in (args.replace_day or [])]
    total_t0 = time.perf_counter()
    total_rows = total_records = 0
    skipped = 0
    with Store(args.state) as store:
        for path in files:
            if not Path(path).exists():
                print(f"warning: {path} does not exist, skipping", file=sys.stderr)
                continue
            sha = file_sha256(path)
            if store.conn.execute("SELECT 1 FROM batch WHERE sha256 = ?", (sha,)).fetchone():
                skipped += 1
                if not args.quiet:
                    print(f"skip  {path}  (already imported)")
                continue
            # A corrected re-export of a day: drop the old ledger for those
            # days *before* re-reading, or the import would add to the bad data
            # instead of replacing it.
            for day in replace_days:
                store.delete_day(day)
            stats = ParseStats()
            default_day = to_day_int(args.day) if args.day else None
            t0 = time.perf_counter()
            records = iter_records(path, field_map=fm, default_day=default_day,
                                   ip_prefix=args.ip_prefix, card_salt=salt,
                                   stats=stats, limit=args.limit)
            res = store.import_records(records, source=str(path), sha256=sha)
            dt = time.perf_counter() - t0
            total_rows += stats.rows
            total_records += res.n_records
            if not args.quiet:
                extra = ""
                if stats.missing_account:
                    extra += f"  skipped_no_account={stats.missing_account}"
                if stats.no_resource:
                    extra += f"  skipped_no_resource={stats.no_resource}"
                print(f"import {path}  rows={stats.rows:,}  records={res.n_records:,}  "
                      f"new_accounts={res.n_new_accounts:,}  "
                      f"new_entities={res.n_new_entities:,}  {dt:.1f}s{extra}")
        counts = store.counts()
        a, b = store.day_range()
        print(f"\nstate: {counts['accounts']:,} accounts, {counts['entities']:,} entities, "
              f"{counts['links']:,} links, {counts['batches']:,} batches, "
              f"days {a}..{b}  ({total_records:,} records this run, "
              f"{skipped} file(s) skipped, {time.perf_counter() - total_t0:.1f}s)")
    return 0


def _cmd_build(args) -> int:
    from .build import run_build
    from .graph import BuildParams

    params = BuildParams(gamma=args.gamma, min_size=args.min_size,
                         pair_degree_cap=args.pair_degree_cap,
                         pair_budget=args.pair_budget)
    if args.type_weight:
        from .schema import TYPE_BY_NAME
        for spec in args.type_weight:
            name, _, val = spec.partition("=")
            if name.strip() not in TYPE_BY_NAME:
                raise ConfigError(f"unknown entity type {name!r}")
            params.type_weight[TYPE_BY_NAME[name.strip()]] = float(val)
    outcome = run_build(args.state, args.out, params,
                        min_score=args.min_score, verbose=not args.quiet)
    s = outcome.summary
    print(f"\n{outcome.n_rings:,} rings reported in {outcome.seconds:.1f}s "
          f"({outcome.n_vertices_in_graph:,} accounts in graph, "
          f"{outcome.n_pairs:,} pairs)")
    sc = s["score"]
    print(f"scores: max={sc['max']}  p90={sc['p90']}  p50={sc['p50']}  "
          f">=60: {sc['n_ge_60']}  >=75: {sc['n_ge_75']}")
    print(f"wrote {Path(args.out) / 'results.sqlite'} and summary.json")
    return 0


def _cmd_info(args) -> int:
    with Store(args.state, read_only=True) as store:
        print(json.dumps({
            "counts": store.counts(),
            "entities_by_type": store.entity_type_counts(),
            "day_range": store.day_range(),
        }, indent=2))
        for b in store.batches(limit=10):
            print(f"  {b['imported_at']}  {b['source']}  {b['n_records']:,} records")
    if args.out:
        print(json.dumps(json.loads((Path(args.out) / "summary.json").read_text()),
                         indent=2))
    return 0


def _date_arg(text: str):
    try:
        to_day_int(text)
    except StoreError:
        raise argparse.ArgumentTypeError(f"not a date: {text!r}")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
