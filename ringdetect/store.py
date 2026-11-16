"""SQLite-backed state store with incremental, idempotent daily imports.

Layout
------
One row in a daily file = one *record* (an order, a signup, a login...).  A
record belongs to one account and may reference up to four resources.

``account`` / ``entity``
    interned ids plus lifetime stats.
``link``
    rolled-up per (account, entity) counters -- what the graph build reads.
``link_day``
    per-day ledger, used to make re-imports of a corrected day exact.

Idempotency
-----------
Every batch is content-addressed (sha256 of the raw file bytes); importing the
same file twice is a no-op.  Two *different* files covering the same day are
additive, which is right for genuinely partitioned data.  When a file is a
corrected replacement for a day, ``--replace-day`` deletes that day's ledger
first (``Store.delete_day``), so the result is exactly the new file.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .schema import (
    ACCOUNT,
    TYPE_BY_NAME,
    TYPE_NAMES,
    entity_key,
    normalize_value,
)

SCHEMA_VERSION = 1
_EPOCH = _dt.date(1970, 1, 1)
_COPY_CHUNK = 20_000     # rows per executemany
_IN_CHUNK = 800          # SQLite bound-parameter ceiling is 32766; stay well below


class StoreError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# date handling
# --------------------------------------------------------------------------

def to_day_int(value: object) -> int:
    """Parse a date-ish value into days since 1970-01-01.

    Accepts ``2024-03-01``, ``20240301``, ``2024/03/01``, ISO timestamps, epoch
    seconds/milliseconds, and ``date``/``datetime`` objects.
    """
    if isinstance(value, _dt.datetime):
        return (value.date() - _EPOCH).days
    if isinstance(value, _dt.date):
        return (value - _EPOCH).days
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = int(value)
        if 19000101 <= num <= 29991231:  # YYYYMMDD
            return (_dt.date(num // 10000, num // 100 % 100, num % 100) - _EPOCH).days
        if num > 10_000_000_000:  # epoch millis
            num //= 1000
        if num > 0:
            try:
                return (_dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=num)).date() \
                    .toordinal() - _EPOCH.toordinal()
            except (OverflowError, ValueError):
                raise StoreError(f"day value out of range: {value!r}")
        raise StoreError(f"unparseable day value {value!r}")
    text = str(value).strip()
    if not text:
        raise StoreError("empty day value")
    digits = text.replace("-", "").replace("/", "").replace(".", "")
    if len(digits) == 8 and digits.isdigit():
        try:
            return (_dt.date(int(digits[:4]), int(digits[4:6]),
                             int(digits[6:])) - _EPOCH).days
        except ValueError:
            pass
    try:
        return (_dt.date.fromisoformat(text[:10]) - _EPOCH).days
    except ValueError:
        pass
    try:
        return (_dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date() - _EPOCH).days
    except ValueError as exc:
        raise StoreError(f"unparseable day value {value!r}") from exc


def day_to_str(day: int) -> str:
    return (_EPOCH + _dt.timedelta(days=int(day))).isoformat()


def datetime_now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# input record
# --------------------------------------------------------------------------

@dataclass
class ParsedRecord:
    """One input record after parsing, before interning.

    ``resources`` maps entity type -> raw value.  ``None`` values are skipped.
    """

    account: object
    day: int
    resources: dict = None  # type: ignore[assignment]
    signup_day: int | None = None
    ip_prefix: int = 0
    card_salt: bytes | None = None

    def __post_init__(self):
        if self.resources is None:
            self.resources = {}


@dataclass
class ImportResult:
    batch_id: str
    skipped: bool
    n_records: int
    n_links: int
    n_new_accounts: int = 0
    n_new_entities: int = 0


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS batch (
    batch_id    TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    source      TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    n_records   INTEGER NOT NULL,
    n_new_accounts INTEGER NOT NULL DEFAULT 0,
    n_new_entities INTEGER NOT NULL DEFAULT 0,
    days        TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS batch_sha ON batch (sha256) WHERE sha256 <> '';

CREATE TABLE IF NOT EXISTS account (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,
    first_day   INTEGER,
    last_day    INTEGER,
    signup_day  INTEGER,
    n_days      INTEGER NOT NULL DEFAULT 0,
    n_records   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entity (
    id        INTEGER PRIMARY KEY,
    type      INTEGER NOT NULL,
    value     TEXT NOT NULL,
    first_day INTEGER,
    last_day  INTEGER,
    n_accounts INTEGER NOT NULL DEFAULT 0,
    n_records INTEGER NOT NULL DEFAULT 0,
    UNIQUE (type, value)
);

CREATE TABLE IF NOT EXISTS link (
    account_id INTEGER NOT NULL,
    entity_id  INTEGER NOT NULL,
    n          INTEGER NOT NULL,
    n_days     INTEGER NOT NULL,
    first_day  INTEGER NOT NULL,
    last_day   INTEGER NOT NULL,
    PRIMARY KEY (account_id, entity_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS link_day (
    account_id INTEGER NOT NULL,
    entity_id  INTEGER NOT NULL,
    day        INTEGER NOT NULL,
    n          INTEGER NOT NULL,
    PRIMARY KEY (account_id, entity_id, day)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS day_ledger (
    day        INTEGER PRIMARY KEY,
    n_records  INTEGER NOT NULL,
    n_links    INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS entity_by_type_value ON entity (type, value);
CREATE INDEX IF NOT EXISTS link_by_entity ON link (entity_id);
"""


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

class Store:
    """The on-disk state.  Not thread-safe; one connection per process."""

    def __init__(self, path: str | os.PathLike, *, read_only: bool = False):
        self.path = Path(path)
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.exists():
            raise StoreError(f"state database not found: {self.path}")
        mode = "ro" if read_only else "rwc"
        self.conn = sqlite3.connect(
            f"file:{self.path}?mode={mode}", uri=True, timeout=120.0,
            isolation_level=None,  # explicit transactions
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-524288")   # 512 MiB page cache
        self.conn.execute("PRAGMA mmap_size=1073741824")  # 1 GiB mmap
        self._read_only = read_only
        if not read_only:
            self.conn.executescript(_SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta (k, v) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _tx(self) -> "_Transaction":
        return _Transaction(self.conn)

    def _bump_revision(self) -> None:
        self.conn.execute(
            "INSERT INTO meta (k, v) VALUES ('revision', '1') "
            "ON CONFLICT(k) DO UPDATE SET v = CAST(CAST(v AS INTEGER) + 1 AS TEXT)"
        )

    @property
    def revision(self) -> int:
        row = self.conn.execute("SELECT v FROM meta WHERE k = 'revision'").fetchone()
        return int(row["v"]) if row else 0

    # -- stats -------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        q = self.conn.execute
        return {
            "accounts": q("SELECT COUNT(*) FROM account").fetchone()[0],
            "entities": q("SELECT COUNT(*) FROM entity").fetchone()[0],
            "links": q("SELECT COUNT(*) FROM link").fetchone()[0],
            "link_days": q("SELECT COUNT(*) FROM link_day").fetchone()[0],
            "batches": q("SELECT COUNT(*) FROM batch").fetchone()[0],
        }

    def entity_type_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT type, COUNT(*) AS c FROM entity GROUP BY type ORDER BY type"
        )
        return {TYPE_NAMES[r["type"]]: r["c"] for r in rows}

    def day_range(self) -> tuple[int | None, int | None]:
        row = self.conn.execute(
            "SELECT MIN(day) AS a, MAX(day) AS b FROM day_ledger"
        ).fetchone()
        return row["a"], row["b"]

    def batches(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM batch ORDER BY imported_at DESC, rowid DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    # -- import ------------------------------------------------------------

    def import_records(
        self,
        records: Iterable[ParsedRecord],
        *,
        source: str,
        sha256: str | None = None,
        batch_id: str | None = None,
    ) -> ImportResult:
        """Import one file's worth of records in a single transaction."""
        if self._read_only:
            raise StoreError("store is read-only")
        if sha256:
            hit = self.conn.execute(
                "SELECT batch_id FROM batch WHERE sha256 = ?", (sha256,)
            ).fetchone()
            if hit:
                return ImportResult(batch_id=hit["batch_id"], skipped=True,
                                    n_records=0, n_links=0)

        batch_id = batch_id or (
            _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-" + os.urandom(3).hex()
        )
        st = _ImportState()
        try:
            with self._tx():
                chunk: list[tuple[str, list[str], int, int | None]] = []
                for rec in records:
                    chunk.append(self._prepare(rec))
                    if len(chunk) >= 10_000:
                        self._apply_chunk(chunk, st)
                        chunk = []
                if chunk:
                    self._apply_chunk(chunk, st)

                days = sorted(st.days_touched)
                self.conn.execute(
                    "INSERT INTO batch (batch_id, sha256, source, imported_at, n_records, "
                    "n_new_accounts, n_new_entities, days) VALUES (?,?,?,?,?,?,?,?)",
                    (batch_id, sha256 or "", source, datetime_now_utc(), st.records,
                     st.new_accounts, st.new_entities,
                     ",".join(day_to_str(d) for d in days)),
                )
                self._ledger_many(st, days)
                self._bump_revision()
        except sqlite3.Error as exc:
            raise StoreError(f"import failed: {exc}") from exc

        return ImportResult(batch_id=batch_id, skipped=False, n_records=st.records,
                            n_links=st.links_created + st.links_updated,
                            n_new_accounts=st.new_accounts,
                            n_new_entities=st.new_entities)

    def _ledger_many(self, st: "_ImportState", days: list[int]) -> None:
        self.conn.executemany(
            "INSERT INTO day_ledger (day, n_records, n_links, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(day) DO UPDATE SET n_records = n_records + excluded.n_records, "
            "n_links = n_links + excluded.n_links, updated_at = excluded.updated_at",
            [(d, st.day_records.get(d, 0), st.day_links.get(d, 0), datetime_now_utc())
             for d in days],
        )

    def _prepare(self, rec: ParsedRecord) -> tuple[str, list[str], int, int | None]:
        acct = normalize_value(ACCOUNT, rec.account)
        if acct is None:
            raise StoreError(f"record without an account value: {rec!r}")
        ak = entity_key(ACCOUNT, acct)
        eks: list[str] = []
        for etype, raw in rec.resources.items():
            val = normalize_value(etype, raw, ip_prefix=rec.ip_prefix,
                                  card_salt=rec.card_salt)
            if val is not None:
                eks.append(entity_key(etype, val))
        return ak, eks, rec.day, rec.signup_day

    def _apply_chunk(self, chunk, st: "_ImportState") -> None:
        acct_keys = sorted({c[0] for c in chunk})
        ent_keys = sorted({k for c in chunk for k in c[1]})
        acct_ids = self._intern_accounts(acct_keys, st)
        ent_ids = self._intern_entities(ent_keys, st)

        agg: dict[tuple[int, int, int], int] = {}
        acct_days: dict[int, set[int]] = {}
        acct_recs: dict[int, int] = {}
        ent_recs: dict[int, int] = {}
        ent_days: dict[int, set[int]] = {}
        signups: dict[int, int | None] = {}
        for ak, eks, day, signup in chunk:
            aid = acct_ids[ak]
            acct_recs[aid] = acct_recs.get(aid, 0) + 1
            acct_days.setdefault(aid, set()).add(day)
            if signup is not None:
                signups[aid] = signup
            st.records += 1
            st.days_touched.add(day)
            st.day_records[day] = st.day_records.get(day, 0) + 1
            for ek in eks:
                eid = ent_ids[ek]
                key = (aid, eid, day)
                agg[key] = agg.get(key, 0) + 1
                ent_recs[eid] = ent_recs.get(eid, 0) + 1
                ent_days.setdefault(eid, set()).add(day)

        # 1) day ledger
        self.conn.executemany(
            "INSERT INTO link_day (account_id, entity_id, day, n) VALUES (?,?,?,?) "
            "ON CONFLICT(account_id, entity_id, day) DO UPDATE SET n = n + excluded.n",
            [(a, e, d, n) for (a, e, d), n in agg.items()],
        )

        # 2) roll up every touched pair from the ledger in one statement
        pairs = sorted({(a, e) for (a, e, _d) in agg})
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _p (a INTEGER, e INTEGER, PRIMARY KEY(a,e)) WITHOUT ROWID")
        self.conn.execute("DELETE FROM _p")
        self.conn.executemany("INSERT INTO _p (a, e) VALUES (?,?)", pairs)
        self.conn.execute(
            "INSERT INTO link (account_id, entity_id, n, n_days, first_day, last_day) "
            "SELECT l.account_id, l.entity_id, SUM(l.n), COUNT(*), MIN(l.day), MAX(l.day) "
            "FROM link_day l JOIN _p p ON p.a = l.account_id AND p.e = l.entity_id "
            "GROUP BY l.account_id, l.entity_id "
            "ON CONFLICT(account_id, entity_id) DO UPDATE SET n = excluded.n, "
            "n_days = excluded.n_days, first_day = excluded.first_day, "
            "last_day = excluded.last_day"
        )
        st.links_updated += len(pairs)

        # 3) account stats.  signup_day is taken from whichever row carries it;
        # COALESCE(?, signup_day) keeps the first value seen for an account and
        # never lets a later file blank it out.
        self.conn.executemany(
            "UPDATE account SET first_day = MIN(COALESCE(first_day, ?), ?), "
            "last_day = MAX(COALESCE(last_day, ?), ?), "
            "signup_day = COALESCE(signup_day, ?), "
            "n_records = n_records + ? WHERE id = ?",
            [(min(ds), min(ds), max(ds), max(ds), signups.get(a), acct_recs[a], a)
             for a, ds in acct_days.items()],
        )
        # account activity days (distinct) are recomputed on demand in build()

        # 4) entity stats
        self.conn.executemany(
            "UPDATE entity SET first_day = MIN(COALESCE(first_day, ?), ?), "
            "last_day = MAX(COALESCE(last_day, ?), ?), n_records = n_records + ? "
            "WHERE id = ?",
            [(min(ds), min(ds), max(ds), max(ds), ent_recs[e], e)
             for e, ds in ent_days.items()],
        )
        self._refresh_entity_degrees(list(ent_recs))

    def _refresh_entity_degrees(self, entity_ids: list[int]) -> None:
        if not entity_ids:
            return
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _ents (id INTEGER PRIMARY KEY) "
                          "WITHOUT ROWID")
        self.conn.execute("DELETE FROM _ents")
        self.conn.executemany("INSERT OR IGNORE INTO _ents (id) VALUES (?)",
                              [(e,) for e in entity_ids])
        self.conn.execute(
            "UPDATE entity SET n_accounts = ("
            "  SELECT COUNT(*) FROM link l WHERE l.entity_id = entity.id"
            ") WHERE id IN (SELECT id FROM _ents)"
        )

    def _intern_accounts(self, keys: list[str], st: "_ImportState") -> dict[str, int]:
        if not keys:
            return {}
        out = self._lookup_ids("account", keys)
        missing = [k for k in keys if k not in out]
        if missing:
            self.conn.executemany(
                "INSERT OR IGNORE INTO account (key) VALUES (?)", [(k,) for k in missing]
            )
            found = self._lookup_ids("account", missing)
            out.update(found)
            st.new_accounts += len(found)
        return out

    def _intern_entities(self, keys: list[str], st: "_ImportState") -> dict[str, int]:
        if not keys:
            return {}
        out = self._entity_lookup(keys)
        missing = [k for k in keys if k not in out]
        if missing:
            self.conn.executemany(
                "INSERT OR IGNORE INTO entity (type, value) VALUES (?,?)",
                [self._split_key(k) for k in missing],
            )
            found = self._entity_lookup(missing)
            out.update(found)
            st.new_entities += len(found)
        return out

    def _entity_lookup(self, keys: Sequence[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in range(0, len(keys), _IN_CHUNK):
            part = [self._split_key(k) for k in keys[i:i + _IN_CHUNK]]
            marks = ",".join(["(?,?)"] * len(part))
            flat = [v for pair in part for v in pair]
            for row in self.conn.execute(
                f"SELECT id, type, value FROM entity WHERE (type, value) IN ({marks})",
                flat,
            ):
                out[entity_key(int(row["type"]), row["value"])] = row["id"]
        return out

    def _lookup_ids(self, table: str, keys: Sequence[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in range(0, len(keys), _IN_CHUNK):
            part = keys[i:i + _IN_CHUNK]
            marks = ",".join("?" * len(part))
            for row in self.conn.execute(
                f"SELECT id, key FROM {table} WHERE key IN ({marks})", part
            ):
                out[row["key"]] = row["id"]
        return out

    # -- day replacement / deletion ----------------------------------------

    def delete_day(self, day: int) -> None:
        """Remove one day from the ledger and re-derive the affected links.

        Accounts and entities that end up with *no* links are intentionally left
        in place: they are still real accounts in the source data, they cost
        nothing, and an entity that reappears in a later file must resolve to
        the same id.  The graph build selects on ``link``, so an orphan never
        reaches the analysis.
        """
        if self._read_only:
            raise StoreError("store is read-only")
        with self._tx():
            pairs = self.conn.execute(
                "SELECT DISTINCT account_id, entity_id FROM link_day WHERE day = ?", (day,)
            ).fetchall()
            self.conn.execute("DELETE FROM link_day WHERE day = ?", (day,))
            self.conn.execute("DELETE FROM day_ledger WHERE day = ?", (day,))
            self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _p (a INTEGER, e INTEGER, "
                              "PRIMARY KEY(a,e)) WITHOUT ROWID")
            self.conn.execute("DELETE FROM _p")
            self.conn.executemany("INSERT INTO _p (a, e) VALUES (?,?)",
                                  [(r[0], r[1]) for r in pairs])
            self.conn.execute(
                "DELETE FROM link WHERE EXISTS ("
                "SELECT 1 FROM _p p WHERE p.a = link.account_id AND p.e = link.entity_id)"
            )
            self.conn.execute(
                "INSERT INTO link (account_id, entity_id, n, n_days, first_day, last_day) "
                "SELECT account_id, entity_id, SUM(n), COUNT(*), MIN(day), MAX(day) "
                "FROM link_day l JOIN _p p ON p.a = l.account_id AND p.e = l.entity_id "
                "GROUP BY account_id, entity_id"
            )
            # Recompute account/entity stats touched by this day.  Cheaper to
            # rebuild them wholesale than to track deltas, and deletes are rare.
            self._rebuild_entity_stats()
            self._rebuild_account_stats()
            self._bump_revision()

    def _rebuild_entity_stats(self) -> None:
        self.conn.execute(
            "UPDATE entity SET first_day = NULL, last_day = NULL, n_accounts = 0, n_records = 0"
        )
        self.conn.execute(
            "UPDATE entity SET n_accounts = COALESCE(("
            "  SELECT COUNT(*) FROM link l WHERE l.entity_id = entity.id), 0), "
            "first_day = (SELECT MIN(first_day) FROM link l WHERE l.entity_id = entity.id), "
            "last_day = (SELECT MAX(last_day) FROM link l WHERE l.entity_id = entity.id), "
            "n_records = COALESCE((SELECT SUM(n) FROM link l WHERE l.entity_id = entity.id), 0)"
        )

    def _rebuild_account_stats(self) -> None:
        self.conn.execute(
            "UPDATE account SET first_day = NULL, last_day = NULL, n_records = 0"
        )
        self.conn.execute(
            "UPDATE account SET first_day = (SELECT MIN(first_day) FROM link l "
            "  WHERE l.account_id = account.id), "
            "last_day = (SELECT MAX(last_day) FROM link l WHERE l.account_id = account.id), "
            "n_records = COALESCE((SELECT SUM(n) FROM link l WHERE l.account_id = account.id), 0)"
        )

    # -- read paths ---------------------------------------------------------

    def link_arrays(self) -> tuple:
        """Return numpy int64 arrays (account_id, entity_id, n, n_days, first, last)."""
        import numpy as np

        n = self.conn.execute("SELECT COUNT(*) FROM link").fetchone()[0]
        if n == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z, z, z, z
        cur = self.conn.execute(
            "SELECT account_id, entity_id, n, n_days, first_day, last_day FROM link"
        )
        buf = np.fromiter((v for row in cur for v in row), dtype=np.int64, count=n * 6)
        return tuple(buf[i::6].copy() for i in range(6))

    def entity_meta(self) -> tuple:
        """Arrays (entity_id, type, value_str_list, n_accounts)."""
        rows = self.conn.execute(
            "SELECT id, type, value, n_accounts FROM entity ORDER BY id"
        ).fetchall()
        import numpy as np

        n = len(rows)
        ids = np.empty(n, dtype=np.int64)
        types = np.empty(n, dtype=np.int8)
        deg = np.empty(n, dtype=np.int32)
        values: list[str] = [""] * n
        for i, r in enumerate(rows):
            ids[i] = r["id"]
            types[i] = r["type"]
            values[r["id"]] = r["value"]
            deg[i] = r["n_accounts"]
        return ids, types, values, deg

    def account_meta(self) -> tuple:
        """Arrays (account_id, n_records, first_day, last_day, n_distinct_days)."""
        import numpy as np

        rows = self.conn.execute(
            "SELECT a.id, a.n_records, a.first_day, a.last_day, "
            "  (SELECT COUNT(DISTINCT day) FROM link_day d WHERE d.account_id = a.id) AS nd "
            "FROM account a ORDER BY a.id"
        ).fetchall()
        n = len(rows)
        ids = np.empty(n, dtype=np.int64)
        recs = np.empty(n, dtype=np.int32)
        first = np.empty(n, dtype=np.int32)
        last = np.empty(n, dtype=np.int32)
        ndays = np.empty(n, dtype=np.int32)
        for i, r in enumerate(rows):
            ids[i] = r[0]
            recs[i] = r[1] or 0
            first[i] = r[2] if r[2] is not None else -1
            last[i] = r[3] if r[3] is not None else -1
            ndays[i] = r[4] or 0
        return ids, recs, first, last, ndays

    def accounts_by_id(self, ids: Sequence[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        idlist = list(ids)
        for i in range(0, len(idlist), _IN_CHUNK):
            part = idlist[i:i + _IN_CHUNK]
            marks = ",".join("?" * len(part))
            for row in self.conn.execute(
                "SELECT * FROM account WHERE id IN (%s)" % marks, part
            ):
                out[row["id"]] = dict(row)
        return out

    def entities_by_id(self, ids: Sequence[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        idlist = list(ids)
        for i in range(0, len(idlist), _IN_CHUNK):
            part = idlist[i:i + _IN_CHUNK]
            marks = ",".join("?" * len(part))
            for row in self.conn.execute(
                "SELECT * FROM entity WHERE id IN (%s)" % marks, part
            ):
                out[row["id"]] = dict(row)
        return out

    def links_for_accounts(self, account_ids: Sequence[int]) -> list[tuple]:
        rows: list[tuple] = []
        ids = list(account_ids)
        for i in range(0, len(ids), _IN_CHUNK):
            part = ids[i:i + _IN_CHUNK]
            marks = ",".join("?" * len(part))
            rows.extend(
                tuple(r) for r in self.conn.execute(
                    "SELECT account_id, entity_id, n, n_days, first_day, last_day "
                    "FROM link WHERE account_id IN (%s)" % marks, part
                )
            )
        return rows

    def accounts_for_entities(self, entity_ids: Sequence[int]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {e: [] for e in entity_ids}
        ids = list(entity_ids)
        for i in range(0, len(ids), _IN_CHUNK):
            part = ids[i:i + _IN_CHUNK]
            marks = ",".join("?" * len(part))
            for r in self.conn.execute(
                "SELECT entity_id, account_id FROM link WHERE entity_id IN (%s)" % marks, part
            ):
                out[r[0]].append(r[1])
        return out

    def account_keys(self, ids: Sequence[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        idlist = list(ids)
        for i in range(0, len(idlist), _IN_CHUNK):
            part = idlist[i:i + _IN_CHUNK]
            marks = ",".join("?" * len(part))
            for row in self.conn.execute(
                "SELECT id, key FROM account WHERE id IN (%s)" % marks, part
            ):
                out[row["id"]] = row["key"]
        return out


    # -- batch lookups used by the result writer ---------------------------

    def account_meta_by_id(self, ids: Sequence[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        idlist = sorted(set(int(i) for i in ids))
        for i in range(0, len(idlist), _IN_CHUNK):
            part = idlist[i:i + _IN_CHUNK]
            marks = ",".join("?" * len(part))
            for row in self.conn.execute(
                f"SELECT id, key, first_day, last_day, signup_day, n_records "
                f"FROM account WHERE id IN ({marks})", part
            ):
                out[row["id"]] = dict(row)
        return out

    @staticmethod
    def _split_key(key: str) -> tuple[int, str]:
        tname, value = key.split("\x1f", 1)
        return TYPE_BY_NAME[tname], value


class _Transaction:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.conn.execute("COMMIT" if exc_type is None else "ROLLBACK")
        return False


@dataclass
class _ImportState:
    records: int = 0
    links_created: int = 0
    links_updated: int = 0
    new_accounts: int = 0
    new_entities: int = 0

    def __post_init__(self):
        self.days_touched: set[int] = set()
        self.day_records: dict[int, int] = {}
        self.day_links: dict[int, int] = {}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
