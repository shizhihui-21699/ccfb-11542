"""Readers for daily data files (CSV/TSV/JSONL, plain or gzip).

Column names vary between shops, so every field is mapped explicitly.  The
defaults cover the column names our generator writes; anything else is
``--account-col`` etc. on the command line.

Deliberate behaviours:

* unknown columns are ignored (daily exports grow columns over time);
* a row missing the account id is *counted and reported*, not silently dropped;
* a row whose day cannot be parsed aborts the import -- a wrong day is worse
  than no import, because it corrupts the temporal evidence;
* the file's sha256 is returned so the store can make re-imports idempotent.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .schema import (
    ADDRESS,
    CARD,
    DEVICE,
    IP,
)
from .store import ParsedRecord, StoreError, to_day_int

_RESOURCE_KEYS = {
    DEVICE: ("device", "device_id", "imei", "idfa", "oaid", "mac", "fingerprint"),
    CARD: ("card", "card_id", "card_no", "pan", "bank_card", "payment_card"),
    ADDRESS: ("address", "addr", "ship_address", "receiver_address", "address_id"),
    IP: ("ip", "ip_addr", "client_ip", "login_ip"),
}
_ACCOUNT_KEYS = ("account", "account_id", "user_id", "uid", "buyer_id", "member_id")
_DAY_KEYS = ("day", "date", "order_date", "dt", "stat_date", "event_date", "created_at")
_SIGNUP_KEYS = ("signup_day", "signup_date", "register_date", "reg_date", "register_time")


class ParseError(StoreError):
    pass


@dataclass
class FieldMap:
    account: str = "account_id"
    day: str | None = None          # auto-detected when None
    signup: str | None = None
    device: str | None = "device_id"
    card: str | None = "card_id"
    address: str | None = "address_id"
    ip: str | None = "ip"

    def columns(self) -> dict[int, str]:
        out = {}
        for etype, col in ((DEVICE, self.device), (CARD, self.card),
                           (ADDRESS, self.address), (IP, self.ip)):
            if col:
                out[etype] = col
        return out

    def resolve(self, headers: list[str]) -> "FieldMap":
        """Fill in only the unset columns from the header row.

        A column explicitly named in the configuration always wins; ``None``
        means "look for one of the known aliases".
        """
        lower = {h.strip().lower(): h for h in headers}

        def pick(explicit, candidates, default):
            if explicit:
                return explicit
            for c in candidates:
                if c in lower:
                    return lower[c]
            return default

        return FieldMap(
            account=self.account or pick(None, _ACCOUNT_KEYS, "account_id"),
            day=pick(self.day, _DAY_KEYS, None),
            signup=pick(self.signup, _SIGNUP_KEYS, None),
            device=pick(self.device, _RESOURCE_KEYS[DEVICE], None),
            card=pick(self.card, _RESOURCE_KEYS[CARD], None),
            address=pick(self.address, _RESOURCE_KEYS[ADDRESS], None),
            ip=pick(self.ip, _RESOURCE_KEYS[IP], None),
        )

    @classmethod
    def from_headers(cls, headers: list[str], *, account: str | None = None,
                     day: str | None = None, signup: str | None = None,
                     device: str | None = None, card: str | None = None,
                     address: str | None = None, ip: str | None = None) -> "FieldMap":
        """Fill unset columns from the header row, so defaults are permissive."""
        lower = {h.strip().lower(): h for h in headers}

        def pick(explicit, candidates, default):
            if explicit is not None:
                return explicit
            for c in candidates:
                if c in lower:
                    return lower[c]
            return default

        fm = cls()
        fm.account = pick(account, _ACCOUNT_KEYS, "account_id")
        fm.day = pick(day, _DAY_KEYS, None)
        fm.signup = pick(signup, _SIGNUP_KEYS, None)
        fm.device = pick(device, _RESOURCE_KEYS[DEVICE], None)
        fm.card = pick(card, _RESOURCE_KEYS[CARD], None)
        fm.address = pick(address, _RESOURCE_KEYS[ADDRESS], None)
        fm.ip = pick(ip, _RESOURCE_KEYS[IP], None)
        return fm

    def validate(self) -> None:
        if not self.account:
            raise ParseError("no account column configured")
        if not self.columns():
            raise ParseError("no resource column configured (device/card/address/ip)")


@dataclass
class ParseStats:
    rows: int = 0
    kept: int = 0
    missing_account: int = 0
    no_resource: int = 0
    bad_day: int = 0
    errors: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "rows": self.rows, "kept": self.kept,
            "missing_account": self.missing_account,
            "no_resource": self.no_resource, "bad_day": self.bad_day,
        }


def iter_records(path: str | os.PathLike, *, field_map: FieldMap | None = None,
                 default_day: int | None = None, ip_prefix: int = 0,
                 card_salt: bytes | None = None,
                 stats: ParseStats | None = None,
                 limit: int | None = None) -> Iterator[ParsedRecord]:
    """Yield ParsedRecord for every usable row of ``path``.

    ``default_day`` supplies the day for files with no date column (the usual
    case for a per-day export whose name carries the date).
    """
    path = Path(path)
    stats = stats if stats is not None else ParseStats()
    suffix = path.name.lower()
    opener = gzip.open if suffix.endswith((".gz", ".gzip")) else open

    if suffix.endswith((".jsonl", ".json", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
        yield from _iter_jsonl(path, opener, field_map, default_day, ip_prefix,
                               card_salt, stats, limit)
    else:
        yield from _iter_csv(path, opener, field_map, default_day, ip_prefix,
                             card_salt, stats, limit)


def _iter_csv(path, opener, field_map, default_day, ip_prefix, card_salt, stats, limit):
    with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(64 * 1024)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        if not reader.fieldnames:
            raise ParseError(f"{path}: empty file")
        fm = field_map or FieldMap.from_headers(reader.fieldnames)
        fm = fm.resolve(reader.fieldnames)
        fm.validate()
        cols = fm.columns()
        for row in reader:
            stats.rows += 1
            if limit is not None and stats.kept >= limit:
                return
            rec = _row_to_record(row, fm, cols, stats, default_day, ip_prefix, card_salt)
            if rec is not None:
                stats.kept += 1
                yield rec


def _iter_jsonl(path, opener, field_map, default_day, ip_prefix, card_salt, stats, limit):
    fm = field_map
    with opener(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            stats.rows += 1
            if limit is not None and stats.kept >= limit:
                return
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if fm is None:
                fm = FieldMap().resolve(list(obj.keys()))
                fm.validate()
            cols = fm.columns()
            rec = _row_to_record(obj, fm, cols, stats, default_day, ip_prefix, card_salt,
                                 line_no=line_no)
            if rec is not None:
                stats.kept += 1
                yield rec


def _row_to_record(row, fm: FieldMap, cols, stats: ParseStats, default_day,
                   ip_prefix, card_salt, line_no: int | None = None) -> ParsedRecord | None:
    account = _get(row, fm.account)
    if account is None or str(account).strip() == "":
        stats.missing_account += 1
        return None

    day_raw = _get(row, fm.day) if fm.day else None
    if day_raw is None or str(day_raw).strip() == "":
        if default_day is None:
            raise ParseError(
                "row has no day value and no --day/--default-day was given"
                + (f" (line {line_no})" if line_no else "")
            )
        day = default_day
    else:
        try:
            day = to_day_int(day_raw)
        except StoreError as exc:
            stats.bad_day += 1
            raise ParseError(f"bad day {day_raw!r} (line {line_no})") from exc

    resources = {}
    for etype, col in cols.items():
        val = _get(row, col)
        if val is not None and str(val).strip() != "":
            resources[etype] = val
    if not resources:
        stats.no_resource += 1
        return None

    signup = None
    if fm.signup:
        raw = _get(row, fm.signup)
        if raw is not None and str(raw).strip():
            try:
                signup = to_day_int(raw)
            except StoreError:
                signup = None

    return ParsedRecord(account=account, day=day, resources=resources,
                        signup_day=signup, ip_prefix=ip_prefix, card_salt=card_salt)


def _get(row, key: str | None):
    if not key:
        return None
    try:
        return row.get(key)
    except AttributeError:
        return None


def load_field_map(spec: dict) -> FieldMap:
    """Build a FieldMap from CLI-style kwargs (None means auto-detect)."""
    return FieldMap(
        account=spec.get("account") or "account_id",
        day=spec.get("day"),
        signup=spec.get("signup"),
        device=spec.get("device", "device_id"),
        card=spec.get("card", "card_id"),
        address=spec.get("address", "address_id"),
        ip=spec.get("ip", "ip"),
    )
