"""Correctness tests for normalization, day parsing, and the store.

These run without generating data, so they are fast enough for a pre-commit.
"""

from __future__ import annotations

import datetime as dt

import pytest

from ringdetect.schema import (
    ACCOUNT, ADDRESS, CARD, DEVICE, IP, ConfigError, entity_key, normalize_value,
)
from ringdetect.store import (
    ParsedRecord, Store, StoreError, day_to_str, to_day_int,
)
from ringdetect.ingest import FieldMap, ParseError, ParseStats, iter_records


# ---------------------------------------------------------------- schema

def test_normalize_lowercases_and_collapses_space():
    assert normalize_value(ACCOUNT, "  U123  ABC ") == "u123 abc"
    assert normalize_value(DEVICE, "DEV-1") == "dev-1"
    assert normalize_value(ADDRESS, " 1 Main St. , ") == "1 main st."


def test_normalize_unicode_width():
    # fullwidth characters must fold to ASCII, or the same device splits in two
    assert normalize_value(DEVICE, "ＤＥＶ－１") == "dev-1"


def test_empty_values_are_none():
    for etype in (ACCOUNT, DEVICE, CARD, ADDRESS, IP):
        assert normalize_value(etype, None) is None
        assert normalize_value(etype, "   ") is None


def test_card_is_hashed_and_stable():
    salt = b"pepper"
    a = normalize_value(CARD, "4111111111111111", card_salt=salt)
    b = normalize_value(CARD, "4111111111111111", card_salt=salt)
    c = normalize_value(CARD, "4111111111111111", card_salt=b"other")
    assert a == b and a != c
    assert "4111" not in a
    assert a.startswith("pan:")


def test_raw_card_without_salt_is_refused():
    with pytest.raises(ConfigError):
        normalize_value(CARD, "4111111111111111")


def test_tokenized_card_passes_through():
    assert normalize_value(CARD, "tok_abc") == "tok_abc"


def test_ip_normalization_and_prefix():
    assert normalize_value(IP, "10.1.2.3") == "10.1.2.3"
    assert normalize_value(IP, "  10.1.2.3 ") == "10.1.2.3"
    assert normalize_value(IP, "2001:DB8::1") == "2001:db8::1"
    assert normalize_value(IP, "10.1.2.3", ip_prefix=24) == "10.1.2.0/24"
    assert normalize_value(IP, "not-an-ip") == "not-an-ip"


def test_entity_key_is_type_scoped():
    assert entity_key(DEVICE, "x") != entity_key(CARD, "x")


# ------------------------------------------------------------- day parsing

@pytest.mark.parametrize("value", [
    "2026-09-01", "20260901", "2026/09/01", "2026-09-01T12:00:00Z",
    "2026-09-01 12:00:00", dt.date(2026, 9, 1),
    dt.datetime(2026, 9, 1, 6, 30),
])
def test_to_day_int_variants_agree(value):
    assert to_day_int(value) == (dt.date(2026, 9, 1) - dt.date(1970, 1, 1)).days


def test_day_int_roundtrip():
    d = to_day_int("2024-02-29")
    assert day_to_str(d) == "2024-02-29"


def test_epoch_seconds_and_millis():
    sec = to_day_int(1700000000)
    ms = to_day_int(1700000000000)
    assert sec == ms


def test_bad_day_raises():
    with pytest.raises(StoreError):
        to_day_int("not a date")
    with pytest.raises(StoreError):
        to_day_int("")


# ------------------------------------------------------------ field mapping

def test_field_map_autodetects_generator_headers():
    fm = FieldMap(account=None, device="device_id", card="card_id",
                 address="address_id", ip="ip")
    resolved = fm.resolve(["account_id", "device_id", "card_id",
                           "address_id", "ip", "signup_date", "event_date"])
    assert resolved.account == "account_id"
    assert resolved.day == "event_date"
    assert resolved.signup == "signup_date"
    assert resolved.columns() == {DEVICE: "device_id", CARD: "card_id",
                                  ADDRESS: "address_id", IP: "ip"}


def test_field_map_rejects_no_resources():
    with pytest.raises(ParseError):
        FieldMap(account="a", device=None, card=None, address=None,
                 ip=None).validate()


# ---------------------------------------------------------------- store

def _rec(account="u1", day=1, resources=None):
    return ParsedRecord(account=account, day=day, resources=dict(resources or {}))


def test_import_and_counts(tmp_path):
    with Store(tmp_path / "s.db") as st:
        res = st.import_records(
            [_rec("u1", 1, {DEVICE: "d1", IP: "1.1.1.1"}),
             _rec("u2", 1, {DEVICE: "d1"}),
             _rec("u1", 2, {DEVICE: "d1"})],
            source="x.csv", sha256="abc")
        assert res.n_records == 3
        counts = st.counts()
        assert counts["accounts"] == 2
        assert counts["entities"] == 2
        # u1-d1, u2-d1, u1-1.1.1.1
        assert counts["links"] == 3


def test_reimport_same_sha_is_noop(tmp_path):
    with Store(tmp_path / "s.db") as st:
        recs = [_rec("u1", 1, {DEVICE: "d1"})]
        st.import_records(recs, source="a.csv", sha256="same")
        before = st.counts()
        res = st.import_records([_rec("u9", 1, {DEVICE: "d9"})],
                                source="a.csv", sha256="same")
        assert res.skipped
        assert st.counts() == before


def test_link_rollup_tracks_days_and_span(tmp_path):
    with Store(tmp_path / "s.db") as st:
        st.import_records([
            _rec("u1", 1, {DEVICE: "d1"}),
            _rec("u1", 1, {DEVICE: "d1"}),
            _rec("u1", 5, {DEVICE: "d1"}),
        ], source="a.csv", sha256="a")
        row = st.conn.execute(
            "SELECT n, n_days, first_day, last_day FROM link").fetchone()
        assert tuple(row) == (3, 2, 1, 5)


def test_link_day_ledger_is_additive_across_files(tmp_path):
    with Store(tmp_path / "s.db") as st:
        st.import_records([_rec("u1", 3, {DEVICE: "d1"})], source="a", sha256="a")
        st.import_records([_rec("u1", 3, {DEVICE: "d1"})], source="b", sha256="b")
        n = st.conn.execute("SELECT SUM(n) FROM link_day").fetchone()[0]
        assert n == 2


def test_delete_day_restores_prior_state(tmp_path):
    with Store(tmp_path / "s.db") as st:
        st.import_records([_rec("u1", 1, {DEVICE: "d1"})], source="a", sha256="a")
        st.import_records([_rec("u1", 2, {DEVICE: "d1"})], source="b", sha256="b")
        st.delete_day(2)
        row = st.conn.execute(
            "SELECT n, n_days, first_day, last_day FROM link").fetchone()
        assert tuple(row) == (1, 1, 1, 1)


def test_delete_day_removes_link_when_last_day_goes(tmp_path):
    with Store(tmp_path / "s.db") as st:
        st.import_records([_rec("u1", 1, {DEVICE: "d1"})], source="a", sha256="a")
        st.delete_day(1)
        assert st.conn.execute("SELECT COUNT(*) FROM link").fetchone()[0] == 0


def test_signup_day_persists_and_sticks(tmp_path):
    with Store(tmp_path / "s.db") as st:
        st.import_records([ParsedRecord("u1", 5, {DEVICE: "d1"}, signup_day=0)],
                          source="a", sha256="a")
        st.import_records([ParsedRecord("u1", 6, {DEVICE: "d1"}, signup_day=None)],
                          source="b", sha256="b")
        row = st.conn.execute("SELECT signup_day FROM account").fetchone()
        assert row[0] == 0


def test_missing_account_is_counted_not_silently_dropped(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("account_id,device_id,event_date\n"
                    "u1,d1,2026-09-01\n"
                    ",d2,2026-09-01\n"
                    "u3,,2026-09-01\n")
    stats = ParseStats()
    recs = list(iter_records(path, stats=stats))
    assert stats.rows == 3
    assert stats.missing_account == 1
    assert stats.no_resource == 1
    assert len(recs) == 1


def test_bad_day_aborts_rather_than_mis_dating(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("account_id,device_id,event_date\nu1,d1,2026-13-99\n")
    with pytest.raises(ParseError):
        list(iter_records(path))


def test_jsonl_input(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"account_id":"u1","device_id":"d1","event_date":"2026-09-01"}\n'
        '{"account_id":"u2","device_id":"d1","event_date":"2026-09-02"}\n')
    recs = list(iter_records(path))
    assert len(recs) == 2
    assert recs[0].day != recs[1].day


def test_gzip_input(tmp_path):
    import gzip
    path = tmp_path / "d.csv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("account_id,device_id,event_date\nu1,d1,2026-09-01\n")
    assert len(list(iter_records(path))) == 1


def test_store_is_read_only_when_asked(tmp_path):
    p = tmp_path / "s.db"
    Store(p).close()
    with Store(p, read_only=True) as st:
        with pytest.raises(StoreError):
            st.import_records([_rec()], source="x", sha256="y")
