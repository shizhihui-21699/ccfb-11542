"""SQLite 存储 + 每日数据增量导入 (幂等: 同一文件重复导入自动跳过)."""
import os
import sqlite3

import pandas as pd

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS accounts(
    account_id TEXT PRIMARY KEY,
    first_seen TEXT, last_seen TEXT);
CREATE TABLE IF NOT EXISTS links(
    account_id TEXT, entity_type TEXT, entity_id TEXT,
    first_seen TEXT, last_seen TEXT, n_days INTEGER,
    PRIMARY KEY(account_id, entity_type, entity_id));
CREATE INDEX IF NOT EXISTS idx_links_entity ON links(entity_type, entity_id);
CREATE TABLE IF NOT EXISTS coupons(
    account_id TEXT, coupon_id TEXT, day TEXT, amount REAL,
    PRIMARY KEY(account_id, coupon_id, day));
CREATE TABLE IF NOT EXISTS ingested_files(path TEXT PRIMARY KEY, n_rows INTEGER);
-- 分析结果表 (analyze 时重建)
CREATE TABLE IF NOT EXISTS entity_stats(
    entity_type TEXT, entity_id TEXT, df INTEGER,
    PRIMARY KEY(entity_type, entity_id));
CREATE TABLE IF NOT EXISTS proj_edges(a TEXT, b TEXT, weight REAL);
CREATE INDEX IF NOT EXISTS idx_proj_a ON proj_edges(a);
CREATE INDEX IF NOT EXISTS idx_proj_b ON proj_edges(b);
CREATE TABLE IF NOT EXISTS communities(
    id INTEGER PRIMARY KEY, size INTEGER, n_edges INTEGER,
    avg_weight REAL, density REAL, coupon_total INTEGER,
    coupon_lift REAL, score REAL, summary TEXT);
CREATE TABLE IF NOT EXISTS community_members(
    community_id INTEGER, account_id TEXT,
    PRIMARY KEY(community_id, account_id));
CREATE TABLE IF NOT EXISTS evidence(
    community_id INTEGER, entity_type TEXT, entity_id TEXT,
    df INTEGER, members INTEGER, weight REAL);
CREATE TABLE IF NOT EXISTS evidence_edges(
    community_id INTEGER, a TEXT, b TEXT, weight REAL, shared TEXT);
"""


def connect(db_path):
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    return db


def _ingest_accounts(db, df, day):
    rows = [(r.account_id, day, day) for r in df.itertuples()]
    db.executemany(
        "INSERT INTO accounts(account_id, first_seen, last_seen) VALUES(?,?,?) "
        "ON CONFLICT(account_id) DO UPDATE SET last_seen=MAX(last_seen, excluded.last_seen)",
        rows)
    return len(rows)


def _ingest_links(db, df, day):
    rows = [(r.account_id, r.entity_type, r.entity_id, day, day)
            for r in df.itertuples()]
    db.executemany(
        "INSERT INTO links(account_id, entity_type, entity_id, first_seen, last_seen, n_days)"
        " VALUES(?,?,?,?,?,1) "
        "ON CONFLICT(account_id, entity_type, entity_id) DO UPDATE SET "
        "last_seen=MAX(last_seen, excluded.last_seen), n_days=n_days+1",
        rows)
    return len(rows)


def _ingest_coupons(db, df, day):
    rows = [(r.account_id, r.coupon_id, str(r.day), float(r.amount))
            for r in df.itertuples()]
    db.executemany(
        "INSERT OR IGNORE INTO coupons(account_id, coupon_id, day, amount)"
        " VALUES(?,?,?,?)", rows)
    return len(rows)


INGESTERS = {"accounts": _ingest_accounts, "links": _ingest_links,
             "coupons": _ingest_coupons}


def ingest_file(db, path, day):
    """导入单个 csv; 已导入过的文件直接跳过."""
    if db.execute("SELECT 1 FROM ingested_files WHERE path=?", (path,)).fetchone():
        return 0
    kind = os.path.basename(path).replace(".csv", "")
    n = 0
    for chunk in pd.read_csv(path, chunksize=200_000, dtype=str):
        n += INGESTERS[kind](db, chunk, day)
    db.execute("INSERT INTO ingested_files(path, n_rows) VALUES(?,?)", (path, n))
    db.commit()
    return n


def ingest_dir(db, incoming_dir):
    """扫描 incoming/day_* 目录, 导入所有新文件. 返回 (文件数, 行数)."""
    n_files = n_rows = 0
    for day_dir in sorted(os.listdir(incoming_dir)):
        dpath = os.path.join(incoming_dir, day_dir)
        if not os.path.isdir(dpath) or not day_dir.startswith("day_"):
            continue
        day = day_dir.removeprefix("day_")
        for fname in sorted(os.listdir(dpath)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(dpath, fname)
            n = ingest_file(db, fpath, day)
            if n:
                n_files += 1
                n_rows += n
                print(f"[ingest] {fpath}: {n} rows")
    return n_files, n_rows
