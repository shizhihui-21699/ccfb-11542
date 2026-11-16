"""Read-only HTTP API over a built result set.

Deliberately stdlib-only (``http.server``): the deployment story is "copy the
result directory to the analyst's laptop and run one command".  Adding a web
framework would mean a dependency tree next to data that is often sensitive.

Endpoints
---------
``GET /``                     the viewer page
``GET /api/summary``          summary.json plus a score histogram
``GET /api/rings?…``          ranked ring list with filters and paging
``GET /api/rings/<id>``       one ring with members, evidence, bridges
``GET /api/rings/<id>/graph`` bipartite / account graph for the ring view
``GET /api/accounts/<key>``   every ring an account appears in
``GET /api/entities?q=``      prefix/contains search over evidence entities
``GET /api/batches``          import history from the state database
``GET /api/health``           liveness + what the server can see
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .schema import TYPE_NAMES
from .store import day_to_str

WEB_DIR = Path(__file__).parent / "web"
MAX_PAGE = 500


class DataSource:
    """Lazily-opened, read-only handles onto the result set and state DB."""

    def __init__(self, out_dir: Path, state_path: Path | None = None):
        self.out_dir = Path(out_dir)
        self.results_path = self.out_dir / "results.sqlite"
        if not self.results_path.exists():
            raise FileNotFoundError(
                f"{self.results_path} not found -- run `ringdetect build` first")
        self.state_path = Path(state_path) if state_path else None
        if self.state_path and not self.state_path.exists():
            self.state_path = None
        self._local = threading.local()
        self._summary = json.loads((self.out_dir / "summary.json").read_text()) \
            if (self.out_dir / "summary.json").exists() else {}
        self._hist = None

    # -- connections -------------------------------------------------------

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(f"file:{self.results_path}?mode=ro", uri=True,
                                timeout=10.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    def state(self) -> sqlite3.Connection | None:
        if not self.state_path:
            return None
        c = getattr(self._local, "state_conn", None)
        if c is None:
            c = sqlite3.connect(f"file:{self.state_path}?mode=ro", uri=True,
                                timeout=10.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            self._local.state_conn = c
        return c

    # -- queries -----------------------------------------------------------

    def summary(self) -> dict:
        out = dict(self._summary)
        out["histogram"] = self.histogram()
        st = self.state()
        out["has_state"] = st is not None
        # Surface a stale result set: importing more days without rebuilding
        # leaves the rings behind the data, and an analyst reading old numbers
        # without knowing it is the worst failure mode this tool can have.
        out["stale"] = False
        if st is not None:
            row = st.execute("SELECT v FROM meta WHERE k = 'revision'").fetchone()
            state_rev = int(row[0]) if row else None
            built_rev = out.get("state_revision")
            if state_rev is not None and built_rev is not None and state_rev != built_rev:
                out["stale"] = True
                out["state_revision_now"] = state_rev
        return out

    def histogram(self) -> list[dict]:
        if self._hist is None:
            rows = self.conn().execute(
                "SELECT CAST(score / 5 AS INTEGER) * 5 AS bucket, COUNT(*) n, "
                "SUM(n_accounts) accounts FROM ring GROUP BY bucket ORDER BY bucket"
            ).fetchall()
            self._hist = [{"score": r[0], "rings": r[1], "accounts": r[2] or 0}
                          for r in rows]
        return self._hist

    def rings(self, *, min_score: float = 0.0, min_size: int = 2,
              max_size: int | None = None, sort: str = "score",
              limit: int = 50, offset: int = 0, weak_only: bool | None = None,
              q: str | None = None) -> dict:
        where = ["score >= ?", "n_accounts >= ?"]
        args: list = [min_score, min_size]
        if max_size:
            where.append("n_accounts <= ?")
            args.append(max_size)
        if weak_only is not None:
            where.append("weak_only = ?")
            args.append(int(weak_only))
        if q:
            where.append("id IN (SELECT ring_id FROM ring_entity WHERE value LIKE ?)")
            args.append(f"%{q}%")
        order = {"score": "score DESC, n_accounts DESC",
                 "size": "n_accounts DESC, score DESC",
                 "recent": "last_day DESC, score DESC"}.get(sort, "score DESC")
        clause = " AND ".join(where)
        c = self.conn()
        total = c.execute(f"SELECT COUNT(*) FROM ring WHERE {clause}", args).fetchone()[0]
        rows = c.execute(
            f"SELECT id, score, n_accounts, cartel_size, cartel_mass, closure, "
            f"weak_only, type_counts, first_day, last_day, signup_med_day "
            f"FROM ring WHERE {clause} ORDER BY {order} LIMIT ? OFFSET ?",
            args + [min(limit, MAX_PAGE), offset],
        ).fetchall()
        items = []
        for r in rows:
            d = dict(r)
            d["first_day"] = _day(d["first_day"])
            d["last_day"] = _day(d["last_day"])
            d["signup_med_day"] = _day(d["signup_med_day"])
            d["type_counts"] = json.loads(d["type_counts"] or "{}")
            items.append(d)
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    def ring(self, ring_id: int) -> dict | None:
        c = self.conn()
        row = c.execute("SELECT * FROM ring WHERE id = ?", (ring_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["parts"] = json.loads(d["parts"] or "{}")
        d["type_counts"] = json.loads(d["type_counts"] or "{}")
        for k in ("first_day", "last_day", "signup_med_day"):
            d[k] = _day(d.get(k))

        d["members"] = [
            {**dict(m), **{k: _day(m[k]) for k in ("first_day", "last_day", "signup_day")}}
            for m in c.execute(
                "SELECT account_key, account_id, n_records, first_day, last_day, "
                "signup_day, n_strong, n_weak FROM ring_member WHERE ring_id = ? "
                "ORDER BY n_strong DESC, n_records DESC", (ring_id,))
        ]
        d["evidence_entities"] = [
            dict(e) for e in c.execute(
                "SELECT type, value, accounts_in_ring, accounts_total, rarity, "
                "type_weight FROM ring_entity WHERE ring_id = ? ORDER BY rank",
                (ring_id,))
        ]
        d["evidence_edges"] = [
            dict(e) for e in c.execute(
                "SELECT a_key, b_key, weight FROM ring_edge WHERE ring_id = ? "
                "ORDER BY rank", (ring_id,))
        ]
        d["bridges"] = [
            {**dict(b), "other_ring": b["ring_b"] if b["ring_a"] == ring_id else b["ring_a"]}
            for b in c.execute(
                "SELECT * FROM ring_link WHERE ring_a = ? OR ring_b = ? "
                "ORDER BY n_a + n_b DESC LIMIT 50", (ring_id, ring_id))
        ]
        return d

    def ring_graph(self, ring_id: int) -> dict | None:
        """Nodes and edges for the ring view.

        With the state DB available we can draw the *bipartite* graph: accounts
        plus the devices/cards/addresses/IPs they share, which is what makes a
        ring legible.  Without it we fall back to the account projection, still
        useful but less persuasive.
        """
        ring = self.ring(ring_id)
        if ring is None:
            return None
        keys = [m["account_key"] for m in ring["members"]]
        nodes: list[dict] = []
        edges: list[dict] = []
        st = self.state()
        in_ring = set(keys)
        if st is not None and keys:
            ids = {}
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                marks = ",".join("?" * len(part))
                for row in st.execute(
                    f"SELECT id, key FROM account WHERE key IN ({marks})",
                    [f"account\x1f{k}" for k in part],
                ):
                    ids[row["key"].split("\x1f", 1)[1]] = row["id"]
            acc_ids = list(ids.values())
            if acc_ids:
                marks = ",".join("?" * len(acc_ids))
                link_rows = st.execute(
                    f"SELECT l.account_id, e.id, e.type, e.value, l.n, l.n_days "
                    f"FROM link l JOIN entity e ON e.id = l.entity_id "
                    f"WHERE l.account_id IN ({marks})", acc_ids).fetchall()
                # entity -> accounts in the ring, to size nodes
                ent_ids = sorted({r[1] for r in link_rows})
                ent_deg: dict[int, int] = {}
                ent_total: dict[int, int] = {}
                if ent_ids:
                    for i in range(0, len(ent_ids), 500):
                        part = ent_ids[i:i + 500]
                        marks = ",".join("?" * len(part))
                        for row in st.execute(
                            f"SELECT entity_id, COUNT(*) c FROM link WHERE entity_id "
                            f"IN ({marks}) GROUP BY entity_id", part):
                            ent_total[row[0]] = row[1]
                id_to_key = {v: k for k, v in ids.items()}
                for r in link_rows:
                    ent_deg[r[1]] = ent_deg.get(r[1], 0) + 1
                # Only entities two or more accounts *in this ring* share carry
                # evidence.  A device one member touched once, and an IP only
                # they used, are noise that would triple the graph without
                # telling the analyst anything.
                shared_ent = {e for e, c in ent_deg.items() if c >= 2}

                for k in keys:
                    nodes.append({"id": f"a:{k}", "kind": "account", "label": k})
                seen_ent = set()
                for account_id, eid, etype, value, n, n_days in link_rows:
                    k = id_to_key.get(account_id)
                    if k is None or eid not in shared_ent:
                        continue
                    node_id = f"e:{eid}"
                    if eid not in seen_ent:
                        seen_ent.add(eid)
                        nodes.append({
                            "id": node_id, "kind": TYPE_NAMES[etype],
                            "label": value,
                            "accounts_here": ent_deg.get(eid, 0),
                            "accounts_total": ent_total.get(eid, 0),
                        })
                    edges.append({"source": f"a:{k}", "target": node_id,
                                  "weight": float(n), "days": int(n_days)})

                for m in ring["members"]:
                    if m["account_key"] in in_ring:
                        for n in nodes:
                            if n["id"] == f"a:{m['account_key']}":
                                n["n_strong"] = m.get("n_strong")
                                n["n_weak"] = m.get("n_weak")
                                n["n_records"] = m.get("n_records")
                                n["signup_day"] = m.get("signup_day")
                                break
        else:
            for k in keys:
                nodes.append({"id": f"a:{k}", "kind": "account", "label": k})
            for e in ring["evidence_edges"]:
                edges.append({"source": f"a:{e['a_key']}", "target": f"a:{e['b_key']}",
                              "weight": e["weight"]})
        return {"ring_id": ring_id, "mode": "bipartite" if st else "projection",
                "nodes": nodes, "edges": edges}

    def account(self, key: str) -> dict | None:
        rows = self.conn().execute(
            "SELECT r.id, r.score, r.n_accounts, r.cartel_size, m.n_strong, m.n_weak, "
            "m.n_records, m.first_day, m.last_day, m.signup_day "
            "FROM ring_member m JOIN ring r ON r.id = m.ring_id "
            "WHERE m.account_key = ? ORDER BY r.score DESC", (key,)).fetchall()
        st = self.state()
        detail = None
        if st is not None:
            row = st.execute("SELECT * FROM account WHERE key = ?",
                             (f"account\x1f{key}",)).fetchone()
            if row:
                detail = dict(row)
                detail.pop("key", None)
                for k in ("first_day", "last_day", "signup_day"):
                    detail[k] = _day(detail.get(k))
        if not rows and detail is None:
            return None
        return {
            "account": key,
            "rings": [{**dict(r), **{k: _day(r[k]) for k in
                                     ("first_day", "last_day", "signup_day")}}
                      for r in rows],
            "detail": detail,
        }

    def entities(self, q: str, limit: int = 50) -> list[dict]:
        if not q or len(q) < 2:
            return []
        rows = self.conn().execute(
            "SELECT type, value, SUM(accounts_in_ring) n, COUNT(*) rings, "
            "MAX(accounts_total) total, AVG(rarity) rarity "
            "FROM ring_entity WHERE value LIKE ? "
            "GROUP BY type, value ORDER BY rings DESC, n DESC LIMIT ?",
            (f"%{q}%", min(limit, MAX_PAGE)),
        )
        return [dict(r) for r in rows]

    def batches(self) -> list[dict]:
        st = self.state()
        if st is None:
            return []
        return [dict(r) for r in st.execute(
            "SELECT batch_id, source, imported_at, n_records, n_new_accounts, "
            "n_new_entities, days FROM batch ORDER BY imported_at DESC, rowid DESC "
            "LIMIT 100")]


def _day(value) -> str | None:
    if value is None:
        return None
    try:
        return day_to_str(int(value))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    source: DataSource
    server_version = "ringdetect"
    protocol_version = "HTTP/1.1"

    # quieter logs: one line per API call, none for static assets
    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path.startswith("/api/"):
                self._api(path, query)
            else:
                self._static(path)
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 - surface as JSON, never 500 silently
            self._json({"error": str(exc), "type": type(exc).__name__}, status=500)

    # -- routes ------------------------------------------------------------

    def _api(self, path: str, q: dict) -> None:
        src = self.source
        if path == "/api/health":
            return self._json({"ok": True, "summary": src.summary()})
        if path == "/api/summary":
            return self._json(src.summary())
        if path == "/api/batches":
            return self._json({"items": src.batches()})
        if path == "/api/entities":
            return self._json({"items": src.entities(_one(q, "q", ""),
                                                     _int(q, "limit", 50))})
        if path == "/api/rings":
            return self._json(src.rings(
                min_score=_float(q, "min_score", 0.0),
                min_size=_int(q, "min_size", 2),
                max_size=_int(q, "max_size", 0) or None,
                sort=_one(q, "sort", "score"),
                limit=_int(q, "limit", 50),
                offset=_int(q, "offset", 0),
                weak_only=(None if "weak_only" not in q
                           else _one(q, "weak_only") == "1"),
                q=_one(q, "q") or None,
            ))
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[1] == "rings":
            try:
                rid = int(parts[2])
            except ValueError:
                return self._json({"error": "bad ring id"}, status=400)
            if len(parts) == 4 and parts[3] == "graph":
                out = src.ring_graph(rid)
            else:
                out = src.ring(rid)
            return self._json(out if out is not None else {"error": "not found"},
                              status=200 if out is not None else 404)
        if len(parts) >= 3 and parts[1] == "accounts":
            out = src.account("/".join(parts[2:]))
            return self._json(out if out is not None else {"error": "not found"},
                              status=200 if out is not None else 404)
        return self._json({"error": "no such endpoint"}, status=404)

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (WEB_DIR / rel).resolve()
        try:
            target.relative_to(WEB_DIR.resolve())
        except ValueError:
            return self._json({"error": "forbidden"}, status=403)
        if not target.is_file():
            return self._json({"error": "not found"}, status=404)
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
            ".json": "application/json",
        }.get(target.suffix, "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=_json_default).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _one(q: dict, name: str, default=None):
    v = q.get(name)
    return v[0] if v else default


def _int(q: dict, name: str, default: int) -> int:
    try:
        return int(_one(q, name, default))
    except (TypeError, ValueError):
        return default


def _float(q: dict, name: str, default: float) -> float:
    try:
        return float(_one(q, name, default))
    except (TypeError, ValueError):
        return default


def make_server(out_dir, state_path=None, host="127.0.0.1", port=8765
                ) -> ThreadingHTTPServer:
    source = DataSource(Path(out_dir), Path(state_path) if state_path else None)
    handler = type("BoundHandler", (Handler,), {"source": source})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def serve(args) -> int:
    import webbrowser

    for attempt in range(20):
        try:
            httpd = make_server(args.out, args.state, args.host, args.port + attempt)
            break
        except OSError as exc:
            if attempt == 19:
                print(f"cannot bind {args.host}:{args.port}: {exc}")
                return 2
    port = httpd.server_address[1]
    url = f"http://{args.host}:{port}/"
    print(f"ringdetect viewer: {url}")
    print(f"  results: {Path(args.out).resolve()}")
    print(f"  state:   {Path(args.state).resolve() if args.state else '(none: bipartite view off)'}")
    if args.open:
        threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)),
                         daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0
