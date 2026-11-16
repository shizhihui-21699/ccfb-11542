"""Flask 网页查看: 团伙列表(按可疑分排序) + 团伙详情(成员/证据/关系图SVG). 纯本地, 无外部依赖."""
import html
import json
import math

import igraph as ig
from flask import Flask, abort, jsonify

from .db import connect
from .score import SUSPECT_THRESHOLD

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font:14px/1.5 -apple-system,"PingFang SC",sans-serif;margin:0;background:#f5f6f8;color:#222}}
.wrap{{max-width:1100px;margin:0 auto;padding:24px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;width:100%;background:#fff}}
th,td{{padding:8px 10px;border-bottom:1px solid #eee;text-align:left;font-size:13px}}
th{{background:#fafafa}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:12px}}
.high{{background:#d33}} .mid{{background:#e90}} .low{{background:#8a8}}
.card{{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:16px;margin:12px 0}}
.mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
a{{color:#06c;text-decoration:none}}
</style></head><body><div class="wrap">{body}</div></body></html>"""


def _esc(s):
    return html.escape(str(s))


def _badge(score):
    cls = "high" if score >= SUSPECT_THRESHOLD else ("mid" if score >= 40 else "low")
    return f'<span class="badge {cls}">{score:.1f}</span>'


def create_app(db_path):
    app = Flask(__name__)

    def db():
        return connect(db_path)

    @app.get("/")
    def index():
        rows = db().execute(
            "SELECT id, size, n_edges, avg_weight, coupon_lift, score, summary"
            " FROM communities ORDER BY score DESC").fetchall()
        trs = "".join(
            f"<tr><td><a href='/community/{r[0]}'>#{r[0]}</a></td><td>{r[1]}</td>"
            f"<td>{r[2]}</td><td>{r[3]:.1f}</td><td>{r[4]:.1f}x</td>"
            f"<td>{_badge(r[5])}</td><td class='mono'>{_esc(r[6] or '')}</td></tr>"
            for r in rows)
        body = (f"<h1>团伙列表 <small>({len(rows)} 个社区, 按可疑分排序)</small></h1>"
                "<table><tr><th>ID</th><th>人数</th><th>边数</th><th>均边权</th>"
                "<th>券强度</th><th>可疑分</th><th>头号证据</th></tr>"
                f"{trs}</table>")
        return PAGE.format(title="团伙列表", body=body)

    @app.get("/community/<int:cid>")
    def detail(cid):
        conn = db()
        c = conn.execute("SELECT * FROM communities WHERE id=?", (cid,)).fetchone()
        if not c:
            abort(404)
        members = [r[0] for r in conn.execute(
            "SELECT account_id FROM community_members WHERE community_id=?",
            (cid,))]
        coupons = {a: n for a, n in conn.execute(
            "SELECT account_id, COUNT(*) FROM coupons GROUP BY account_id")}
        ev = conn.execute(
            "SELECT entity_type, entity_id, df, members, weight FROM evidence"
            " WHERE community_id=? ORDER BY df ASC", (cid,)).fetchall()
        ev_edges = conn.execute(
            "SELECT a, b, weight, shared FROM evidence_edges WHERE community_id=?",
            (cid,)).fetchall()

        ev_rows = "".join(
            f"<tr><td>{_esc(t)}</td><td class='mono'>{_esc(e)}</td><td>{d}</td>"
            f"<td>{m}</td><td>{w:.1f}</td></tr>" for t, e, d, m, w in ev)
        edge_rows = "".join(
            f"<tr><td class='mono'>{_esc(a)}</td><td class='mono'>{_esc(b)}</td>"
            f"<td>{w:.1f}</td><td class='mono'>{_esc(', '.join(json.loads(s)))}</td></tr>"
            for a, b, w, s in ev_edges)
        mem_rows = "".join(
            f"<tr><td class='mono'>{_esc(a)}</td><td>{coupons.get(a, 0)}</td></tr>"
            for a in sorted(members, key=lambda x: -coupons.get(x, 0)))

        body = f"""<h1><a href='/'>←</a> 团伙 #{cid} {_badge(c[7])}</h1>
<div class='card'>人数 <b>{c[1]}</b> · 内部边 <b>{c[2]}</b> · 均边权 <b>{c[3]:.1f}</b> ·
密度 <b>{c[4]:.2f}</b> · 券核销 <b>{c[5]}</b> 张 · 券强度 <b>{c[6]:.1f}x</b> 全站均值</div>
<h2>关系图</h2><div class='card'>{_community_svg(conn, cid, members, coupons)}</div>
<h2>关键证据: 共享稀有实体</h2>
<table><tr><th>类型</th><th>实体</th><th>全站df</th><th>团伙内共享人数</th><th>IDF权重</th></tr>{ev_rows}</table>
<h2>关键证据: 最强边</h2>
<table><tr><th>账号A</th><th>账号B</th><th>边权</th><th>共享实体</th></tr>{edge_rows}</table>
<h2>成员 ({len(members)})</h2>
<table><tr><th>账号</th><th>券核销数</th></tr>{mem_rows}</table>"""
        return PAGE.format(title=f"团伙 #{cid}", body=body)

    @app.get("/api/communities")
    def api_list():
        rows = db().execute(
            "SELECT id, size, n_edges, avg_weight, density, coupon_total,"
            " coupon_lift, score, summary FROM communities ORDER BY score DESC"
        ).fetchall()
        keys = ["id", "size", "n_edges", "avg_weight", "density",
                "coupon_total", "coupon_lift", "score", "summary"]
        return jsonify([dict(zip(keys, r)) for r in rows])

    @app.get("/api/community/<int:cid>")
    def api_detail(cid):
        conn = db()
        c = conn.execute("SELECT * FROM communities WHERE id=?", (cid,)).fetchone()
        if not c:
            abort(404)
        return jsonify({
            "id": cid, "score": c[7],
            "members": [r[0] for r in conn.execute(
                "SELECT account_id FROM community_members WHERE community_id=?", (cid,))],
            "evidence": [dict(zip(["entity_type", "entity_id", "df", "members", "weight"], r))
                         for r in conn.execute(
                             "SELECT entity_type, entity_id, df, members, weight"
                             " FROM evidence WHERE community_id=?", (cid,))],
        })

    return app


def _community_svg(conn, cid, members, coupons, width=1000, height=520):
    """用 igraph 布局直接渲染 SVG, 不依赖任何前端库."""
    placeholders = ",".join("?" for _ in members)
    edges = conn.execute(
        f"SELECT a, b, weight FROM proj_edges WHERE a IN ({placeholders})"
        f" AND b IN ({placeholders})", members + members).fetchall()
    if not edges:
        return "<p>无内部边</p>"
    idx = {a: i for i, a in enumerate(members)}
    g = ig.Graph(n=len(members),
                 edges=[(idx[a], idx[b]) for a, b, _ in edges],
                 edge_attrs={"weight": [w for _, _, w in edges]})
    layout = g.layout_fruchterman_reingold(weights="weight")
    xs = [p[0] for p in layout]
    ys = [p[1] for p in layout]

    def norm(v, lo, hi, out_lo, out_hi):
        span = (hi - lo) or 1.0
        return [out_lo + (x - lo) / span * (out_hi - out_lo) for x in v]

    pad = 40
    X = norm(xs, min(xs), max(xs), pad, width - pad)
    Y = norm(ys, min(ys), max(ys), pad, height - pad)
    max_w = max(w for _, _, w in edges) or 1.0
    max_c = max(coupons.get(a, 0) for a in members) or 1

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'xmlns="http://www.w3.org/2000/svg">']
    for (a, b, w) in edges:
        lw = 0.5 + 3.5 * w / max_w
        parts.append(f'<line x1="{X[idx[a]]:.1f}" y1="{Y[idx[a]]:.1f}" '
                     f'x2="{X[idx[b]]:.1f}" y2="{Y[idx[b]]:.1f}" '
                     f'stroke="#99a" stroke-width="{lw:.1f}" stroke-opacity="0.55"/>')
    for a in members:
        i = idx[a]
        heat = min(1.0, math.log1p(coupons.get(a, 0)) / math.log1p(max_c))
        r = int(255 * heat)
        b = int(200 * (1 - heat))
        parts.append(f'<circle cx="{X[i]:.1f}" cy="{Y[i]:.1f}" r="7" '
                     f'fill="rgb({r},80,{b})" fill-opacity="0.9"/>'
                     f'<text x="{X[i]:.1f}" y="{Y[i]-11:.1f}" font-size="9" '
                     f'text-anchor="middle" fill="#555">{_esc(a)}</text>')
    parts.append("</svg>")
    parts.append("<p class='mono'>节点颜色越红 = 券核销越多; 边越粗 = 共享实体越稀有</p>")
    return "".join(parts)
