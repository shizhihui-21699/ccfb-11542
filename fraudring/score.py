"""团伙打分 + 关键证据提取.

可疑分 (0-100) 由四个可解释分量加权:
  - size_pts    团伙规模 (对数, 50 人封顶)
  - weight_pts  平均边权 / 30 (共享实体越稀有越大)
  - coupon_pts  券核销强度相对全站均值的提升 (lift-1, 对数, 9 倍封顶)
  - density_pts 内部边密度 / 0.5
"""
import json
import math

import numpy as np

W_SIZE, W_WEIGHT, W_COUPON, W_DENSITY = 0.25, 0.30, 0.30, 0.15
SUSPECT_THRESHOLD = 60.0
# 券强度 lift 的加-k 平滑: 2 人小团偶然多核销几张券不该得高分,
# 样本(券数)越少 lift 越向全站均值收缩. k=20 相当于先验 20 个普通账号.
LIFT_SMOOTH_K = 20.0


def coupon_counts(db):
    """每个账号的券核销数 + 全站均值."""
    cnt = {}
    for aid, c in db.execute("SELECT account_id, COUNT(*) FROM coupons GROUP BY account_id"):
        cnt[aid] = c
    n_acc = db.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    mean = sum(cnt.values()) / max(n_acc, 1)
    return cnt, mean


def score_communities(members_by_cid, edges_by_cid, coupon_cnt, coupon_mean):
    """members_by_cid: {cid: [account_id]}, edges_by_cid: {cid: [(a,b,w)]}.
    返回 {cid: {...}}."""
    out = {}
    for cid, members in members_by_cid.items():
        size = len(members)
        edges = edges_by_cid.get(cid, [])
        n_edges = len(edges)
        avg_w = float(np.mean([w for _, _, w in edges])) if edges else 0.0
        density = 2.0 * n_edges / (size * (size - 1)) if size > 1 else 0.0
        coupon_total = sum(coupon_cnt.get(a, 0) for a in members)
        lift = (coupon_total / size) / max(coupon_mean, 1e-9)
        lift_adj = ((coupon_total + LIFT_SMOOTH_K * coupon_mean)
                    / (size + LIFT_SMOOTH_K)) / max(coupon_mean, 1e-9)

        size_pts = min(1.0, math.log1p(size) / math.log1p(50))
        weight_pts = min(1.0, avg_w / 30.0)
        coupon_pts = min(1.0, math.log1p(max(0.0, lift_adj - 1.0)) / math.log1p(9.0))
        density_pts = min(1.0, density / 0.5)
        score = 100.0 * (W_SIZE * size_pts + W_WEIGHT * weight_pts
                         + W_COUPON * coupon_pts + W_DENSITY * density_pts)
        out[cid] = {
            "size": size, "n_edges": n_edges, "avg_weight": round(avg_w, 2),
            "density": round(density, 3), "coupon_total": coupon_total,
            "coupon_lift": round(lift, 1), "score": round(score, 1),
        }
    return out


def evidence_for_community(db, members, df_lookup, n_accounts, top_entities=8, top_edges=5):
    """关键证据: 成员间共享的稀有实体 + 最强边及其共享实体."""
    placeholders = ",".join("?" for _ in members)
    rows = db.execute(
        f"SELECT entity_type, entity_id, COUNT(DISTINCT account_id) c "
        f"FROM links WHERE account_id IN ({placeholders}) "
        f"GROUP BY entity_type, entity_id HAVING c >= 2", members).fetchall()
    ents = []
    for et, eid, c in rows:
        df = df_lookup.get((et, eid), 0)
        ents.append({"entity_type": et, "entity_id": eid, "df": df,
                     "members": c, "weight": round(math.log1p(n_accounts / max(df, 1)), 2)})
    # 按稀有度(df 升序) × 覆盖人数(降序) 排序
    ents.sort(key=lambda e: (e["df"], -e["members"]))
    ents = ents[:top_entities]

    # 最强边: 投影边表已在 proj_edges
    edges = db.execute(
        f"SELECT a, b, weight FROM proj_edges WHERE a IN ({placeholders}) "
        f"AND b IN ({placeholders}) ORDER BY weight DESC LIMIT ?",
        members + members + [top_edges]).fetchall()
    ev_edges = []
    for a, b, w in edges:
        shared = db.execute(
            "SELECT entity_type, entity_id FROM links WHERE account_id=? "
            "INTERSECT SELECT entity_type, entity_id FROM links WHERE account_id=?",
            (a, b)).fetchall()
        ev_edges.append({"a": a, "b": b, "weight": round(w, 2),
                         "shared": [f"{t}:{e}" for t, e in shared]})
    return ents, ev_edges


def summarize(ents):
    if not ents:
        return ""
    top = ents[0]
    return f"{top['entity_type']}:{top['entity_id']} (df={top['df']}, {top['members']}人共享)"


def load_df_lookup(db):
    return {(t, e): df for t, e, df in
            db.execute("SELECT entity_type, entity_id, df FROM entity_stats")}


def to_json(obj):
    return json.dumps(obj, ensure_ascii=False)
