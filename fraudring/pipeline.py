"""分析流水线: 构图 -> Leiden -> 打分 -> 证据 -> 落库, 可选对照 ground truth 评估."""
import json
import os
import time

import numpy as np

from . import community as comm_mod
from . import graph as graph_mod
from . import score as score_mod


def _entity_stats(db, ent_list, df_full):
    db.execute("DELETE FROM entity_stats")
    db.executemany(
        "INSERT INTO entity_stats(entity_type, entity_id, df) VALUES(?,?,?)",
        [(t, e, int(df_full[i])) for i, (t, e) in enumerate(ent_list)])


def analyze(db, data_dir=None, resolution=1.0, evidence_top=1000, verbose=True):
    t0 = time.time()

    # 1. 构图
    accounts, ent_list, df_full, src, dst, wgt, msk = graph_mod.build_projection(db)
    _entity_stats(db, ent_list, df_full)
    if verbose:
        print(f"[analyze] 账号={len(accounts)} 实体={len(ent_list)} "
              f"投影边={len(src)} ({time.time()-t0:.1f}s)")

    # 2. 存投影边 (网页查看/证据用)
    db.execute("DELETE FROM proj_edges")
    acc_arr = np.array(accounts, dtype=object)
    db.executemany(
        "INSERT INTO proj_edges(a, b, weight) VALUES(?,?,?)",
        zip(acc_arr[src].tolist(), acc_arr[dst].tolist(),
            np.round(wgt, 4).tolist()))
    db.commit()

    # 3. Leiden 社区发现
    g, node_accounts = graph_mod.build_igraph(accounts, src, dst, wgt)
    membership = comm_mod.find_communities(g, resolution=resolution)
    if verbose:
        print(f"[analyze] 图节点={g.vcount()} 边={g.ecount()} "
              f"社区={max(membership)+1 if membership else 0} ({time.time()-t0:.1f}s)")

    # 4. 按社区聚合成员与边
    members_by_cid, cid_of_node = {}, {}
    for v, cid in enumerate(membership):
        members_by_cid.setdefault(cid, []).append(node_accounts[v])
        cid_of_node[v] = cid
    # 只保留 >=2 人的社区
    members_by_cid = {c: m for c, m in members_by_cid.items() if len(m) >= 2}
    keep_cids = set(members_by_cid)
    edges_by_cid = {}
    nodes = np.unique(np.concatenate([src, dst])) if len(src) else np.array([], dtype=np.int64)
    remap = np.zeros(len(accounts), dtype=np.int64)
    remap[nodes] = np.arange(len(nodes))
    for a, b, w in zip(remap[src] if len(src) else [], remap[dst] if len(dst) else [],
                       wgt):
        cid = cid_of_node.get(int(a))
        if cid is None or cid not in keep_cids or cid != cid_of_node.get(int(b)):
            continue
        edges_by_cid.setdefault(cid, []).append(
            (node_accounts[int(a)], node_accounts[int(b)], float(w)))

    # 5. 打分
    coupon_cnt, coupon_mean = score_mod.coupon_counts(db)
    stats = score_mod.score_communities(members_by_cid, edges_by_cid,
                                        coupon_cnt, coupon_mean)

    # 6. 落库
    for t in ("communities", "community_members", "evidence", "evidence_edges"):
        db.execute(f"DELETE FROM {t}")
    ranked = sorted(stats, key=lambda c: -stats[c]["score"])
    new_cid = {}
    for rank, cid in enumerate(ranked):
        new_cid[cid] = rank
        s = stats[cid]
        db.execute(
            "INSERT INTO communities(id, size, n_edges, avg_weight, density,"
            " coupon_total, coupon_lift, score, summary)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (rank, s["size"], s["n_edges"], s["avg_weight"], s["density"],
             s["coupon_total"], s["coupon_lift"], s["score"], ""))
        db.executemany("INSERT INTO community_members(community_id, account_id)"
                       " VALUES(?,?)",
                       [(rank, a) for a in members_by_cid[cid]])
    db.commit()

    # 7. 证据 (只对分数最高的前 evidence_top 个社区, 避免对几万个合租对逐个查)
    df_lookup = score_mod.load_df_lookup(db)
    n_acc = len(accounts)
    for cid in ranked[:evidence_top]:
        members = members_by_cid[cid]
        ents, ev_edges = score_mod.evidence_for_community(
            db, members, df_lookup, n_acc)
        rid = new_cid[cid]
        db.executemany(
            "INSERT INTO evidence(community_id, entity_type, entity_id, df,"
            " members, weight) VALUES(?,?,?,?,?,?)",
            [(rid, e["entity_type"], e["entity_id"], e["df"], e["members"],
              e["weight"]) for e in ents])
        db.executemany(
            "INSERT INTO evidence_edges(community_id, a, b, weight, shared)"
            " VALUES(?,?,?,?,?)",
            [(rid, e["a"], e["b"], e["weight"],
              score_mod.to_json(e["shared"])) for e in ev_edges])
        db.execute("UPDATE communities SET summary=? WHERE id=?",
                   (score_mod.summarize(ents), rid))
    db.commit()
    if verbose:
        n_suspect = sum(1 for c in stats if stats[c]["score"] >= score_mod.SUSPECT_THRESHOLD)
        print(f"[analyze] 社区(>=2人)={len(stats)} 可疑(>=60分)={n_suspect} "
              f"总耗时 {time.time()-t0:.1f}s")

    # 8. 对照 ground truth 评估
    if data_dir:
        rings_path = os.path.join(data_dir, "rings.json")
        if os.path.exists(rings_path):
            evaluate(db, rings_path)
    return stats


def evaluate(db, rings_path):
    with open(rings_path) as f:
        rings = json.load(f)
    comm_members = {}
    for cid, aid in db.execute("SELECT community_id, account_id FROM community_members"):
        comm_members.setdefault(cid, set()).add(aid)
    hits, jaccards = [], []
    for rid, info in rings.items():
        truth = set(info["members"])
        best = 0.0
        for members in comm_members.values():
            inter = len(truth & members)
            if not inter:
                continue
            best = max(best, inter / len(truth | members))
        jaccards.append(best)
        hits.append(best >= 0.8)
    print(f"[eval] 埋入团伙={len(rings)} 检出(Jaccard>=0.8)={sum(hits)}/{len(hits)} "
          f"平均最佳Jaccard={np.mean(jaccards):.3f}")
