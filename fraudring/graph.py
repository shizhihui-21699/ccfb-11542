"""构图: 账号-实体二部图 -> IDF 加权 -> 账号-账号投影图 (numpy 向量化).

权重设计:
  - 每个实体 e 的权重 w(e) = type_weight[type] * ln(1 + N_accounts / df(e))
    df 越小(越稀有)权重越大 —— 共用一台冷门设备远比共用公司出口 IP 可疑.
  - 账号对边权 = 所有共享实体权重之和, 同时记录共享实体的类型位掩码.
  - 剪枝: df 超过类型上限的实体(公司 NAT 等)不参与投影;
    只被单一类型弱证据连接的边(如仅碰撞同一个 IP)被丢弃.
"""
import numpy as np
import igraph as ig

ETYPES = ["device", "ip", "address", "card"]
ET_CODE = {t: i for i, t in enumerate(ETYPES)}

# 各类实体的可信度: 设备/卡最强, IP 最弱(公共出口太多)
TYPE_WEIGHT = {"device": 1.0, "card": 1.0, "address": 0.9, "ip": 0.3}
# df 超过该值的实体不参与投影 (正常高共享, 如公司/校园出口 IP)
DF_CAP = {"device": 100, "card": 100, "address": 100, "ip": 50}
# 单类型弱证据边的最低权重 (仅碰撞一个 IP 的边约 3.9, 被过滤)
MIN_EDGE_WEIGHT = 8.0


def load_links(db):
    """links 表 -> (account_ids, 实体整数id数组, 账号下标数组, 类型码数组)."""
    accounts = [r[0] for r in db.execute("SELECT account_id FROM accounts ORDER BY account_id")]
    aidx = {a: i for i, a in enumerate(accounts)}
    ent_map, ent_list = {}, []
    e_ent, e_acct, e_type = [], [], []
    cur = db.execute("SELECT entity_type, entity_id, account_id FROM links")
    while True:
        rows = cur.fetchmany(500_000)
        if not rows:
            break
        for et, eid, aid in rows:
            key = (et, eid)
            k = ent_map.get(key)
            if k is None:
                k = len(ent_list)
                ent_map[key] = k
                ent_list.append(key)
            e_ent.append(k)
            e_acct.append(aidx[aid])
            e_type.append(ET_CODE[et])
    arrays = (np.asarray(e_ent, dtype=np.int64),
              np.asarray(e_acct, dtype=np.int64),
              np.asarray(e_type, dtype=np.int8))
    return accounts, ent_list, arrays


def build_projection(db, type_weight=None, df_cap=None, min_weight=MIN_EDGE_WEIGHT):
    """返回 (accounts, ent_list, df_per_entity, src, dst, weight, type_mask)."""
    type_weight = type_weight or TYPE_WEIGHT
    df_cap = df_cap or DF_CAP
    tw = np.array([type_weight[t] for t in ETYPES])
    cap = np.array([df_cap[t] for t in ETYPES])

    accounts, ent_list, (ent, acct, etype) = load_links(db)
    n = len(accounts)

    # 按实体分组
    order = np.argsort(ent, kind="stable")
    ent_s, acct_s, type_s = ent[order], acct[order], etype[order]
    uniq, starts, counts = np.unique(ent_s, return_index=True, return_counts=True)
    df = counts  # df_per_entity, 下标与 uniq 一致; uniq 即实体整数id(升序)
    df_full = np.zeros(ent.max() + 1 if len(ent) else 0, dtype=np.int64)
    df_full[uniq] = counts

    # 每个共享实体 -> 账号对 (笛卡尔对), 权重 = type_weight * ln(1 + N/df)
    src_l, dst_l, w_l, m_l = [], [], [], []
    for i in range(len(uniq)):
        k = counts[i]
        if k < 2:
            continue
        t = type_s[starts[i]]
        if k > cap[t]:
            continue
        # ln(1 + N/df): df 越小权重越大
        w = float(tw[t] * np.log1p(n / k))
        members = acct_s[starts[i]:starts[i] + k]
        if k == 2:
            a, b = members[0], members[1]
            if a == b:
                continue
            src_l.append(np.array([min(a, b)], dtype=np.int64))
            dst_l.append(np.array([max(a, b)], dtype=np.int64))
        else:
            ii, jj = np.triu_indices(k, 1)
            pa, pb = members[ii], members[jj]
            same = pa == pb
            if same.any():
                pa, pb = pa[~same], pb[~same]
            if len(pa) == 0:
                continue
            lo = np.minimum(pa, pb)
            hi = np.maximum(pa, pb)
            src_l.append(lo)
            dst_l.append(hi)
        bit = np.int64(1 << int(t))
        m_l.append(np.full(len(src_l[-1]), bit, dtype=np.int64))
        w_l.append(np.full(len(src_l[-1]), w, dtype=np.float64))

    if not src_l:
        return accounts, ent_list, df_full, *(np.array([], dtype=np.int64) for _ in range(2)), \
            np.array([]), np.array([], dtype=np.int64)

    src = np.concatenate(src_l)
    dst = np.concatenate(dst_l)
    wgt = np.concatenate(w_l)
    msk = np.concatenate(m_l)

    # 同一账号对的多条贡献合并: 权重求和, 类型位掩码取或
    key = src * n + dst
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    w_s = wgt[order]
    m_s = msk[order]
    ukey, starts = np.unique(key_s, return_index=True)
    w_sum = np.add.reduceat(w_s, starts)
    m_or = np.bitwise_or.reduceat(m_s, starts)

    # 剪枝: 保留 (涉及 >=2 类实体) 或 (权重足够大) 的边
    n_types = np.bitwise_count(m_or)
    keep = (n_types >= 2) | (w_sum >= min_weight)
    ukey = ukey[keep]
    w_sum = w_sum[keep]
    m_or = m_or[keep]
    src_f = (ukey // n).astype(np.int64)
    dst_f = (ukey % n).astype(np.int64)
    return accounts, ent_list, df_full, src_f, dst_f, w_sum, m_or


def build_igraph(accounts, src, dst, weight):
    """投影边 -> igraph 图 (只含有边账号, 孤立账号不参与社区发现)."""
    nodes = np.unique(np.concatenate([src, dst]))
    remap = np.zeros(len(accounts), dtype=np.int64)
    remap[nodes] = np.arange(len(nodes))
    g = ig.Graph(n=len(nodes),
                 edges=list(zip(remap[src].tolist(), remap[dst].tolist())),
                 edge_attrs={"weight": weight.tolist()})
    node_accounts = [accounts[i] for i in nodes]
    return g, node_accounts
