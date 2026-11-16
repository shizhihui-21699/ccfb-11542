"""生成模拟数据: 正常用户 + 埋入的羊毛团伙 + 正常共享(合租/家庭卡/公司网络).

输出目录结构 (每日一个子目录, 支持增量导入):
    out/day_00/accounts.csv   account_id,register_day
    out/day_00/links.csv      account_id,entity_type,entity_id,day
    out/day_00/coupons.csv    account_id,coupon_id,day,amount
    out/rings.json            团伙 ground truth (评估用)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ETYPES = ["device", "ip", "address", "card"]
ET_CODE = {t: i for i, t in enumerate(ETYPES)}


def _rand_days(rng, reg_day, n_days):
    """每个链接落在 [注册日, n_days) 的随机一天."""
    span = n_days - reg_day
    return reg_day + (rng.random(len(reg_day)) * span).astype(np.int32)


class EntityRegistry:
    """按类型注册实体 id, 返回整数下标, 链接用下标表示, 最后统一映射回字符串."""

    def __init__(self):
        self.ids = {t: [] for t in ETYPES}

    def add(self, etype, entity_id):
        self.ids[etype].append(entity_id)
        return len(self.ids[etype]) - 1


def generate(out_dir, n_accounts=1_000_000, n_days=7, n_rings=12, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    reg = EntityRegistry()

    acc_idx = np.arange(n_accounts, dtype=np.int64)
    account_ids = np.array([f"u{i:07d}" for i in range(n_accounts)])
    reg_day = rng.integers(0, n_days, n_accounts).astype(np.int32)

    # (account_idx, etype_code, entity_idx, day) 分块收集
    L_a, L_t, L_e, L_d = [], [], [], []

    def add_links(acct, etype, ents, days=None):
        acct = np.asarray(acct, dtype=np.int64)
        ents = np.asarray(ents, dtype=np.int64)
        if days is None:
            days = _rand_days(rng, reg_day[acct], n_days)
        L_a.append(acct)
        L_t.append(np.full(len(acct), ET_CODE[etype], dtype=np.int8))
        L_e.append(ents)
        L_d.append(np.asarray(days, dtype=np.int32))

    # ---------- 正常用户: 每人一台设备 ----------
    dev_personal = np.array([reg.add("device", f"d{i:07d}") for i in range(n_accounts)])
    add_links(acc_idx, "device", dev_personal)

    # ---------- IP: 25万池均匀采样 + 200 个公司/公共出口热点 ----------
    ip_pool = max(1000, n_accounts // 4)
    ip_base = np.array([reg.add("ip", f"ip{j:06d}") for j in range(ip_pool)])
    add_links(acc_idx, "ip", ip_base[rng.integers(0, ip_pool, n_accounts)])
    has_second = rng.random(n_accounts) < 0.2
    second_acct = acc_idx[has_second]
    add_links(second_acct, "ip", ip_base[rng.integers(0, ip_pool, len(second_acct))])
    # 公司/校园网出口 IP: 大量账号共享 (df 高 -> 分析时按超帽过滤, 属正常共享)
    hotspot_ips = rng.choice(ip_pool, 200, replace=False)
    for hip in hotspot_ips:
        n_users = int(rng.integers(100, 2000))
        add_links(rng.integers(0, n_accounts, n_users), "ip", np.full(n_users, hip))

    # ---------- 地址: 默认每人一个; 3% 合租(2-4人共用一个地址+一个家庭IP) ----------
    addr_ent = np.array([reg.add("address", f"addr{i:07d}") for i in range(n_accounts)])
    n_roomies = int(n_accounts * 0.03)
    roomie_pool = rng.choice(acc_idx, n_roomies, replace=False)
    pos, g = 0, 0
    while pos < n_roomies:
        size = int(rng.integers(2, 5))
        members = roomie_pool[pos:pos + size]
        pos += size
        if len(members) < 2:
            break
        shared_addr = reg.add("address", f"raddr{g:06d}")
        addr_ent[members] = shared_addr
        home_ip = reg.add("ip", f"hip{g:06d}")
        add_links(members, "ip", np.full(len(members), home_ip))
        g += 1
    add_links(acc_idx, "address", addr_ent)

    # ---------- 支付卡: 默认每人一张; 2% 家庭共享(2-3人一张卡) ----------
    card_ent = np.array([reg.add("card", f"card{i:07d}") for i in range(n_accounts)])
    n_family = int(n_accounts * 0.02)
    family_pool = rng.choice(acc_idx, n_family, replace=False)
    pos, g = 0, 0
    while pos < n_family:
        size = int(rng.integers(2, 4))
        members = family_pool[pos:pos + size]
        pos += size
        if len(members) < 2:
            break
        card_ent[members] = reg.add("card", f"famcard{g:06d}")
        g += 1
    add_links(acc_idx, "card", card_ent)

    # ---------- 埋入羊毛团伙 ----------
    # 每个团伙: 共享 1 个收货地址(全员) + 2 台设备 + 2 张卡 + 3 个 IP, 成员各用其中一部分
    is_ring = np.zeros(n_accounts, dtype=bool)
    rings = {}
    ring_sizes = rng.integers(8, 41, n_rings)
    ring_members_all = rng.choice(acc_idx, int(ring_sizes.sum()), replace=False)
    pos = 0
    for r, size in enumerate(ring_sizes):
        members = ring_members_all[pos:pos + size]
        pos += size
        is_ring[members] = True
        rid = f"ring_{r:02d}"
        ents = []
        ring_addr = reg.add("address", f"rgaddr{r:03d}")
        add_links(members, "address", np.full(len(members), ring_addr))
        ents.append(f"rgaddr{r:03d}")
        ring_devs = [reg.add("device", f"rgdev{r:03d}_{k}") for k in range(2)]
        ring_cards = [reg.add("card", f"rgcard{r:03d}_{k}") for k in range(2)]
        ring_ips = [reg.add("ip", f"rgip{r:03d}_{k}") for k in range(3)]
        ents += [f"rgdev{r:03d}_{k}" for k in range(2)]
        ents += [f"rgcard{r:03d}_{k}" for k in range(2)]
        ents += [f"rgip{r:03d}_{k}" for k in range(3)]
        # 每个成员用 1-2 台团伙设备 / 1 张团伙卡 / 1-2 个团伙 IP
        for m in members:
            for dev in rng.choice(ring_devs, rng.integers(1, 3), replace=False):
                add_links([m], "device", [dev])
            add_links([m], "card", [rng.choice(ring_cards)])
            for ip in rng.choice(ring_ips, rng.integers(1, 3), replace=False):
                add_links([m], "ip", [ip])
        rings[rid] = {"members": account_ids[members].tolist(), "entities": ents}

    # ---------- 优惠券核销: 团伙高频, 正常用户低频 ----------
    lam = np.where(is_ring, 8.0, 0.1)
    rows_a, rows_c, rows_d, rows_amt = [], [], [], []
    coupon_seq = 0
    for day in range(n_days):
        active = reg_day <= day
        cnt = rng.poisson(lam[active])
        accts = acc_idx[active]
        accts = np.repeat(accts, cnt)
        n_c = len(accts)
        rows_a.append(accts)
        rows_c.append(np.arange(coupon_seq, coupon_seq + n_c, dtype=np.int64))
        rows_d.append(np.full(n_c, day, dtype=np.int32))
        rows_amt.append(np.round(rng.uniform(5, 200, n_c), 2))
        coupon_seq += n_c
    cp_a = np.concatenate(rows_a)
    coupons = pd.DataFrame({
        "account_id": account_ids[cp_a],
        "coupon_id": "c" + np.concatenate(rows_c).astype(str),
        "day": np.concatenate(rows_d),
        "amount": np.concatenate(rows_amt),
    })

    # ---------- 汇总链接表 ----------
    la = np.concatenate(L_a)
    lt = np.concatenate(L_t)
    le = np.concatenate(L_e)
    ld = np.concatenate(L_d)
    etype_str = np.array(ETYPES, dtype=object)[lt]
    # entity_idx -> entity_id 字符串 (按类型分别映射)
    eid = np.empty(len(la), dtype=object)
    for t, name in enumerate(ETYPES):
        mask = lt == t
        eid[mask] = np.array(reg.ids[name], dtype=object)[le[mask]]
    links = pd.DataFrame({
        "account_id": account_ids[la],
        "entity_type": etype_str,
        "entity_id": eid,
        "day": ld,
    })

    # ---------- 按天落盘 ----------
    accounts = pd.DataFrame({"account_id": account_ids, "register_day": reg_day})
    for day in range(n_days):
        ddir = os.path.join(out_dir, f"day_{day:02d}")
        os.makedirs(ddir, exist_ok=True)
        accounts[accounts.register_day == day][["account_id"]].to_csv(
            os.path.join(ddir, "accounts.csv"), index=False)
        links[links.day == day].to_csv(os.path.join(ddir, "links.csv"), index=False)
        coupons[coupons.day == day].to_csv(os.path.join(ddir, "coupons.csv"), index=False)

    with open(os.path.join(out_dir, "rings.json"), "w") as f:
        json.dump(rings, f, indent=1)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"n_accounts": int(n_accounts), "n_days": n_days,
                   "n_rings": n_rings, "n_links": len(links),
                   "n_coupons": len(coupons), "seed": seed}, f, indent=1)
    print(f"[generate] accounts={n_accounts} links={len(links)} coupons={len(coupons)} "
          f"rings={n_rings} -> {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="生成模拟数据")
    p.add_argument("--out", default="data/incoming")
    p.add_argument("--accounts", type=int, default=1_000_000)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--rings", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    generate(a.out, a.accounts, a.days, a.rings, a.seed)


if __name__ == "__main__":
    main()
