# fraudring — 羊毛党团伙挖掘

把账号、设备、IP、收货地址、支付卡连成关系图，用 **IDF 加权投影 + Leiden 社区发现**
找出"单看每个账号都正常、合在一起是团伙"的羊毛党，并给出每个团伙的可疑分和关键证据。

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas igraph flask

# 一键演示 (20万账号, 埋 12 个团伙 + 合租/家庭卡/公司网络等正常共享)
.venv/bin/python -m fraudring demo

# 或分步执行 (默认 100 万账号)
.venv/bin/python -m fraudring generate --accounts 1000000 --days 7 --rings 12
.venv/bin/python -m fraudring ingest            # 增量导入 data/incoming/day_*
.venv/bin/python -m fraudring analyze           # 构图 + 社区发现 + 打分
.venv/bin/python -m fraudring serve             # http://127.0.0.1:8000
```

每日新数据放入 `data/incoming/day_XX/{accounts,links,coupons}.csv` 后重跑
`ingest`（幂等，已导入文件自动跳过）再跑 `analyze` 即可。

## 方法

**构图**：账号—实体二部图投影为账号—账号图。每个共享实体贡献权重

```
w(e) = type_weight[type] * ln(1 + N_accounts / df(e))
```

即共享实体越稀有（df 越小）越可疑：共用一台冷门设备 ≫ 共用公司出口 IP。
类型权重 device/card=1.0、address=0.9、ip=0.3；df 超帽（如出口 IP>50）的实体不参与投影；
仅由单一弱证据（如碰巧同 IP）连接的边被剪枝。

**社区发现**：Leiden（igraph C 实现）。选它而不是 Louvain，因为 Leiden 保证切出的
社区内部连通（不会把团伙切碎或粘连），支持边权与分辨率参数，百万级边秒级完成。

**可疑分**（0–100，≥60 标记可疑）= 0.25·规模 + 0.30·均边权 + 0.30·券强度 + 0.15·密度。
券强度是相对全站均值的核销提升，经加-k 平滑（k=20），避免 2 人小团偶然多核销
几张券被误判。

**证据**：每个团伙给出成员间共享的稀有实体（按 df 升序）和最强边及其共享实体，
详情页有 SVG 关系图（节点越红用券越多，边越粗共享实体越稀有）。

## 性能（M 系列 Mac，100 万账号 / 440 万链接）

| 阶段 | 耗时 |
|---|---|
| 数据生成 | ~4s |
| 增量导入（580 万行） | ~16s |
| 分析（构图+Leiden+打分） | ~13s |

## 模拟数据验证

- 埋入 12 个团伙（8–40 人，共享设备/地址/卡/IP，高频用券）：全部检出，
  占据可疑分 top-12（91–98 分），平均最佳 Jaccard 0.93；
- 正常共享（3% 合租、2% 家庭卡、200 个公司出口 IP）：最高分 46.6，无一误报；
- 增量加入第 8 天数据（含新团伙）：重跑后新团伙 20/20 人完整成社区，得分 94.4。

## 文件

```
fraudring/
  generate.py   模拟数据生成（团伙 + 正常共享）
  db.py         SQLite schema + 幂等增量导入
  graph.py      IDF 加权投影（numpy 向量化）
  community.py  Leiden 社区发现
  score.py      可疑分 + 证据提取
  pipeline.py   分析流水线 + ground truth 评估
  webapp.py     Flask 查看页（列表/详情/SVG 关系图/JSON API）
```
