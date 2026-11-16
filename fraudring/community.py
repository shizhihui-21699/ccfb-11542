"""社区发现: Leiden 算法 (igraph C 实现).

选型理由:
  - Leiden 是 Louvain 的修正版, 保证切出的社区内部连通, 不会把团伙切碎/粘连;
  - C 实现, 百万级边秒级完成; 支持边权(我们的 IDF 权重)和分辨率参数;
  - 团伙是 5-50 人的小稠密团, modularity 目标即可; 若团伙偏小可调大 resolution.
"""
import igraph as ig


def find_communities(g, resolution=1.0):
    """返回 membership 列表 (顶点下标 -> 社区id)."""
    clustering = g.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,  # 跑到收敛
    )
    return clustering.membership
