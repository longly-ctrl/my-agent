from copy import deepcopy  # 复制资料内容，避免融合结果与输入共享可变对象。
from math import isfinite  # 检查权重是否包含 NaN 或无穷大。

def fuse_rrf(  # 按可选通道权重融合已有排名。
    rankings: dict[str, list[dict]],  # 接收各通道已经排序的结果。
    top_k: int = 3,  # 指定最终保留数量。
    rank_constant: int = 60,  # 控制排名贡献的平滑程度。
    weights: dict[str, float] | None = None,  # 不传时每个通道仍使用权重一。
) -> list[dict]:
    if type(top_k) is not int or top_k <= 0:  # 检查最终返回数量。
        raise ValueError("top_k 必须是正整数")  # 拒绝非法数量。

    if type(rank_constant) is not int or rank_constant <= 0:  # 当前实现要求排名常数为正整数。
        raise ValueError("rank_constant 必须是正整数")  # 拒绝非法融合参数。

    channel_weights = {name: 1.0 for name in rankings} if weights is None else dict(weights)  # 建立本次融合使用的权重快照。
    if set(channel_weights) != set(rankings):  # 检查权重名称是否恰好覆盖所有输入通道。
        raise ValueError("权重必须与召回通道一一对应")  # 避免漏配或拼错名称。

    for weight in channel_weights.values():  # 逐个检查通道权重。
        if type(weight) not in (int, float) or not isfinite(weight) or weight <= 0:  # 当前实现只接受有限正数。
            raise ValueError("权重必须是有限正数")  # 拒绝无效权重。

    merged = {}  # 按 chunk_id 保存唯一的融合条目。

    for channel, items in sorted(rankings.items()):  # 固定通道处理顺序，让重复运行保持一致。
        seen = set()  # 记录当前通道已经出现的文本块。
        for rank, item in enumerate(items, start=1):  # 输入列表顺序就是排名，从一开始计数。
            cid = item["chunk_id"]  # 获取跨通道关联同一资料的标识。
            if cid in seen:  # 检查同一通道是否重复提供相同文本块。
                raise ValueError(f"通道内出现重复文本块：{channel}，{cid}")  # 防止一个通道重复贡献分数。
            seen.add(cid)  # 标记当前块已经在本通道出现。

            content = {key: value for key, value in item.items() if key != "score"}  # 分离原始评分与资料内容。
            if cid not in merged:  # 首次遇到当前文本块时创建融合条目。
                merged[cid] = {**deepcopy(content), "score": 0.0, "ranks": {}, "retrieval_scores": {}}  # 保存资料副本及各通道诊断信息。
            else:  # 再次遇到同一标识时，核对两路资料是否一致。
                stored = {  # 提取此前保存的原始资料字段。
                    key: value for key, value in merged[cid].items()  # 遍历已有条目。
                    if key not in {"score", "ranks", "retrieval_scores"}  # 排除融合过程新增的字段。
                }
                if stored != content:  # 相同标识必须对应相同正文和来源信息。
                    raise ValueError(f"同一文本块内容不一致：{cid}")  # 拒绝混合冲突数据。

            result = merged[cid]  # 取得当前文本块唯一的融合记录。
            result["score"] += channel_weights[channel] / (rank_constant + rank)  # 累加当前通道加权后的排名贡献。
            result["ranks"][channel] = rank  # 记录当前块在该通道中的排名。
            result["retrieval_scores"][channel] = item["score"]  # 保存原始分数，供观察和排错。

    results = list(merged.values())  # 将去重后的条目转换为列表。
    results.sort(key=lambda item: (-item["score"], item["chunk_id"]))  # 按融合分数降序排列，同分时按标识排序。
    return results[:top_k]  # 返回最终前 top_k 条候选。