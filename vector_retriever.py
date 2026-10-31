from math import fsum, hypot, isfinite  # 分别用于稳定求和、计算向量长度和检查有限数值。
from copy import deepcopy  # 复制返回条目，避免调用方修改结果时影响原始文本块。

def cosine_similarity(left: list[float], right: list[float]) -> float:  # 计算两个非零、同维向量的余弦相似度。
    if not left or len(left) != len(right):  # 检查向量非空，并且对应位置能够一一配对。
        raise ValueError("向量必须非空且维度相同")  # 拒绝空输入或维度不一致。

    if not all(isfinite(x) for x in left) or not all(isfinite(x) for x in right):  # 检查是否包含 NaN 或无穷大。
        raise ValueError("向量元素必须是有限数")  # 避免非法数值污染最终分数。

    left_norm = hypot(*left)  # 计算左向量的欧几里得长度，减少直接平方带来的溢出问题。
    right_norm = hypot(*right)  # 使用相同方法计算右向量长度。

    if left_norm == 0 or right_norm == 0:  # 零向量没有可用于比较的方向。
        raise ValueError("不能计算零向量的余弦相似度")  # 避免除以零，也避免将非法输入伪装成零分。

    if not isfinite(left_norm) or not isfinite(right_norm):  # 检查计算出的长度是否超出浮点数可表示范围。
        raise ValueError("向量长度超出有限浮点数范围")  # 明确拒绝无法可靠归一化的输入。

    score = fsum((a / left_norm) * (b / right_norm) for a, b in zip(left, right))  # 先归一化再求点积，减少大数乘法溢出。
    return max(-1.0, min(1.0, score))  # 修正浮点舍入可能造成的轻微越界。
    
def search_vectors(chunks: list[dict], vectors: dict[str, list[float]], query_vector: list[float], top_k: int = 3) -> list[dict]:  # 按余弦相似度返回前若干文本块。
    if type(top_k) is not int or top_k <= 0:  # 要求返回数量为正整数，拒绝布尔值。
        raise ValueError("top_k 必须是正整数")  # 明确拒绝非法结果数量。

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]  # 收集本次检索范围内的文本块标识。
    if len(set(chunk_ids)) != len(chunk_ids):  # 检查是否有两个条目共用同一标识。
        raise ValueError("chunk_id 不能重复")  # 防止向量与正文的对应关系出现歧义。

    if set(vectors) != set(chunk_ids):  # 检查每个文本块恰好有对应向量，且没有额外条目。
        raise ValueError("向量标识必须与文本块标识完全一致")  # 拒绝缺失或混入其他语料的向量集合。

    cosine_similarity(query_vector, query_vector)  # 校验查询向量，即使语料为空也拒绝非法查询。
    results = []  # 收集所有文本块及其相似度。

    for chunk in chunks:  # 遍历待检索文本块。
        score = cosine_similarity(query_vector, vectors[chunk["chunk_id"]])  # 按标识取得正确向量并评分。
        result = deepcopy(chunk)  # 复制文本及来源元数据。
        result["score"] = score  # 添加当前查询对应的相似度。
        results.append(result)  # 收集候选结果，保留零分和负分。

    results.sort(key=lambda item: (-item["score"], item["chunk_id"]))  # 按分数降序排列，同分时按标识排序。
    return results[:top_k]  # 返回最接近的若干候选，数量不足时返回全部。