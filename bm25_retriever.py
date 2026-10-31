import re  # 使用标准库正则表达式识别英文术语和连续中文。
from collections import Counter  # 统计各个词项出现的次数。
from copy import deepcopy  # 复制文本块及其内部列表，建立独立的数据快照。
from math import isfinite, log1p  # 检查数值是否有限，并计算自然对数 log(1+x)。

# 第一部分匹配字母、数字和下划线，并允许末尾带 ++ 或 #；第二部分匹配连续的基本汉字。
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+(?:\+\+|#)?|[\u4e00-\u9fff]+")  # 预编译规则，避免每次调用时重新构造。


def tokenize(text: str) -> list[str]:  # 将正文或查询转换为使用相同规则的词项列表。
    tokens = []  # 保存切分结果，保留重复词项。

    for piece in TOKEN_PATTERN.findall(text.casefold()):  # 先统一大小写，再依次提取术语或中文片段。
        if piece.isascii():  # 判断当前片段是否由 ASCII 字符组成。
            tokens.append(piece)  # 将英文术语、数字或标识符作为一个完整词项。
        else:  # 按当前正则规则，此分支处理连续中文片段。
            tokens.extend(piece)  # 加入每个汉字，使单字查询也能够匹配。
            tokens.extend(piece[i:i + 2] for i in range(len(piece) - 1))  # 加入相邻双字串，补充局部顺序信息。

    return tokens  # 返回词项列表；空文本或纯标点返回空列表。
    
def build_bm25_index(chunks: list[dict]) -> dict:  # 为文本块建立 BM25 所需的内存统计。
    documents = deepcopy(chunks)  # 复制输入，避免调用方后续修改原数据而影响索引。
    chunk_ids = [chunk["chunk_id"] for chunk in documents]  # 收集所有文本块标识。

    if len(set(chunk_ids)) != len(chunk_ids):  # 检查不同条目是否错误地使用了相同标识。
        raise ValueError("chunk_id 不能重复")  # 防止后续合并检索结果时混淆条目。

    term_freqs = []  # 按文本块顺序保存每块的词频。
    doc_lengths = []  # 按相同顺序保存每块的词项数量。
    doc_freqs = Counter()  # 统计每个词项出现在多少个文本块中。

    for chunk in documents:  # 逐块建立统计，保证各个列表的位置一一对应。
        text = "\n".join([*chunk["heading_path"], chunk["text"]])  # 将标题路径和正文一起用于检索。
        frequencies = Counter(tokenize(text))  # 对当前块分词，并统计每个词项的出现次数。
        term_freqs.append(frequencies)  # 保存当前块的词频表。
        doc_lengths.append(sum(frequencies.values()))  # 使用词项总数作为长度，包含重复项。
        doc_freqs.update(frequencies.keys())  # 每个不同词项只给文档频率增加一次。

    count = len(documents)  # 记录参与统计的文本块数量。

    return {  # 返回原始条目及其对应的统计信息。
        "chunks": documents,  # 保存独立的文本块快照。
        "term_freqs": term_freqs,  # 下标 i 对应 chunks[i] 的词频。
        "doc_lengths": doc_lengths,  # 下标 i 对应 chunks[i] 的词项数量。
        "doc_freqs": doc_freqs,  # 保存跨文本块的文档频率。
        "doc_count": count,  # 保存文本块总数。
        "avgdl": sum(doc_lengths) / count if count else 0.0,  # 计算平均长度，空语料返回零。
    }
    
def search_bm25(index: dict, query: str, top_k: int = 3, k1: float = 1.2, b: float = 0.75) -> list[dict]:  # 返回分数最高的若干文本块。
    if type(top_k) is not int or top_k <= 0:  # 要求结果数量为正整数，同时拒绝布尔值。
        raise ValueError("top_k 必须是正整数")  # 避免非法切片或没有意义的数量。

    if not isfinite(k1) or k1 <= 0:  # 当前实现要求词频参数为有限正数。
        raise ValueError("k1 必须是有限正数")  # 拒绝无穷大、非数值结果和非正数。

    if not isfinite(b) or not 0 <= b <= 1:  # 长度归一化强度必须位于零到一之间。
        raise ValueError("b 必须是 0 到 1 之间的有限数")  # 拒绝无效的归一化参数。

    terms = sorted(set(tokenize(query)))  # 查询词去重并固定计算顺序，本版不按查询中的重复次数加权。
    if not terms or index["doc_count"] == 0 or index["avgdl"] == 0:  # 检查是否具备可评分的查询和语料。
        return []  # 无有效查询词或语料时直接返回空结果。

    results = []  # 收集至少命中一个查询词的文本块。

    for i, chunk in enumerate(index["chunks"]):  # 按索引顺序扫描所有文本块。
        frequencies = index["term_freqs"][i]  # 获取当前块的词频表。
        length = index["doc_lengths"][i]  # 获取包含标题在内的词项数量。
        norm = k1 * (1 - b + b * length / index["avgdl"])  # 计算评分分母中的长度调整项。
        score = 0.0  # 从零开始累计查询词的贡献。

        for term in terms:  # 分别计算每个不同查询词的贡献。
            tf = frequencies.get(term, 0)  # 查询当前块中该词的出现次数。
            if tf == 0:  # 当前块没有这个词时，不贡献分数。
                continue  # 跳过当前词，继续处理下一个查询词。

            df = index["doc_freqs"][term]  # 获取整个语料中包含该词的文本块数量。
            idf = log1p((index["doc_count"] - df + 0.5) / (df + 0.5))  # 计算逆文档频率，较少见的词权重较高。
            score += idf * tf * (k1 + 1) / (tf + norm)  # 累计词频饱和与长度归一化后的贡献。

        if score > 0:  # 只保留实际命中查询词的文本块。
            result = deepcopy(chunk)  # 复制原始条目，防止返回结果与索引共享可变数据。
            result["score"] = score  # 为返回条目添加分数，保留原有 chunk_id 和来源信息。
            results.append(result)  # 收集当前命中结果。

    results.sort(key=lambda item: (-item["score"], item["chunk_id"]))  # 按分数降序排列，同分时按标识确定顺序。
    return results[:top_k]  # 返回前 top_k 条；命中不足时返回全部命中项。