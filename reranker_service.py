from copy import deepcopy  # 复制返回结果，避免修改原始候选。
from math import isfinite  # 检查精排分数是否包含非法数值。
from pathlib import Path  # 定位本地模型缓存。
from sentence_transformers import CrossEncoder  # 加载文本对精排模型。
from torch.nn import Identity  # 保留模型原始相关性分数。

RERANKER_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # 使用已经验收的精排模型。
RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"  # 固定模型版本。


def load_reranker() -> CrossEncoder:  # 显式加载模型，供调用方复用。
    cache_dir = Path(__file__).resolve().parent / ".cache" / "models"  # 定位已有模型文件。
    return CrossEncoder(  # 创建并返回精排模型。
        RERANKER_ID,  # 指定模型名称。
        revision=RERANKER_REVISION,  # 固定权重和分词器版本。
        device="cpu",  # 使用 CPU 推理。
        cache_folder=str(cache_dir),  # 使用项目内缓存。
        local_files_only=True,  # 仅加载已经下载的文件。
        max_length=512,  # 设置问题与原文合并后的输入上限。
        activation_fn=Identity(),  # 输出原始分数。
        trust_remote_code=False,  # 使用库内置模型实现。
        model_kwargs={"use_safetensors": True},  # 加载已验收的权重格式。
    )


def rerank_candidates(model: CrossEncoder, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:  # 对 RRF 候选重新评分并保留来源。
    if type(top_k) is not int or top_k <= 0:  # 检查返回数量。
        raise ValueError("top_k 必须是正整数")  # 拒绝非法数量。

    question = query.strip()  # 去除问题两端的空白。
    if not question:  # 检查是否为空问题。
        raise ValueError("查询不能为空")  # 避免无意义的评分。

    ids = [c["chunk_id"] for c in candidates]  # 收集候选标识并保留输入顺序。
    if len(set(ids)) != len(ids):  # 检查候选是否已经去重。
        raise ValueError("候选 chunk_id 不能重复")  # 避免重复资料占用结果位置。

    if not candidates:  # 检查是否没有候选。
        return []  # 空候选直接返回，不调用模型。

    texts = ["\n".join([*c["heading_path"], c["text"]]) for c in candidates]  # 将每块的标题和正文作为待评分原文。
    tokenized = model.tokenizer([question] * len(texts), texts, add_special_tokens=True, padding=False, truncation=False)  # 检查完整问题与原文的联合长度。

    for cid, tokens in zip(ids, tokenized["input_ids"]):  # 逐个核对文本对长度。
        if len(tokens) > model.max_seq_length:  # 使用当前版本的输入长度属性。
            raise ValueError(f"精排文本对超限：{cid}")  # 在模型计算前拒绝超限输入。

    scores = model.predict(  # 为每个问题与原文组合计算分数。
        [(question, text) for text in texts],  # 保持输出分数与候选顺序一致。
        batch_size=1,  # 控制 CPU 推理的批次内存占用。
        convert_to_numpy=True,  # 返回可检查形状的数组。
        show_progress_bar=True,  # 显示候选处理进度。
    )

    if scores.shape != (len(candidates),) or not all(isfinite(float(s)) for s in scores):  # 检查分数数量和数值有效性。
        raise ValueError("精排分数数量或数值异常")  # 阻止异常结果进入排序。

    results = []  # 收集带精排分数的独立结果。
    for candidate, score in zip(candidates, scores):  # 按输入顺序绑定候选和分数。
        result = deepcopy(candidate)  # 保留原文、来源和通道排名的独立副本。
        result.setdefault("rrf_score", result["score"])  # 保留最初的 RRF 分数，便于比较。
        result["score"] = float(score)  # 将通用评分字段更新为精排分数。
        results.append(result)  # 收集当前结果。

    results.sort(key=lambda item: (-item["score"], item["chunk_id"]))  # 按精排分数降序排列，同分时固定顺序。
    return results[:top_k]  # 返回最终候选，保留零分和负分。