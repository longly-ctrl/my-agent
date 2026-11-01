import json  # 读取已经生成的文本块与向量缓存。
from hashlib import sha256  # 计算当前 Word 文件的字节指纹。
from math import hypot, isclose, isfinite  # 检查向量长度、归一化状态和数值有效性。
from pathlib import Path  # 表示缓存与原始文档路径。
from embedding_service import MODEL_ID, MODEL_REVISION, DIMENSION, CHUNK_PARAMS  # 复用当前嵌入配置，不加载模型权重。
from threading import Lock  # 让同一服务实例的查询串行使用共享模型。
from bm25_retriever import build_bm25_index, search_bm25  # 构建并复用关键词索引。
from embedding_service import load_embedding_model, encode_query  # 加载查询模型并编码问题。
from vector_retriever import search_vectors  # 根据查询向量召回资料。
from hybrid_retriever import fuse_rrf  # 合并两路召回结果。
from reranker_service import load_reranker, rerank_candidates  # 提供可选的本地精排能力。

def load_retrieval_cache(cache_file: Path, source_file: Path) -> dict:  # 加载并校验检索缓存，返回独立读取的数据。
    payload = json.loads(cache_file.read_text(encoding="utf-8"))  # 读取缓存；文件或 JSON 错误直接向调用方传播。

    expected = {  # 定义当前检索链路支持的缓存配置。
        "schema_version": 1,  # 要求当前缓存结构版本。
        "model_id": MODEL_ID,  # 要求资料向量来自同一个嵌入模型。
        "model_revision": MODEL_REVISION,  # 要求模型权重版本一致。
        "dimension": DIMENSION,  # 要求向量维度一致。
        "normalize_embeddings": True,  # 要求缓存声明采用归一化向量。
        "text_format": "heading-path-newline-body-v1",  # 要求标题与正文的连接方式一致。
        "chunk_params": CHUNK_PARAMS,  # 要求切块长度与重叠参数一致。
    }
    for field, value in expected.items():  # 逐项核对缓存配置。
        if payload.get(field) != value:  # 检查配置是否缺失或不匹配。
            raise ValueError(f"缓存配置不匹配：{field}")  # 拒绝混用不同实验条件的缓存。

    chunks = payload.get("chunks")  # 获取原文块列表。
    vectors = payload.get("vectors")  # 获取文本块标识到向量的映射。
    if not isinstance(chunks, list) or not chunks or not isinstance(vectors, dict):  # 检查缓存基本结构及非空语料。
        raise ValueError("缓存必须包含非空文本块列表和向量字典")  # 阻止无效结构进入建索引阶段。

    if not all(isinstance(chunk, dict) for chunk in chunks):  # 检查每个文本块是否为字典记录。
        raise ValueError("缓存中的文本块结构错误")  # 避免后续字段访问产生含糊错误。

    ids = [chunk.get("chunk_id") for chunk in chunks]  # 按原始顺序收集文本块标识。
    if any(not isinstance(cid, str) or not cid for cid in ids):  # 要求每个标识都是非空字符串。
        raise ValueError("文本块标识必须是非空字符串")  # 拒绝缺失或非法标识。

    if len(set(ids)) != len(ids) or set(vectors) != set(ids):  # 检查标识唯一且向量恰好覆盖所有文本块。
        raise ValueError("文本块标识重复，或向量与文本块未一一对应")  # 防止覆盖、缺失或混入其他资料。

    source = source_file.resolve()  # 解析调用方明确指定的原始文档路径。
    source_hash = sha256(source.read_bytes()).hexdigest()  # 对当前文档的一份字节快照计算指纹。

    for chunk in chunks:  # 逐块核对资料来源与向量。
        if Path(chunk["source"]) != source or chunk["source_hash"] != source_hash:  # 检查路径和文件版本是否一致。
            raise ValueError("缓存与当前原始文档不一致，需要重新构建缓存")  # 拒绝使用旧资料的段落位置和向量。

        vector = vectors[chunk["chunk_id"]]  # 根据唯一标识取得对应向量。
        if not isinstance(vector, list) or len(vector) != DIMENSION:  # 检查向量结构与维度。
            raise ValueError(f"向量维度错误：{chunk['chunk_id']}")  # 定位异常文本块。

        if any(type(value) not in (int, float) or not isfinite(value) for value in vector):  # 拒绝布尔值、字符串、NaN 和无穷大。
            raise ValueError(f"向量包含非法数值：{chunk['chunk_id']}")  # 阻止异常数值污染相似度计算。

        if not isclose(hypot(*vector), 1.0, abs_tol=1e-5):  # 检查实际向量是否接近单位长度。
            raise ValueError(f"向量未正确归一化：{chunk['chunk_id']}")  # 拒绝零向量或不符合编码约定的向量。

    return payload  # 返回已经校验的数据，供后续服务复用。
    
class RetrievalService:  # 集中管理检索资源，并提供统一查询入口。
    def __init__(  # 创建服务时初始化资源，供后续多次查询复用。
        self,  # 接收当前服务实例。
        cache_file: Path,  # 指定已经生成的检索缓存。
        source_file: Path,  # 指定缓存对应的原始 Word 文件。
        *,  # 后面的配置必须使用参数名传入。
        enable_rerank: bool = False,  # 默认关闭精排，允许显式开启。
        candidate_k: int = 50,  # 指定每路召回数量。
        rank_constant: int = 60,  # 指定 RRF 排名平滑常数。
        weights: dict[str, float] | None = None,  # 不传权重时沿用默认等权融合。
    ):
        if type(enable_rerank) is not bool:  # 避免将字符串等对象误当成开关。
            raise ValueError("enable_rerank 必须是布尔值")  # 拒绝含糊配置。

        if type(candidate_k) is not int or candidate_k <= 0:  # 检查召回数量，同时拒绝布尔值。
            raise ValueError("candidate_k 必须是正整数")  # 避免无效的召回范围。

        self._weights = None if weights is None else dict(weights)  # 保存权重副本，避免调用方修改原字典影响服务。
        fuse_rrf({"bm25": [], "vector": []}, top_k=1, rank_constant=rank_constant, weights=self._weights)  # 用空候选提前校验融合参数。

        self._candidate_k = candidate_k  # 保存每路召回数量。
        self._rank_constant = rank_constant  # 保存排名常数。
        payload = load_retrieval_cache(cache_file, source_file)  # 校验缓存配置、资料版本和向量。
        self._index = build_bm25_index(payload["chunks"])  # 建立独立的文本块快照和 BM25 统计。
        self._vectors = payload["vectors"]  # 持有本次读取的资料向量。
        self._embedding_model = load_embedding_model()  # 加载一次查询编码模型。
        self._reranker = load_reranker() if enable_rerank else None  # 仅在开启时加载精排权重。
        self._lock = Lock()  # 为当前实例建立查询互斥锁。

    def search(self, query: str, top_k: int = 3) -> list[dict]:  # 返回保留正文、来源和排名信息的候选。
        if type(top_k) is not int or top_k <= 0:  # 检查最终返回数量。
            raise ValueError("top_k 必须是正整数")  # 拒绝非法数量。

        if not isinstance(query, str) or not query.strip():  # 要求问题是包含实际内容的字符串。
            raise ValueError("查询必须是非空字符串")  # 拒绝空问题和错误类型。

        question = query.strip()  # 统一两路检索实际使用的问题文本。
        with self._lock:  # 串行执行同一实例的查询，保护共享模型的使用边界。
            query_vector = encode_query(self._embedding_model, question)  # 复用模型生成查询向量。
            sources = {  # 保存两路独立召回结果。
                "bm25": search_bm25(self._index, question, top_k=self._candidate_k),  # 召回关键词候选。
                "vector": search_vectors(self._index["chunks"], self._vectors, query_vector, top_k=self._candidate_k),  # 召回语义候选。
            }
            candidates = fuse_rrf(  # 对完整候选并集去重并融合排名。
                sources,  # 只融合两路原始结果。
                top_k=2 * self._candidate_k,  # 保留完整并集，避免精排前丢失目标。
                rank_constant=self._rank_constant,  # 使用实例保存的排名常数。
                weights=self._weights,  # 使用实例保存的权重配置。
            )

            if self._reranker is not None:  # 检查当前实例是否开启精排。
                return rerank_candidates(self._reranker, question, candidates, top_k=top_k)  # 精排完整并集后返回结果。

            return candidates[:top_k]  # 未开启精排时返回融合排名靠前的结果。