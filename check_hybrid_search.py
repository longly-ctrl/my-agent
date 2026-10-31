import json  # 读取向量缓存并保存对照报告。
from datetime import datetime, timezone  # 为每次报告生成独立的 UTC 时间标识。
from pathlib import Path  # 定位缓存和报告目录。
from bm25_retriever import build_bm25_index, search_bm25  # 构建关键词索引并召回候选。
from embedding_service import MODEL_ID, MODEL_REVISION, DIMENSION, CHUNK_PARAMS, load_embedding_model, encode_query, QUERY_INSTRUCTION  # 导入模型配置和查询编码能力。
from vector_retriever import search_vectors  # 根据查询向量召回资料。
from hybrid_retriever import fuse_rrf  # 融合两路候选的排名。

root = Path(__file__).resolve().parent  # 以脚本所在目录作为项目根目录。
payload = json.loads((root / ".cache" / "knowledge_vectors.json").read_text(encoding="utf-8"))  # 加载同一份资料快照。

expected = {  # 定义当前程序支持的缓存配置。
    "schema_version": 1,  # 要求当前缓存结构。
    "model_id": MODEL_ID,  # 要求模型名称一致。
    "model_revision": MODEL_REVISION,  # 要求模型版本一致。
    "dimension": DIMENSION,  # 要求向量维度一致。
    "normalize_embeddings": True,  # 要求资料向量已归一化。
    "text_format": "heading-path-newline-body-v1",  # 要求资料文本连接方式一致。
    "chunk_params": CHUNK_PARAMS,  # 要求分块参数一致。
}
for field, value in expected.items():  # 逐项检查缓存。
    if payload.get(field) != value:  # 检查配置是否不匹配。
        raise ValueError(f"缓存配置不匹配：{field}")  # 阻止混用不同配置的索引。

query = "显存不足时，如何用较少资源微调大模型？"  # 固定问题，便于比较不同检索方式。
candidate_k = 50  # 每路先召回五十条，最终展示数量另行控制。
index = build_bm25_index(payload["chunks"])  # 直接使用缓存里的文本块建立 BM25 索引。
model = load_embedding_model()  # 加载一次本地嵌入模型。
query_vector = encode_query(model, query)  # 为同一个问题生成查询向量。

rankings = {  # 保存两路完整候选，供融合与诊断。
    "bm25": search_bm25(index, query, top_k=candidate_k),  # 按关键词相关性召回。
    "vector": search_vectors(index["chunks"], payload["vectors"], query_vector, top_k=candidate_k),  # 在完全相同的文本块集合上召回。
}
hybrid = fuse_rrf(rankings, top_k=3, rank_constant=60)  # 将两路候选融合成最终三条结果。

top3 = {name: items[:3] for name, items in rankings.items()}  # 取得两个单独检索方式的前三条。
top3["hybrid"] = hybrid  # 加入混合检索的前三条。
target_hits = {  # 分别判断既定目标章节是否进入前三。
    name: any(r["section_start_paragraph"] == 4040 and r["section_end_paragraph"] == 4058 for r in items)  # 按已核对的章节范围判断命中。
    for name, items in top3.items()  # 对三种方式采用相同的判断标准。
}

report = {  # 保存本次比较的配置、判断结果和证据。
    "query": query,  # 记录原始问题。
    "index_config": expected,  # 记录本次使用的索引约定。
    "query_instruction": QUERY_INSTRUCTION,  # 显式记录查询指令，方便后续复现。
    "candidate_k": candidate_k,  # 记录每路候选数量。
    "rank_constant": 60,  # 记录融合参数。
    "target_section": [4040, 4058],  # 记录本条用例的目标章节。
    "target_in_top3": target_hits,  # 记录三种方式是否命中目标。
    "rankings": rankings,  # 保留两路候选、原始分数及来源。
    "hybrid": hybrid,  # 保留融合结果与各通道排名。
}

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")  # 生成不会轻易重名的报告时间标识。
output = root / "eval_reports" / f"hybrid_comparison_{stamp}.json"  # 每次生成独立报告。
output.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
temporary = output.with_suffix(".json.tmp")  # 先写临时文件。
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 保存完整对照数据。
temporary.replace(output)  # 写入完成后替换正式文件。

for name, hit in target_hits.items():  # 展示三种方式的目标命中情况。
    print(name.upper() + "_TARGET_IN_TOP3=", hit)  # 输出可直接比较的布尔结果。
print("REPORT=", output)  # 输出报告位置。
print("HYBRID_REPORT_SAVED")  # 只表示报告保存完成，不表示检索质量通过。