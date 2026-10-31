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

def is_target(chunk: dict, case: dict) -> bool:  # 判断一个返回块是否满足当前用例的完整证据要求。
    same_section = [chunk["section_start_paragraph"], chunk["section_end_paragraph"]] == case["target_section"]  # 核对所属章节。
    has_evidence = all(fragment in chunk["text"] for fragment in case["evidence_contains"])  # 检查同一块包含全部证据片段。
    return same_section and has_evidence  # 位置与正文条件必须同时满足。


dataset = json.loads((root / "retrieval_holdout_cases.json").read_text(encoding="utf-8-sig"))  # 加载未参与选参的留出集。
cases = dataset["cases"]  # 取得待执行的问题列表。
if dataset["schema_version"] != 1 or not cases:  # 检查评估结构版本和用例数量。
    raise ValueError("评估集版本错误或没有用例")  # 避免错误输入产生误导性统计。

if {c["source_hash"] for c in payload["chunks"]} != {dataset["source_hash"]}:  # 检查证据标签对应的资料版本。
    raise ValueError("评估集与缓存的资料版本不一致")  # 阻止使用过期的位置标签。

case_ids = [case["id"] for case in cases]  # 收集用例标识。
if len(set(case_ids)) != len(case_ids):  # 检查是否重复统计了同一个标识。
    raise ValueError("评估用例 ID 不能重复")  # 拒绝含糊的用例记录。

for case in cases:  # 在加载模型前核对各条目标证据。
    if not case["evidence_contains"] or not any(is_target(c, case) for c in payload["chunks"]):  # 检查证据条件非空且目标确实存在。
        raise ValueError(f"目标证据无效：{case['id']}")  # 将标签错误与检索未命中分开处理。

if dataset.get("split") != "holdout":  # 检查当前文件确实是留出验证集。
    raise ValueError("当前评估文件不是留出集")  # 防止误用开发集。

config = dataset["frozen_config"]  # 读取查看新题排名前已经固定的参数。
if config != {"candidate_k": 50, "final_k": 3, "rank_constant": 60, "weights": {"bm25": 0.3, "vector": 0.7}}:  # 核对本次冻结方案。
    raise ValueError("冻结配置发生变化，需要核对实验条件")  # 防止无意中改变比较条件。

candidate_k = config["candidate_k"]  # 使用固定的每路召回数量。
final_k = config["final_k"]  # 使用固定的命中统计范围。
rank_constant = config["rank_constant"]  # 使用固定的排名常数。
weights = config["weights"]  # 使用开发阶段选出的候选权重。
index = build_bm25_index(payload["chunks"])  # 所有问题共用同一份 BM25 索引。
model = load_embedding_model()  # 整批评估只加载一次模型。
rows = []  # 收集每条问题的结果和诊断证据。

for case in cases:  # 按固定顺序执行所有问题。
    query_vector = encode_query(model, case["query"])  # 为当前问题生成查询向量。
    sources = {  # 保存两路原始候选，作为两个融合方案共同的输入。
        "bm25": search_bm25(index, case["query"], top_k=candidate_k),  # 取得关键词候选。
        "vector": search_vectors(index["chunks"], payload["vectors"], query_vector, top_k=candidate_k),  # 取得向量候选。
    }
    rankings = {  # 保留单路结果和两个融合对照。
        **sources,  # 加入原始 BM25 与向量排名。
        "hybrid_equal": fuse_rrf(sources, top_k=2 * candidate_k, rank_constant=rank_constant),  # 使用默认等权融合。
        "hybrid_weighted": fuse_rrf(sources, top_k=2 * candidate_k, rank_constant=rank_constant, weights=weights),  # 使用冻结权重融合。
    }

    first_ranks = {  # 查找每种方式首次命中完整证据的位置。
        name: next((rank for rank, chunk in enumerate(items, start=1) if is_target(chunk, case)), None)  # 候选中没有目标时记录空值。
        for name, items in rankings.items()  # 对三种方式使用相同判断规则。
    }
    hits = {name: rank is not None and rank <= final_k for name, rank in first_ranks.items()}  # 命中范围仍严格限制为前三。
    rows.append({  # 保存当前问题的统计及完整候选。
        "id": case["id"],  # 关联评估用例。
        "query": case["query"],  # 保留实际问题。
        "first_target_ranks": first_ranks,  # 记录目标在保留候选中的首次排名。
        "hit_at_3": hits,  # 记录各路是否命中前三。
        "rankings": rankings,  # 保留正文、来源、分数和融合排名。
    })
    print(case["id"], "FIRST_RANKS=", first_ranks)  # 展示当前问题的诊断结果。

summary = {}  # 保存三种方式的汇总指标。
for name in ("bm25", "vector", "hybrid_equal", "hybrid_weighted"):  # 分别统计四个对照方案。
    hit_count = sum(row["hit_at_3"][name] for row in rows)  # 布尔值按命中一次计一。
    summary[name] = {"hits": hit_count, "total": len(rows), "hit_at_3": hit_count / len(rows)}  # 计算查询级目标命中率。

report = {  # 保存后续比较需要的配置、标签和实际结果。
    "dataset": dataset,  # 将本次使用的评估标签一并保存。
    "index_config": expected,  # 记录模型与分块配置。
    "frozen_config": config,  # 完整记录本次冻结方案，便于复查实验条件。
    "query_instruction": QUERY_INSTRUCTION,  # 记录查询编码前缀。
    "candidate_k": candidate_k,  # 记录每路候选范围。
    "final_k": final_k,  # 记录命中统计范围。
    "rank_constant": rank_constant,  # 记录融合参数。
    "results": rows,  # 保存逐题结果。
    "summary": summary,  # 保存三路汇总。
}

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")  # 生成不会轻易重名的报告时间标识。
output = root / "eval_reports" / f"retrieval_holdout_eval_{stamp}.json"  # 单独保存留出集结果。
output.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
temporary = output.with_suffix(".json.tmp")  # 先写临时文件。
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 保存完整对照数据。
temporary.replace(output)  # 写入完成后替换正式文件。

for name, metrics in summary.items():  # 展示三种方式的整体命中情况。
    print(name, "HITS=", metrics["hits"], "TOTAL=", metrics["total"], "HIT_AT_3=", metrics["hit_at_3"])  # 同时展示分子、分母和比例。
print("REPORT=", output)  # 输出报告位置。
print("HOLDOUT_EVAL_SAVED")  # 表示验证报告已保存，实际质量以逐题结果为准。