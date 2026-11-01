import json  # 读取历史候选并保存本次评估结果。
from datetime import datetime, timezone  # 生成独立报告的 UTC 时间标识。
from pathlib import Path  # 定位项目文件与报告目录。
from time import perf_counter  # 测量模型加载和逐题重排耗时。
from hybrid_retriever import fuse_rrf  # 从两路原始候选重建融合基线。
from reranker_service import RERANKER_ID, RERANKER_REVISION, load_reranker, rerank_candidates  # 复用已经验收的重排服务。


def is_target(chunk: dict, case: dict) -> bool:  # 沿用此前固定的目标证据判断规则。
    same_section = [chunk["section_start_paragraph"], chunk["section_end_paragraph"]] == case["target_section"]  # 检查章节范围。
    has_evidence = all(fragment in chunk["text"] for fragment in case["evidence_contains"])  # 检查同一块是否包含全部指定证据。
    return same_section and has_evidence  # 章节与证据条件必须同时满足。


root = Path(__file__).resolve().parent  # 使用当前脚本所在目录作为项目根目录。
inputs = {  # 固定输入报告，避免自动选中不同实验产生的文件。
    "development": "retrieval_eval_20261031T021200_202968Z.json",  # 使用原来的四道开发题。
    "previous_holdout_regression": "retrieval_holdout_eval_20261031T045133_624126Z.json",  # 将原留出题用于回归检查。
}
methods = ("bm25", "vector", "hybrid_equal", "hybrid_weighted", "reranked")  # 固定五种比较方式。
weights = {"bm25": 0.3, "vector": 0.7}  # 保留此前实验使用的融合权重。
tasks = []  # 在加载模型前准备并检查全部任务。
snapshots = []  # 收集两份报告的数据与索引配置。
seen_ids = set()  # 防止重复统计同一道题。

for split, filename in inputs.items():  # 分别读取两组固定问题。
    saved = json.loads((root / "eval_reports" / filename).read_text(encoding="utf-8"))  # 加载历史报告。
    if (saved["candidate_k"], saved["final_k"], saved["rank_constant"]) != (50, 3, 60):  # 核对原始召回与评估参数。
        raise ValueError(f"报告参数不一致：{filename}")  # 阻止混用不同实验条件。

    dataset = saved["dataset"]  # 读取报告内保存的原始标签。
    snapshots.append((dataset["source_hash"], saved["index_config"], saved["query_instruction"]))  # 记录资料版本与编码配置。
    cases = {case["id"]: case for case in dataset["cases"]}  # 按标识关联目标证据。
    rows = {row["id"]: row for row in saved["results"]}  # 按标识关联实际候选。
    if len(cases) != 4 or len(dataset["cases"]) != 4 or len(rows) != 4 or len(saved["results"]) != 4 or set(cases) != set(rows):  # 检查数量、重复标识和关联关系。
        raise ValueError(f"用例或结果不完整：{filename}")  # 拒绝缺失或重复的记录。

    for cid, case in cases.items():  # 准备当前报告中的四道题。
        row = rows[cid]  # 取得同一题的历史检索结果。
        if cid in seen_ids or row["query"] != case["query"] or not case["evidence_contains"]:  # 检查问题、标识与证据条件。
            raise ValueError(f"用例信息异常：{cid}")  # 避免错误配对或空证据条件。
        seen_ids.add(cid)  # 标记当前题已经加入本次评估。

        sources = {name: row["rankings"][name] for name in ("bm25", "vector")}  # 只取两路原始召回结果。
        if any(len(items) > 50 for items in sources.values()):  # 检查每路候选是否超出原实验范围。
            raise ValueError(f"原始候选数量异常：{cid}")  # 防止无意扩大召回范围。
        pool = fuse_rrf(sources, top_k=100, rank_constant=60, weights=weights)  # 保留完整去重并集，最多一百块。
        union_ids = {chunk["chunk_id"] for items in sources.values() for chunk in items}  # 计算两路候选的真实并集。
        if {chunk["chunk_id"] for chunk in pool} != union_ids:  # 检查精排输入是否遗漏候选。
            raise ValueError(f"候选并集不完整：{cid}")  # 避免提前截断导致目标证据丢失。
        if {chunk["source_hash"] for chunk in pool} != {dataset["source_hash"]}:  # 核对候选资料版本。
            raise ValueError(f"候选资料版本不一致：{cid}")  # 阻止使用错位的章节标签。

        tasks.append({"split": split, "case": case, "sources": sources, "pool": pool, "source_report": filename})  # 保存已经检查的任务。

if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):  # 比较两组问题使用的资料、索引和查询编码配置。
    raise ValueError("两份输入报告的资料或索引配置不一致")  # 保证本次比较使用同一套语料条件。

started = perf_counter()  # 开始单独测量模型加载耗时。
model = load_reranker()  # 整批只加载一次本地模型。
load_seconds = perf_counter() - started  # 记录加载耗时，不计入逐题重排时间。
results = []  # 收集逐题结果与人工复核材料。

for task in tasks:  # 串行执行八道固定问题。
    case = task["case"]  # 取得当前问题及证据标签。
    pool = task["pool"]  # 使用完整候选并集。
    rankings = {**task["sources"]}  # 保留两个单路检索基线。
    rankings["hybrid_equal"] = fuse_rrf(task["sources"], top_k=100, rank_constant=60)  # 重建等权融合基线。
    rankings["hybrid_weighted"] = pool  # 保留固定权重的融合基线。

    started = perf_counter()  # 从联合分词检查开始测量重排耗时。
    rankings["reranked"] = rerank_candidates(model, case["query"], pool, top_k=len(pool))  # 对完整候选重新评分并保留完整排名。
    rerank_seconds = perf_counter() - started  # 包含分词、模型推理、复制和排序，不包含原始召回。

    first_ranks = {  # 对五种方式使用完全相同的目标证据规则。
        name: next((rank for rank, chunk in enumerate(rankings[name], 1) if is_target(chunk, case)), None)  # 查找首个目标；未找到时记录空值。
        for name in methods  # 依次检查各个基线与重排结果。
    }
    hits = {name: rank is not None and rank <= 3 for name, rank in first_ranks.items()}  # 严格按前三名统计命中。
    results.append({  # 保存当前题的诊断信息。
        "split": task["split"],  # 保留问题分组。
        "case": case,  # 保存问题与目标标签。
        "source_report": task["source_report"],  # 关联原始候选报告。
        "candidate_count": len(pool),  # 记录实际参与重排的块数。
        "rerank_seconds": rerank_seconds,  # 保存本题重排耗时。
        "first_target_ranks": first_ranks,  # 保存目标在五种方式中的首次排名。
        "hit_at_3": hits,  # 保存五种方式的前三命中情况。
        "top3": {name: rankings[name][:3] for name in methods},  # 保留各方案的前三原文供人工核对。
        "reranked": rankings["reranked"],  # 保存完整精排顺序与原始 RRF 分数。
        "review_status": "pending",  # 自动命中统计不能替代人工相关性复核。
    })
    print(case["id"], "CANDIDATES=", len(pool), "FIRST_RANKS=", first_ranks, "RERANK_SECONDS=", round(rerank_seconds, 3))  # 展示逐题变化。

summary = {}  # 分组保存统计，避免把回归题包装成新的独立验证集。
for split in inputs:  # 分别汇总开发题和原留出回归题。
    group = [row for row in results if row["split"] == split]  # 取得当前组的四道题。
    summary[split] = {}  # 初始化当前组的统计。
    for name in methods:  # 对五种方式分别计数。
        hit_count = sum(row["hit_at_3"][name] for row in group)  # 统计命中前三的题数。
        summary[split][name] = {"hits": hit_count, "total": len(group), "hit_at_3": hit_count / len(group)}  # 保存分子、分母和比例。
        print(split, name, summary[split][name])  # 显示分组对照结果。

report = {  # 保存复查实验所需的信息。
    "input_reports": inputs,  # 记录固定输入文件。
    "source_hash": snapshots[0][0],  # 记录资料快照。
    "index_config": snapshots[0][1],  # 记录原始召回使用的索引配置。
    "query_instruction": snapshots[0][2],  # 记录原始向量查询的编码前缀。
    "reranker": {"model_id": RERANKER_ID, "revision": RERANKER_REVISION, "max_seq_length": model.max_seq_length, "batch_size": 1},  # 记录精排配置。
    "fusion": {"candidate_k": 50, "rank_constant": 60, "weights": weights, "final_k": 3, "rerank_input": "full_union"},  # 记录比较范围。
    "load_seconds": load_seconds,  # 单独保存模型加载耗时。
    "timing_scope": "逐题耗时仅包含重排服务，不包含原始召回、模型加载或回答生成。",  # 明确耗时数据的边界。
    "evaluation_role": "开发题与已用于错误分析的原留出回归题；不是新的独立测试集。",  # 记录本次评估的适用范围。
    "results": results,  # 保存逐题排名、正文和耗时。
    "summary": summary,  # 保存分组命中统计。
}
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")  # 生成独立报告标识。
output = root / "eval_reports" / f"rerank_eval_{stamp}.json"  # 确定新报告路径。
output.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
temporary = output.with_suffix(".json.tmp")  # 使用临时文件避免产生半份正式报告。
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 保存完整结果并拒绝非法数值。
temporary.replace(output)  # 写入成功后生成正式报告。
print("REPORT=", output)  # 显示保存位置。
print("RERANK_EVAL_SAVED")  # 表示评估执行完成，不表示所有问题都命中。