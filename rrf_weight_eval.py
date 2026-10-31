import json  # 读取已有候选并保存权重实验结果。
from datetime import datetime, timezone  # 为实验报告生成独立时间标识。
from pathlib import Path  # 定位基线报告和输出目录。
from hybrid_retriever import fuse_rrf  # 复用修改后的融合函数。

root = Path(__file__).resolve().parent  # 定位项目根目录。
baseline_path = root / "eval_reports" / "retrieval_eval_20261031T021200_202968Z.json"  # 固定本次比较的基线。
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))  # 读取已经保存的两路候选。
cases = {case["id"]: case for case in baseline["dataset"]["cases"]}  # 按用例标识查找证据标签。


def is_target(chunk: dict, case: dict) -> bool:  # 沿用原评估的命中口径。
    same_section = [chunk["section_start_paragraph"], chunk["section_end_paragraph"]] == case["target_section"]  # 核对章节。
    return same_section and all(s in chunk["text"] for s in case["evidence_contains"])  # 同时核对全部证据片段。


for stored in baseline["results"]:  # 先检查修改没有改变默认行为。
    inputs = {name: stored["rankings"][name] for name in ("bm25", "vector")}  # 只取原始两路，避免把旧融合结果当成第三路。
    actual = fuse_rrf(inputs, top_k=2 * baseline["candidate_k"], rank_constant=baseline["rank_constant"])  # 不传权重，重放原融合。
    assert actual == stored["rankings"]["hybrid"], stored["id"]  # 要求排名、分数和来源与基线完全一致。

print("DEFAULT_BASELINE_IDENTICAL_OK")  # 表示默认行为的回归检查通过。

weight_options = [  # 固定几组对照比例，保留改善和退化结果。
    {"bm25": 0.75, "vector": 0.25},  # 偏重关键词排名。
    {"bm25": 0.5, "vector": 0.5},  # 等权对照，排名应与原基线一致。
    {"bm25": 0.3, "vector": 0.7},  # 适度提高向量排名的影响。
    {"bm25": 0.25, "vector": 0.75},  # 进一步提高向量影响，观察其他题是否退化。
]
trials = []  # 收集所有权重组合的结果。

for weights in weight_options:  # 对相同候选逐组调整权重。
    rows = []  # 收集当前权重下的逐题表现。
    for stored in baseline["results"]:  # 遍历同一批固定问题。
        inputs = {name: stored["rankings"][name] for name in ("bm25", "vector")}  # 复用原候选及顺序。
        results = fuse_rrf(inputs, top_k=2 * baseline["candidate_k"], rank_constant=baseline["rank_constant"], weights=weights)  # 只改变融合权重。
        case = cases[stored["id"]]  # 取得当前问题的证据标签。
        first_rank = next((rank for rank, item in enumerate(results, start=1) if is_target(item, case)), None)  # 查找首次命中位置。
        hit = first_rank is not None and first_rank <= baseline["final_k"]  # 仍按原来的前三条统计。
        rows.append({"id": stored["id"], "first_target_rank": first_rank, "hit_at_3": hit})  # 保存逐题结果。

    hits = sum(row["hit_at_3"] for row in rows)  # 统计命中问题数。
    trials.append({"weights": weights, "hits": hits, "total": len(rows), "hit_at_3": hits / len(rows), "results": rows})  # 保存当前权重的汇总和明细。
    print("WEIGHTS=", weights, "HITS=", hits, "TOTAL=", len(rows))  # 展示不同权重的表现。

report = {  # 保存实验来源、固定条件与实际结果。
    "baseline_report": str(baseline_path),  # 记录被重放的候选来源。
    "dataset": baseline["dataset"],  # 保留问题及证据标签。
    "index_config": baseline["index_config"],  # 保留索引配置。
    "candidate_k": baseline["candidate_k"],  # 保留召回范围。
    "final_k": baseline["final_k"],  # 保留命中统计范围。
    "rank_constant": baseline["rank_constant"],  # 保留排名常数。
    "trials": trials,  # 保存各组权重的结果。
}
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")  # 生成本次实验的时间标识。
output = root / "eval_reports" / f"rrf_weight_eval_{stamp}.json"  # 创建独立报告路径。
temporary = output.with_suffix(".json.tmp")  # 先写临时文件。
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 保存合法 JSON。
temporary.replace(output)  # 完整写入后替换正式报告。

print("REPORT=", output)  # 输出实验报告位置。
print("WEIGHT_EVAL_SAVED")  # 表示对照实验已经完成并保存。