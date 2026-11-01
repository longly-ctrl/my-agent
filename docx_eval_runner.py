import argparse  # 解析原始 Word 文件路径。
import json  # 读取固定用例并保存评估报告。
from datetime import datetime, timezone  # 生成带 UTC 标识的报告文件名。
from pathlib import Path  # 定位缓存、用例和输出目录。
from time import perf_counter  # 单独测量检索服务的构造耗时。
from openai import OpenAI  # 创建整批复用的云端客户端。
from app_config import load_config  # 读取本地模型配置。
from eval_runner import run_case  # 复用单例执行、用量记录和轨迹保存能力。


def main(docx_path: Path) -> Path:  # 执行外部资料生成回归评估。
    root = Path(__file__).resolve().parent  # 以脚本目录定位项目文件。
    dataset = json.loads((root / "docx_eval_cases.json").read_text(encoding="utf-8-sig"))  # 加载固定用例及资料版本。
    if dataset.get("schema_version") != 1 or dataset.get("split") != "regression" or dataset.get("expected_sources_policy") != "required_subset":  # 核对评估结构和标签规则。
        raise ValueError("评估集版本、用途或引用规则不受支持")  # 在调用模型前停止错误实验。

    cases = dataset.get("cases")  # 获取本批问题。
    if not isinstance(cases, list) or not cases or not all(isinstance(case, dict) for case in cases):  # 要求非空对象列表。
        raise ValueError("评估用例必须是非空对象列表")  # 拒绝错误结构。
    ids = [case.get("id") for case in cases]  # 收集用例标识。
    if any(not isinstance(cid, str) or not cid.strip() for cid in ids) or len(set(ids)) != len(ids):  # 检查标识有效且不重复。
        raise ValueError("评估用例标识为空或重复")  # 避免含糊或重复统计。

    from retrieval_service import RetrievalService, load_retrieval_cache  # 实际执行时才导入检索组件。
    from external_knowledge_tool import ExternalKnowledgeTool  # 将检索服务适配为工具。
    from embedding_service import QUERY_INSTRUCTION  # 记录查询编码使用的前缀。

    cache_file = root / ".cache" / "knowledge_vectors.json"  # 定位已有向量缓存。
    payload = load_retrieval_cache(cache_file, docx_path)  # 校验当前文档、缓存配置和向量。
    if {chunk["source_hash"] for chunk in payload["chunks"]} != {dataset["source_hash"]}:  # 检查标签对应的资料版本。
        raise ValueError("生成评估集与缓存的资料版本不一致")  # 阻止使用过期标签。
    known = {f"[docx:{chunk['chunk_id']}]" for chunk in payload["chunks"]}  # 建立有效引用集合。

    for case in cases:  # 在模型调用前检查所有用例。
        if not isinstance(case.get("question"), str) or not case["question"].strip():  # 要求有效问题。
            raise ValueError(f"问题无效：{case['id']}")  # 定位错误用例。
        checks = case.get("answer_checks")  # 读取人工检查标准。
        if not isinstance(checks, list) or not checks or not all(isinstance(check, str) and check.strip() for check in checks):  # 要求非空字符串列表。
            raise ValueError(f"回答检查标准无效：{case['id']}")  # 拒绝没有质量标准的用例。
        refs = case.get("expected_sources")  # 读取必须包含的主要引用。
        if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) for ref in refs) or not set(refs) <= known:  # 检查引用真实存在。
            raise ValueError(f"预期来源无效：{case['id']}")  # 阻止错误标识进入评估。

    config = load_config()  # 标签检查通过后读取云端模型配置。
    retrieval_options = {  # 固定本轮检索条件。
        "enable_rerank": False,  # 与当前默认服务一致，关闭精排。
        "candidate_k": 50,  # 每路最多召回五十个候选。
        "rank_constant": 60,  # 使用已经验收的排名常数。
        "weights": {"bm25": 1.0, "vector": 1.0},  # 显式记录等权融合。
    }
    started = perf_counter()  # 开始测量服务构造耗时。
    retrieval = RetrievalService(cache_file, docx_path, **retrieval_options)  # 整批创建一次索引和查询模型。
    tool = ExternalKnowledgeTool(retrieval)  # 整批复用同一个工具。
    initialization_seconds = perf_counter() - started  # 与逐题耗时分开记录。

    started_at = datetime.now(timezone.utc)  # 记录本批问答开始时间。
    stamp = started_at.strftime("%Y%m%dT%H%M%S_%fZ")  # 生成独立报告标识。
    output = root / "eval_reports" / f"docx_generation_{stamp}.json"  # 使用生成评估专用前缀。
    output.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    report = {  # 保存复核所需的输入、配置和结果。
        "schema_version": 1,  # 标记报告结构。
        "started_at": started_at.isoformat(),  # 保存带时区的开始时间。
        "model": config["LLM_MODEL_ID"],  # 记录模型名称，不保存密钥。
        "dataset": dataset,  # 保存问题、标签、资料版本及引用规则。
        "source_name": Path(payload["chunks"][0]["source"]).name,  # 记录实际资料文件名。
        "index_config": {field: payload[field] for field in ("schema_version", "model_id", "model_revision", "dimension", "normalize_embeddings", "text_format", "chunk_params")},  # 保存索引配置，省略全部正文和向量。
        "query_instruction": QUERY_INSTRUCTION,  # 保存查询编码前缀。
        "retrieval_config": {**retrieval_options, "final_k": 3},  # 记录召回、融合和工具返回数量。
        "service_initialization_seconds": initialization_seconds,  # 保存服务构造耗时。
        "results": [],  # 逐题追加完整记录。
    }

    with OpenAI(  # 整批复用客户端，退出时关闭。
        api_key=config["LLM_API_KEY"],  # 仅用于认证，不进入报告。
        base_url=config["LLM_BASE_URL"],  # 使用本地配置中的地址。
        timeout=30.0,  # 保留现有请求超时。
        max_retries=0,  # 保留现有调试重试策略。
    ) as client:
        for case in cases:  # 每题由 run_case 创建独立历史。
            print(f"正在评估：{case['id']}", flush=True)  # 显示当前进度。
            row = run_case(client, config["LLM_MODEL_ID"], case, external_search=tool.search)  # 执行问答并记录轨迹。
            report["results"].append(row)  # 同时保留成功和失败记录。
            temporary = output.with_suffix(".json.tmp")  # 准备临时文件。
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 每完成一题保存一次。
            temporary.replace(output)  # 写完后替换正式报告。
            print(f"执行状态：{row['execution_status']}，耗时：{row['elapsed_seconds']:.2f} 秒")  # 显示执行结果。
            if row["execution_status"] == "completed":  # 检查是否产生最终回答。
                print("最终回答：\n", row["answer"])  # 展示答案供人工复核。
            else:  # 当前题执行失败时保留错误记录。
                print("执行异常类型：", row["error"]["type"])  # 控制台显示异常类型。

    print("REPORT=", output)  # 显示报告路径。
    print("DOCX_GENERATION_REPORT_SAVED")  # 仅表示结果已保存，不代表质量通过。
    return output  # 返回报告路径。


if __name__ == "__main__":  # 导入模块时不执行模型调用。
    parser = argparse.ArgumentParser(description="外部 Word 资料生成回归评估")  # 创建解析器。
    parser.add_argument("--docx", type=Path, required=True, help="缓存对应的原始 Word 文件")  # 要求明确指定资料。
    args = parser.parse_args()  # 解析参数。
    main(args.docx)  # 启动评估。