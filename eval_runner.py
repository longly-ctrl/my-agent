from time import perf_counter

from openai import OpenAI, APIError
from agent import build_initial_messages, run_agent
import json
from datetime import datetime, timezone
from pathlib import Path

from app_config import load_config


def run_case(client: OpenAI, model: str, case: dict) -> dict:
    """独立执行一个用例，保留结果供后续评审。"""
    # 每个用例从空会话开始，不加载个人存档。
    messages = build_initial_messages()
    messages.append({
        "role": "user",
        "content": case["question"],
    })

    answer = None
    error = None
    # 每个用例独立持有自己的用量记录。
    usage_records = []
    started = perf_counter()

    try:
        answer = run_agent(
            client,
            model,
            messages,
            max_steps=5,
            usage_records=usage_records,
        )
    except (APIError, OSError, ValueError, TypeError, RuntimeError) as exc:
        # 记录执行失败，保留已有轨迹，方便定位失败发生在哪一步。
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    elapsed = perf_counter() - started

    return {
        "id": case["id"],
        # 完成执行不等于答案正确，质量需要另外评审。
        "execution_status": "completed" if error is None else "error",
        "review_status": "pending",
        "answer": answer,
        "error": error,
        "elapsed_seconds": elapsed,
        "usage_records": usage_records,
        "expected_sources": case["expected_sources"],
        "answer_checks": case["answer_checks"],
        # SDK 消息转为普通字典，方便后续写入 JSON 报告。
        "messages": [
            message if isinstance(message, dict)
            else message.model_dump(mode="json")
            for message in messages
        ],
    }
 
def main():
    root = Path(__file__).resolve().parent
    cases = json.loads(
        (root / "eval_cases.json").read_text(encoding="utf-8-sig")
    )
    if not isinstance(cases, list) or not cases:
        raise ValueError("评估用例必须是非空列表")

    config = load_config()
    started_at = datetime.now(timezone.utc)

    # 每次运行生成独立报告，方便保留和比较结果。
    report_dir = root / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = report_dir / f"report_{stamp}.json"

    report = {
        "started_at": started_at.isoformat(),
        "model": config["LLM_MODEL_ID"],
        "results": [],
    }

    # 复用网络客户端；每个用例的消息历史仍由 run_case 独立创建。
    with OpenAI(
        api_key=config["LLM_API_KEY"],
        base_url=config["LLM_BASE_URL"],
        timeout=30.0,
        max_retries=0,
    ) as client:
        for case in cases:
            print(f"\n正在评估：{case['id']}", flush=True)
            result = run_case(client, config["LLM_MODEL_ID"], case)
            report["results"].append(result)

            # 每完成一条就保存，后续失败时仍保留前面的记录。
            # 先写临时文件，再替换本次报告，减少半写入风险。
            temporary = report_path.with_suffix(".json.tmp")    #替换路径中的最后一个文件后缀，返回新路径。
            temporary.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            temporary.replace(report_path)

            print(
                f"执行状态：{result['execution_status']}，"
                f"耗时：{result['elapsed_seconds']:.2f} 秒"
            )
            # 执行日志与答案质量分开观察，最终回答必须单独展示。
            if result["execution_status"] == "completed":
                print("最终回答：\n", result["answer"])
            else:
                print("执行异常：", result["error"])

    print(f"\n报告已保存：{report_path}")


if __name__ == "__main__":
    main()