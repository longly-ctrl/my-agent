import json

from openai import OpenAI
from tools import (
    search_notes, read_note,
    search_knowledge, read_knowledge,
    ToolInputError,
)
from tool_schemas import TOOL_SCHEMAS

def build_initial_messages() -> list[dict[str, str]]:
    """为一个新会话创建初始消息。"""
    # 每次调用都创建新的列表和字典，避免不同会话共用可变历史。
    return [
        {
            "role": "system",
            "content": (
                "你是学习助手。回答本地资料问题前，默认使用"
                "search_knowledge查询knowledge目录。"
                "查询使用简短关键词，多个词用空格分隔。"
                "预览足以回答时直接回答；不足时调用read_knowledge，"
                "原样使用搜索返回的doc_id及起止行号。"
                "续读使用next_offset；为null时停止。"
                "仅依据工具资料回答，使用"
                "[文件名:L起始行-L结束行]标注来源。"
                "最终回答仅包含工具原文支持的事实和必要的资料不足说明。"
                "可以复述用户问题来说明缺口，但不得补充问题及工具原文之外的"
                "专业术语、具体分级、机制或示例；"
                "即使以“例如”或“资料没有介绍”的形式列举，也不例外。"
                "只有用户明确要求根目录notes.md时，"
                "才使用search_notes和read_note。"
            ),
        },
    ]
    
def build_context(messages: list, keep_turns: int = 3) -> list:
    """前提：第一条消息是 system；每条 user 消息开始一轮。"""
    if keep_turns < 1:
        raise ValueError("keep_turns 必须至少为 1")

    # 历史中同时存在字典和 SDK 消息对象，分别读取 role。
    user_starts = [
        index
        for index, message in enumerate(messages)
        if (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        ) == "user"
    ]

    # 轮数不足时，返回列表副本。
    if len(user_starts) <= keep_turns:
        return messages[:]

    # 从保留范围内最早的用户问题开始截取，另外保留系统规则。
    start = user_starts[-keep_turns]
    return messages[:1] + messages[start:]


def run_agent(
    client: OpenAI,
    model: str,
    messages: list,
    max_steps: int = 5,
    usage_records: list[dict] | None = None,
) -> str:
    """围绕当前消息历史执行 Agent，返回最终回答。"""
    if max_steps < 1:
        raise ValueError("max_steps 必须至少为 1")
    #***version 0.7 固定两次请求--最多5次的Agent循环***
    # 每项依次保存：本地函数、必需参数名、允许的全部参数名。
    tool_functions = {
        "search_notes": (
            search_notes, {"query"}, {"query"},
        ),
        "read_note": (
            read_note, {"line_number"}, {"line_number", "offset"},
        ),
        "search_knowledge": (
            search_knowledge, {"query"}, {"query"},
        ),
        "read_knowledge": (
            read_knowledge,
            {"doc_id", "start_line", "end_line"},
            {"doc_id", "start_line", "end_line", "offset"},
        ),
    }

    for step in range(1, max_steps + 1):
        print(f"--- 第 {step}/{max_steps} 次模型调用 ---", flush=True)

        response = client.chat.completions.create(
            model=model,
            messages=build_context(messages, keep_turns=3),
            tools=TOOL_SCHEMAS,
            tool_choice="auto",  # 每轮都允许选择工具或直接回答。
            stream=False,
            max_tokens=256,
            extra_body={"thinking": {"type": "disabled"}},
        )
        # 每次请求分别记录实际用量。
        # 测试中的模拟响应可能没有 usage，因此允许缺失。
        usage = getattr(response, "usage", None)
        if usage_records is not None:
            # 在检查结束原因之前记录：
            # 即使响应被截断，本次已返回的用量也应保留。
            usage_records.append({
                "step": step,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            })
        if usage is not None:
            print(
                f"Token 用量：输入={usage.prompt_tokens}，"
                f"输出={usage.completion_tokens}，"
                f"总计={usage.total_tokens}"
            )
        choice = response.choices[0]
        message = choice.message

        # 截断等异常结束，不能当作完整响应继续处理。
        if choice.finish_reason not in ("stop", "tool_calls"):
            raise RuntimeError(
                f"模型未正常完成输出：{choice.finish_reason}"
            )

        # 保存本轮模型消息，包括它提出的工具调用。
        messages.append(message)

        # 没有工具调用时，检查正文并结束整个 main()。
        if not message.tool_calls:
            answer = message.content
            if choice.finish_reason != "stop" or not answer or not answer.strip():
                raise RuntimeError("模型没有返回有效的最终回答")

            return answer

        if choice.finish_reason != "tool_calls":
            raise RuntimeError("工具调用与结束原因不一致")

        # 最后一轮仍要求工具，意味着预算内没有完成任务。
        if step == max_steps:
            raise RuntimeError(
                "已达到模型调用上限，模型仍请求工具，任务未完成"
            )

        # 执行本轮全部工具调用，补齐结果后进入下一轮。
        for call in message.tool_calls:
            if call.type != "function":
                raise ValueError(f"不支持的工具类型：{call.type}")
                
#                          未添加新工具read_note                                          0.2
            # tool_function = tool_functions.get(call.function.name)
            # if tool_function is None:
                # raise ValueError(f"未知工具：{call.function.name}")

            # arguments = json.loads(call.function.arguments)
            # if not isinstance(arguments, dict) or set(arguments) != {"query"}:
                # raise ValueError("工具参数必须是只包含 query 的对象")
#                                                                                        0.2

            registered = tool_functions.get(call.function.name)
            if registered is None:
                raise ValueError(f"未知工具：{call.function.name}")

            tool_function, required, allowed = registered
            arguments = json.loads(call.function.arguments)

            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            keys = set(arguments)
            # 必需字段不能少，允许范围之外的字段不能多。
            if not required <= keys or not keys <= allowed:
                raise ValueError(
                    f"工具参数缺失或包含未知字段：{call.function.name}"
                )
            # 观察模型选择了什么工具，以及实际查询的关键词。
            print("工具请求：", call.function.name, arguments)
            try:
                observation = tool_function(**arguments)
            except ToolInputError as exc:
                # 明确返回失败状态，供模型修改参数或说明无法完成。
                observation = json.dumps(
                    {
                        "ok": False,
                        "error_type": "ToolInputError",
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                )
            print("工具执行结果：", observation)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": observation,
            })