from openai import OpenAI, APIError

from agent import run_agent, build_initial_messages
from app_config import load_config
from session_store import (
    SESSION_PATH,
    get_session_path,
    save_session,
    load_session,
)
import argparse
from pathlib import Path  # 处理命令行传入的文档路径及项目缓存路径。

#                       未将工具说明拆成独立模块                                        0.3
# # 每一轮都向模型提供相同的工具说明。
# TOOL_SCHEMAS = [
    # {
        # "type": "function",
        # "function": {
            # "name": "search_notes",
            # "description": "按关键词查询用户的本地学习笔记。",
            # "parameters": {
                # "type": "object",
                # "properties": {
                    # "query": {
                        # "type": "string",
                        # "description": "简短关键词，例如 epoll，不要传完整问题。",
                    # }
                # },
                # "required": ["query"],
                # "additionalProperties": False,
            # },
        # },
    # }
# ]
# TOOL_SCHEMAS.append({
    # "type": "function",
    # "function": {
        # "name": "read_note",
        # "description": (
            # "按搜索结果行号读取原文，每次最多300字符。"
            # "需要补充被截断的内容时使用；按续读提示继续读取。"
        # ),
        # "parameters": {
            # "type": "object",
            # "properties": {
                # "line_number": {
                    # "type": "integer",
                    # "minimum": 1,
                    # "description": "搜索结果中的原始行号，从1开始。",
                # },
                # "offset": {
                    # "type": "integer",
                    # "minimum": 0,
                    # "description": "字符偏移，从0开始；首次读取可省略。",
                # },
            # },
            # "required": ["line_number"],
            # "additionalProperties": False,
        # },
    # },
# })
#                                                                                             0.3

def main(session_id: str | None = None, *, docx_path: Path | None = None):  # 可选地启用外部 Word 检索。
    # 不传ID时兼容原存档；指定ID时使用独立存档。
    session_file = (
        SESSION_PATH if session_id is None
        else get_session_path(session_id)
    )
    print(f"当前会话文件：{session_file.name}")
    # 从当前脚本旁边读取配置。
    config = load_config()
    # 整个会话共用这一个列表，不能放进下面的 while 循环。
    # 同一个会话复用这份历史；新会话重新调用函数创建。
    messages = build_initial_messages(external_enabled=docx_path is not None)  # 按本次启动参数选择系统规则。
    try:
        history = load_session(path=session_file)
    except (OSError, ValueError) as exc:
        print(f"会话恢复失败：{exc}")
        return  # 停止启动，避免后续保存覆盖损坏的记录。

    if history:
        # 跳过旧 system，使用当前代码里的系统规则。
        messages.extend(history[1:])
        print(f"已恢复 {len(history) - 1} 条历史消息。")

    external_search = None  # 默认不提供外部检索工具。
    if docx_path is not None:  # 只有明确指定文档时才初始化检索资源。
        from retrieval_service import RetrievalService  # 按需导入本地检索服务。
        from external_knowledge_tool import ExternalKnowledgeTool  # 按需导入工具适配层。

        root = Path(__file__).resolve().parent  # 定位项目目录，不依赖终端当前目录。
        retrieval = RetrievalService(root / ".cache" / "knowledge_vectors.json", docx_path)  # 校验缓存并加载一次模型，默认关闭精排。
        external_tool = ExternalKnowledgeTool(retrieval)  # 让工具持有同一个检索服务。
        external_search = external_tool.search  # 保存绑定方法，供每轮 Agent 调用复用。
        print("外部 Word 混合检索已启用，精排关闭。")  # 显示当前资料模式。
    # 创建客户端；退出 with 时自动关闭网络连接资源。
    # 整个会话复用客户端，退出会话后再关闭。
    with OpenAI(
        api_key=config["LLM_API_KEY"],
        base_url=config["LLM_BASE_URL"],
        timeout=30.0,
        max_retries=0,  # 首次调试关闭自动重试，便于观察一次请求的结果。
    ) as client:
        print("学习助手已启动，输入 exit 或 quit 退出。")

        while True:
            question = input("\n你：").strip()

            if question.lower() in ("exit", "quit"):
                print("会话结束。")
                break

            if not question:
                print("问题不能为空，请重新输入。")
                continue
            # 必须在追加当前问题之前，记录已有历史的长度。
            history_start = len(messages)

            messages.append({
                "role": "user",
                "content": question,
            })

            try:
                answer = run_agent(  # 执行当前问题，复用已经初始化的工具。
                    client,  # 复用当前 CLI 客户端。
                    config["LLM_MODEL_ID"],  # 使用本地配置中的模型。
                    messages,  # 保留当前会话历史。
                    max_steps=5,  # 沿用模型调用预算。
                    external_search=external_search,  # 传入绑定方法；未启用时传入 None。
                )
            except (APIError, OSError, ValueError, TypeError, RuntimeError) as exc:
                # 删除当前问题及其执行过程中追加的所有消息。
                # 之前已经完成的问答继续保留。
                del messages[history_start:]

                print(f"本轮失败（{type(exc).__name__}）：{exc}")
                print("已撤销本轮对话，可以继续提问。")
                continue

            print("助手：", answer)
            
            # 回答已经完成，保存失败单独报告，不撤销成功的问答。
            try:
                saved_path = save_session(messages, path=session_file)
            except (OSError, TypeError, ValueError) as exc:
                print(f"回答已完成，但会话保存失败：{exc}")
            else:
                print(f"会话已保存：{saved_path.name}")
            
    #***把当前问题加入历史，执行一次 Agent，再打印最终回答***                                          0.1
            # # 每收到一个新问题，只追加这一条用户消息。                       
            # messages.append({
                # "role": "user",
                # "content": question,
            # })

            # # 每个新问题，重新获得最多 5 次模型调用的额度。
            # answer = run_agent(
                # client,
                # config["LLM_MODEL_ID"],
                # messages,
                # max_steps=5,
            # )
            # print("助手：", answer)                                                             0.1
            
        # ***version 0.6 一次调用模型--模型生成请求调用--调用本地工具--本地工具回填模型给出最终答案***
        # print("正在请求 DeepSeek……", flush=True)

        # # 这一行才真正发送网络请求。
        # response = client.chat.completions.create(
        # model=config["LLM_MODEL_ID"],
        # messages=messages,
        # # 工具说明会发给模型，函数实现仍保留在本地。
        # tools=[
            # {
                # "type": "function",
                # "function": {
                    # "name": "search_notes",
                    # "description": "按关键词查询用户的本地学习笔记。",
                    # "parameters": {
                        # "type": "object",
                        # "properties": {
                            # "query": {
                                # "type": "string",
                                # "description": "简短关键词，例如 epoll，不要传完整问题。",
                            # }
                        # },
                        # "required": ["query"],
                        # "additionalProperties": False,
                    # },
                # },
            # }
        # ],
        # tool_choice="auto",
        # stream=False,
        # max_tokens=256,
        # extra_body={"thinking": {"type": "disabled"}},
    # )

        # choice = response.choices[0]
        # message = choice.message
        # print("结束原因：", choice.finish_reason)

       # # 输出被截断时，参数可能不完整，不继续执行工具。
        # if choice.finish_reason == "length":
            # raise RuntimeError("模型输出达到长度限制，无法可靠执行工具。")

        # # 工具注册表：把模型使用的名称映射到本地函数。
        # tool_functions = {
            # "search_notes": search_notes,
        # }

        # if message.tool_calls:
            # # 先记录模型提出的完整工具调用请求，只追加一次。
            # messages.append(message)
            # for call in message.tool_calls:
                # if call.type != "function":
                    # raise ValueError(f"不支持的工具类型：{call.type}")

                # # 只允许调用注册表中明确提供的工具。
                # tool_function = tool_functions.get(call.function.name)
                # if tool_function is None:
                    # raise ValueError(f"未知工具：{call.function.name}")

                # # arguments 是 JSON 字符串，先解析为 Python 对象。
                # arguments = json.loads(call.function.arguments)

                # # 参数必须是字典，并且只包含 query。
                # if not isinstance(arguments, dict) or set(arguments) != {"query"}:
                    # raise ValueError("工具参数必须是只包含 query 的对象")

                # print("工具名称：", call.function.name)
                # print("解析后的参数：", arguments)

                # # ** 将字典展开为关键字参数。
                # # 例如 {"query": "epoll"} → search_notes(query="epoll")
                # observation = tool_function(**arguments)
                # print("工具执行结果：", observation)
                # # 每个工具结果，都对应模型此前提出的一次调用。
                # messages.append({
                    # "role": "tool",
                    # "tool_call_id": call.id,
                    # "content": observation,
                # })
            # print("正在根据工具结果生成回答……", flush=True)

            # # 第二次请求携带完整历史，让模型看到真实执行结果。
            # final_response = client.chat.completions.create(
                # model=config["LLM_MODEL_ID"],
                # messages=messages,
                # tool_choice="none",  # 本轮直接生成回答。
                # stream=False,
                # max_tokens=256,
                # extra_body={"thinking": {"type": "disabled"}},
            # )

            # final_choice = final_response.choices[0]
            # if final_choice.finish_reason != "stop":
                # raise RuntimeError(
                    # f"第二轮未正常结束：{final_choice.finish_reason}"
                # )

            # final_answer = final_choice.message.content
            # if not final_answer or not final_answer.strip():
                # raise RuntimeError("第二轮没有返回有效正文")

            # print("最终回答：", final_answer)  
        # else:
            # answer = message.content
            # if not answer or not answer.strip():
                # raise RuntimeError("响应既没有工具调用，也没有非空正文。")
            # print("模型直接回答：", answer)


# 直接运行此文件时执行；被其他文件导入时不发送请求。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="本地知识库学习助手"
    )
    parser.add_argument(
        "--session",
        default=None,
        help="会话ID；省略时使用原来的session.json",
    )
    parser.add_argument(  # 增加外部资料启动选项。
        "--docx",  # 接收原始 Word 文件路径。
        type=Path,  # 将命令行字符串转换为 Path。
        default=None,  # 不传时继续使用原来的 Markdown 模式。
        help="启用外部 Word 混合检索，使用已有向量缓存",  # 说明需要事先生成缓存。
    )
    args = parser.parse_args()  # 解析会话与资料选项。
    main(session_id=args.session, docx_path=args.docx)  # 将启动配置传入入口。