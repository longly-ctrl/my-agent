import json
from pathlib import Path

SESSION_PATH = Path(__file__).resolve().parent / "session.json"


def save_session(messages: list, path: Path = SESSION_PATH) -> Path:
    """将当前完整消息历史保存为 JSON。"""

    # 历史中既有普通字典，也有 SDK 返回的消息对象。
    data = [
        message
        if isinstance(message, dict)
        else message.model_dump(mode="json")
        for message in messages
    ]
    #上文等价写法                                 0.1
    # data = []
    # for message in messages:
        # if isinstance(message, dict):
            # # 已经是字典，直接放入结果列表。
            # data.append(message)
        # else:
            # # SDK 消息对象先转换成可用于 JSON 序列化的字典。
            # data.append(message.model_dump(mode="json"))
    #                                            0.1
    
    #把代码编码成JSON字符串
    text = json.dumps(
        data,
        ensure_ascii=False,  # 保留可读的中文。
        indent=2,            #嵌套缩进2个空格
        allow_nan=False,     #遇到浮点数NaN、正无穷、负无穷时抛出ValueError，避免生成不符合标准的JSON
    )

    # 先写临时文件，成功后再替换记录，减少半写入的风险。
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)

    return path
def load_session(path: Path = SESSION_PATH) -> list:
    """读取保存的消息；文件不存在时返回空列表。"""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("会话文件的顶层必须是列表")

    for message in data:
        if not isinstance(message, dict):
            raise ValueError("每条消息必须是对象")
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("会话包含未知的消息角色")

    if data:
        if data[0]["role"] != "system":
            raise ValueError("会话首条消息必须是 system")
        if any(m["role"] == "system" for m in data[1:]):
            raise ValueError("会话只能在开头包含一条 system 消息")

    if len(data) > 1:
        last = data[-1]
        if last["role"] != "assistant" or last.get("tool_calls"):
            raise ValueError("会话必须以完成的助手回答结束")

        content = last.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("最后一条助手回答不能为空")

    return data