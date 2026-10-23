# 每一轮都向模型提供相同的工具说明。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "按关键词查询用户的本地学习笔记。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "简短关键词，例如 epoll，不要传完整问题。",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]
TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "read_note",
        "description": (
            "按搜索结果行号读取原文，每次最多300字符。"
            "需要补充被截断的内容时使用；按续读提示继续读取。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "line_number": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "搜索结果中的原始行号，从1开始。",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "字符偏移，从0开始；首次读取可省略。",
                },
            },
            "required": ["line_number"],
            "additionalProperties": False,
        },
    },
})