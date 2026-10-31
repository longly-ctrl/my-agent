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

TOOL_SCHEMAS.extend([
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索knowledge目录，返回前三个相关片段及来源。"
                "truncated表示预览不完整。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "简短关键词，以空格分隔，例如 epoll Linux。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_knowledge",
            "description": (
                "读取搜索返回的文档和行号范围，每次最多300字符。"
                "使用next_offset续读，null表示读完。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "搜索返回的文档标识。",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "搜索返回的起始行号。",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "搜索返回的结束行号，包含该行。",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "所选正文的字符偏移，默认0。",
                    },
                },
                "required": ["doc_id", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    },
])