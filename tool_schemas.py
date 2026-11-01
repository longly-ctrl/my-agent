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

EXTERNAL_KNOWLEDGE_SCHEMA = {  # 仅在调用方提供外部检索能力时使用。
    "type": "function",  # 声明这是函数工具。
    "function": {  # 定义模型可见的调用接口。
        "name": "search_external_knowledge",  # 与本地注册表中的名称保持一致。
        "description": (  # 说明资料范围、返回内容和引用方式。
            "混合检索外部 Word 学习资料，返回前三个候选的完整文本块。"
            "文本块不代表整个章节；回答来源应原样复制返回的 citation。"
        ),
        "parameters": {  # 描述允许模型提交的参数。
            "type": "object",  # 参数必须是 JSON 对象。
            "properties": {  # 声明允许的字段。
                "query": {  # 接收检索问题。
                    "type": "string",  # 要求字符串。
                    "minLength": 1,  # 声明非空要求；纯空白仍由本地检查。
                    "description": "自然语言问题或明确的技术术语，例如 QLoRA 中的 NF4 是什么。",  # 支持语义检索表达。
                },
            },
            "required": ["query"],  # 必须提供查询。
            "additionalProperties": False,  # 不允许传入模型路径等额外配置。
        },
    },
}