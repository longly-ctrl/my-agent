import json  # 将检索结果序列化为工具消息。
from pathlib import Path  # 从来源路径提取文件名。
from retrieval_service import RetrievalService  # 使用已经验收的混合检索服务。
from tools import ToolInputError  # 复用可反馈给模型的参数错误类型。


class ExternalKnowledgeTool:  # 将检索服务适配为模型工具接口。
    def __init__(self, service: RetrievalService):  # 接收应用已经创建的服务实例。
        self._service = service  # 持有服务引用，后续查询复用其索引和模型。

    def search(self, query: str) -> str:  # 接收查询，返回可作为工具消息的 JSON 字符串。
        if not isinstance(query, str) or not query.strip():  # 检查查询类型及实际内容。
            raise ToolInputError("query 必须是非空字符串")  # 允许 Agent 将这类参数错误反馈给模型。

        hits = self._service.search(query.strip(), top_k=3)  # 调用混合检索，保留前三个候选。
        matches = []  # 收集模型回答所需的正文及来源。

        for hit in hits:  # 保持检索服务返回的排名顺序。
            matches.append({  # 为当前文本块建立工具返回记录。
                "chunk_id": hit["chunk_id"],  # 保留绑定资料版本和位置的唯一标识。
                "citation": f"[docx:{hit['chunk_id']}]",  # 由程序生成引用标记，供模型原样复制。
                "source_name": Path(hit["source"]).name,  # 返回来源文件名，避免发送无关的本机目录。
                "source_hash": hit["source_hash"],  # 保留原始文件版本指纹。
                "heading_path": hit["heading_path"],  # 保留完整标题路径，帮助理解正文语境。
                "section_start_paragraph": hit["section_start_paragraph"],  # 标记所在章节正文的起始段落。
                "section_end_paragraph": hit["section_end_paragraph"],  # 标记所在章节正文的结束段落。
                "start_char": hit["start_char"],  # 标记块在章节拼接正文中的起始字符偏移。
                "end_char": hit["end_char"],  # 标记结束偏移，该位置本身不包含在块内。
                "text": hit["text"],  # 返回完整文本块，不再次截断正文。
            })

        return json.dumps({  # 返回统一的工具观察结果。
            "matches": matches,  # 没有候选时保留空列表，不编造资料。
            "location_note": (  # 明确位置字段与正文的解释范围。
                "段落范围属于整个章节正文；字符范围是章节拼接正文中的"
                "[start_char, end_char)。text 是完整文本块，不代表整个章节。"
            ),
        }, ensure_ascii=False, allow_nan=False)  # 保留中文，并拒绝不可序列化为标准 JSON 的非法数值。