import re
import json
from knowledge_base import load_documents, chunk_documents, search_chunks
#*** 本地工具查Python字典***
# # 先用少量内置笔记验证工具机制，后续再接入文件。
# NOTES = {
    # "epoll": "epoll 是 Linux 提供的 I/O 事件通知机制。",
    # "reactor": "Reactor 模式通过事件循环等待事件，并分发事件给处理函数。",
# }


# def search_notes(query: str) -> str:
    # """按关键词查询学习笔记，返回匹配内容或明确的未找到提示。"""

    # # 类型注解不会自动检查实参，工具边界需要显式校验。
    # if not isinstance(query, str):
        # raise TypeError("query 必须是字符串")

    # keyword = query.strip().lower()
    # if not keyword:
        # raise ValueError("query 不能为空")

    # results = []
    # for title, content in NOTES.items():
        # # 忽略英文大小写，检查标题和正文是否包含关键词。
        # if keyword in (title + " " + content).lower():
            # results.append(f"[{title}] {content}")

    # return "\n".join(results) if results else "未找到相关笔记。"


# # 直接运行时测试；被其他模块导入时只提供函数。
# if __name__ == "__main__":
    # for query in ("epoll", "Reactor", "Redis"):
        # print(f"查询：{query}")
        # print(search_notes(query))
        # print()
        
        
#***本地工具查本地文件*** 
from pathlib import Path

# 固定查询脚本旁边的笔记文件，不受终端当前目录影响。
NOTES_PATH = Path(__file__).resolve().parent / "notes.md"
# 最多向模型返回五条匹配记录。
MAX_NOTE_HITS = 5
# 每条记录展示的正文长度，不包含来源标记和截断提示。
MAX_NOTE_CHARS = 300
class ToolInputError(ValueError):
    """工具参数不合法，可反馈给模型修正。"""
def search_notes(query: str) -> str:
    """按关键词检索本地笔记，返回匹配行及来源位置。"""

    if not isinstance(query, str):
        raise ToolInputError("query 必须是字符串")

    keyword = query.strip()
    if not keyword:
        raise ToolInputError("query 不能为空")
    if len(keyword) > MAX_NOTE_CHARS:
        raise ToolInputError("查询关键词不能长于正文展示上限")

    # 每次查询都重新读取文件，因此保存后的修改能在下次查询生效。
    # read_text 会完成文件的打开、读取和关闭。
    text = NOTES_PATH.read_text(encoding="utf-8-sig")
    results = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        content = line.strip()

        # 按原文位置查找；escape 让 C++、a+b 等作为普通文字匹配。
        match = re.search(re.escape(keyword), content, re.IGNORECASE)
        if match is not None:
            if len(results) >= MAX_NOTE_HITS:
                results.append(
                    f"【检索提示】仅展示前 {MAX_NOTE_HITS} 条匹配，"
                    "还有其他匹配内容；请缩小查询范围。"
                )
                break

            if len(content) > MAX_NOTE_CHARS:
                # 尽量保留关键词前 60 字符，同时保证完整关键词在窗口内。
                start = max(
                    0,
                    match.start() - 60,
                    match.end() - MAX_NOTE_CHARS,
                )
                end = min(len(content), start + MAX_NOTE_CHARS)

                # 标出原文在哪一侧有省略，来源行号保持不变。
                left = "…" if start > 0 else ""
                right = "…" if end < len(content) else ""
                content = (
                    left + content[start:end] + right
                    + "【本行已截断】"
                )

            results.append(
                f"[{NOTES_PATH.name}:L{line_number}] {content}"
            )

    return "\n".join(results) if results else "未找到相关笔记。"
    #   未限制关键词长度                                                        0.3
    # results = []
    # for line_number, line in enumerate(text.splitlines(), start=1):
        # if keyword in line.lower():
            # # 找到第六条时，才确认前五条之外还有匹配。
            # if len(results) >= MAX_NOTE_HITS:
                # results.append(
                    # f"【检索提示】仅展示前 {MAX_NOTE_HITS} 条匹配，"
                    # "还有其他匹配内容；请缩小查询范围。"
                # )
                # break
            # content = line.strip()

            # # 先限制正文长度，再添加来源，避免切断引用标记。
            # if len(content) > MAX_NOTE_CHARS:
                # content = (
                    # content[:MAX_NOTE_CHARS]
                    # + "…【本行已截断】"
                # )

            # results.append(
                # f"[{NOTES_PATH.name}:L{line_number}] {content}"
            # )
            # #               未限制正文长度                                      0.2
            # # # 每条实际返回的记录继续保留文件名和原始行号。                          
            # # results.append(
                # # f"[{NOTES_PATH.name}:L{line_number}] {line.strip()}"
            # # )                                                               0.2

    # return "\n".join(results) if results else "未找到相关笔记。"
#                                                                               0.3
def read_note(line_number: int, offset: int = 0) -> str:
    """按原始行号读取正文；offset 是从 0 开始的字符位置。"""
    # 排除浮点数、字符串，以及 Python 中属于 int 子类的 bool。
    if type(line_number) is not int or type(offset) is not int:
        raise ToolInputError("line_number 和 offset 必须是整数")
    if line_number < 1 or offset < 0:
        raise ToolInputError("line_number 必须大于 0，offset 不能为负数")

    lines = NOTES_PATH.read_text(encoding="utf-8-sig").splitlines()
    if line_number > len(lines):
        raise ToolInputError("行号超出笔记范围")

    # 不使用 strip，避免删掉空格后改变原文的字符位置。
    content = lines[line_number - 1]
    if offset > len(content):
        raise ToolInputError("offset 超出该行长度")

    end = min(len(content), offset + MAX_NOTE_CHARS)
    result = f"[{NOTES_PATH.name}:L{line_number}] {content[offset:end]}"

    if end < len(content):
        result += f"\n【续读提示】本行尚未读完，下次使用 offset={end}。"
    else:
        result += "\n【读取提示】已到本行末尾。"

    return result
    
def search_knowledge(query: str) -> str:
    # 工具边界继续使用统一的输入异常。
    if not isinstance(query, str):
        raise ToolInputError("query 必须是字符串")
    keyword = query.strip()
    if not keyword:
        raise ToolInputError("query 不能为空")
    if len(keyword) > MAX_NOTE_CHARS:
        raise ToolInputError("查询关键词不能长于正文展示上限")

    documents = load_documents()
    chunks = chunk_documents(documents)
    hits = search_chunks(keyword, chunks, top_k=3)

    matches = []
    for score, chunk in hits:
        matches.append({
            "doc_id": chunk.doc_id,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "score": score,
            # 预览长度受限，来源范围描述的是完整片段。
            "preview": chunk.text[:MAX_NOTE_CHARS],
            "truncated": len(chunk.text) > MAX_NOTE_CHARS,
        })

    return json.dumps({"matches": matches}, ensure_ascii=False)
    
def read_knowledge(
    doc_id: str,
    start_line: int,
    end_line: int,
    offset: int = 0,
) -> str:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ToolInputError("doc_id 必须是非空字符串")
    if any(type(value) is not int for value in (start_line, end_line, offset)):
        raise ToolInputError("行号和 offset 必须是整数")
    if start_line < 1 or end_line < start_line or offset < 0:
        raise ToolInputError("行号范围或 offset 不合法")

    # 只接受已加载资料的标识，不把模型输入拼成磁盘路径。
    documents = load_documents()
    if doc_id not in documents:
        raise ToolInputError("未知文档标识，请使用搜索返回的 doc_id")

    lines = documents[doc_id].splitlines()
    if end_line > len(lines):
        raise ToolInputError("行号范围超出文档")

    # 行号从1开始，范围包含首尾两行。
    selected = "\n".join(lines[start_line - 1:end_line])
    if offset > len(selected):
        raise ToolInputError("offset 超出所选正文长度")

    end = min(len(selected), offset + MAX_NOTE_CHARS)
    return json.dumps({
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end_line,
        "offset": offset,
        "text": selected[offset:end],
        "next_offset": end if end < len(selected) else None,
    }, ensure_ascii=False)
    
    
if __name__ == "__main__":
    for query in ("epoll", "Reactor", "Redis"):
        print(f"查询：{query}")
        print(search_notes(query))
        print()