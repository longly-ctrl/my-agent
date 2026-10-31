from hashlib import sha256  # 根据文本块的来源和位置生成确定性的标识。
import json  # 将标识所需的多个字段编码为结构明确的字符串。


def split_sections(document: dict) -> list[dict]:  # 按标题切分正文，保留章节路径和来源。
    sections = []  # 收集已经完成的章节单元。
    heading_stack = []  # 保存当前从父标题到子标题的路径。
    buffer = []  # 暂存当前标题路径下的正文段落。

    def flush_section() -> None:  # 将当前正文保存成一个独立章节单元。
        if not buffer:  # 判断当前标题路径下是否实际存在正文。
            return  # 连续标题之间没有正文时，不生成空单元。

        sections.append({  # 将当前来源、标题路径和正文一起保存。
            "source": document["source"],  # 保留原始文档路径。
            "source_hash": document["source_hash"],  # 绑定原始文档版本。
            "heading_path": [h["text"] for h in heading_stack],  # 保存从父级到子级的标题文字。
            "heading_paragraphs": [h["paragraph"] for h in heading_stack],  # 保留各级标题的原始位置。
            "start_paragraph": buffer[0]["paragraph"],  # 记录正文起始段落。
            "end_paragraph": buffer[-1]["paragraph"],  # 记录正文结束段落。
            "paragraphs": [record.copy() for record in buffer],  # 复制正文记录，避免共享可变字典。
        })

        buffer.clear()  # 清空当前缓存；已经复制到章节单元中的记录不受影响。

    for record in document["paragraphs"]:  # 按原文顺序处理标题和正文。
        level = record["heading_level"]  # 读取上一阶段识别的标题级别。

        if level is not None:  # 当前记录是一个标题。
            flush_section()  # 先把旧正文归入旧标题路径，再修改路径。

            while heading_stack and heading_stack[-1]["level"] >= level:  # 查找新标题应归属的父级。
                heading_stack.pop()  # 移除旧的同级标题及其更深层标题。

            heading_stack.append({  # 将当前标题加入新的路径。
                "level": level,  # 保存真实级别，允许标题出现跳级。
                "paragraph": record["paragraph"],  # 保存标题在原文中的位置。
                "text": record["text"],  # 保存标题文字。
            })

        else:  # 当前记录是普通正文。
            buffer.append(record)  # 加入当前章节，等待遇到下一标题或文档结束。

    flush_section()  # 保存文档末尾尚未提交的正文。
    return sections  # 返回带标题路径与来源位置的章节单元。
    
def split_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[dict]:  # 将文本切成长度受限、相邻重叠的片段。
    if max_chars <= 0:  # 检查每块允许的字符数是否为正数。
        raise ValueError("max_chars 必须大于 0")  # 拒绝无法生成有效文本块的长度限制。

    if not 0 <= overlap < max_chars:  # 重叠长度必须非负，并保证窗口能够向前移动。
        raise ValueError("overlap 必须满足 0 <= overlap < max_chars")  # 避免窗口停留原地或向后移动。

    chunks = []  # 保存切分后的文本及其位置。
    start = 0  # 从原始文本的第一个字符开始。

    while start < len(text):  # 只要还有文本未处理，就继续生成片段。
        end = min(start + max_chars, len(text))  # 限制结束位置，避免超过原文长度。

        chunks.append({  # 保存文本块及其在原文中的字符范围。
            "start_char": start,  # 起点包含在当前片段内，从 0 开始计数。
            "end_char": end,  # 终点不包含在当前片段内，与 Python 切片规则一致。
            "text": text[start:end],  # 按字符范围提取原文，不删除空格或换行。
        })

        if end == len(text):  # 判断当前片段是否已经覆盖原文末尾。
            break  # 覆盖末尾后立即结束，避免额外生成重复尾块。

        start = end - overlap  # 下一块回退指定长度，让相邻片段保留共同上下文。

    return chunks  # 返回切分结果；空字符串会返回空列表。
    
def build_chunks(document: dict, max_chars: int = 800, overlap: int = 100) -> list[dict]:  # 将文档转换为可建立检索索引的文本块。
    split_text("", max_chars=max_chars, overlap=overlap)  # 复用参数校验，保证空文档也会拒绝非法参数。
    chunks = []  # 收集所有章节生成的文本块。

    for section in split_sections(document):  # 逐个处理章节，避免文本块跨越章节边界。
        text = "\n".join(record["text"] for record in section["paragraphs"])  # 按原顺序连接正文，用换行分隔段落。

        for part in split_text(text, max_chars=max_chars, overlap=overlap):  # 在当前章节内部生成重叠窗口。
            identity = json.dumps([  # 将共同决定文本块身份的字段编码为 JSON 数组。
                "docx-char-v1",  # 标记当前切分规则的版本。
                section["source"],  # 区分不同路径下的来源文档。
                section["source_hash"],  # 区分同一来源文档的不同文件版本。
                section["start_paragraph"],  # 区分文档内的不同章节单元。
                part["start_char"],  # 区分同一章节内的不同窗口起点。
                part["end_char"],  # 区分同一起点但长度不同的窗口。
            ], ensure_ascii=False)  # 保留中文路径，随后统一按 UTF-8 编码。

            chunks.append({  # 保存索引、融合和来源核对所需的数据。
                "chunk_id": sha256(identity.encode("utf-8")).hexdigest(),  # 相同来源版本及位置生成相同标识。
                "source": section["source"],  # 保留原始文档的绝对路径。
                "source_hash": section["source_hash"],  # 保留原始文件指纹。
                "heading_path": section["heading_path"].copy(),  # 复制标题路径，避免与其他块共享可变列表。
                "heading_paragraphs": section["heading_paragraphs"].copy(),  # 保存各级标题在原文中的段落编号。
                "section_start_paragraph": section["start_paragraph"],  # 保存所属章节正文的起始段落。
                "section_end_paragraph": section["end_paragraph"],  # 保存所属章节正文的结束段落。
                **part,  # 展开窗口字典，加入 start_char、end_char 和 text。
            })

    return chunks  # 返回统一格式的文本块，供两路检索共用。