from pathlib import Path
from dataclasses import dataclass

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def load_documents(root: Path | None = None) -> dict[str, str]:
    # 默认位置不受终端当前目录影响；测试可以传入临时目录。
    root = (KNOWLEDGE_DIR if root is None else root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在：{root}")
    if not root.is_dir():
        raise NotADirectoryError(f"知识库路径不是目录：{root}")

    documents = {}
    # 固定遍历顺序，只加载当前目录第一层的 Markdown 文件。
    for path in sorted(root.glob("*.md")):  #找到 root 目录下名称匹配 *.md 的路径，排序后逐个处理。
        if not path.is_file():
            continue

        # 检查解析后的真实路径，防止链接指向目录之外。
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"文档位于知识库目录之外：{path.name}")

        # 保留空行和缩进，方便后续生成准确的来源行号。
        documents[path.name] = resolved.read_text(
            encoding="utf-8-sig"
        )
    return documents
    
@dataclass(frozen=True) #把下面的类定义成数据类，并限制对象创建后的字段修改。
#frozen=True 表示创建后禁止正常修改字段
class Chunk:
    """一个可检索片段，以及它在原文件中的位置。"""
    doc_id: str
    start_line: int
    end_line: int
    text: str


def chunk_documents(documents: dict[str, str]) -> list[Chunk]:
    chunks = []

    for doc_id, text in documents.items():
        buffer = []
        start_line = 0

        # 末尾补一个空行，确保最后一段也能被提交。
        for line_number, line in enumerate(
            text.splitlines() + [""], start=1
        ):
            if line.strip():
                if not buffer:
                    start_line = line_number
                # 保留原始缩进，strip 只用于判断是否为空行。
                buffer.append(line)
            elif buffer:
                chunks.append(Chunk(
                    doc_id=doc_id,
                    start_line=start_line,
                    end_line=line_number - 1,
                    text="\n".join(buffer),
                ))
                buffer.clear()

    return chunks

def search_chunks(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[tuple[int, Chunk]]:
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")

    # 按空格拆词、忽略大小写、去重。
    terms = set(query.casefold().split())
    if not terms:
        raise ValueError("query 不能为空")

    ranked = []
    for chunk in chunks:
        text = chunk.text.casefold()

        # 每命中一个不同的查询词得1分，重复出现不额外加分。
        score = sum(term in text for term in terms)
        if score > 0:
            ranked.append((score, chunk))

    # 分数降序；同分按文件名、起始行号排序。
    ranked.sort(key=lambda item: (
        -item[0],
        item[1].doc_id,
        item[1].start_line,
    ))
    return ranked[:top_k]