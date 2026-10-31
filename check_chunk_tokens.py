import sys  # 获取命令行传入的 Word 文件路径。
from pathlib import Path  # 构造文档路径和模型缓存路径。
from transformers import AutoTokenizer  # 只加载模型对应的分词器。
from docx_loader import load_docx  # 读取 Word 原始文本及标题信息。
from docx_chunker import build_chunks  # 使用当前规则生成文本块。

MODEL_ID = "BAAI/bge-small-zh-v1.5"  # 使用已经验收的嵌入模型。
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"  # 与模型编码检查保持相同版本。
cache_dir = Path(__file__).resolve().parent / ".cache" / "models"  # 定位已有模型缓存。

tokenizer = AutoTokenizer.from_pretrained(  # 从缓存加载当前模型的分词规则。
    MODEL_ID,  # 指定模型名称。
    revision=MODEL_REVISION,  # 固定分词器版本。
    cache_dir=str(cache_dir),  # 使用此前下载模型的缓存目录。
    local_files_only=True,  # 本次仅使用已经缓存的文件。
    trust_remote_code=False,  # 使用库内置的分词器实现。
)

limit = tokenizer.model_max_length  # 读取当前分词器声明的模型输入上限。
document = load_docx(Path(sys.argv[1]))  # 加载命令行指定的真实资料。
        #chunks = build_chunks(document)  # 按当前默认参数生成文本块。
chunks = build_chunks(document, max_chars=400, overlap=50)  # 缩小正文窗口，为标题和特殊标记保留输入空间。
texts = ["\n".join([*c["heading_path"], c["text"]]) for c in chunks]  # 将标题与正文连接为实际待编码文本。

encoded = tokenizer(  # 对全部待编码文本进行长度统计。
    texts,  # 按文本块顺序处理。
    add_special_tokens=True,  # 把模型需要的特殊标记计入输入长度。
    padding=False,  # 不填充到相同长度，保留每条文本的真实长度。
    truncation=False,  # 不截断，才能发现超过上限的文本。
)

lengths = [len(ids) for ids in encoded["input_ids"]]  # 统计各块完整输入的 Token 数。
over_limit = [(c, n) for c, n in zip(chunks, lengths) if n > limit]  # 收集超限块及对应长度。

print("CHUNKS=", len(chunks))  # 输出本次审计的文本块数量。
print("MAX_TOKENS=", max(lengths, default=0))  # 输出最长输入的 Token 数，空语料输出零。
print("OVER_LIMIT=", len(over_limit))  # 输出超过模型上限的块数。

for chunk, count in sorted(over_limit, key=lambda item: -item[1])[:3]:  # 展示最长的三个超限块。
    print("TOKENS=", count, "SECTION_P=", chunk["section_start_paragraph"], chunk["section_end_paragraph"])  # 输出长度和所属章节范围。
    print("HEADING=", " > ".join(chunk["heading_path"]))  # 输出标题路径，方便定位原文。

        #print("TOKEN_AUDIT_DONE")  # 表示审计执行完成，是否合规需要看超限数量。
assert not over_limit, "仍有文本块超过模型输入上限"  # 存在超限块时明确失败，阻止误判为可以完整编码。
print("TOKEN_LIMIT_OK")  # 表示本次生成的全部文本块均通过实际 Token 长度检查。