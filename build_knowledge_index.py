import argparse  # 解析原始文档和输出路径。
import json  # 序列化文本块、模型配置与向量。
from pathlib import Path  # 定位文档和缓存文件。
from tempfile import NamedTemporaryFile  # 创建本次构建独占的临时文件。
from docx_loader import load_docx  # 从同一份文档字节快照提取正文与指纹。
from docx_chunker import build_chunks  # 生成带来源信息的文本块。


def build_index(docx_path: Path, output_path: Path) -> Path:  # 从 Word 构建新的向量缓存。
    output = output_path.resolve()  # 将输出位置解析为绝对路径。
    if output.exists():  # 在读取和编码前检查目标。
        raise FileExistsError(f"输出文件已存在，请使用新的输出路径：{output}")  # 拒绝覆盖已有缓存。

    document = load_docx(docx_path)  # 读取原始资料及文件指纹。
    from embedding_service import MODEL_ID, MODEL_REVISION, DIMENSION, CHUNK_PARAMS, load_embedding_model, encode_chunks  # 复用已验收的配置和编码函数。
    from retrieval_service import load_retrieval_cache  # 复用发布前的缓存校验。

    chunks = build_chunks(document, **CHUNK_PARAMS)  # 显式使用当前四百字符窗口和五十字符重叠。
    if not chunks:  # 检查是否产生可编码正文。
        raise ValueError("原始资料没有可建立索引的文本块")  # 避免发布空索引。
    model = load_embedding_model()  # 从本地缓存加载一次固定版本的 CPU 模型。
    vectors = encode_chunks(model, chunks)  # 检查完整 Token 长度并生成归一化向量。

    payload = {  # 使用当前检索服务支持的缓存结构。
        "schema_version": 1,  # 标记缓存结构版本。
        "model_id": MODEL_ID,  # 记录嵌入模型。
        "model_revision": MODEL_REVISION,  # 记录固定模型版本。
        "dimension": DIMENSION,  # 记录向量维度。
        "normalize_embeddings": True,  # 记录归一化约定。
        "text_format": "heading-path-newline-body-v1",  # 记录标题和正文连接方式。
        "chunk_params": CHUNK_PARAMS.copy(),  # 保存实际分块参数的副本。
        "chunks": chunks,  # 保存完整文本块及来源信息。
        "vectors": vectors,  # 按块标识保存对应向量。
    }

    output.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = None  # 保存本次临时路径，供失败时清理。
    try:  # 依次完成写入、校验和发布。
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False) as stream:  # 在目标目录创建独占临时文件。
            temporary = Path(stream.name)  # 保存实际临时文件路径。
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)  # 完整写入并拒绝非法浮点值。
        load_retrieval_cache(temporary, docx_path)  # 关闭文件后验证缓存，并检查原文是否在编码期间变化。
        temporary.rename(output)  # 在当前 Windows 环境发布新文件，目标已存在时会报错。
    finally:  # 成功或失败都只清理本次临时路径。
        if temporary is not None:  # 检查是否创建过临时文件。
            temporary.unlink(missing_ok=True)  # 发布成功后旧路径已消失；失败时删除残留文件。

    print("CHUNKS=", len(chunks))  # 显示生成的块数。
    print("VECTORS=", len(vectors))  # 显示向量数量。
    print("DIMENSION=", DIMENSION)  # 显示向量维度。
    print("OUTPUT=", output)  # 显示新缓存位置。
    print("KNOWLEDGE_INDEX_BUILT")  # 表示构建、结构校验和发布完成。
    return output  # 返回新文件路径。


if __name__ == "__main__":  # 导入模块时不执行编码或写入。
    parser = argparse.ArgumentParser(description="从 Word 重建本地知识库向量缓存")  # 创建解析器。
    parser.add_argument("--docx", type=Path, required=True, help="原始 Word 文件")  # 要求指定输入资料。
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / ".cache" / "knowledge_vectors.json", help="新缓存路径；拒绝覆盖已有文件")  # 设置默认缓存位置。
    args = parser.parse_args()  # 解析参数。
    build_index(args.docx, args.output)  # 执行构建。