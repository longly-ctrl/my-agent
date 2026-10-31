from math import hypot, isclose, isfinite  # 检查输出向量是否有限且已归一化。
from pathlib import Path  # 构造本地模型缓存路径。
from sentence_transformers import SentenceTransformer  # 加载模型并批量编码文本。

MODEL_ID = "BAAI/bge-small-zh-v1.5"  # 使用已经验收的中文嵌入模型。
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"  # 固定模型文件版本。
DIMENSION = 512  # 记录当前模型的输出维度。
CHUNK_PARAMS = {"max_chars": 400, "overlap": 50}  # 集中保存已经通过长度检查的切块参数。


def load_embedding_model() -> SentenceTransformer:  # 显式加载模型，避免导入模块时立即占用模型资源。
    cache_dir = Path(__file__).resolve().parent / ".cache" / "models"  # 定位已下载的模型文件。
    return SentenceTransformer(  # 创建模型对象并交给调用方复用。
        MODEL_ID,  # 指定模型名称。
        revision=MODEL_REVISION,  # 固定模型版本。
        device="cpu",  # 使用 CPU 推理。
        cache_folder=str(cache_dir),  # 使用项目内的模型缓存。
        local_files_only=True,  # 本次只加载已经下载的文件。
        trust_remote_code=False,  # 使用库内置的模型实现。
        model_kwargs={"use_safetensors": True},  # 加载已经验收的权重格式。
    )


def encode_chunks(model: SentenceTransformer, chunks: list[dict]) -> dict[str, list[float]]:  # 返回文本块标识到向量的映射。
    chunk_ids = [c["chunk_id"] for c in chunks]  # 保留文本块的输入顺序。
    if len(set(chunk_ids)) != len(chunk_ids):  # 检查是否存在重复标识。
        raise ValueError("chunk_id 不能重复")  # 避免构造字典时覆盖已有向量。

    if not chunks:  # 检查是否为空语料。
        return {}  # 空语料不调用模型。

    texts = ["\n".join([*c["heading_path"], c["text"]]) for c in chunks]  # 使用与长度审计相同的标题和正文格式。
    tokenized = model.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)  # 获取未经截断的完整输入长度。

    for cid, ids in zip(chunk_ids, tokenized["input_ids"]):  # 逐块核对实际输入长度。
        if len(ids) > model.max_seq_length:  # 检查是否超过当前模型限制。
            raise ValueError(f"文本块超限：{cid}，Token 数：{len(ids)}")  # 在编码前终止，防止遗漏尾部内容。

    embeddings = model.encode(  # 批量生成全部文本块的向量。
        texts,  # 输出行与输入文本顺序对应。
        batch_size=8,  # 每批处理八条，控制 CPU 推理时的批次规模。
        normalize_embeddings=True,  # 将输出向量归一化为单位长度。
        convert_to_numpy=True,  # 返回便于检查形状的数组。
        show_progress_bar=True,  # 显示整批编码的进度。
    )

    if embeddings.shape != (len(chunks), DIMENSION):  # 检查输出数量和维度。
        raise ValueError("编码结果数量或维度不符合预期")  # 阻止错误结果进入索引。

    vectors = embeddings.tolist()  # 转成普通浮点列表，便于检索和 JSON 保存。
    for vector in vectors:  # 逐个检查输出向量。
        if not all(isfinite(v) for v in vector) or not isclose(hypot(*vector), 1.0, abs_tol=1e-5):  # 检查数值有效性和单位长度。
            raise ValueError("编码结果包含非法数值或未正确归一化")  # 拒绝异常模型输出。

    return dict(zip(chunk_ids, vectors))  # 按相同顺序将每个标识绑定到对应向量。
    
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："  # 使用当前模型建议的查询前缀。


def encode_query(model: SentenceTransformer, query: str) -> list[float]:  # 将一个用户问题编码为查询向量。
    question = query.strip()  # 去除问题两端的空白。
    if not question:  # 检查问题是否包含实际内容。
        raise ValueError("查询不能为空")  # 拒绝空问题。

    text = QUERY_INSTRUCTION + question  # 将检索指令添加到问题前面。
    tokenized = model.tokenizer(text, add_special_tokens=True, truncation=False)  # 统计包含指令和特殊标记的完整长度。
    if len(tokenized["input_ids"]) > model.max_seq_length:  # 检查整个查询输入是否超限。
        raise ValueError("查询超过模型输入上限")  # 避免问题尾部被截断。

    embeddings = model.encode(  # 使用与资料相同的模型生成查询向量。
        [text],  # 单条问题也使用列表，保持二维输出。
        normalize_embeddings=True,  # 与资料向量采用相同的归一化约定。
        convert_to_numpy=True,  # 返回可检查形状的数组。
        show_progress_bar=False,  # 单条查询关闭进度条。
    )

    if embeddings.shape != (1, DIMENSION):  # 检查输出是一条规定维度的向量。
        raise ValueError("查询向量形状不符合预期")  # 拒绝数量或维度异常的结果。

    vector = embeddings[0].tolist()  # 取出唯一一行，转换为普通列表。
    if not all(isfinite(v) for v in vector) or not isclose(hypot(*vector), 1.0, abs_tol=1e-5):  # 检查数值和归一化结果。
        raise ValueError("查询向量无效")  # 阻止异常向量参与排序。

    return vector  # 返回可直接交给向量检索函数的查询向量。