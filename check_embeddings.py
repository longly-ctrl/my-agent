from math import hypot, isclose, isfinite  # 检查向量长度、浮点误差和异常数值。
from pathlib import Path  # 根据当前脚本位置构造缓存路径。
from sentence_transformers import SentenceTransformer  # 加载嵌入模型并执行文本编码。

MODEL_ID = "BAAI/bge-small-zh-v1.5"  # 指定本次使用的中文嵌入模型。
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"  # 固定模型仓库版本，减少后续下载时的变化。
cache_dir = Path(__file__).resolve().parent / ".cache" / "models"  # 将模型缓存放在当前项目中。

model = SentenceTransformer(  # 首次运行下载模型，随后将模型加载到内存。
    MODEL_ID,  # 指定模型仓库名称。
    revision=MODEL_REVISION,  # 使用已经核对的固定版本。
    device="cpu",  # 明确使用 CPU 推理。
    cache_folder=str(cache_dir),  # 指定模型文件的本地缓存目录。
    trust_remote_code=False,  # 使用库内置的模型实现。
    model_kwargs={"use_safetensors": True},  # 明确加载仓库中的 safetensors 权重。
)

texts = [  # 使用少量短文本验证编码链路。
    "检索增强生成通过外部资料辅助回答问题。",  # 第一条样本文本。
    "知识库检索可以为回答提供相关依据。",  # 第二条样本文本。
    "今天晚上准备去公园散步。",  # 第三条样本文本。
]

encoded = model.tokenizer(texts, add_special_tokens=True, truncation=False)  # 获取包含特殊标记的完整 Token 序列。
assert all(len(ids) <= model.max_seq_length for ids in encoded["input_ids"])  # 检查样本均未超过模型输入上限。

embeddings = model.encode(  # 将三条文本分别转换为向量。
    texts,  # 按输入顺序编码，输出行与文本一一对应。
    batch_size=3,  # 本次三条样本作为一个批次处理。
    normalize_embeddings=True,  # 将每个输出向量归一化为单位长度。
    convert_to_numpy=True,  # 返回便于检查形状的 NumPy 数组。
    show_progress_bar=False,  # 关闭本次短编码任务的进度条。
)

assert embeddings.shape == (3, 512)  # 检查三条文本各自得到一个 512 维向量。
for vector in embeddings.tolist():  # 转成普通列表，使用标准库检查每个向量。
    assert all(isfinite(value) for value in vector)  # 检查所有分量都是有限数。
    assert isclose(hypot(*vector), 1.0, abs_tol=1e-5)  # 检查归一化后的长度接近一。

print("MAX_SEQ_LENGTH=", model.max_seq_length)  # 输出模型实际使用的输入长度上限。
print("EMBEDDING_SHAPE=", embeddings.shape)  # 输出编码结果的形状。
print("EMBEDDING_MODEL_OK")  # 输出真实模型编码验收标记。