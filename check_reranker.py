from math import isfinite  # 检查模型输出是否包含非法数值。
from pathlib import Path  # 构造项目内的模型缓存路径。
from sentence_transformers import CrossEncoder  # 加载联合处理问题与原文的精排模型。
from torch.nn import Identity  # 保留模型原始分数，便于明确评分含义。

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # 指定本次验证的多语言精排模型。
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"  # 固定已经核对的模型仓库版本。
cache_dir = Path(__file__).resolve().parent / ".cache" / "models"  # 复用项目内的模型缓存目录。

model = CrossEncoder(  # 首次运行下载模型，然后加载到 CPU。
    MODEL_ID,  # 指定模型名称。
    revision=MODEL_REVISION,  # 固定模型版本。
    device="cpu",  # 使用 CPU 推理。
    cache_folder=str(cache_dir),  # 将下载文件放入已有缓存目录。
    max_length=512,  # 限制问题与原文合并后的总输入长度。
    activation_fn=Identity(),  # 直接输出原始相关性分数。
    trust_remote_code=False,  # 使用库内置的模型实现。
    model_kwargs={"use_safetensors": True},  # 明确加载 safetensors 权重。
)

query = "相邻文本块保留重叠内容，主要缓解什么问题？"  # 构造一个含义明确的中文问题。
pairs = [  # 为同一问题准备相关和不相关的原文。
    (query, "在相邻文本块之间保留重叠内容，可以缓解切分边界造成的信息断裂。"),  # 直接回答问题。
    (query, "Transformer 的每层通常包含自注意力和前馈网络。"),  # 讨论其他主题。
]

tokenized = model.tokenizer(  # 按文本对的形式检查完整输入长度。
    [pair[0] for pair in pairs],  # 提供每对输入中的问题。
    [pair[1] for pair in pairs],  # 提供每对输入中的候选原文。
    add_special_tokens=True,  # 将模型需要的特殊标记计入长度。
    padding=False,  # 保留每对输入的实际长度。
    truncation=False,  # 检查前不截断，避免隐藏超限情况。
)
assert all(len(ids) <= model.max_length for ids in tokenized["input_ids"])  # 检查每个完整文本对均符合输入限制。

scores = model.predict(  # 为每个问题与原文组合计算相关性分数。
    pairs,  # 输出顺序与输入文本对一致。
    batch_size=1,  # 起步时逐对处理，控制内存占用。
    convert_to_numpy=True,  # 返回便于检查形状的数组。
    show_progress_bar=False,  # 两条样本关闭进度条。
)

assert scores.shape == (2,)  # 检查两个文本对分别得到一个分数。
assert all(isfinite(float(score)) for score in scores)  # 检查所有输出均为有限数。
assert scores[0] > scores[1], "相关原文没有获得更高分数"  # 验证最基本的中文相关性区分。

print("SCORES=", scores.tolist())  # 展示两条原文的实际分数，不预设具体数值。
print("RERANKER_MODEL_OK")  # 表示模型加载、评分和本条相关性检查通过。