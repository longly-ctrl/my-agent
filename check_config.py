from pathlib import Path
from dotenv import dotenv_values

# 根据脚本所在目录定位配置，避免受终端当前目录影响。
env_path = Path(__file__).resolve().parent / ".env"
config = dotenv_values(env_path)

# 这里只读取本地文件，不发送网络请求。
key = (config.get("LLM_API_KEY") or "").strip()
key_ready = bool(key) and key != "替换为你的DeepSeek API Key"

print("配置文件存在：", env_path.is_file())
print("模型：", config.get("LLM_MODEL_ID"))
print("服务地址：", config.get("LLM_BASE_URL"))
print("密钥已填写：", key_ready)  # 只输出检查结果，不打印密钥。