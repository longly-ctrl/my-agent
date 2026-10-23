from pathlib import Path
from dotenv import dotenv_values


def load_config(env_path: Path | None = None) -> dict[str, str]:
    # 默认读取本模块旁边的配置，也允许测试传入其他路径。
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"

    values = dotenv_values(env_path, encoding="utf-8-sig")
    config = {}

    for name in ("LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL"):
        # 校验后返回清理过首尾空白的值。
        value = (values.get(name) or "").strip()
        if not value:
            raise ValueError(f"配置缺失：{name}")
        config[name] = value

    return config