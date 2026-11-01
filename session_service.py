from openai import OpenAI

from agent import build_initial_messages, run_agent
from session_store import get_session_path, load_session, save_session
from threading import Lock
from collections.abc import Callable  # 描述调用方提供的检索函数接口。

def _run_session_unlocked(
    client: OpenAI,
    model: str,
    session_id: str,
    question: str,
    *,  # 后续参数必须按名称传入，保持原有位置参数兼容。
    external_search: Callable[[str], str] | None = None,  # 接收共享检索工具的绑定方法。
) -> dict:
    """顺序执行一个具名会话的当前问题。"""
    path = get_session_path(session_id)

    # 每次从磁盘恢复，使用当前代码中的系统规则。
    # 读取失败直接传播，避免覆盖损坏存档。
    history = load_session(path)
    messages = build_initial_messages(external_enabled=external_search is not None)  # 根据实际提供的工具选择当前系统规则。
    if history:
        messages.extend(history[1:])

    messages.append({
        "role": "user",
        "content": question,
    })

    # 执行失败直接传播；尚未保存，原存档保持不变。
    answer = run_agent(  # 执行当前会话的问题。
        client,  # 使用调用方提供的客户端。
        model,  # 使用调用方指定的模型。
        messages,  # 传入恢复历史并追加当前问题后的消息。
        max_steps=5,  # 保留现有调用预算。
        external_search=external_search,  # 将检索函数继续传给 Agent。
    )

    # 回答已经完成，保存失败需要单独表达。
    saved = True
    save_error = None
    try:
        save_session(messages, path)
    except (OSError, TypeError, ValueError) as exc:
        saved = False
        save_error = type(exc).__name__

    return {
        "session_id": session_id,
        "answer": answer,
        "saved": saved,
        "save_error": save_error,
    }

class SessionBusyError(RuntimeError):
    """同一会话已有请求正在执行。"""


# 保护锁表本身，确保同一个 ID 始终取得同一把锁。
_registry_lock = Lock() # 保护“会话锁登记表”的访问与创建
_session_locks = {}     # 保存 session_id → 对应的锁


def run_session(
    client: OpenAI,
    model: str,
    session_id: str,
    question: str,
    *,  # 新增参数必须通过名称传入，兼容原来的四个位置参数。
    external_search: Callable[[str], str] | None = None,  # 接收外部检索函数；省略时使用 None。
) -> dict:
    # 先校验 ID，避免为非法输入创建锁表记录。
    get_session_path(session_id)

    with _registry_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = Lock()
        session_lock = _session_locks[session_id]

    # 不排队等待：忙碌时交给调用方明确处理。
    if not session_lock.acquire(blocking=False):    #blocking=False 表示获取不到时立即返回。获取成功返回 True，失败返回 False。
        raise SessionBusyError("该会话正在处理中，请稍后重试")

    try:
        # 必须保护完整的“读取—执行—保存”，避免并发覆盖历史。
        return _run_session_unlocked(  # 在持有当前会话锁期间执行完整流程。
            client,  # 转发客户端。
            model,  # 转发模型配置。
            session_id,  # 转发已经校验的会话标识。
            question,  # 转发当前问题。
            external_search=external_search,  # 转发同一个检索函数，不重新创建模型。
        )
    finally:
        # 只有取得锁的请求才会进入这里；成功或异常都释放。
        session_lock.release()