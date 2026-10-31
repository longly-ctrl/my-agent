from openai import OpenAI

from agent import build_initial_messages, run_agent
from session_store import get_session_path, load_session, save_session
from threading import Lock

def _run_session_unlocked(
    client: OpenAI,
    model: str,
    session_id: str,
    question: str,
) -> dict:
    """顺序执行一个具名会话的当前问题。"""
    path = get_session_path(session_id)

    # 每次从磁盘恢复，使用当前代码中的系统规则。
    # 读取失败直接传播，避免覆盖损坏存档。
    history = load_session(path)
    messages = build_initial_messages()
    if history:
        messages.extend(history[1:])

    messages.append({
        "role": "user",
        "content": question,
    })

    # 执行失败直接传播；尚未保存，原存档保持不变。
    answer = run_agent(client, model, messages, max_steps=5)

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
        return _run_session_unlocked(
            client, model, session_id, question
        )
    finally:
        # 只有取得锁的请求才会进入这里；成功或异常都释放。
        session_lock.release()