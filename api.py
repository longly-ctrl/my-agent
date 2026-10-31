from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI, APIError

from agent import build_initial_messages, run_agent
from app_config import load_config
from fastapi.responses import JSONResponse
from session_service import run_session, SessionBusyError

# 应用对象保存路由等配置，由 Uvicorn 加载。
app = FastAPI(title="Local Knowledge Agent")

class UTF8JSONResponse(JSONResponse):
    # 正文使用 UTF-8，同时明确告知客户端解码方式。
    media_type = "application/json; charset=utf-8"

@app.get("/health")
def health() -> dict[str, str]:
    # 只检查 HTTP 服务能否响应，不调用模型或读写会话。
    return {"status": "ok"}
    
class ChatRequest(BaseModel):       #定义请求的数据规则    
    # 去除字符串首尾空白，拒绝未声明的字段。
    model_config = ConfigDict(      #配置整个模型的处理方式
        str_strip_whitespace=True,  #去掉字符串首尾空白
        extra="forbid",             #拒绝未声明的字段
    )
    # 这里限制字符数，不是 Token 数。
    question: str = Field(min_length=1, max_length=2000)    #声明字段及其约束
#   字段名称：字段预期是字符串   为字段声明约束等信息  长度至少为 1 长度最多为 2000，包含 2000
    # 省略 ID 时保持单轮问答；提供 ID 时恢复并保存该会话。
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9_-]+$",
    )

#                              增加了“可选具名会话”和“会话忙碌处理”。                                           0.1
# @app.post("/chat", response_class=UTF8JSONResponse)
# def chat(body: ChatRequest) -> dict[str, str]:
    # # 当前 SDK 和 Agent 都是同步调用，因此使用普通 def。
    # # FastAPI 会在线程池中执行这个同步路由。
    # try:
        # config = load_config()

        # # 每个请求拥有独立历史，本步不加载或保存会话。
        # messages = build_initial_messages()
        # messages.append({
            # "role": "user",
            # "content": body.question,
        # })

        # # 请求结束时关闭客户端；异常退出也会执行清理。
        # with OpenAI(
            # api_key=config["LLM_API_KEY"],
            # base_url=config["LLM_BASE_URL"],
            # timeout=30.0,
            # max_retries=0,
        # ) as client:
            # answer = run_agent(
                # client,
                # config["LLM_MODEL_ID"],
                # messages,
                # max_steps=5,
            # )

    # except APIError as exc:
        # # 只记录异常类型，不输出密钥或原始服务端错误内容。
        # print("模型调用失败类型：", type(exc).__name__, flush=True)
        # raise HTTPException(
            # status_code=502,
            # detail="模型服务调用失败",
        # ) from exc

    # except (OSError, ValueError, TypeError, RuntimeError) as exc:
        # print("Agent执行失败类型：", type(exc).__name__, flush=True)
        # raise HTTPException(
            # status_code=500,
            # detail="Agent执行失败",
        # ) from exc

    # return {"answer": answer}
#                                                                                                           0.1

@app.post("/chat", response_class=UTF8JSONResponse)
def chat(body: ChatRequest) -> dict:
    # 返回值现在可能包含 bool 和 None，因此不再声明为 dict[str, str]。
    try:
        config = load_config()

        with OpenAI(
            api_key=config["LLM_API_KEY"],
            base_url=config["LLM_BASE_URL"],
            timeout=30.0,
            max_retries=0,
        ) as client:
            if body.session_id is None:
                # 未指定 ID：独立问答，不写入存档。
                messages = build_initial_messages()
                messages.append({
                    "role": "user",
                    "content": body.question,
                })
                answer = run_agent(
                    client,
                    config["LLM_MODEL_ID"],
                    messages,
                    max_steps=5,
                )
                result = {"answer": answer}
            else:
                # 指定 ID：通过会话层完成加锁、恢复、执行和保存。
                result = run_session(
                    client,
                    config["LLM_MODEL_ID"],
                    body.session_id,
                    body.question,
                )

    except SessionBusyError as exc:
        # 该异常继承 RuntimeError，必须先于通用处理捕获。不然会把会话忙碌也当成普通执行失败，返回 500，而不是预期的 409。
        raise HTTPException(
            status_code=409,
            detail="该会话正在处理中，请稍后重试",
        ) from exc

    except APIError as exc:
        print("模型调用失败类型：", type(exc).__name__, flush=True)
        raise HTTPException(
            status_code=502,
            detail="模型服务调用失败",
        ) from exc

    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print("Agent执行失败类型：", type(exc).__name__, flush=True)
        raise HTTPException(
            status_code=500,
            detail="Agent执行失败",
        ) from exc

    return result