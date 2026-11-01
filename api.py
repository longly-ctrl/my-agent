from fastapi import FastAPI, HTTPException, Request  # 获取当前请求所属的应用实例。
from contextlib import asynccontextmanager  # 定义应用启动与停机的上下文。
from collections.abc import AsyncIterator  # 标注异步生成器的产出类型。
from pathlib import Path  # 定位缓存与原始文档。
import os  # 读取服务器启动环境中的资料路径。
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI, APIError

from agent import build_initial_messages, run_agent
from app_config import load_config
from fastapi.responses import JSONResponse
from session_service import run_session, SessionBusyError

# 应用对象保存路由等配置，由 Uvicorn 加载。
@asynccontextmanager  # 将异步生成器转换为应用生命周期上下文。
async def lifespan(application: FastAPI) -> AsyncIterator[None]:  # 管理当前应用实例持有的检索资源。
    application.state.external_search = None  # 默认使用原来的 Markdown 模式。

    try:  # 确保正常停机或初始化异常时清理应用保存的引用。
        source_path = os.environ.get("RAG_DOCX_PATH", "").strip()  # 在启动时读取外部资料配置。
        if source_path:  # 只有明确配置 Word 路径时才加载检索资源。
            from retrieval_service import RetrievalService  # 按需导入检索服务。
            from external_knowledge_tool import ExternalKnowledgeTool  # 按需导入工具适配层。

            root = Path(__file__).resolve().parent  # 使用项目目录定位缓存。
            retrieval = RetrievalService(root / ".cache" / "knowledge_vectors.json", Path(source_path))  # 校验缓存并加载一次模型。
            external_tool = ExternalKnowledgeTool(retrieval)  # 让工具持有检索服务。
            application.state.external_search = external_tool.search  # 保存供所有请求复用的绑定方法。
            print("HTTP 外部 Word 混合检索已启用，精排关闭。")  # 显示当前服务模式。

        yield  # 启动准备完成，允许应用处理请求。
    finally:  # 应用停机或初始化异常时执行。
        application.state.external_search = None  # 移除应用对工具的引用。


app = FastAPI(title="Local Knowledge Agent", lifespan=lifespan)  # 将生命周期管理函数交给 FastAPI。

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
def chat(body: ChatRequest, request: Request) -> dict:  # 由 FastAPI 注入当前请求对象。
    # 返回值现在可能包含 bool 和 None，因此不再声明为 dict[str, str]。
    try:
        external_search = request.app.state.external_search  # 获取启动时创建的共享工具。
        config = load_config()

        with OpenAI(
            api_key=config["LLM_API_KEY"],
            base_url=config["LLM_BASE_URL"],
            timeout=30.0,
            max_retries=0,
        ) as client:
            if body.session_id is None:
                # 未指定 ID：独立问答，不写入存档。
                messages = build_initial_messages(external_enabled=external_search is not None)  # 按当前服务能力选择系统规则。
                messages.append({
                    "role": "user",
                    "content": body.question,
                })
                answer = run_agent(  # 执行独立问答。
                    client,  # 使用当前请求的云端客户端。
                    config["LLM_MODEL_ID"],  # 使用本地模型配置。
                    messages,  # 使用本次独立历史。
                    max_steps=5,  # 保留调用预算。
                    external_search=external_search,  # 注入共享检索工具。
                )
                result = {"answer": answer}
            else:
                # 指定 ID：通过会话层完成加锁、恢复、执行和保存。
                result = run_session(  # 通过会话层完成加锁、恢复、执行和保存。
                    client,  # 使用当前请求的客户端。
                    config["LLM_MODEL_ID"],  # 转发模型配置。
                    body.session_id,  # 转发会话标识。
                    body.question,  # 转发当前问题。
                    external_search=external_search,  # 与单轮分支使用同一个检索工具。
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
        if isinstance(exc, RuntimeError):  # 仅为本次运行异常补充具体原因。
            print("Agent运行失败原因：", str(exc), flush=True)  # 立即输出异常说明，供本地诊断。
        raise HTTPException(
            status_code=500,
            detail="Agent执行失败",
        ) from exc

    return result