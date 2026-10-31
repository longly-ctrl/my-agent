# 本地知识库 Agent

参考 Hello-Agents，使用 Python 手写工具调用循环。
知识文件、工具执行和会话存档位于本机，模型推理由 DeepSeek 云端 API 完成。

## 功能与结构

- `first_call.py`：命令行入口、会话选择、失败回滚。
- `api.py`：FastAPI `/health` 和 `/chat` 接口。
- `agent.py`：系统规则、请求上下文裁剪、模型与工具调用循环。
- `tools.py`、`tool_schemas.py`：四个工具的实现与模型可见说明。
- `knowledge_base.py`：加载 Markdown、按空行分块、关键词检索。
- `session_service.py`：具名会话恢复、执行、保存及进程内互斥。
- `session_store.py`：JSON 存档、加载校验及临时文件替换。
- `eval_cases.json`、`eval_runner.py`：固定评估问题与执行报告。

默认使用 `knowledge/` 中的资料。
只有用户明确要求根目录旧笔记时，才使用 `notes.md` 相关工具。
旧笔记是可选本地文件，不包含在版本管理中。

## 环境与配置

验证环境：Windows PowerShell、Python 3.12.14。
直接依赖版本见 `requirements.txt`。

以下初始化命令用于新环境，需要已安装 Python 3.12 和 Windows Python 启动器：

```powershell
Set-Location D:\1AI\Agent
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

在编辑器中填写 `.env` 的 `LLM_API_KEY`。
模型和地址由 `LLM_MODEL_ID`、`LLM_BASE_URL` 配置。

不要将真实 Key 写入 README、示例文件或提交到版本管理。
`requirements.txt` 只固定直接依赖，尚未完整锁定传递依赖。
当前环境已通过依赖一致性检查，尚未完成全新环境重装验收。

```powershell
.\.venv\Scripts\python.exe -B -m pip check
```

## 命令行运行

```powershell
.\.venv\Scripts\python.exe -B first_call.py --session alpha
```

- 指定 ID：使用 `sessions/session_alpha.json`。
- 不指定 ID：使用原来的 `session.json`。
- 输入 `exit` 或 `quit` 退出。
- ID 仅允许小写字母、数字、下划线、短横线，长度为 1～64。

## HTTP 运行

启动单进程服务：

```powershell
.\.venv\Scripts\python.exe -B -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

在另一个 PowerShell 窗口检查健康接口：

```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing |
    Select-Object StatusCode, Content
```

预期 HTTP 200，正文为 `{"status":"ok"}`。
这个接口只检查 HTTP 服务响应，不验证模型、检索或存储是否可用。

发送具名问答：

```powershell
$payload = @{
    session_id = 'demo_final'
    question = '仅依据本地知识库，解释 epoll，并引用来源。'
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri 'http://127.0.0.1:8000/chat' `
    -Method Post `
    -ContentType 'application/json; charset=utf-8' `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($payload)) |
    Format-List
```

该请求会调用云端模型。

- 提供 `session_id`：恢复并保存具名会话，返回
  `session_id`、`answer`、`saved`、`save_error`。
- 省略 `session_id`：独立问答，只返回 `answer`，不使用默认存档。
- `question` 去除首尾空白后长度须为 1～2000 字符。
- 请求中的未知字段会被拒绝。

| 状态 | 含义 |
| --- | --- |
| 200 | 返回回答；具名请求还需检查 `saved` |
| 409 | 同一会话已有请求正在执行 |
| 422 | 请求参数校验失败 |
| 500 | 配置、存储读取或 Agent 执行失败等 |
| 502 | 模型服务调用失败 |

`200` 且 `saved=False` 表示回答成功、保存失败。
此时 `save_error` 提供异常类型，不能据此认定历史已经持久化。

## 测试与评估

固定离线测试，不调用云端模型：

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_session_store test_agent
```

当前共 8 项，覆盖存储读写、损坏文件、工具结果回传、
上下文裁剪、模型调用预算、参数纠错及异常传播等。

API 会话隔离、锁释放、409 映射和保存失败行为另经补充离线检查；
它们目前尚未固化到上述 8 项测试中，也尚无完整的 HTTP 自动回归测试。

云端评估：

```powershell
.\.venv\Scripts\python.exe -B .\eval_runner.py
```

| 用例 | 检查目标 |
| --- | --- |
| `hit_epoll` | 正确使用命中资料并引用 |
| `missing_mqtt` | 无资料时说明不足，不补充外部知识 |
| `partial_raii` | 回答已知部分，说明资料缺口 |
| `continuation` | 读取预览之外的代号并引用 |
| `prompt_injection` | 接收到伪造指令后仍回答正式代号 |

当前为 5 条云端用例，每条最多 5 次模型请求。
调用预算耗尽的确定性检查由离线测试完成。

报告写入 `eval_reports/`，每次生成独立文件，记录：
回答、异常、消息轨迹、耗时及逐响应 Token 用量。

- `execution_status=completed` 只表示执行完成。
- `review_status=pending` 不代表质量通过。
- 按 `answer_checks` 人工核对事实、引用和工具读取证据。
- Token 用量缺失时记录为 `null`，不推断为零。
- 批处理中断时可能只留下部分结果，应检查结果数量。

五条样本曾在本地人工核对通过，完整报告为：
`eval_reports/report_20261026T064443_459338Z.json`。
该结论仅针对这些样本，原始报告的评审字段仍为 `pending`。

## 最终演示顺序

1. 运行 8 项离线测试，确认全部通过。
2. 启动 HTTP 服务，检查 `/health` 返回 200。
3. 使用 `demo_final` 提问 epoll，检查回答、引用及 `saved=True`。
4. 停止并重启服务，使用同一 ID 提问 RAII。
5. 检查存档保留前后两轮问答，工具调用与结果完整配对。
6. 展示已有完整评估报告中的续读证据和干扰指令处理结果。

重启恢复已在 `api_demo` 会话中实际验收：
存档同时保留了 epoll 与 RAII 两轮完整问答。

## 当前机制与边界

- 每轮最多 5 次模型请求，一次响应可以提出多个工具调用。
- 请求上下文保留当前问题及前两轮，完整历史仍用于存档；
  这不是严格的 Token 总量上限。
- 关键词按空格拆分、忽略大小写并去重，采用子串匹配。
  每命中一个不同查询词得一分，返回 Top-3。
- 搜索预览和单次读取最多返回 300 字符，预览可能省略答案。
- 尚未实现中文分词、BM25、向量检索及超长段落进一步切分。
- 读取结果不足以支持答案时，应说明资料不足；
  搜索无结果不能普遍证明整个资料库没有相关知识。
- 会话锁只保护同一进程内通过会话层执行的请求。
  多 worker、多个服务实例或 CLI/API 同写同一 ID 不受该锁统一保护。
- 当前用于本机单进程学习演示，尚未实现认证、全局并发限额，
  会话锁表也尚未回收。
- 提示词注入样本通过，不代表任意攻击都能被阻止。