# 本地知识库 Agent

参考 Hello-Agents，使用 Python 手写工具调用循环。

知识文件、工具执行和会话存档位于本机。
回答生成由 DeepSeek 云端 API 完成；
Word 检索使用本地 CPU 运行嵌入模型，精排模型可选，默认关闭。
检索到的资料片段会作为工具结果发送给云端模型。

## 功能与结构

### 两种检索模式

- 默认 Markdown 模式：检索 `knowledge/` 中的资料，
  按空行分块，使用关键词子串匹配，返回前三个结果。
- 外部 Word 模式：通过 CLI 的 `--docx` 或 API 启动环境变量
  `RAG_DOCX_PATH` 启用，使用 BM25 与向量检索，再通过 RRF 融合排名。
- Word 默认每路最多召回 50 个候选，采用等权 RRF，
  排名常数为 60，最终向模型返回前三个完整文本块。
- Word 正文切块上限为 400 字符，相邻块重叠 50 字符。
  字符数不等于模型 Token 数，编码前另行检查实际输入长度。
- 精排默认关闭，目前没有用于启用精排的 `--rerank` 命令行参数。

### 主要模块

- `first_call.py`：命令行入口、检索模式选择、会话选择及失败回滚。
- `api.py`：FastAPI `/health` 和 `/chat` 接口，以及检索服务生命周期。
- `agent.py`：系统规则、请求上下文裁剪、模型与工具调用循环。
- `tools.py`、`tool_schemas.py`：四个基础工具及工具接口说明。
- `knowledge_base.py`：Markdown 加载、分块和关键词检索。
- `docx_loader.py`、`docx_chunker.py`：Word 解析、标题路径与重叠切块。
- `embedding_service.py`：本地嵌入模型加载、输入长度检查及向量编码。
- `bm25_retriever.py`、`vector_retriever.py`：关键词召回与向量召回。
- `hybrid_retriever.py`、`reranker_service.py`：排名融合与可选精排。
- `retrieval_service.py`：缓存校验、检索资源复用及统一查询入口。
- `external_knowledge_tool.py`：将 Word 检索结果转换为模型工具消息。
- `build_knowledge_index.py`：构建并校验向量缓存，拒绝覆盖已有目标。
- `session_service.py`、`session_store.py`：会话互斥、恢复及持久化。
- `eval_runner.py`、`docx_eval_runner.py`：Markdown 与 Word 生成评估。

四个基础工具为 `search_notes`、`read_note`、
`search_knowledge`、`read_knowledge`。
只有调用方注入外部检索能力时，才向模型注册第五个工具
`search_external_knowledge`。

只有用户明确要求根目录旧笔记时，才使用 `notes.md` 相关工具。
旧笔记、外部 Word、模型缓存和向量缓存不随仓库发布。

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

### Word 模式的额外依赖

仅使用 Markdown 模式时，安装 `requirements.txt` 即可。
使用 Word 混合检索时，还需安装 `requirements-rag.txt`。

`requirements-rag.txt` 已通过 `-r requirements.txt` 包含基础依赖，
并固定当前使用的 PyTorch、Sentence Transformers 等直接依赖版本。

以下命令用于已创建 `.venv` 的 Windows x64、Python 3.12 新环境，
在项目根目录执行，需要联网下载安装包：

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu  # 从官方 CPU 索引安装项目固定的 PyTorch 版本。
.\.venv\Scripts\python.exe -m pip install -r .\requirements-rag.txt  # 从默认索引安装其余依赖，并检查已有 PyTorch 是否满足版本要求。
.\.venv\Scripts\python.exe -m pip check  # 检查已安装包声明的依赖关系是否存在冲突。
```

CPU 索引仅用于第一条 PyTorch 安装命令；
不要将整份 `requirements-rag.txt` 都限定到该索引。

当前已有环境已通过依赖一致性检查和离线安装计划检查。
这两份依赖文件尚未完整锁定传递依赖，
也尚未完成全新环境的安装验收。

安装 Python 依赖后，仍需另行准备固定版本的模型权重和资料向量缓存。
运行期检索服务只读取本地模型缓存，不自动下载模型。
安装精排相关依赖不会自动开启精排。

### 首次准备 Word 检索资源

完成 Word 模式依赖安装后，还需要准备嵌入模型和资料向量缓存。
以下操作用于新环境，原始 Word 文件由使用者自行提供。

先准备嵌入模型：

```powershell
.\.venv\Scripts\python.exe -B .\check_embeddings.py  # 首次可联网下载固定版本模型，并执行三条文本的编码检查。
```

该脚本使用 `BAAI/bge-small-zh-v1.5`，
固定 revision 为 `7999e1d3359715c523056ef9478215996d62a620`，
模型文件缓存到 `.cache/models/`。

预期输出包含：

```text
MAX_SEQ_LENGTH= 512
EMBEDDING_SHAPE= (3, 512)
EMBEDDING_MODEL_OK
```

随后构建资料向量缓存。执行前，将下面的占位文字替换为实际
Word 文件的绝对路径；后续启动时应使用同一个资料路径：

```powershell
.\.venv\Scripts\python.exe -B .\build_knowledge_index.py --docx "<你的 Word 文件绝对路径>"  # 从本地模型缓存加载模型，解析资料、编码并校验后发布新缓存。
```

默认输出为 `.cache/knowledge_vectors.json`。
成功时输出块数、向量数、维度、输出位置以及 `KNOWLEDGE_INDEX_BUILT`。
向量数量应与块数相同，当前模型的向量维度为 512。

构建脚本拒绝覆盖已有目标。
如果默认文件已经存在，应先确认是否可以直接使用；
需要另外构建验证副本时，通过 `--output` 指定尚不存在的新路径。

CLI 和 API 当前读取 `.cache/knowledge_vectors.json`，
不会自动使用另一个名称的验证副本。

模型准备脚本允许联网获取权重；
索引构建与运行期检索只加载本地模型文件。
缺少模型缓存时，应先完成模型准备。

修改 Word 内容、移动原文件或改变切块及嵌入配置后，
应重新构建匹配的缓存，并重启或重新创建检索服务。
资料绝对路径参与块标识生成，移动资料后也需重新核对评估引用标签。

原始 Word、模型权重和资料向量缓存均不随仓库发布。
默认关闭精排，因此上述准备流程不要求下载精排模型。

## 命令行运行

```powershell
.\.venv\Scripts\python.exe -B first_call.py --session alpha
```

- 指定 ID：使用 `sessions/session_alpha.json`。
- 不指定 ID：使用原来的 `session.json`。
- 输入 `exit` 或 `quit` 退出。
- ID 仅允许小写字母、数字、下划线、短横线，长度为 1～64。

### 外部 Word 命令行模式

前提：已完成嵌入模型准备，并生成与原始 Word 匹配的
`.cache/knowledge_vectors.json`。

将占位文字替换为构建缓存时使用的 Word 绝对路径：

```powershell
.\.venv\Scripts\python.exe -B .\first_call.py --session rag_cli_readme_01 --docx "<你的 Word 文件绝对路径>"  # 使用独立会话启动 Word 混合检索，复用已有模型与资料缓存。
```

演示时使用尚未创建过的会话 ID。
如果 `rag_cli_readme_01` 已有存档，请换一个新 ID，避免旧历史影响观察。

检索初始化成功后，终端会显示：

```text
外部 Word 混合检索已启用，精排关闭。
```

如果资料包含 QLoRA 和 NF4，可输入：

```text
仅依据外部 Word 资料，用两句话解释 QLoRA 中的 NF4，并引用来源。
```

提交问题会调用云端模型。
检查回答是否得到原文支持，引用格式应为 `[docx:<完整 chunk_id>]`。
引用标记存在本身不能证明回答事实正确。

示例会话保存到 `sessions/session_rag_cli_readme_01.json`。
输入 `exit` 或 `quit` 退出。

`--docx` 用于选择 Word 检索模式，不会自动构建或替换缓存。
资料路径或文件内容与缓存不匹配时，启动会报错，应核对资料与缓存。
省略 `--docx` 时，继续使用默认 Markdown 模式。

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

### 外部 Word HTTP 模式

HTTP 服务在启动时决定资料模式：
`RAG_DOCX_PATH` 未设置或为空时使用 Markdown；
设置为 Word 路径时启用外部 Word 混合检索。

前提：已准备本地嵌入模型，以及与原始 Word 匹配的
`.cache/knowledge_vectors.json`。

在启动服务的 PowerShell 窗口执行，
先将占位文字替换为构建缓存时使用的 Word 绝对路径。
如果已有服务占用 8000 端口，先停止原服务：

```powershell
$env:RAG_DOCX_PATH = '<你的 Word 文件绝对路径>'  # 设置当前 PowerShell 及其后续子进程可读取的资料路径。
.\.venv\Scripts\python.exe -B -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1  # 启动单 worker 服务，在应用启动阶段初始化 Word 检索。
```

检索初始化成功后，终端会显示：

```text
HTTP 外部 Word 混合检索已启用，精排关闭。
```

`RAG_DOCX_PATH` 从启动进程的操作系统环境变量读取，
仅将它写入 `.env` 不会启用 Word 模式。
修改资料配置后，需要停止并重新启动服务。

应用启动时创建一个共享检索服务，后续请求复用索引和本地模型；
每个问答请求仍创建并关闭自己的云端客户端。
本地检索查询通过实例内的锁串行执行。

缺少模型、缓存无效或资料与缓存不一致时，应用启动失败，
不会自动退回 Markdown 模式。

`/health`、`/chat` 的调用方式及响应字段沿用上面的说明。
Word 演示应使用新的会话 ID，例如 `rag_api_readme_01`，
并将问题换为该 Word 中包含的内容。
回答应使用 `[docx:<完整 chunk_id>]` 格式引用资料。

保持 `--workers 1`。
会话锁只保护同一进程内通过会话层执行的请求，
不能统一保护多进程、多个服务实例或 CLI/API 同写同一会话。

切回 Markdown 模式时，先停止服务，
再在启动窗口清空变量并重新运行上述服务启动命令：

```powershell
$env:RAG_DOCX_PATH = ''  # 清空后续启动进程使用的 Word 配置；已经运行的服务需要重启才会切换模式。
```

## 测试与评估

固定离线测试，不调用云端模型：

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_session_store test_agent
```

当前共 8 项，覆盖存储读写、损坏文件、工具结果回传、
上下文裁剪、模型调用预算、参数纠错及异常传播等。

API 会话隔离、锁释放、409 映射和保存失败行为另经补充离线检查；
它们目前尚未固化到上述 8 项测试中，也尚无完整的 HTTP 自动回归测试。

### Markdown 生成评估

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

Markdown 评估当前为 5 条云端用例，每条最多 5 次模型请求。
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

### Word 生成回归评估

`docx_eval_cases.json` 保存针对固定 Word 资料版本建立的三条回归用例，
由 `docx_eval_runner.py` 执行。

| 用例 | 检查目标 |
| --- | --- |
| `docx_nf4` | 用两句话解释 NF4，事实与引用均得到原文支持 |
| `docx_rrf_grounding` | 根据资料解释 RRF 直觉，不补写原文没有的公式或参数 |
| `docx_rrf_scope_missing` | 只判断指定段落的资料缺口，不扩大为全文断言 |

运行前，原始资料、向量缓存及评估标签必须匹配。
该评估集不能直接套用到任意 Word；
修改资料或移动原文件后，需要重新核对资料版本及引用标签。

```powershell
.\.venv\Scripts\python.exe -B .\docx_eval_runner.py --docx "<评估集对应的原始 Word 绝对路径>"  # 校验资料与标签后，执行三条云端生成回归并保存报告。
```

该命令会调用云端模型。
每条用例使用独立历史，最多进行 5 次模型请求，
每次模型请求的最大输出为 1024 Tokens。

整批复用一个本地检索服务和一个云端客户端。
检索条件为每路最多召回 50 个候选、等权 RRF、排名常数 60、
最终返回前三个文本块，精排关闭。

报告写入 `eval_reports/docx_generation_<时间戳>.json`，
包含资料版本、模型与检索配置、回答、消息轨迹、
逐响应 Token 用量及耗时；检索服务构造耗时单独记录。

`DOCX_GENERATION_REPORT_SAVED` 只表示报告已保存。
`execution_status=completed` 只表示执行完成，
不等于事实、引用或回答范围已经通过质量审核。

评审时，应按 `required_subset` 规则检查回答包含预期主要引用，
并核对所有实际引用及事实是否得到本题工具原文支持。
`expected_sources` 和 `answer_checks` 是评审依据，
不会作为证据提供给回答模型，脚本也不自动裁定回答的语义正确性。

已有真实生成报告：

- `eval_reports/docx_generation_20261101T055838_823203Z.json`
- `eval_reports/docx_generation_20261101T055838_823203Z_review.json`

该报告使用 `deepseek-flash`，三条用例均执行完成；
独立审核文件记录三条通过、零条失败，并通过 SHA256 绑定原报告。
原报告中的 `review_status=pending` 保留，审核结论记录在独立文件中。

这三条结果属于已知问题的回归证据，
不能据此宣称系统整体准确率为 100%，也不代表独立测试集表现。

### 检索权重与精排实验

检索实验使用 Hit@3（前三命中率）：
每个问题的前三个结果中，至少有一个文本块同时满足
指定章节与全部证据片段要求，才记为该问题命中。

该指标衡量指定目标证据是否进入前三，
不等于回答准确率；指定目标未命中，也不代表其他候选完全没有用。

已有固定样本结果如下：

| 检索方式 | 开发集：4 题 | 原留出回归集：4 题 |
| --- | --- | --- |
| BM25 | 3/4 | 3/4 |
| 向量检索 | 3/4 | 2/4 |
| 等权 RRF | 3/4 | 2/4 |
| 加权 RRF：BM25 0.3、向量 0.7 | 4/4 | 2/4 |
| 对两路完整候选并集进行精排 | 3/4 | 3/4 |

加权 RRF 的权重是在开发集上比较选出的。
开发集的 4/4 不能作为泛化能力证明，
该权重也没有自动成为 CLI/API 的默认配置。

原留出题后来已参与错误分析及精排方案比较，
现在按 `previous_holdout_regression` 使用，
不再作为未接触过的独立测试集；历史文件名和原始报告保留。

精排实验使用每路最多 50 个候选的完整去重并集，
没有先截取前三个结果再精排。
该次 CPU 实验中，8 题的单题精排耗时平均约为 5.664 秒，
仅包含精排处理，不包含原始召回、模型加载或回答生成。

加权 RRF 与精排在这 8 题上均合计命中 6/8，
但具体命中的问题不同，不能据此宣称精排普遍提升质量。
当前 CLI/API 仍默认使用等权 RRF，精排关闭。

实验入口为 `retrieval_eval.py`、`rrf_weight_eval.py`、
`retrieval_holdout_eval.py` 和 `rerank_eval.py`。
对应历史报告：

- `eval_reports/retrieval_eval_20261031T021200_202968Z.json`
- `eval_reports/rrf_weight_eval_20261031T025248_685709Z.json`
- `eval_reports/retrieval_holdout_eval_20261031T045133_624126Z.json`
- `eval_reports/rerank_eval_20261031T093712_574725Z.json`

以上结果仅适用于这些资料、标签、候选范围及实验配置。
检索目标命中统计不能替代对候选相关性和生成回答的人工审核。

## 最终演示顺序

演示使用尚未创建过的会话 ID。
Markdown 与 Word 分别使用不同 ID；
验证重启恢复时，则必须继续使用重启前的同一个 ID。

提交问答或重新运行生成评估会调用云端模型。
展示已有报告无需重新执行生成评估。

### Markdown 基础行为与重启恢复

1. 运行前文的 8 项离线测试，确认全部通过。
2. 停止已有服务，在启动窗口清空 Word 配置，再启动服务：

```powershell
$env:RAG_DOCX_PATH = ''  # 清空 Word 配置，使后续启动的服务使用 Markdown 模式。
.\.venv\Scripts\python.exe -B -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1  # 启动单 worker 服务。
```

3. 检查 `/health` 返回 HTTP 200。
4. 使用前文 `/chat` 请求示例，将会话 ID 改为新的
   `demo_md_01`，提问 epoll，检查事实、Markdown 来源引用及 `saved=True`。
5. 停止并重新启动服务，保持 Word 配置为空；
   使用同一个 `demo_md_01` 提问 RAII。
6. 检查 `sessions/session_demo_md_01.json`，
   确认两轮用户问题、工具调用、工具结果和最终回答均保留，
   工具结果的 `tool_call_id` 与对应调用 ID 匹配。
7. 展示已有 Markdown 评估报告中的续读及干扰指令处理证据。

若示例 ID 已有存档，开始演示前换一个新 ID；
重启后的第二次提问继续使用这个新 ID。

### Word 混合检索与回答依据

1. 停止 Markdown 服务，按“外部 Word HTTP 模式”设置资料路径，
   使用匹配的模型和向量缓存启动服务。
2. 确认终端显示 Word 混合检索已启用、精排关闭，
   再检查 `/health` 返回 HTTP 200。
3. 使用新的会话 ID `demo_docx_01`，
   对已验收的面试资料提问：
   “仅依据外部 Word 资料，用两句话解释 QLoRA 中的 NF4，并引用来源。”
4. 检查回答事实是否得到对应原文支持，
   引用是否采用 `[docx:<完整 chunk_id>]` 格式，并检查 `saved=True`。
5. 检查 `sessions/session_demo_docx_01.json`，
   核对最终引用是否确实来自本轮检索工具返回的结果。
6. 展示已有 Word 三条生成回归报告及独立审核文件，
   说明它们属于固定样本的回归证据。

`/health` 成功不能代替问答验证；
HTTP 200 且 `saved=False` 表示回答成功、保存失败，
不能认定会话持久化验收通过。

### 已有验收范围

Markdown 重启恢复已在历史 `api_demo` 会话中实际验收。
Word 模式已有 CLI/API 真实问答及三条生成回归证据；
Word 模式的专门重启恢复尚未单独验收。

## 当前机制与边界

- 每轮最多 5 次模型请求，一次响应可以提出多个工具调用。
- 请求上下文保留当前问题及前两轮，完整历史仍用于存档；
  这不是严格的 Token 总量上限。
- Markdown 基线按空格拆分查询词、忽略大小写并去重，
  每命中一个不同查询词得一分，返回前三个结果。
  Markdown 搜索预览和单次读取最多返回 300 字符，可按偏移续读。
- Word 工具返回前三个完整正文块，每块最多 400 字符，
  不使用 Markdown 的 300 字符预览与续读协议。
- Word BM25 使用英文术语、汉字单字及相邻双字串进行启发式切分，
  尚未接入专门的中文词法分词器。
- Word 引用包含资料指纹、标题路径、章节正文段落范围和字符偏移。
  段落范围属于整节正文，字符偏移定位节内文本块，不表示 Word 页码。
- Word 图片尚未进行 OCR，表格仅提取为段落文本，不保证行列语义。
- 检索排名和分数不代表事实正确或证据充分；
  无关查询也可能返回向量候选，模型仍需根据原文判断能否回答。
- 读取结果不足以支持答案时，应说明资料不足；
  搜索无结果不能普遍证明整个资料库没有相关知识。
- 会话锁只保护同一进程内通过会话层执行的请求。
  多 worker、多个服务实例或 CLI/API 同写同一 ID 不受该锁统一保护。
- 当前用于本机单进程学习演示，尚未实现认证、全局并发限额，
  会话锁表也尚未回收。
- 提示词注入样本通过，不代表任意攻击都能被阻止。