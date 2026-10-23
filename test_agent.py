import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletionMessage
from agent import run_agent
import json
from tools import ToolInputError


def make_response(message, finish_reason):
    # 构造 run_agent 实际读取的响应属性。
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message, finish_reason=finish_reason)
        ]
    )


class AgentTests(unittest.TestCase):
    def test_tool_result_is_sent_back(self):
        messages = [
            {"role": "system", "content": "查询笔记后回答。"}
        ]

        # 加入三轮已完成的问答，使当前问题成为第四轮。
        for number in range(1, 4):
            messages.extend([
                {"role": "user", "content": f"旧问题 {number}"},
                {"role": "assistant", "content": f"旧回答 {number}"},
            ])

        messages.append({
            "role": "user",
            "content": "epoll 是什么？",
        })

        # 保存调用前的完整历史，用于检查原始记录没有被删除。
        history_before = deepcopy(messages)

        # 第一次模拟响应：模型要求查询笔记。
        tool_message = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_notes",
                    "arguments": '{"query": "epoll"}',
                },
            }],
        )

        # 第二次模拟响应：模型返回最终答案。
        final_message = ChatCompletionMessage(
            role="assistant",
            content="测试最终回答",
        )
        replies = iter([
            make_response(tool_message, "tool_calls"),
            make_response(final_message, "stop"),
        ])
        requests = []

        def fake_create(**kwargs):
            # 保存发送瞬间的快照，避免后续 append 改变观察结果。
            requests.append(deepcopy(kwargs["messages"]))
            return next(replies)

        client = Mock()
        client.chat.completions.create.side_effect = fake_create
        result = "[notes.md:L1] epoll 是 I/O 事件通知机制。"

        # 临时替换工具，排除真实笔记内容变化对本测试的影响。
        with patch("agent.search_notes", return_value=result) as search:
            answer = run_agent(client, "offline-model", messages)

        search.assert_called_once_with(query="epoll")
        self.assertEqual(len(requests), 2)
        # 第一轮请求：保留 system，去掉最早的一组问答。
        self.assertEqual(
            requests[0],
            [history_before[0], *history_before[3:]],
        )

        # 第二轮请求：在同一窗口后追加工具调用和工具结果。
        self.assertEqual(requests[1][:-2], requests[0])

        # 发送窗口变短，但原始历史的已有部分必须完整保留。
        self.assertEqual(
            messages[:len(history_before)],
            history_before,
        )
        self.assertEqual(requests[1][-2].tool_calls[0].id, "call_1")
        self.assertEqual(requests[1][-1], {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": result,
        })
        self.assertEqual(answer, "测试最终回答")

    def test_stops_at_model_call_limit(self):
        client = Mock()
        replies = []

        # 两次响应都要求查询工具，始终不给最终答案。
        for index in range(2):
            message = ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[{
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": "search_notes",
                        "arguments": '{"query": "epoll"}',
                    },
                }],
            )
            replies.append(make_response(message, "tool_calls"))

        client.chat.completions.create.side_effect = replies
        messages = [
            {"role": "system", "content": "查询笔记后回答。"},
            {"role": "user", "content": "epoll 是什么？"},
        ]

        with patch("agent.search_notes", return_value="模拟笔记") as search:
            with self.assertRaisesRegex(RuntimeError, "模型调用上限"):
                run_agent(client, "offline-model", messages, max_steps=2)

        self.assertEqual(client.chat.completions.create.call_count, 2)
        search.assert_called_once_with(query="epoll")
        
 #                      未进行修改参数后读取成功的测试                                         0.1   
    # def test_tool_input_error_is_sent_back(self):
        # client = Mock()
        # request = ChatCompletionMessage(
            # role="assistant",
            # content=None,
            # tool_calls=[{
                # "id": "bad_line",
                # "type": "function",
                # "function": {
                    # "name": "read_note",
                    # "arguments": '{"line_number": 999}',
                # },
            # }],
        # )
        # final = ChatCompletionMessage(
            # role="assistant", content="无法读取该行。"
        # )
        # client.chat.completions.create.side_effect = [
            # make_response(request, "tool_calls"),
            # make_response(final, "stop"),
        # ]
        # messages = [
            # {"role": "system", "content": "读取笔记后回答。"},
            # {"role": "user", "content": "读取第999行。"},
        # ]

        # with patch(
            # "agent.read_note",
            # side_effect=ToolInputError("行号超出笔记范围"),
        # ):
            # answer = run_agent(client, "offline-model", messages)

        # result = messages[-2]
        # self.assertEqual(result["role"], "tool")
        # self.assertEqual(result["tool_call_id"], "bad_line")
        # error = json.loads(result["content"])
        # self.assertIs(error["ok"], False)
        # self.assertEqual(error["error_type"], "ToolInputError")
        # self.assertEqual(client.chat.completions.create.call_count, 2)
        # self.assertEqual(answer, "无法读取该行。")
#                                                                                                   0.1
    def test_tool_input_error_is_sent_back(self):
        client = Mock()
        request = ChatCompletionMessage(
            role="assistant", content=None,
            tool_calls=[{
                "id": "bad_line",
                "type": "function",
                "function": {
                    "name": "read_note",
                    "arguments": '{"line_number": 999}',
                },
            }],
        )

        # 深复制，避免修改第二次请求时破坏第一次的错误请求。
        corrected = deepcopy(request)
        corrected.tool_calls[0].id = "good_line"
        corrected.tool_calls[0].function.arguments = '{"line_number": 1}'

        final = ChatCompletionMessage(
            role="assistant", content="读取成功。"
        )
        client.chat.completions.create.side_effect = [
            make_response(request, "tool_calls"),
            make_response(corrected, "tool_calls"),
            make_response(final, "stop"),
        ]
        messages = [
            {"role": "system", "content": "读取笔记后回答。"},
            {"role": "user", "content": "读取笔记，行号错误时修正后重试。"},
        ]
        observation = "[notes.md:L1] epoll 是 I/O 事件通知机制。"

        # 第一次工具执行失败，第二次返回有效记录。
        with patch(
            "agent.read_note",
            side_effect=[ToolInputError("行号超出笔记范围"), observation],
        ) as reader:
            answer = run_agent(client, "offline-model", messages)

        results = [
            item for item in messages
            if isinstance(item, dict) and item.get("role") == "tool"
        ]
        self.assertEqual(len(results), 2)

        # 错误与成功结果分别对应各自的调用 ID。
        self.assertEqual(results[0]["tool_call_id"], "bad_line")
        error = json.loads(results[0]["content"])
        self.assertIs(error["ok"], False)
        self.assertEqual(error["error_type"], "ToolInputError")

        self.assertEqual(results[1]["tool_call_id"], "good_line")
        self.assertEqual(results[1]["content"], observation)
        self.assertEqual(
            [item.kwargs for item in reader.call_args_list],
            [{"line_number": 999}, {"line_number": 1}],
        )
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual(answer, "读取成功。")     
        
    def test_file_error_is_not_hidden(self):
        client = Mock()
        request = ChatCompletionMessage(
            role="assistant", content=None,
            tool_calls=[{
                "id": "read_missing", "type": "function",
                "function": {
                    "name": "read_note",
                    "arguments": '{"line_number": 1}',
                },
            }],
        )
        client.chat.completions.create.return_value = make_response(
            request, "tool_calls"
        )
        messages = [
            {"role": "system", "content": "读取笔记后回答。"},
            {"role": "user", "content": "读取第1行。"},
        ]
        failure = FileNotFoundError("simulated missing notes")

        # 模拟底层读取失败，不删除或修改真实笔记。
        with patch("tools.Path.read_text", side_effect=failure) as reader:
            with self.assertRaises(FileNotFoundError) as raised:
                run_agent(client, "offline-model", messages)

        # 原异常向外传播，并且没有继续请求模型。
        self.assertIs(raised.exception, failure)
        reader.assert_called_once_with(encoding="utf-8-sig")
        self.assertEqual(client.chat.completions.create.call_count, 1)        
if __name__ == "__main__":
    unittest.main()