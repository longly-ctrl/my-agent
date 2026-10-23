import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openai.types.chat import ChatCompletionMessage
from session_store import load_session, save_session


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        # 每个测试使用独立临时目录。
        self.directory = TemporaryDirectory(
            prefix="session-test-",
            dir=Path(__file__).resolve().parent,
        )
        # 即使测试失败，结束后也会清理临时目录。
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "session.json"

    def test_missing_file_returns_empty(self):
        # 首次运行没有记录，应返回空历史，也不创建文件。
        self.assertEqual(load_session(self.path), [])
        self.assertFalse(self.path.exists())

    def test_save_and_load_sdk_message(self):
        # 覆盖普通字典与 SDK 消息对象混合保存的情况。
        messages = [
            {"role": "system", "content": "你是学习助手。"},
            {"role": "user", "content": "epoll 是什么？"},
            ChatCompletionMessage(
                role="assistant",
                content="epoll 是 I/O 事件通知机制。",
            ),
        ]

        save_session(messages, self.path)
        restored = load_session(self.path)

        self.assertEqual(
            [message["role"] for message in restored],
            ["system", "user", "assistant"],
        )
        self.assertEqual(restored[2]["content"], messages[2].content)

    def test_corrupt_file_raises_and_preserves_content(self):
        # 损坏的记录必须报错，并保留原文件供排查。
        original = '{"broken":'
        self.path.write_text(original, encoding="utf-8")

        with self.assertRaises(ValueError):
            load_session(self.path)

        self.assertEqual(
            self.path.read_text(encoding="utf-8"),
            original,
        )


if __name__ == "__main__":
    unittest.main()