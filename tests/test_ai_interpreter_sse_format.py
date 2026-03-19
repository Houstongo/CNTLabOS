import json
import unittest

from backend.core.ai_interpreter import AIInterpreter


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeCompletions:
    def create(self, **kwargs):
        return [_Chunk("你好")]


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


class AIInterpreterSSEFormatTests(unittest.TestCase):
    def setUp(self):
        self.interpreter = AIInterpreter.__new__(AIInterpreter)
        self.interpreter.model = "glm-4-flash"
        self.interpreter.client = _FakeClient()

    def test_interpret_stream_uses_real_sse_newline_separator(self):
        messages = list(
            self.interpreter.interpret_stream(
                features={},
                params={},
                similar_exps=[],
                pdf_passages=[],
                knowledge_links=[],
                temperature=0.5,
            )
        )

        self.assertTrue(messages[0].endswith("\n\n"))
        self.assertNotIn("\\n\\n", messages[0])
        payload = json.loads(messages[0][len("data: ") : -2])
        self.assertEqual(payload["type"], "content")
        self.assertEqual(payload["text"], "你好")

    def test_chat_stream_uses_real_sse_newline_separator(self):
        messages = list(
            self.interpreter.chat_stream(
                history=[],
                user_message="测试",
                context=None,
                temperature=0.5,
            )
        )

        self.assertTrue(messages[0].endswith("\n\n"))
        self.assertNotIn("\\n\\n", messages[0])
        payload = json.loads(messages[0][len("data: ") : -2])
        self.assertEqual(payload["type"], "content")
        self.assertEqual(payload["text"], "你好")


if __name__ == "__main__":
    unittest.main()
