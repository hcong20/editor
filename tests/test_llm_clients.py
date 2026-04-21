from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_brief.config import GeminiSettings, OllamaSettings, OpenAISettings
from movie_brief.llm_clients import (
    GeminiJSONClient,
    OllamaJSONClient,
    OpenAIResponsesJSONClient,
    parse_json_from_text,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class LLMClientTests(unittest.TestCase):
    def test_parse_json_from_text_wraps_top_level_segments_array(self) -> None:
        payload = """
        [
            {"section":"背景介绍","narration":"开场","scene_ids":["scene_001"]},
            {"section":"冲突爆发","narration":"升级","scene_ids":["scene_002"]}
        ]
        """
        parsed = parse_json_from_text(payload)

        self.assertIn("segments", parsed)
        self.assertEqual(len(parsed["segments"]), 2)
        self.assertEqual(parsed["segments"][0]["section"], "背景介绍")

    def test_parse_json_from_text_wraps_top_level_beats_array(self) -> None:
        payload = """
        [
            {"beat_id":"beat_setup","summary":"开场","scene_ids":["scene_001"],"importance":1.2}
        ]
        """
        parsed = parse_json_from_text(payload)

        self.assertIn("beats", parsed)
        self.assertEqual(len(parsed["beats"]), 1)
        self.assertEqual(parsed["beats"][0]["beat_id"], "beat_setup")

    def test_openai_client_builds_multimodal_structured_request(self) -> None:
        settings = OpenAISettings()
        client = OpenAIResponsesJSONClient(settings)
        captured: dict = {}

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeHTTPResponse({"output_text": '{"summary":"测试","events":[],"characters":[],"visual_cues":[],"emotion_intensity":0.4,"core_character_score":0.5,"conflict_score":0.6,"plot_progression_score":0.7}'})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with patch("movie_brief.llm_clients.request.urlopen", side_effect=fake_urlopen):
                    result = client.generate_json_with_model(
                        model="gpt-4o",
                        schema_name="scene_analysis",
                        schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
                        system_prompt="system prompt",
                        user_prompt="user prompt",
                        image_paths=[image_path],
                        max_output_tokens=123,
                        temperature=0.4,
                    )

        self.assertEqual(result["summary"], "测试")
        self.assertEqual(captured["url"], settings.base_url)
        self.assertEqual(captured["body"]["model"], "gpt-4o")
        self.assertEqual(captured["body"]["text"]["format"]["type"], "json_schema")
        self.assertEqual(captured["body"]["max_output_tokens"], 123)
        self.assertEqual(captured["body"]["temperature"], 0.4)
        self.assertEqual(captured["body"]["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(captured["body"]["input"][0]["content"][1]["type"], "input_image")
        self.assertIn("data:image/jpeg;base64,", captured["body"]["input"][0]["content"][1]["image_url"])

    def test_gemini_client_builds_inline_data_json_request(self) -> None:
        settings = GeminiSettings()
        client = GeminiJSONClient(settings)
        captured: dict = {}

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeHTTPResponse({"candidates": [{"content": {"parts": [{"text": '{"segments":[]}' }]}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}):
                with patch("movie_brief.llm_clients.request.urlopen", side_effect=fake_urlopen):
                    result = client.generate_json(
                        model="gemini-2.5-flash",
                        schema={"type": "object", "properties": {"segments": {"type": "array"}}, "required": ["segments"]},
                        system_prompt="system prompt",
                        user_prompt="user prompt",
                        image_paths=[image_path],
                        max_output_tokens=456,
                        temperature=0.7,
                    )

        self.assertEqual(result["segments"], [])
        self.assertEqual(
            captured["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        )
        self.assertEqual(captured["body"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(captured["body"]["generationConfig"]["maxOutputTokens"], 456)
        self.assertEqual(captured["body"]["generationConfig"]["temperature"], 0.7)
        self.assertEqual(captured["body"]["contents"][0]["parts"][0]["text"], "user prompt")
        self.assertIn("inline_data", captured["body"]["contents"][0]["parts"][1])
        self.assertEqual(captured["body"]["systemInstruction"]["parts"][0]["text"], "system prompt")

    def test_ollama_client_builds_local_json_chat_request(self) -> None:
        settings = OllamaSettings()
        client = OllamaJSONClient(settings)
        captured: dict = {}

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeHTTPResponse({"message": {"content": '{"segments":[]}'}})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            with patch.object(client, "_chat_via_sdk", return_value=None):
                with patch("movie_brief.llm_clients.request.urlopen", side_effect=fake_urlopen):
                    result = client.generate_json(
                        model="qwen2.5:7b",
                        schema={"type": "object", "properties": {"segments": {"type": "array"}}, "required": ["segments"]},
                        system_prompt="system prompt",
                        user_prompt="user prompt",
                        image_paths=[image_path],
                        max_output_tokens=321,
                        temperature=0.5,
                    )

        self.assertEqual(result["segments"], [])
        self.assertEqual(captured["url"], settings.base_url)
        self.assertEqual(captured["body"]["model"], "qwen2.5:7b")
        self.assertEqual(captured["body"]["stream"], False)
        self.assertEqual(captured["body"]["format"]["type"], "object")
        self.assertEqual(captured["body"]["options"]["num_predict"], 321)
        self.assertEqual(captured["body"]["options"]["temperature"], 0.5)
        self.assertEqual(captured["body"]["messages"][0]["role"], "system")
        self.assertEqual(captured["body"]["messages"][1]["role"], "user")
        self.assertTrue(captured["body"]["messages"][1]["images"][0])

    def test_ollama_client_prefers_sdk_when_available(self) -> None:
        settings = OllamaSettings()
        client = OllamaJSONClient(settings)

        with patch.object(
            client,
            "_chat_via_sdk",
            return_value={"message": {"content": '{"segments":[]}' }},
        ) as sdk_mock:
            with patch(
                "movie_brief.llm_clients.request.urlopen",
                side_effect=AssertionError("HTTP fallback should not be called when SDK succeeds."),
            ):
                result = client.generate_json(
                    model="qwen2.5:7b",
                    schema={"type": "object", "properties": {"segments": {"type": "array"}}, "required": ["segments"]},
                    system_prompt="system prompt",
                    user_prompt="user prompt",
                    image_paths=None,
                    max_output_tokens=321,
                    temperature=0.5,
                )

        self.assertEqual(result["segments"], [])
        sdk_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
