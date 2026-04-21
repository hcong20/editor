from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from movie_brief.config import GeminiSettings, OllamaSettings, OpenAISettings
from movie_brief.media import guess_mime_type, image_to_base64, image_to_data_url


class LLMRequestError(RuntimeError):
    pass


def _coerce_non_object_json(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, list):
        sample = parsed[0] if parsed else None
        if isinstance(sample, dict):
            keys = set(sample.keys())
            # Script generators occasionally return a top-level segment array.
            if {"narration", "scene_ids"}.issubset(keys) or "section" in keys:
                return {"segments": parsed}
            # Story modeling can occasionally return a top-level beats array.
            if "beat_id" in keys or {"summary", "scene_ids", "importance"}.issubset(keys):
                return {"beats": parsed}
    raise LLMRequestError("Expected a JSON object in model output.")


def parse_json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise LLMRequestError("Model returned an empty response.")

    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        if isinstance(parsed, dict):
            return parsed
        return _coerce_non_object_json(parsed)

    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start == -1 or end == -1 or end < start:
            continue
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        return _coerce_non_object_json(parsed)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMRequestError(f"Failed to parse JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        return _coerce_non_object_json(parsed)
    return parsed


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMRequestError(f"HTTP {exc.code} calling {url}: {detail}") from exc
    except error.URLError as exc:
        raise LLMRequestError(f"Network error calling {url}: {exc.reason}") from exc


def _extract_openai_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"]

    if payload.get("error"):
        raise LLMRequestError(str(payload["error"]))

    texts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
                continue
            if isinstance(text, dict) and isinstance(text.get("value"), str):
                texts.append(text["value"])
    if texts:
        return "\n".join(texts)
    raise LLMRequestError("OpenAI response did not contain any text output.")


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    prompt_feedback = payload.get("promptFeedback") or {}
    if prompt_feedback.get("blockReason"):
        raise LLMRequestError(f"Gemini prompt was blocked: {prompt_feedback}")

    texts: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
    if texts:
        return "\n".join(texts)
    raise LLMRequestError("Gemini response did not contain any text output.")


def _extract_ollama_text(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        raise LLMRequestError(str(payload["error"]))

    message = payload.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content

    # Some Ollama endpoints return "response" instead of a chat message.
    response_text = payload.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return response_text

    raise LLMRequestError("Ollama response did not contain any text output.")


def _normalize_ollama_host(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    for suffix in ("/api/chat", "/api/generate"):
        if cleaned.endswith(suffix):
            return cleaned[: -len(suffix)]
    return cleaned


def _coerce_ollama_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return {"message": {"content": content}}

    response_text = getattr(response, "response", None)
    if isinstance(response_text, str):
        return {"response": response_text}

    raise LLMRequestError("Ollama SDK response did not contain any text output.")


class OpenAIResponsesJSONClient:
    def __init__(self, settings: OpenAISettings) -> None:
        self.settings = settings

    def generate_json(
        self,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        api_key = os.getenv(self.settings.api_key_env)
        if not api_key:
            raise LLMRequestError(
                f"Missing API key in environment variable {self.settings.api_key_env}."
            )

        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        for image_path in image_paths or []:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(image_path),
                    "detail": self.settings.image_detail,
                }
            )

        payload = {
            "model": self.settings.scene_model,
            "instructions": system_prompt,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }

        response = _post_json(
            self.settings.base_url,
            payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout_seconds=self.settings.timeout_seconds,
        )
        return parse_json_from_text(_extract_openai_text(response))

    def generate_json_with_model(
        self,
        model: str,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        original_model = self.settings.scene_model
        self.settings.scene_model = model
        try:
            return self.generate_json(
                schema_name=schema_name,
                schema=schema,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        finally:
            self.settings.scene_model = original_model


class GeminiJSONClient:
    def __init__(self, settings: GeminiSettings) -> None:
        self.settings = settings

    def generate_json(
        self,
        model: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        api_key = os.getenv(self.settings.api_key_env)
        if not api_key:
            raise LLMRequestError(
                f"Missing API key in environment variable {self.settings.api_key_env}."
            )

        model_path = model if model.startswith("models/") else f"models/{model}"
        url = f"{self.settings.base_url}/{model_path}:generateContent"
        parts: list[dict[str, Any]] = [{"text": user_prompt}]
        for image_path in image_paths or []:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": guess_mime_type(image_path),
                        "data": image_to_base64(image_path),
                    }
                }
            )

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
            },
        }
        response = _post_json(
            url,
            payload,
            headers={"x-goog-api-key": api_key},
            timeout_seconds=self.settings.timeout_seconds,
        )
        return parse_json_from_text(_extract_gemini_text(response))


class OllamaJSONClient:
    def __init__(self, settings: OllamaSettings) -> None:
        self.settings = settings

    def generate_json(
        self,
        model: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        sdk_error: LLMRequestError | None = None
        try:
            response = self._chat_via_sdk(
                model=model,
                schema=schema,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except LLMRequestError as exc:
            response = None
            sdk_error = exc
            print(f"Ollama SDK request failed: {exc}. Falling back to HTTP API.")

        if response is None:
            try:
                response = self._chat_via_http(
                    model=model,
                    schema=schema,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=image_paths,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            except LLMRequestError as exc:
                if sdk_error is None:
                    raise
                raise LLMRequestError(
                    f"Ollama SDK failed ({sdk_error}); HTTP fallback also failed ({exc})."
                ) from exc

        return parse_json_from_text(_extract_ollama_text(response))

    def _chat_via_sdk(
        self,
        model: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any] | None:
        try:
            ollama_module = importlib.import_module("ollama")
        except ImportError:
            return None

        host = _normalize_ollama_host(self.settings.base_url)

        request_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
            "stream": False,
            "keep_alive": self.settings.keep_alive,
        }
        if image_paths:
            request_payload["messages"][1]["images"] = [
                image_to_base64(path) for path in image_paths
            ]
        if not self.settings.keep_alive:
            request_payload.pop("keep_alive")

        chat = getattr(ollama_module, "chat", None)
        if callable(chat):
            try:
                response = chat(host=host, **request_payload)
            except TypeError:
                # Some SDK versions do not accept host in the module-level function.
                response = chat(**request_payload)
            except Exception as exc:
                raise LLMRequestError(f"Ollama SDK error calling {host}: {exc}") from exc
            return _coerce_ollama_payload(response)

        Client = getattr(ollama_module, "Client", None)
        if Client is None:
            raise LLMRequestError("Installed ollama package does not expose chat or Client.")

        try:
            client = Client(host=host, timeout=self.settings.timeout_seconds)
        except TypeError:
            # Older ollama-python versions do not expose a timeout constructor arg.
            client = Client(host=host)

        try:
            response = client.chat(**request_payload)
        except Exception as exc:
            raise LLMRequestError(f"Ollama SDK error calling {host}: {exc}") from exc

        return _coerce_ollama_payload(response)

    def _chat_via_http(
        self,
        model: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        user_message: dict[str, Any] = {
            "role": "user",
            "content": user_prompt,
        }
        if image_paths:
            user_message["images"] = [image_to_base64(path) for path in image_paths]

        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
            "keep_alive": self.settings.keep_alive,
        }

        if not self.settings.keep_alive:
            payload.pop("keep_alive")

        response = _post_json(
            self.settings.base_url,
            payload,
            headers={},
            timeout_seconds=self.settings.timeout_seconds,
        )
        return response
