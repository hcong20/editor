from __future__ import annotations

import os
from typing import Any

from movie_brief.config import PipelineConfig
from movie_brief.llm_clients import (
    GeminiJSONClient,
    LLMRequestError,
    OllamaJSONClient,
    OpenAIResponsesJSONClient,
)
from movie_brief.models import Scene, ScriptSegment, StoryBeat
from movie_brief.prompts import build_bilibili_script_prompt, script_segments_schema
from movie_brief.utils import trim_text


SCRIPT_SYSTEM_PROMPT = """你是一个成熟的中文电影解说编剧，擅长写 B 站风格的口播稿。

你的文案必须满足：
- 能直接拿去配音
- 有悬念、有节奏，但不悬浮
- 重点讲人物动机、冲突升级、转折和高潮
- 不要像按 scene 列表做流水账
- 只输出 JSON，不要 Markdown，不要解释"""


class ScriptGenerator:
    def generate(
        self,
        beats: list[StoryBeat],
        scenes: list[Scene],
        config: PipelineConfig,
    ) -> list[ScriptSegment]:
        raise NotImplementedError


class TemplateScriptGenerator(ScriptGenerator):
    ACT_WEIGHTS = {
        "背景介绍": 0.18,
        "冲突爆发": 0.34,
        "高潮对抗": 0.32,
        "结局收束": 0.16,
    }

    def generate(
        self,
        beats: list[StoryBeat],
        scenes: list[Scene],
        config: PipelineConfig,
    ) -> list[ScriptSegment]:
        scene_map = {scene.scene_id: scene for scene in scenes}
        total_seconds = config.project.target_minutes * 60
        budget = config.script.max_chars
        segments: list[ScriptSegment] = []

        for index, beat in enumerate(beats, start=1):
            target_seconds = int(total_seconds * self.ACT_WEIGHTS.get(beat.title, 0.25))
            related_scenes = [scene_map[scene_id] for scene_id in beat.scene_ids if scene_id in scene_map]
            narration = self._build_narration(beat, related_scenes, config)
            segments.append(
                ScriptSegment(
                    segment_id=f"segment_{index:02d}",
                    title=beat.title,
                    narration=narration,
                    scene_ids=beat.scene_ids,
                    target_seconds=target_seconds,
                )
            )

        total_chars = sum(len(item.narration) for item in segments)
        if total_chars > budget and segments:
            ratio = budget / total_chars
            for item in segments:
                item.narration = trim_text(item.narration, max(80, int(len(item.narration) * ratio)))
        return segments

    def _build_narration(
        self,
        beat: StoryBeat,
        scenes: list[Scene],
        config: PipelineConfig,
    ) -> str:
        top_scenes = sorted(scenes, key=lambda scene: scene.importance_score, reverse=True)[:3]
        detail_lines = []
        for scene in top_scenes:
            detail_lines.append(
                f"{scene.summary} 这一段的重要性分数达到 {scene.importance_score}。"
            )
        detail = " ".join(detail_lines)
        narration = (
            f"{beat.title}这部分要用更{config.script.tone}的方式讲。"
            f"{beat.summary} {detail} 解说时不要陷进细碎对白，"
            "而是抓住人物为什么被逼到这一步，以及局势是怎样一步步失控的。"
        )
        return trim_text(narration, 1000)


class OpenAIScriptGenerator(ScriptGenerator):
    def __init__(self) -> None:
        self.fallback = TemplateScriptGenerator()
        self.client: OpenAIResponsesJSONClient | None = None

    def generate(
        self,
        beats: list[StoryBeat],
        scenes: list[Scene],
        config: PipelineConfig,
    ) -> list[ScriptSegment]:
        fallback_segments = self.fallback.generate(beats, scenes, config)
        self.client = OpenAIResponsesJSONClient(config.openai)
        payload = self.client.generate_json_with_model(
            model=config.openai.script_model,
            schema_name="bilibili_script_segments",
            schema=script_segments_schema(),
            system_prompt=SCRIPT_SYSTEM_PROMPT,
            user_prompt=build_bilibili_script_prompt(
                story_payload=_story_payload(beats),
                scene_payload=_script_scene_payload(beats, scenes),
                max_chars=config.script.max_chars,
                tone=config.script.tone,
            ),
            image_paths=None,
            max_output_tokens=config.openai.script_max_output_tokens,
            temperature=config.openai.script_temperature,
        )
        return _materialize_script_segments(payload, beats, scenes, fallback_segments, config)


class GeminiScriptGenerator(ScriptGenerator):
    def __init__(self) -> None:
        self.fallback = TemplateScriptGenerator()
        self.client: GeminiJSONClient | None = None

    def generate(
        self,
        beats: list[StoryBeat],
        scenes: list[Scene],
        config: PipelineConfig,
    ) -> list[ScriptSegment]:
        fallback_segments = self.fallback.generate(beats, scenes, config)
        self.client = GeminiJSONClient(config.gemini)
        payload = self.client.generate_json(
            model=config.gemini.script_model,
            schema=script_segments_schema(),
            system_prompt=SCRIPT_SYSTEM_PROMPT,
            user_prompt=build_bilibili_script_prompt(
                story_payload=_story_payload(beats),
                scene_payload=_script_scene_payload(beats, scenes),
                max_chars=config.script.max_chars,
                tone=config.script.tone,
            ),
            image_paths=None,
            max_output_tokens=config.gemini.script_max_output_tokens,
            temperature=config.gemini.script_temperature,
        )
        return _materialize_script_segments(payload, beats, scenes, fallback_segments, config)


class OllamaScriptGenerator(ScriptGenerator):
    def __init__(self) -> None:
        self.fallback = TemplateScriptGenerator()
        self.client: OllamaJSONClient | None = None

    def generate(
        self,
        beats: list[StoryBeat],
        scenes: list[Scene],
        config: PipelineConfig,
    ) -> list[ScriptSegment]:
        fallback_segments = self.fallback.generate(beats, scenes, config)
        self.client = OllamaJSONClient(config.ollama)
        payload = self.client.generate_json(
            model=config.ollama.script_model,
            schema=script_segments_schema(),
            system_prompt=SCRIPT_SYSTEM_PROMPT,
            user_prompt=build_bilibili_script_prompt(
                story_payload=_story_payload(beats),
                scene_payload=_script_scene_payload(beats, scenes),
                max_chars=config.script.max_chars,
                tone=config.script.tone,
            ),
            image_paths=None,
            max_output_tokens=config.ollama.script_max_output_tokens,
            temperature=config.ollama.script_temperature,
        )
        return _materialize_script_segments(payload, beats, scenes, fallback_segments, config)


class AutoScriptGenerator(ScriptGenerator):
    def __init__(self) -> None:
        self.template = TemplateScriptGenerator()
        self.openai = OpenAIScriptGenerator()
        self.gemini = GeminiScriptGenerator()
        self.ollama = OllamaScriptGenerator()

    def generate(
        self,
        beats: list[StoryBeat],
        scenes: list[Scene],
        config: PipelineConfig,
    ) -> list[ScriptSegment]:
        if os.getenv(config.openai.api_key_env):
            try:
                return self.openai.generate(beats, scenes, config)
            except LLMRequestError:
                pass
        if os.getenv(config.gemini.api_key_env):
            try:
                return self.gemini.generate(beats, scenes, config)
            except LLMRequestError:
                pass
        try:
            return self.ollama.generate(beats, scenes, config)
        except LLMRequestError:
            pass
        return self.template.generate(beats, scenes, config)


def _story_payload(beats: list[StoryBeat]) -> list[dict[str, Any]]:
    return [
        {
            "beat_id": beat.beat_id,
            "title": beat.title,
            "summary": beat.summary,
            "scene_ids": beat.scene_ids,
            "importance": beat.importance,
            "causal_chain": beat.causal_chain,
            "climax_scene_id": beat.climax_scene_id,
        }
        for beat in beats
    ]


def _script_scene_payload(beats: list[StoryBeat], scenes: list[Scene]) -> list[dict[str, Any]]:
    scene_map = {scene.scene_id: scene for scene in scenes}
    selected_ids: list[str] = []
    for beat in beats:
        ranked = sorted(
            (scene_map[scene_id] for scene_id in beat.scene_ids if scene_id in scene_map),
            key=lambda scene: scene.importance_score,
            reverse=True,
        )
        for scene in ranked[:6]:
            if scene.scene_id not in selected_ids:
                selected_ids.append(scene.scene_id)

    if len(selected_ids) < min(len(scenes), 24):
        for scene in sorted(scenes, key=lambda item: item.importance_score, reverse=True):
            if scene.scene_id not in selected_ids:
                selected_ids.append(scene.scene_id)
            if len(selected_ids) >= 24:
                break

    payload: list[dict[str, Any]] = []
    for scene_id in selected_ids:
        scene = scene_map[scene_id]
        payload.append(
            {
                "scene_id": scene.scene_id,
                "summary": scene.summary,
                "events": scene.events,
                "characters": scene.characters,
                "visual_cues": scene.visual_cues,
                "importance_score": scene.importance_score,
                "window": {
                    "start": scene.window.start,
                    "end": scene.window.end,
                    "duration": scene.window.duration,
                },
            }
        )
    return payload


def _materialize_script_segments(
    payload: dict[str, Any],
    beats: list[StoryBeat],
    scenes: list[Scene],
    fallback_segments: list[ScriptSegment],
    config: PipelineConfig,
) -> list[ScriptSegment]:
    segments_payload = payload.get("segments")
    if not isinstance(segments_payload, list):
        return fallback_segments

    scene_ids = {scene.scene_id for scene in scenes}
    output: list[ScriptSegment] = []
    for index, fallback in enumerate(fallback_segments, start=1):
        raw = segments_payload[index - 1] if index - 1 < len(segments_payload) else {}
        title = str(raw.get("title") or fallback.title).strip() or fallback.title
        narration = trim_text(
            str(raw.get("narration") or fallback.narration).strip() or fallback.narration,
            1000,
        )
        selected_scene_ids = [
            scene_id
            for scene_id in raw.get("scene_ids", [])
            if isinstance(scene_id, str) and scene_id in scene_ids
        ]
        if not selected_scene_ids:
            selected_scene_ids = fallback.scene_ids
        try:
            target_seconds = int(raw.get("target_seconds"))
        except (TypeError, ValueError):
            target_seconds = fallback.target_seconds
        target_seconds = max(30, min(config.project.target_minutes * 60, target_seconds))
        output.append(
            ScriptSegment(
                segment_id=f"segment_{index:02d}",
                title=title,
                narration=narration,
                scene_ids=selected_scene_ids,
                target_seconds=target_seconds,
            )
        )

    total_chars = sum(len(item.narration) for item in output)
    if total_chars > config.script.max_chars and output:
        ratio = config.script.max_chars / total_chars
        for item in output:
            item.narration = trim_text(item.narration, max(100, int(len(item.narration) * ratio)))
    return output


def build_script_generator(provider: str) -> ScriptGenerator:
    if provider in {"template", "stub"}:
        return TemplateScriptGenerator()
    if provider in {"openai", "gpt-4o"}:
        return OpenAIScriptGenerator()
    if provider in {"gemini", "google"}:
        return GeminiScriptGenerator()
    if provider in {"ollama", "local"}:
        return OllamaScriptGenerator()
    if provider in {"auto", "default"}:
        return AutoScriptGenerator()
    raise ValueError(
        f"Unsupported script provider: {provider}. "
        "Please implement the provider in stages/script_generation.py."
    )
