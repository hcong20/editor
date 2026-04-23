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
from movie_brief.models import Scene, StoryBeat
from movie_brief.prompts import build_story_beats_prompt, story_beats_schema
from movie_brief.utils import trim_text


class StoryModeler:
    def build(self, scenes: list[Scene], config: PipelineConfig) -> list[StoryBeat]:
        raise NotImplementedError


class HeuristicStoryModeler(StoryModeler):
    def build(self, scenes: list[Scene], config: PipelineConfig) -> list[StoryBeat]:
        if not scenes:
            return []

        scenes = sorted(scenes, key=lambda item: item.window.start)
        total = len(scenes)
        climax_scene_id = _detect_climax_scene_id(scenes)
        causal_links = _build_causal_links(scenes)

        climax_index = max(range(total), key=lambda idx: scenes[idx].importance_score)
        setup_end = max(1, min(total - 3, total // 4))
        conflict_end = max(setup_end + 1, min(climax_index, total // 2))
        climax_end = max(conflict_end + 1, min(total - 1, climax_index + max(1, total // 8)))

        groups = [
            ("beat_setup", "背景介绍", scenes[:setup_end]),
            ("beat_conflict", "冲突爆发", scenes[setup_end:conflict_end]),
            ("beat_climax", "高潮对抗", scenes[conflict_end:climax_end]),
            ("beat_resolution", "结局收束", scenes[climax_end:]),
        ]

        beats: list[StoryBeat] = []
        for beat_id, title, group in groups:
            if not group:
                continue
            scene_ids = [scene.scene_id for scene in group]
            beat_links = _select_links_for_scenes(causal_links, scene_ids, limit=3)
            beat_climax = climax_scene_id if climax_scene_id in scene_ids else None
            summary = self._summarize_group(title, group, beat_links, beat_climax)
            importance = round(
                sum(scene.importance_score for scene in group) / len(group),
                3,
            )
            beats.append(
                StoryBeat(
                    beat_id=beat_id,
                    title=title,
                    summary=summary,
                    scene_ids=scene_ids,
                    importance=importance,
                    causal_chain=beat_links,
                    climax_scene_id=beat_climax,
                )
            )
        return beats

    def _summarize_group(
        self,
        title: str,
        scenes: list[Scene],
        causal_chain: list[str],
        climax_scene_id: str | None,
    ) -> str:
        strongest_scene = max(scenes, key=lambda scene: scene.importance_score)
        core_events = "、".join(strongest_scene.events[:2]) or "剧情推进"
        summary = (
            f"{title}聚焦于 {strongest_scene.summary}"
            f" 这一段的核心作用是 {core_events}。"
        )
        if causal_chain:
            summary = f"{summary} 因果链：{causal_chain[0]}。"
        if title == "高潮对抗" and climax_scene_id:
            summary = f"{summary} 高潮锚点：{climax_scene_id}。"
        return trim_text(summary, 260)


STORY_MODELING_SYSTEM_PROMPT = """你是专业电影叙事分析师。

你会基于 scene 列表完成三件事：
1. 四段式故事结构重建（背景介绍/冲突爆发/高潮对抗/结局收束）
2. 高潮锚点识别
3. 关键因果链梳理

要求：
- 只输出 JSON
- 只使用输入里存在的 scene_id
- summary 必须体现人物动机、冲突升级和因果关系
- causal_chain 必须是明确的“原因 -> 结果"""


class _BaseLLMStoryModeler(StoryModeler):
    def __init__(self) -> None:
        self.fallback = HeuristicStoryModeler()

    def _generate_payload(self, scenes: list[Scene], config: PipelineConfig) -> dict[str, Any]:
        raise NotImplementedError

    def build(self, scenes: list[Scene], config: PipelineConfig) -> list[StoryBeat]:
        if not scenes:
            return []
        fallback_beats = self.fallback.build(scenes, config)
        payload = self._generate_payload(scenes, config)
        return _materialize_story_beats(payload, scenes, fallback_beats)


class OpenAIStoryModeler(_BaseLLMStoryModeler):
    def __init__(self) -> None:
        super().__init__()
        self.client: OpenAIResponsesJSONClient | None = None

    def _generate_payload(self, scenes: list[Scene], config: PipelineConfig) -> dict[str, Any]:
        self.client = OpenAIResponsesJSONClient(config.openai)
        return self.client.generate_json_with_model(
            model=config.openai.scene_model,
            schema_name="story_beats",
            schema=story_beats_schema(),
            system_prompt=STORY_MODELING_SYSTEM_PROMPT,
            user_prompt=build_story_beats_prompt(_story_scene_payload(scenes)),
            image_paths=None,
            max_output_tokens=config.openai.scene_max_output_tokens,
            temperature=config.openai.scene_temperature,
        )


class GeminiStoryModeler(_BaseLLMStoryModeler):
    def __init__(self) -> None:
        super().__init__()
        self.client: GeminiJSONClient | None = None

    def _generate_payload(self, scenes: list[Scene], config: PipelineConfig) -> dict[str, Any]:
        self.client = GeminiJSONClient(config.gemini)
        return self.client.generate_json(
            model=config.gemini.scene_model,
            schema=story_beats_schema(),
            system_prompt=STORY_MODELING_SYSTEM_PROMPT,
            user_prompt=build_story_beats_prompt(_story_scene_payload(scenes)),
            image_paths=None,
            max_output_tokens=config.gemini.scene_max_output_tokens,
            temperature=config.gemini.scene_temperature,
        )


class OllamaStoryModeler(_BaseLLMStoryModeler):
    def __init__(self) -> None:
        super().__init__()
        self.client: OllamaJSONClient | None = None

    def _generate_payload(self, scenes: list[Scene], config: PipelineConfig) -> dict[str, Any]:
        self.client = OllamaJSONClient(config.ollama)
        return self.client.generate_json(
            model=config.ollama.scene_model,
            schema=story_beats_schema(),
            system_prompt=STORY_MODELING_SYSTEM_PROMPT,
            user_prompt=build_story_beats_prompt(_story_scene_payload(scenes)),
            image_paths=None,
            max_output_tokens=config.ollama.scene_max_output_tokens,
            temperature=config.ollama.scene_temperature,
        )


class AutoStoryModeler(StoryModeler):
    def __init__(self) -> None:
        self.heuristic = HeuristicStoryModeler()
        self.openai = OpenAIStoryModeler()
        self.gemini = GeminiStoryModeler()
        self.ollama = OllamaStoryModeler()

    def build(self, scenes: list[Scene], config: PipelineConfig) -> list[StoryBeat]:
        if os.getenv(config.openai.api_key_env):
            try:
                return self.openai.build(scenes, config)
            except LLMRequestError:
                pass
        if os.getenv(config.gemini.api_key_env):
            try:
                return self.gemini.build(scenes, config)
            except LLMRequestError:
                pass
        try:
            return self.ollama.build(scenes, config)
        except LLMRequestError:
            pass
        return self.heuristic.build(scenes, config)


def _story_scene_payload(scenes: list[Scene]) -> list[dict[str, Any]]:
    ordered = sorted(scenes, key=lambda item: item.window.start)
    return [
        {
            "scene_id": scene.scene_id,
            "scene_index": index,
            "summary": scene.summary,
            "events": scene.events,
            "characters": scene.characters,
            "emotion_intensity": scene.emotion_intensity,
            "conflict_score": scene.conflict_score,
            "plot_progression_score": scene.plot_progression_score,
            "importance_score": scene.importance_score,
            "window": {
                "start": scene.window.start,
                "end": scene.window.end,
                "duration": scene.window.duration,
            },
        }
        for index, scene in enumerate(ordered, start=1)
    ]


def _materialize_story_beats(
    payload: dict[str, Any],
    scenes: list[Scene],
    fallback_beats: list[StoryBeat],
) -> list[StoryBeat]:
    beats_payload = payload.get("beats")
    if not isinstance(beats_payload, list):
        return fallback_beats

    scene_ids = {scene.scene_id for scene in scenes}
    global_causal_chain = _normalize_causal_chain(payload.get("causal_chain"), scene_ids)
    global_climax_scene_id = _normalize_climax(payload.get("climax"), scene_ids)

    output: list[StoryBeat] = []
    for index, fallback in enumerate(fallback_beats):
        raw = beats_payload[index] if index < len(beats_payload) else {}
        if not isinstance(raw, dict):
            raw = {}

        beat_id = str(raw.get("beat_id") or fallback.beat_id).strip() or fallback.beat_id
        title = _normalize_beat_title(raw.get("title"), fallback.title)
        summary = str(raw.get("summary") or fallback.summary).strip() or fallback.summary
        selected_scene_ids = [
            scene_id
            for scene_id in raw.get("scene_ids", [])
            if isinstance(scene_id, str) and scene_id in scene_ids
        ]
        if not selected_scene_ids:
            selected_scene_ids = fallback.scene_ids

        importance = _coerce_importance(raw.get("importance"), fallback.importance)
        local_causal_chain = _normalize_str_list(raw.get("causal_chain"), limit=5)
        if not local_causal_chain:
            local_causal_chain = _select_links_for_scenes(
                global_causal_chain,
                selected_scene_ids,
                limit=3,
            )

        climax_scene_id = _normalize_scene_id(raw.get("climax_scene_id"), scene_ids)
        if climax_scene_id is None and title == "高潮对抗":
            if global_climax_scene_id in selected_scene_ids:
                climax_scene_id = global_climax_scene_id

        summary = _compose_story_summary(summary, local_causal_chain, climax_scene_id)
        output.append(
            StoryBeat(
                beat_id=beat_id,
                title=title,
                summary=summary,
                scene_ids=selected_scene_ids,
                importance=importance,
                causal_chain=local_causal_chain,
                climax_scene_id=climax_scene_id,
            )
        )

    return output or fallback_beats


def _detect_climax_scene_id(scenes: list[Scene]) -> str:
    def score(scene: Scene) -> float:
        return (
            scene.conflict_score * 1.7
            + scene.emotion_intensity * 1.2
            + scene.plot_progression_score * 1.1
            + scene.importance_score * 0.4
        )

    return max(scenes, key=score).scene_id


def _build_causal_links(scenes: list[Scene]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for cause, effect in zip(scenes, scenes[1:]):
        links.append(
            {
                "cause_scene_id": cause.scene_id,
                "effect_scene_id": effect.scene_id,
                "reason": _infer_causal_reason(cause, effect),
            }
        )
    return links


def _infer_causal_reason(cause: Scene, effect: Scene) -> str:
    if effect.conflict_score - cause.conflict_score > 0.18:
        return "上一场冲突升级，直接引发下一场更激烈的对抗"

    shared = [name for name in cause.characters if name in effect.characters]
    if shared:
        return f"角色 {shared[0]} 的决策延续，推动局势进入下一阶段"

    if effect.plot_progression_score - cause.plot_progression_score > 0.12:
        return "上一场信息揭示后，剧情因果链继续推进"

    return "上一场形成的处境和压力，推动下一场事件发生"


def _select_links_for_scenes(
    causal_links: list[dict[str, str]],
    scene_ids: list[str],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []

    scene_id_set = set(scene_ids)
    output: list[str] = []
    for link in causal_links:
        cause_id = link["cause_scene_id"]
        effect_id = link["effect_scene_id"]
        if cause_id not in scene_id_set and effect_id not in scene_id_set:
            continue
        output.append(_format_causal_link(link))
        if len(output) >= limit:
            break
    return output


def _format_causal_link(link: dict[str, str]) -> str:
    return (
        f"{link['cause_scene_id']} -> {link['effect_scene_id']}: "
        f"{link['reason']}"
    )


def _normalize_causal_chain(
    value: Any,
    scene_ids: set[str],
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    output: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        cause_scene_id = item.get("cause_scene_id")
        effect_scene_id = item.get("effect_scene_id")
        reason = str(item.get("reason") or "").strip()
        if (
            not isinstance(cause_scene_id, str)
            or not isinstance(effect_scene_id, str)
            or not reason
            or cause_scene_id not in scene_ids
            or effect_scene_id not in scene_ids
        ):
            continue
        output.append(
            {
                "cause_scene_id": cause_scene_id,
                "effect_scene_id": effect_scene_id,
                "reason": reason,
            }
        )
    return output


def _normalize_climax(value: Any, scene_ids: set[str]) -> str | None:
    if not isinstance(value, dict):
        return None
    scene_id = value.get("scene_id")
    if not isinstance(scene_id, str) or scene_id not in scene_ids:
        return None
    return scene_id


def _normalize_scene_id(value: Any, scene_ids: set[str]) -> str | None:
    if not isinstance(value, str) or value not in scene_ids:
        return None
    return value


def _normalize_beat_title(value: Any, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = value.strip()
    if cleaned in {"背景介绍", "冲突爆发", "高潮对抗", "结局收束"}:
        return cleaned
    return fallback


def _normalize_str_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list) or limit <= 0:
        return []
    output: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned not in output:
            output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _coerce_importance(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return round(max(0.0, min(4.0, numeric)), 3)


def _compose_story_summary(
    summary: str,
    causal_chain: list[str],
    climax_scene_id: str | None,
) -> str:
    output = trim_text(summary, 220)
    if causal_chain:
        output = trim_text(
            f"{output} 因果链：{'；'.join(causal_chain[:2])}。",
            300,
        )
    if climax_scene_id:
        output = trim_text(f"{output} 高潮锚点：{climax_scene_id}。", 330)
    return output


def build_story_modeler(provider: str) -> StoryModeler:
    if provider in {"heuristic", "stub"}:
        return HeuristicStoryModeler()
    if provider in {"openai", "gpt-4o"}:
        return OpenAIStoryModeler()
    if provider in {"gemini", "google"}:
        return GeminiStoryModeler()
    if provider in {"ollama", "local"}:
        return OllamaStoryModeler()
    if provider in {"auto", "default"}:
        return AutoStoryModeler()
    raise ValueError(
        f"Unsupported story modeling provider: {provider}. "
        "Please implement the provider in stages/story_modeling.py."
    )

