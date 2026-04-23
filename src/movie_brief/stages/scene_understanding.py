from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from movie_brief.config import PipelineConfig
from movie_brief.llm_clients import (
    GeminiJSONClient,
    LLMRequestError,
    OllamaJSONClient,
    OpenAIResponsesJSONClient,
)
from movie_brief.media import extract_representative_frames
from movie_brief.models import Scene, Shot, TimeRange, Utterance
from movie_brief.prompts import build_scene_analysis_prompt, scene_analysis_schema
from movie_brief.utils import chunked, first_non_empty, overlap_seconds, trim_text


EMOTION_KEYWORDS = {
    "哭",
    "怒",
    "恨",
    "爱",
    "害怕",
    "绝望",
    "崩溃",
    "激动",
    "痛苦",
    "失控",
}

CONFLICT_KEYWORDS = {
    "杀",
    "打",
    "威胁",
    "背叛",
    "冲突",
    "对抗",
    "危险",
    "复仇",
    "追",
    "逃",
}

SCENE_ANALYSIS_SYSTEM_PROMPT = """你是电影分镜和叙事分析助手。

你会同时参考关键帧和对白摘录，输出一个适合后续自动解说系统使用的 scene 结构化分析。

要求：
- 只输出 JSON
- 不要编造镜头里没有明显证据的人物或事件
- 如果信息不足，给出谨慎判断
- summary 要像电影解说作者在做场景拆解，而不是流水账
- events 和 visual_cues 要短、具体、可用于后续脚本和剪辑"""


@dataclass(slots=True)
class SceneBundle:
    scene_id: str
    shot_ids: list[str]
    window: TimeRange
    matched_utterances: list[Utterance]
    transcript_excerpt: str
    scene_index: int
    total_scenes: int


class SceneUnderstandingEngine:
    def analyze(
        self,
        video_path: Path,
        shots: list[Shot],
        transcript: list[Utterance],
        config: PipelineConfig,
        artifacts_dir: Path | None = None,
    ) -> list[Scene]:
        raise NotImplementedError


class HeuristicSceneUnderstandingEngine(SceneUnderstandingEngine):
    def analyze(
        self,
        video_path: Path,
        shots: list[Shot],
        transcript: list[Utterance],
        config: PipelineConfig,
        artifacts_dir: Path | None = None,
    ) -> list[Scene]:
        bundles = _build_scene_bundles(shots, transcript, config)
        return [_build_heuristic_scene(bundle) for bundle in tqdm(bundles, desc="Scene Analysis", unit="scene")]


class OpenAISceneUnderstandingEngine(SceneUnderstandingEngine):
    def __init__(self) -> None:
        self.client: OpenAIResponsesJSONClient | None = None

    def analyze(
        self,
        video_path: Path,
        shots: list[Shot],
        transcript: list[Utterance],
        config: PipelineConfig,
        artifacts_dir: Path | None = None,
    ) -> list[Scene]:
        self.client = OpenAIResponsesJSONClient(config.openai)
        bundles = _build_scene_bundles(shots, transcript, config)
        scenes: list[Scene] = []

        for bundle in tqdm(bundles, desc="OpenAI Scene Analysis", unit="scene"):
            frame_paths = _extract_bundle_frames(video_path, bundle, config, artifacts_dir)
            payload = self.client.generate_json_with_model(
                model=config.openai.scene_model,
                schema_name="scene_analysis",
                schema=scene_analysis_schema(),
                system_prompt=SCENE_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=build_scene_analysis_prompt(
                    scene_id=bundle.scene_id,
                    start_seconds=bundle.window.start,
                    end_seconds=bundle.window.end,
                    scene_index=bundle.scene_index,
                    total_scenes=bundle.total_scenes,
                    transcript_excerpt=bundle.transcript_excerpt,
                ),
                image_paths=frame_paths,
                max_output_tokens=config.openai.scene_max_output_tokens,
                temperature=config.openai.scene_temperature,
            )
            scenes.append(_build_scene_from_llm_payload(bundle, payload, frame_paths))

        return scenes


class GeminiSceneUnderstandingEngine(SceneUnderstandingEngine):
    def __init__(self) -> None:
        self.client: GeminiJSONClient | None = None

    def analyze(
        self,
        video_path: Path,
        shots: list[Shot],
        transcript: list[Utterance],
        config: PipelineConfig,
        artifacts_dir: Path | None = None,
    ) -> list[Scene]:
        self.client = GeminiJSONClient(config.gemini)
        bundles = _build_scene_bundles(shots, transcript, config)
        scenes: list[Scene] = []

        for bundle in tqdm(bundles, desc="Gemini Scene Analysis", unit="scene"):
            frame_paths = _extract_bundle_frames(video_path, bundle, config, artifacts_dir)
            payload = self.client.generate_json(
                model=config.gemini.scene_model,
                schema=scene_analysis_schema(),
                system_prompt=SCENE_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=build_scene_analysis_prompt(
                    scene_id=bundle.scene_id,
                    start_seconds=bundle.window.start,
                    end_seconds=bundle.window.end,
                    scene_index=bundle.scene_index,
                    total_scenes=bundle.total_scenes,
                    transcript_excerpt=bundle.transcript_excerpt,
                ),
                image_paths=frame_paths,
                max_output_tokens=config.gemini.scene_max_output_tokens,
                temperature=config.gemini.scene_temperature,
            )
            scenes.append(_build_scene_from_llm_payload(bundle, payload, frame_paths))

        return scenes


class OllamaSceneUnderstandingEngine(SceneUnderstandingEngine):
    def __init__(self) -> None:
        self.client: OllamaJSONClient | None = None

    def analyze(
        self,
        video_path: Path,
        shots: list[Shot],
        transcript: list[Utterance],
        config: PipelineConfig,
        artifacts_dir: Path | None = None,
    ) -> list[Scene]:
        self.client = OllamaJSONClient(config.ollama)
        bundles = _build_scene_bundles(shots, transcript, config)
        scenes: list[Scene] = []

        for bundle in tqdm(bundles, desc="Ollama Scene Analysis", unit="scene"):
            frame_paths = _extract_bundle_frames(video_path, bundle, config, artifacts_dir)
            payload = self.client.generate_json(
                model=config.ollama.scene_model,
                schema=scene_analysis_schema(),
                system_prompt=SCENE_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=build_scene_analysis_prompt(
                    scene_id=bundle.scene_id,
                    start_seconds=bundle.window.start,
                    end_seconds=bundle.window.end,
                    scene_index=bundle.scene_index,
                    total_scenes=bundle.total_scenes,
                    transcript_excerpt=bundle.transcript_excerpt,
                ),
                image_paths=frame_paths,
                max_output_tokens=config.ollama.scene_max_output_tokens,
                temperature=config.ollama.scene_temperature,
            )
            scenes.append(_build_scene_from_llm_payload(bundle, payload, frame_paths))

        return scenes


class AutoSceneUnderstandingEngine(SceneUnderstandingEngine):
    def __init__(self) -> None:
        self.heuristic = HeuristicSceneUnderstandingEngine()
        self.openai = OpenAISceneUnderstandingEngine()
        self.gemini = GeminiSceneUnderstandingEngine()
        self.ollama = OllamaSceneUnderstandingEngine()

    def analyze(
        self,
        video_path: Path,
        shots: list[Shot],
        transcript: list[Utterance],
        config: PipelineConfig,
        artifacts_dir: Path | None = None,
    ) -> list[Scene]:
        if os.getenv(config.openai.api_key_env):
            try:
                return self.openai.analyze(video_path, shots, transcript, config, artifacts_dir)
            except LLMRequestError:
                pass
        if os.getenv(config.gemini.api_key_env):
            try:
                return self.gemini.analyze(video_path, shots, transcript, config, artifacts_dir)
            except LLMRequestError:
                pass
        try:
            return self.ollama.analyze(video_path, shots, transcript, config, artifacts_dir)
        except LLMRequestError:
            pass
        return self.heuristic.analyze(video_path, shots, transcript, config, artifacts_dir)


def _build_scene_bundles(
    shots: list[Shot],
    transcript: list[Utterance],
    config: PipelineConfig,
) -> list[SceneBundle]:
    if not shots:
        return []

    grouped_shots = chunked(shots, config.compression.scene_group_size)
    total_groups = len(grouped_shots)
    bundles: list[SceneBundle] = []
    for index, shot_group in enumerate(grouped_shots, start=1):
        start = shot_group[0].window.start
        end = shot_group[-1].window.end
        matched_utterances = [
            item
            for item in transcript
            if overlap_seconds(start, end, item.start, item.end) > 0
        ]
        bundles.append(
            SceneBundle(
                scene_id=f"scene_{index:03d}",
                shot_ids=[shot.shot_id for shot in shot_group],
                window=TimeRange(start=start, end=end),
                matched_utterances=matched_utterances,
                transcript_excerpt=trim_text(
                    " ".join(item.text for item in matched_utterances),
                    320,
                ),
                scene_index=index,
                total_scenes=total_groups,
            )
        )
    return bundles


def _extract_bundle_frames(
    video_path: Path,
    bundle: SceneBundle,
    config: PipelineConfig,
    artifacts_dir: Path | None,
) -> list[Path]:
    if artifacts_dir is None:
        artifacts_dir = video_path.parent / ".scene_frames"
    return extract_representative_frames(
        video_path=video_path,
        start=bundle.window.start,
        end=bundle.window.end,
        output_dir=artifacts_dir,
        scene_id=bundle.scene_id,
        frame_count=config.frames.frames_per_scene,
        image_max_width=config.frames.image_max_width,
        image_quality=config.frames.image_quality,
    )


def _build_scene_from_llm_payload(
    bundle: SceneBundle,
    payload: dict[str, Any],
    frame_paths: list[Path],
) -> Scene:
    fallback = _build_heuristic_scene(bundle)
    summary = trim_text(str(payload.get("summary") or fallback.summary), 220)
    events = _normalize_str_list(payload.get("events"), fallback.events, limit=4)
    characters = _normalize_str_list(payload.get("characters"), fallback.characters, limit=5)
    visual_cues = _normalize_str_list(payload.get("visual_cues"), [], limit=5)
    return Scene(
        scene_id=bundle.scene_id,
        shot_ids=bundle.shot_ids,
        window=bundle.window,
        summary=summary,
        events=events,
        characters=characters,
        emotion_intensity=_coerce_score(
            payload.get("emotion_intensity"),
            fallback.emotion_intensity,
        ),
        core_character_score=_coerce_score(
            payload.get("core_character_score"),
            fallback.core_character_score,
        ),
        conflict_score=_coerce_score(
            payload.get("conflict_score"),
            fallback.conflict_score,
        ),
        plot_progression_score=_coerce_score(
            payload.get("plot_progression_score"),
            fallback.plot_progression_score,
        ),
        transcript_excerpt=bundle.transcript_excerpt,
        visual_cues=visual_cues,
        frame_paths=[str(path) for path in frame_paths],
    )


def _build_heuristic_scene(bundle: SceneBundle) -> Scene:
    characters = _extract_characters(bundle.matched_utterances)
    emotion = _score_keywords(bundle.transcript_excerpt, EMOTION_KEYWORDS, base=0.25)
    conflict = _score_keywords(bundle.transcript_excerpt, CONFLICT_KEYWORDS, base=0.2)
    plot_progression = round(
        min(1.0, bundle.scene_index / max(1, bundle.total_scenes) + 0.1),
        3,
    )
    core_character = round(
        min(1.0, 0.25 + 0.08 * len(characters) + 0.03 * len(bundle.matched_utterances)),
        3,
    )
    events = _infer_events(
        bundle.scene_index,
        bundle.total_scenes,
        bundle.transcript_excerpt,
        conflict,
        emotion,
    )
    summary = _build_summary(
        bundle.scene_index,
        bundle.total_scenes,
        bundle.transcript_excerpt,
        events,
    )
    return Scene(
        scene_id=bundle.scene_id,
        shot_ids=bundle.shot_ids,
        window=bundle.window,
        summary=summary,
        events=events,
        characters=characters,
        emotion_intensity=emotion,
        core_character_score=core_character,
        conflict_score=conflict,
        plot_progression_score=plot_progression,
        transcript_excerpt=bundle.transcript_excerpt,
        visual_cues=[],
        frame_paths=[],
    )


def _extract_characters(utterances: list[Utterance]) -> list[str]:
    speakers = [item.speaker for item in utterances if item.speaker]
    if not speakers:
        return []
    counts = Counter(speakers)
    return [name for name, _ in counts.most_common(4)]


def _score_keywords(text: str, keywords: set[str], base: float) -> float:
    hits = sum(text.count(keyword) for keyword in keywords)
    exclamations = text.count("!") + text.count("！")
    return round(min(1.0, base + hits * 0.12 + exclamations * 0.08), 3)


def _infer_events(
    index: int,
    total_groups: int,
    transcript_excerpt: str,
    conflict: float,
    emotion: float,
) -> list[str]:
    events: list[str] = []
    if index <= max(1, total_groups // 4):
        events.append("建立背景和人物关系")
    if conflict >= 0.45:
        events.append("主要矛盾持续升级")
    if emotion >= 0.55:
        events.append("人物情绪明显波动")
    if index >= max(1, int(total_groups * 0.75)):
        events.append("故事逐步走向结局")
    if not events:
        events.append("剧情继续推进")
    if transcript_excerpt:
        events.append("出现可用于解说的关键信息")
    return events[:4]


def _build_summary(
    index: int,
    total_groups: int,
    transcript_excerpt: str,
    events: list[str],
) -> str:
    position_hint = "前段"
    if index > total_groups * 0.66:
        position_hint = "后段"
    elif index > total_groups * 0.33:
        position_hint = "中段"
    transcript_hint = first_non_empty(
        [transcript_excerpt],
        "画面主要通过表演和场面调度推动故事。",
    )
    summary = (
        f"故事{position_hint}的第 {index} 个场景里，"
        f"{'，'.join(events[:2])}。{transcript_hint}"
    )
    return trim_text(summary, 180)


def _normalize_str_list(value: Any, fallback: list[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return fallback[:limit]
    output: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned not in output:
            output.append(cleaned)
    if not output:
        return fallback[:limit]
    return output[:limit]


def _coerce_score(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return round(max(0.0, min(1.0, numeric)), 3)


def build_scene_understanding_engine(provider: str) -> SceneUnderstandingEngine:
    if provider in {"stub", "heuristic"}:
        return HeuristicSceneUnderstandingEngine()
    if provider in {"openai", "gpt-4o"}:
        return OpenAISceneUnderstandingEngine()
    if provider in {"gemini", "google"}:
        return GeminiSceneUnderstandingEngine()
    if provider in {"ollama", "local"}:
        return OllamaSceneUnderstandingEngine()
    if provider in {"auto", "default"}:
        return AutoSceneUnderstandingEngine()
    raise ValueError(
        f"Unsupported scene understanding provider: {provider}. "
        "Please implement the provider in stages/scene_understanding.py."
    )
