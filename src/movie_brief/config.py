from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(slots=True)
class ProjectSettings:
    title: str = "电影一键浓缩系统"
    target_minutes: int = 12
    language: str = "zh"
    fallback_duration_seconds: int = 7200


@dataclass(slots=True)
class ProviderSettings:
    shot_detector: str = "stub"
    asr: str = "stub"
    vision: str = "stub"
    story: str = "heuristic"
    script: str = "template"
    selector: str = "heuristic"


@dataclass(slots=True)
class ShotDetectionSettings:
    target_shot_count: int = 160
    min_shot_length: float = 1.2


@dataclass(slots=True)
class CompressionSettings:
    selection_ratio: float = 0.2
    coverage_per_arc: int = 2
    scene_group_size: int = 8
    short_ratio: float = 0.35
    teaser_ratio: float = 0.12


@dataclass(slots=True)
class ScriptSettings:
    max_chars: int = 2200
    tone: str = "通俗、有感染力、强调冲突和转折"


@dataclass(slots=True)
class FrameExtractionSettings:
    frames_per_scene: int = 2
    image_max_width: int = 896
    image_quality: int = 3


@dataclass(slots=True)
class PySceneDetectSettings:
    detector: str = "content"
    threshold: float = 27.0
    min_scene_len_frames: int = 15


@dataclass(slots=True)
class FasterWhisperSettings:
    model: str = "small"
    device: str = "auto"
    compute_type: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = True


@dataclass(slots=True)
class OpenAISettings:
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1/responses"
    scene_model: str = "gpt-4o"
    script_model: str = "gpt-4o"
    image_detail: str = "low"
    scene_max_output_tokens: int = 900
    script_max_output_tokens: int = 2600
    scene_temperature: float = 0.2
    script_temperature: float = 0.8
    timeout_seconds: int = 120


@dataclass(slots=True)
class GeminiSettings:
    api_key_env: str = "GEMINI_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    scene_model: str = "gemini-2.5-flash"
    script_model: str = "gemini-2.5-flash"
    scene_max_output_tokens: int = 900
    script_max_output_tokens: int = 2600
    scene_temperature: float = 0.2
    script_temperature: float = 0.8
    timeout_seconds: int = 120


@dataclass(slots=True)
class OllamaSettings:
    base_url: str = "http://127.0.0.1:11434/api/chat"
    scene_model: str = "qwen2.5vl:7b"
    script_model: str = "qwen2.5:7b"
    keep_alive: str = "5m"
    scene_max_output_tokens: int = 900
    script_max_output_tokens: int = 2600
    scene_temperature: float = 0.2
    script_temperature: float = 0.8
    timeout_seconds: int = 180


@dataclass(slots=True)
class PipelineConfig:
    project: ProjectSettings = field(default_factory=ProjectSettings)
    providers: ProviderSettings = field(default_factory=ProviderSettings)
    shot_detection: ShotDetectionSettings = field(default_factory=ShotDetectionSettings)
    compression: CompressionSettings = field(default_factory=CompressionSettings)
    script: ScriptSettings = field(default_factory=ScriptSettings)
    frames: FrameExtractionSettings = field(default_factory=FrameExtractionSettings)
    pyscenedetect: PySceneDetectSettings = field(default_factory=PySceneDetectSettings)
    faster_whisper: FasterWhisperSettings = field(default_factory=FasterWhisperSettings)
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)

    @classmethod
    def load(cls, path: Path | None = None) -> "PipelineConfig":
        config = cls()
        if path is None:
            return config
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        _apply_dataclass_overrides(config, raw)
        return config


def _apply_dataclass_overrides(target: Any, overrides: dict[str, Any]) -> None:
    for field_def in fields(target):
        if field_def.name not in overrides:
            continue
        incoming = overrides[field_def.name]
        current = getattr(target, field_def.name)
        if is_dataclass(current) and isinstance(incoming, dict):
            _apply_dataclass_overrides(current, incoming)
            continue
        setattr(target, field_def.name, incoming)
