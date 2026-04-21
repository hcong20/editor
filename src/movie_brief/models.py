from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TimeRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class Shot:
    shot_id: str
    index: int
    window: TimeRange


@dataclass(slots=True)
class Utterance:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(slots=True)
class Scene:
    scene_id: str
    shot_ids: list[str]
    window: TimeRange
    summary: str
    events: list[str]
    characters: list[str]
    emotion_intensity: float
    core_character_score: float
    conflict_score: float
    plot_progression_score: float
    transcript_excerpt: str = ""
    visual_cues: list[str] = field(default_factory=list)
    frame_paths: list[str] = field(default_factory=list)

    @property
    def importance_score(self) -> float:
        return round(
            self.emotion_intensity
            + self.core_character_score
            + self.conflict_score
            + self.plot_progression_score,
            3,
        )


@dataclass(slots=True)
class StoryBeat:
    beat_id: str
    title: str
    summary: str
    scene_ids: list[str]
    importance: float
    causal_chain: list[str] = field(default_factory=list)
    climax_scene_id: str | None = None


@dataclass(slots=True)
class ScriptSegment:
    segment_id: str
    title: str
    narration: str
    scene_ids: list[str]
    target_seconds: int


@dataclass(slots=True)
class ClipPlan:
    clip_id: str
    scene_id: str
    start: float
    end: float
    narration_hint: str


@dataclass(slots=True)
class RenderPlan:
    clips: list[ClipPlan] = field(default_factory=list)
    variants: dict[str, list[str]] = field(default_factory=dict)
    ffmpeg_commands: list[str] = field(default_factory=list)
