from __future__ import annotations

from pathlib import Path
from tqdm import tqdm

from movie_brief.config import PipelineConfig
from movie_brief.models import Shot, Utterance
from movie_brief.utils import (
    discover_sidecar_transcript,
    module_available,
    parse_sidecar_transcript,
)


class ASREngine:
    def transcribe(
        self,
        video_path: Path,
        shots: list[Shot],
        config: PipelineConfig,
    ) -> list[Utterance]:
        raise NotImplementedError


def _load_sidecar(
    video_path: Path,
    config: PipelineConfig,
) -> list[Utterance] | None:
    sidecar_path = discover_sidecar_transcript(video_path)
    if sidecar_path is None:
        return None
    records = parse_sidecar_transcript(
        sidecar_path,
        fallback_duration=config.project.fallback_duration_seconds,
    )
    return [Utterance(**record) for record in records]


class StubASREngine(ASREngine):
    TEMPLATES = [
        "故事从一个看似平静的处境开始，但隐患已经出现。",
        "人物关系逐渐浮出水面，矛盾也开始累积。",
        "局势在一次意外后失控，主角被迫做出选择。",
        "冲突不断升级，每个人都被推向更危险的位置。",
        "真正的对抗终于到来，故事进入最紧张的阶段。",
        "结局并不轻松，但人物终于付出了代价并完成变化。",
    ]

    def transcribe(
        self,
        video_path: Path,
        shots: list[Shot],
        config: PipelineConfig,
    ) -> list[Utterance]:
        sidecar_utterances = _load_sidecar(video_path, config)
        if sidecar_utterances is not None:
            return sidecar_utterances

        if not shots:
            return []

        interval = max(1, len(shots) // len(self.TEMPLATES))
        utterances: list[Utterance] = []
        for index, template in tqdm(enumerate(self.TEMPLATES), desc="Stub ASR", unit="template"):
            shot = shots[min(index * interval, len(shots) - 1)]
            utterances.append(
                Utterance(
                    start=shot.window.start,
                    end=min(shot.window.end + 2.0, shot.window.end + 6.0),
                    text=template,
                    speaker="Narrator",
                )
            )
        return utterances


class FasterWhisperASREngine(ASREngine):
    def transcribe(
        self,
        video_path: Path,
        shots: list[Shot],
        config: PipelineConfig,
    ) -> list[Utterance]:
        sidecar_utterances = _load_sidecar(video_path, config)
        if sidecar_utterances is not None:
            return sidecar_utterances

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'faster-whisper' requires the 'faster-whisper' package."
            ) from exc

        settings = config.faster_whisper
        model = WhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        segments, _ = model.transcribe(
            str(video_path),
            language=config.project.language,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            condition_on_previous_text=settings.condition_on_previous_text,
        )
        utterances: list[Utterance] = []
        for segment in tqdm(segments, desc="ASR", unit="segment"):
            utterances.append(
                Utterance(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    speaker=None,
                )
            )
        return utterances


class AutoASREngine(ASREngine):
    def __init__(self) -> None:
        self.stub = StubASREngine()
        self.real = FasterWhisperASREngine()

    def transcribe(
        self,
        video_path: Path,
        shots: list[Shot],
        config: PipelineConfig,
    ) -> list[Utterance]:
        sidecar_utterances = _load_sidecar(video_path, config)
        if sidecar_utterances is not None:
            return sidecar_utterances
        if module_available("faster_whisper"):
            return self.real.transcribe(video_path, shots, config)
        return self.stub.transcribe(video_path, shots, config)


def build_asr_engine(provider: str) -> ASREngine:
    if provider == "stub":
        return StubASREngine()
    if provider in {"auto", "default"}:
        return AutoASREngine()
    if provider == "faster-whisper":
        return FasterWhisperASREngine()
    raise ValueError(
        f"Unsupported ASR provider: {provider}. "
        "Please implement the provider in stages/asr.py."
    )
