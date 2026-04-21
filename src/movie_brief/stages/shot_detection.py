from __future__ import annotations

from pathlib import Path

from movie_brief.config import PipelineConfig
from movie_brief.models import Shot, TimeRange
from movie_brief.utils import module_available, probe_duration_seconds


class ShotDetector:
    def detect(self, video_path: Path, config: PipelineConfig) -> list[Shot]:
        raise NotImplementedError


class StubShotDetector(ShotDetector):
    def detect(self, video_path: Path, config: PipelineConfig) -> list[Shot]:
        duration = probe_duration_seconds(video_path) or config.project.fallback_duration_seconds
        target_count = max(1, min(config.shot_detection.target_shot_count, 2000))
        step = max(config.shot_detection.min_shot_length, duration / target_count)
        shots: list[Shot] = []
        cursor = 0.0
        index = 0
        while cursor < duration:
            end = min(duration, cursor + step)
            shots.append(
                Shot(
                    shot_id=f"shot_{index + 1:04d}",
                    index=index,
                    window=TimeRange(start=round(cursor, 3), end=round(end, 3)),
                )
            )
            cursor = end
            index += 1
        return shots


class PySceneDetectShotDetector(ShotDetector):
    def detect(self, video_path: Path, config: PipelineConfig) -> list[Shot]:
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import AdaptiveDetector, ContentDetector
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'pyscenedetect' requires the 'scenedetect' package."
            ) from exc

        detector_name = config.pyscenedetect.detector.strip().lower()
        if detector_name == "adaptive":
            detector = AdaptiveDetector(
                adaptive_threshold=config.pyscenedetect.threshold,
                min_scene_len=config.pyscenedetect.min_scene_len_frames,
            )
        elif detector_name == "content":
            detector = ContentDetector(
                threshold=config.pyscenedetect.threshold,
                min_scene_len=config.pyscenedetect.min_scene_len_frames,
            )
        else:
            raise ValueError(
                "pyscenedetect.detector must be either 'content' or 'adaptive'."
            )

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(detector)
        try:
            manager.detect_scenes(video=video, show_progress=False)
        except TypeError:
            manager.detect_scenes(video)

        scene_list = manager.get_scene_list()
        if not scene_list:
            return StubShotDetector().detect(video_path, config)

        shots: list[Shot] = []
        for index, (start_time, end_time) in enumerate(scene_list):
            start_seconds = round(float(start_time.get_seconds()), 3)
            end_seconds = round(float(end_time.get_seconds()), 3)
            if end_seconds <= start_seconds:
                continue
            shots.append(
                Shot(
                    shot_id=f"shot_{index + 1:04d}",
                    index=index,
                    window=TimeRange(start=start_seconds, end=end_seconds),
                )
            )
        if not shots:
            return StubShotDetector().detect(video_path, config)
        return shots


class AutoShotDetector(ShotDetector):
    def __init__(self) -> None:
        self.stub = StubShotDetector()
        self.real = PySceneDetectShotDetector()

    def detect(self, video_path: Path, config: PipelineConfig) -> list[Shot]:
        if module_available("scenedetect"):
            return self.real.detect(video_path, config)
        return self.stub.detect(video_path, config)


def build_shot_detector(provider: str) -> ShotDetector:
    if provider == "stub":
        return StubShotDetector()
    if provider in {"auto", "default"}:
        return AutoShotDetector()
    if provider in {"pyscenedetect", "scenedetect"}:
        return PySceneDetectShotDetector()
    raise ValueError(
        f"Unsupported shot detector provider: {provider}. "
        "Please implement the provider in stages/shot_detection.py."
    )
