from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_brief.config import PipelineConfig
from movie_brief.stages.asr import AutoASREngine, FasterWhisperASREngine
from movie_brief.stages.shot_detection import AutoShotDetector, PySceneDetectShotDetector


class RealProviderTests(unittest.TestCase):
    def test_auto_shot_detector_falls_back_when_package_missing(self) -> None:
        config = PipelineConfig()
        config.project.fallback_duration_seconds = 30
        config.shot_detection.target_shot_count = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "movie.mp4"
            video_path.write_bytes(b"placeholder")
            detector = AutoShotDetector()
            shots = detector.detect(video_path, config)

        self.assertTrue(shots)
        self.assertEqual(shots[0].shot_id, "shot_0001")

    def test_pyscenedetect_provider_with_fake_module(self) -> None:
        class FakeTimecode:
            def __init__(self, seconds: float) -> None:
                self.seconds = seconds

            def get_seconds(self) -> float:
                return self.seconds

        class FakeSceneManager:
            def __init__(self) -> None:
                self.detector = None

            def add_detector(self, detector: object) -> None:
                self.detector = detector

            def detect_scenes(self, video: object = None, show_progress: bool = False) -> None:
                self.video = video
                self.show_progress = show_progress

            def get_scene_list(self) -> list[tuple[FakeTimecode, FakeTimecode]]:
                return [
                    (FakeTimecode(0.0), FakeTimecode(4.0)),
                    (FakeTimecode(4.0), FakeTimecode(9.5)),
                ]

        class FakeContentDetector:
            def __init__(self, threshold: float, min_scene_len: int) -> None:
                self.threshold = threshold
                self.min_scene_len = min_scene_len

        class FakeAdaptiveDetector:
            def __init__(self, adaptive_threshold: float, min_scene_len: int) -> None:
                self.adaptive_threshold = adaptive_threshold
                self.min_scene_len = min_scene_len

        scenedetect_module = types.ModuleType("scenedetect")
        scenedetect_module.SceneManager = FakeSceneManager
        scenedetect_module.open_video = lambda path: {"path": path}

        detectors_module = types.ModuleType("scenedetect.detectors")
        detectors_module.ContentDetector = FakeContentDetector
        detectors_module.AdaptiveDetector = FakeAdaptiveDetector

        with patch.dict(
            sys.modules,
            {
                "scenedetect": scenedetect_module,
                "scenedetect.detectors": detectors_module,
            },
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = Path(tmpdir) / "movie.mp4"
                video_path.write_bytes(b"placeholder")
                detector = PySceneDetectShotDetector()
                shots = detector.detect(video_path, PipelineConfig())

        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[1].window.end, 9.5)

    def test_auto_asr_uses_sidecar_without_external_package(self) -> None:
        config = PipelineConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "movie.mp4"
            sidecar = root / "movie.srt"
            video_path.write_bytes(b"placeholder")
            sidecar.write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:00,000 --> 00:00:02,000",
                        "Narrator: 事情终于开始失控。",
                    ]
                ),
                encoding="utf-8",
            )

            engine = AutoASREngine()
            utterances = engine.transcribe(video_path, shots=[], config=config)

        self.assertEqual(len(utterances), 1)
        self.assertEqual(utterances[0].text, "事情终于开始失控。")

    def test_faster_whisper_provider_with_fake_module(self) -> None:
        class FakeSegment:
            def __init__(self, start: float, end: float, text: str) -> None:
                self.start = start
                self.end = end
                self.text = text

        class FakeWhisperModel:
            def __init__(self, model_name: str, device: str, compute_type: str) -> None:
                self.model_name = model_name
                self.device = device
                self.compute_type = compute_type

            def transcribe(
                self,
                path: str,
                language: str,
                beam_size: int,
                vad_filter: bool,
                condition_on_previous_text: bool,
            ) -> tuple[list[FakeSegment], dict]:
                self.path = path
                self.language = language
                self.beam_size = beam_size
                self.vad_filter = vad_filter
                self.condition_on_previous_text = condition_on_previous_text
                return (
                    [
                        FakeSegment(0.0, 1.5, "第一句台词"),
                        FakeSegment(1.5, 3.0, "第二句台词"),
                    ],
                    {"language": language},
                )

        faster_whisper_module = types.ModuleType("faster_whisper")
        faster_whisper_module.WhisperModel = FakeWhisperModel

        with patch.dict(sys.modules, {"faster_whisper": faster_whisper_module}):
            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = Path(tmpdir) / "movie.mp4"
                video_path.write_bytes(b"placeholder")
                engine = FasterWhisperASREngine()
                utterances = engine.transcribe(video_path, shots=[], config=PipelineConfig())

        self.assertEqual([item.text for item in utterances], ["第一句台词", "第二句台词"])


if __name__ == "__main__":
    unittest.main()
