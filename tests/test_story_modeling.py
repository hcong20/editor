from __future__ import annotations

from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_brief.config import PipelineConfig
from movie_brief.llm_clients import LLMRequestError
from movie_brief.models import Scene, TimeRange
from movie_brief.stages.story_modeling import (
    AutoStoryModeler,
    GeminiStoryModeler,
    HeuristicStoryModeler,
    OllamaStoryModeler,
    OpenAIStoryModeler,
    build_story_modeler,
)


def _build_scenes() -> list[Scene]:
    return [
        Scene(
            scene_id="scene_001",
            shot_ids=["shot_0001"],
            window=TimeRange(start=0.0, end=10.0),
            summary="主角被迫接下任务。",
            events=["任务出现", "人物动机被触发"],
            characters=["主角"],
            emotion_intensity=0.2,
            core_character_score=0.7,
            conflict_score=0.2,
            plot_progression_score=0.25,
        ),
        Scene(
            scene_id="scene_002",
            shot_ids=["shot_0002"],
            window=TimeRange(start=10.0, end=20.0),
            summary="主角的判断引发第一次误会。",
            events=["误会产生", "关系恶化"],
            characters=["主角", "搭档"],
            emotion_intensity=0.35,
            core_character_score=0.75,
            conflict_score=0.45,
            plot_progression_score=0.4,
        ),
        Scene(
            scene_id="scene_003",
            shot_ids=["shot_0003"],
            window=TimeRange(start=20.0, end=30.0),
            summary="双方摊牌，冲突全面爆发。",
            events=["秘密揭露", "正面冲突"],
            characters=["主角", "反派"],
            emotion_intensity=0.95,
            core_character_score=0.9,
            conflict_score=0.98,
            plot_progression_score=0.82,
        ),
        Scene(
            scene_id="scene_004",
            shot_ids=["shot_0004"],
            window=TimeRange(start=30.0, end=40.0),
            summary="代价开始显现，主角必须收拾残局。",
            events=["后果兑现", "立场转变"],
            characters=["主角"],
            emotion_intensity=0.6,
            core_character_score=0.82,
            conflict_score=0.55,
            plot_progression_score=0.9,
        ),
    ]


class StoryModelingTests(unittest.TestCase):
    def test_heuristic_story_modeler_extracts_causal_chain_and_climax(self) -> None:
        config = PipelineConfig()
        scenes = _build_scenes()

        beats = HeuristicStoryModeler().build(scenes, config)

        self.assertTrue(beats)
        self.assertTrue(any(beat.causal_chain for beat in beats))
        climax_beats = [beat for beat in beats if beat.climax_scene_id]
        self.assertTrue(climax_beats)
        self.assertEqual(climax_beats[0].title, "高潮对抗")

    def test_openai_story_modeler_materializes_story_payload(self) -> None:
        payload = {
            "beats": [
                {
                    "beat_id": "beat_setup",
                    "title": "背景介绍",
                    "summary": "主角在压力下接下任务。",
                    "scene_ids": ["scene_001"],
                    "importance": 1.2,
                    "causal_chain": ["scene_001 -> scene_002: 选择埋下误会"],
                },
                {
                    "beat_id": "beat_conflict",
                    "title": "冲突爆发",
                    "summary": "误会累积并迅速扩大。",
                    "scene_ids": ["scene_002"],
                    "importance": 2.3,
                    "causal_chain": ["scene_002 -> scene_003: 双方走向摊牌"],
                },
                {
                    "beat_id": "beat_climax",
                    "title": "高潮对抗",
                    "summary": "真相曝光，冲突抵达峰值。",
                    "scene_ids": ["scene_003"],
                    "importance": 3.9,
                    "causal_chain": ["scene_003 -> scene_004: 决战后果外溢"],
                    "climax_scene_id": "scene_003",
                },
                {
                    "beat_id": "beat_resolution",
                    "title": "结局收束",
                    "summary": "主角承担代价并完成转变。",
                    "scene_ids": ["scene_004"],
                    "importance": 2.6,
                    "causal_chain": ["scene_003 -> scene_004: 冲突后果兑现"],
                },
            ],
            "climax": {"scene_id": "scene_003", "reason": "冲突强度与情绪峰值同时出现"},
            "causal_chain": [
                {
                    "cause_scene_id": "scene_001",
                    "effect_scene_id": "scene_002",
                    "reason": "主角的选择引发误会",
                },
                {
                    "cause_scene_id": "scene_002",
                    "effect_scene_id": "scene_003",
                    "reason": "误会升级成正面冲突",
                },
            ],
        }

        class FakeClient:
            def __init__(self, _settings: object) -> None:
                pass

            def generate_json_with_model(self, **_kwargs: object) -> dict:
                return payload

        with patch("movie_brief.stages.story_modeling.OpenAIResponsesJSONClient", FakeClient):
            beats = OpenAIStoryModeler().build(_build_scenes(), PipelineConfig())

        self.assertEqual(len(beats), 4)
        self.assertEqual(beats[2].title, "高潮对抗")
        self.assertEqual(beats[2].climax_scene_id, "scene_003")
        self.assertIn("scene_003 -> scene_004", " ".join(beats[2].causal_chain))

    def test_auto_story_modeler_falls_back_to_heuristic_when_llms_fail(self) -> None:
        modeler = AutoStoryModeler()
        scenes = _build_scenes()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "x", "GEMINI_API_KEY": "y"}):
            with patch.object(modeler.openai, "build", side_effect=LLMRequestError("openai failed")):
                with patch.object(modeler.gemini, "build", side_effect=LLMRequestError("gemini failed")):
                    with patch.object(modeler.ollama, "build", side_effect=LLMRequestError("ollama failed")):
                        beats = modeler.build(scenes, PipelineConfig())

        self.assertTrue(beats)
        self.assertTrue(any(beat.causal_chain for beat in beats))

    def test_build_story_modeler_supports_llm_providers(self) -> None:
        self.assertIsInstance(build_story_modeler("openai"), OpenAIStoryModeler)
        self.assertIsInstance(build_story_modeler("gemini"), GeminiStoryModeler)
        self.assertIsInstance(build_story_modeler("ollama"), OllamaStoryModeler)
        self.assertIsInstance(build_story_modeler("auto"), AutoStoryModeler)


if __name__ == "__main__":
    unittest.main()
