from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_brief.config import PipelineConfig
from movie_brief.models import Scene, StoryBeat, TimeRange
from movie_brief.stages.selection import HeuristicSceneSelector


def _scene(
    scene_id: str,
    start: float,
    conflict: float,
    progression: float,
) -> Scene:
    return Scene(
        scene_id=scene_id,
        shot_ids=[f"shot_{scene_id}"],
        window=TimeRange(start=start, end=start + 10.0),
        summary=f"{scene_id} summary",
        events=["event"],
        characters=["主角"],
        emotion_intensity=0.45,
        core_character_score=0.55,
        conflict_score=conflict,
        plot_progression_score=progression,
    )


class SelectionTests(unittest.TestCase):
    def test_selector_prioritizes_causal_value_within_beat(self) -> None:
        scenes = [
            _scene("scene_001", 0.0, conflict=0.35, progression=0.40),
            _scene("scene_002", 10.0, conflict=0.35, progression=0.40),
            _scene("scene_003", 20.0, conflict=0.95, progression=0.86),
            _scene("scene_004", 30.0, conflict=0.50, progression=0.70),
        ]
        beats = [
            StoryBeat(
                beat_id="beat_conflict",
                title="冲突爆发",
                summary="conflict",
                scene_ids=["scene_001", "scene_002"],
                importance=2.0,
                causal_chain=["scene_002 -> scene_003: 决策升级引发摊牌"],
            ),
            StoryBeat(
                beat_id="beat_climax",
                title="高潮对抗",
                summary="climax",
                scene_ids=["scene_003"],
                importance=3.1,
                causal_chain=["scene_002 -> scene_003: 决策升级引发摊牌"],
                climax_scene_id="scene_003",
            ),
            StoryBeat(
                beat_id="beat_resolution",
                title="结局收束",
                summary="resolution",
                scene_ids=["scene_004"],
                importance=2.2,
            ),
        ]

        config = PipelineConfig()
        config.compression.coverage_per_arc = 1
        config.compression.selection_ratio = 0.25

        selected = HeuristicSceneSelector().select(scenes, beats, config)

        self.assertEqual(selected, ["scene_002", "scene_003", "scene_004"])

    def test_selector_returns_time_ordered_ids(self) -> None:
        scenes = [
            _scene("scene_003", 20.0, conflict=0.9, progression=0.85),
            _scene("scene_001", 0.0, conflict=0.3, progression=0.35),
            _scene("scene_004", 30.0, conflict=0.55, progression=0.75),
            _scene("scene_002", 10.0, conflict=0.4, progression=0.45),
        ]
        beats = [
            StoryBeat(
                beat_id="beat_a",
                title="冲突爆发",
                summary="a",
                scene_ids=["scene_003", "scene_001"],
                importance=2.8,
            ),
            StoryBeat(
                beat_id="beat_b",
                title="高潮对抗",
                summary="b",
                scene_ids=["scene_004", "scene_002"],
                importance=2.9,
                climax_scene_id="scene_004",
            ),
        ]

        config = PipelineConfig()
        config.compression.coverage_per_arc = 1
        config.compression.selection_ratio = 0.5

        selected = HeuristicSceneSelector().select(scenes, beats, config)
        start_map = {scene.scene_id: scene.window.start for scene in scenes}

        self.assertEqual(selected, sorted(selected, key=lambda scene_id: start_map[scene_id]))


if __name__ == "__main__":
    unittest.main()
