from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_brief.config import PipelineConfig
from movie_brief.pipeline import MovieCommentaryPipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_generates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "sample.mp4"
            output_dir = root / "run"
            subtitle_path = root / "sample.srt"

            video_path.write_bytes(b"not-a-real-movie")
            subtitle_path.write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:01,000 --> 00:00:04,000",
                        "Narrator: 一切从一个普通的夜晚开始。",
                        "",
                        "2",
                        "00:00:05,000 --> 00:00:08,000",
                        "冲突很快爆发，人物再也无法回头。",
                    ]
                ),
                encoding="utf-8",
            )

            config = PipelineConfig()
            config.project.fallback_duration_seconds = 120
            config.shot_detection.target_shot_count = 12
            config.compression.scene_group_size = 3

            pipeline = MovieCommentaryPipeline(config)
            pipeline.run(video_path, output_dir)

            expected = [
                "01_shots.json",
                "02_transcript.json",
                "03_scenes.json",
                "04_story_beats.json",
                "05_script.json",
                "06_selected_scenes.json",
                "07_render_plan.json",
                "summary.md",
            ]
            for filename in expected:
                self.assertTrue((output_dir / filename).exists(), filename)

            story_data = json.loads((output_dir / "04_story_beats.json").read_text(encoding="utf-8"))
            self.assertIn("beats", story_data)
            self.assertIn("cross_beat_causal_graph", story_data)
            self.assertIsInstance(story_data["cross_beat_causal_graph"].get("nodes"), list)
            self.assertIsInstance(story_data["cross_beat_causal_graph"].get("edges"), list)
            edges = story_data["cross_beat_causal_graph"]["edges"]
            self.assertTrue(edges)
            self.assertIn("causal_strength", edges[0])
            self.assertGreaterEqual(edges[0]["causal_strength"], 0.0)
            self.assertLessEqual(edges[0]["causal_strength"], 1.0)

            summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("## 因果链总览", summary_text)
            self.assertIn("强度", summary_text)


if __name__ == "__main__":
    unittest.main()
