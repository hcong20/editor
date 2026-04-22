from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_brief.config import PipelineConfig
from movie_brief.models import ClipPlan, RenderPlan, ScriptSegment
from movie_brief.stages.delivery import (
    AutoTTSEngine,
    HeuristicSubtitleEngine,
    StubTTSEngine,
    _clean_narration_text,
    _build_variant_narration,
    _resolved_tts_provider_name,
    _resolve_variant_clips,
    _should_force_text_subtitles,
)


class DeliveryStageTests(unittest.TestCase):
    def test_stub_tts_generates_silent_wav(self) -> None:
        config = PipelineConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "narration.mp3"
            audio_path = StubTTSEngine().synthesize(
                text="这是一个测试配音。",
                output_path=output_path,
                config=config,
            )
            self.assertEqual(audio_path.suffix, ".wav")
            self.assertTrue(audio_path.exists())
            self.assertGreater(audio_path.stat().st_size, 44)

    def test_heuristic_subtitle_engine_generates_srt(self) -> None:
        config = PipelineConfig()
        config.subtitles.max_chars_per_line = 8
        config.subtitles.min_seconds_per_line = 0.8

        segments = [
            ScriptSegment(
                segment_id="segment_01",
                title="背景介绍",
                narration="第一段",
                scene_ids=["scene_001"],
                target_seconds=10,
            ),
            ScriptSegment(
                segment_id="segment_02",
                title="冲突爆发",
                narration="第二段",
                scene_ids=["scene_002"],
                target_seconds=12,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            subtitle_path = Path(tmpdir) / "narration.srt"
            output = HeuristicSubtitleEngine().build(
                media_path=None,
                narration_text="故事开始看似平静。冲突很快爆发。主角被迫做出选择。",
                script_segments=segments,
                output_path=subtitle_path,
                config=config,
            )
            content = output.read_text(encoding="utf-8")
            self.assertTrue(output.exists())
            self.assertIn("00:00:00,000 -->", content)
            self.assertIn("冲突", content)

    def test_variant_clip_resolution_falls_back_to_commentary(self) -> None:
        clip_a = ClipPlan(
            clip_id="clip_001",
            scene_id="scene_001",
            start=0.0,
            end=10.0,
            narration_hint="背景",
        )
        clip_b = ClipPlan(
            clip_id="clip_002",
            scene_id="scene_002",
            start=10.0,
            end=20.0,
            narration_hint="冲突",
        )
        render_plan = RenderPlan(
            clips=[clip_a, clip_b],
            variants={"commentary_10m": ["clip_002"]},
        )

        selected = _resolve_variant_clips(render_plan, variant="missing_variant")

        self.assertEqual([item.clip_id for item in selected], ["clip_002"])

    def test_build_variant_narration_prefers_selected_scenes(self) -> None:
        segments = [
            ScriptSegment(
                segment_id="segment_01",
                title="背景介绍",
                narration="开场说明。",
                scene_ids=["scene_001"],
                target_seconds=12,
            ),
            ScriptSegment(
                segment_id="segment_02",
                title="高潮对抗",
                narration="关键反转。",
                scene_ids=["scene_003"],
                target_seconds=20,
            ),
        ]
        clips = [
            ClipPlan(
                clip_id="clip_003",
                scene_id="scene_003",
                start=30.0,
                end=40.0,
                narration_hint="高潮",
            )
        ]

        narration = _build_variant_narration(segments, clips)

        self.assertIn("关键反转", narration)
        self.assertNotIn("开场说明", narration)

    def test_clean_narration_removes_meta_instructions(self) -> None:
        raw = (
            "背景介绍这部分要用更通俗的方式讲。"
            "真正的故事从这场误会开始。"
            "这一段的重要性分数达到 1.23。"
            "因果链：scene_001 -> scene_002: 冲突升级。"
            "主角被迫做出选择。"
        )

        cleaned = _clean_narration_text(raw)

        self.assertIn("真正的故事从这场误会开始", cleaned)
        self.assertIn("主角被迫做出选择", cleaned)
        self.assertNotIn("重要性分数", cleaned)
        self.assertNotIn("因果链", cleaned)
        self.assertNotIn("这部分要用更", cleaned)

    def test_resolved_tts_provider_name_uses_auto_last_provider(self) -> None:
        engine = AutoTTSEngine()
        engine.last_provider = "macos-say"

        provider_name = _resolved_tts_provider_name(engine)

        self.assertEqual(provider_name, "macos-say")

    def test_force_text_subtitles_for_stub_and_macos_say(self) -> None:
        self.assertTrue(_should_force_text_subtitles("stub"))
        self.assertTrue(_should_force_text_subtitles("macos-say"))
        self.assertFalse(_should_force_text_subtitles("edge-tts"))


if __name__ == "__main__":
    unittest.main()
