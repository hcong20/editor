from __future__ import annotations

from pathlib import Path

from movie_brief.config import PipelineConfig
from movie_brief.models import ClipPlan, RenderPlan, Scene, ScriptSegment


class RenderPlanner:
    def plan(
        self,
        video_path: Path,
        scenes: list[Scene],
        segments: list[ScriptSegment],
        selected_scene_ids: list[str],
        config: PipelineConfig,
    ) -> RenderPlan:
        scene_map = {scene.scene_id: scene for scene in scenes}
        segment_map = {
            scene_id: segment.title
            for segment in segments
            for scene_id in segment.scene_ids
        }

        clips: list[ClipPlan] = []
        for index, scene_id in enumerate(selected_scene_ids, start=1):
            scene = scene_map[scene_id]
            clips.append(
                ClipPlan(
                    clip_id=f"clip_{index:03d}",
                    scene_id=scene_id,
                    start=scene.window.start,
                    end=scene.window.end,
                    narration_hint=segment_map.get(scene_id, "主体叙事"),
                )
            )

        variants = self._build_variants(clips, config)
        ffmpeg_commands = self._build_ffmpeg_commands(video_path, clips)
        return RenderPlan(clips=clips, variants=variants, ffmpeg_commands=ffmpeg_commands)

    def _build_variants(
        self,
        clips: list[ClipPlan],
        config: PipelineConfig,
    ) -> dict[str, list[str]]:
        if not clips:
            return {"commentary_10m": [], "short_3m": [], "teaser_1m": []}

        short_count = max(1, int(len(clips) * config.compression.short_ratio))
        teaser_count = max(1, int(len(clips) * config.compression.teaser_ratio))
        return {
            "commentary_10m": [clip.clip_id for clip in clips],
            "short_3m": [clip.clip_id for clip in clips[:short_count]],
            "teaser_1m": [clip.clip_id for clip in clips[:teaser_count]],
        }

    def _build_ffmpeg_commands(self, video_path: Path, clips: list[ClipPlan]) -> list[str]:
        commands = []
        for clip in clips:
            commands.append(
                " ".join(
                    [
                        "ffmpeg",
                        "-y",
                        f"-ss {clip.start:.3f}",
                        f"-to {clip.end:.3f}",
                        f'-i "{video_path}"',
                        "-c:v libx264",
                        "-c:a aac",
                        f'"exports/{clip.clip_id}.mp4"',
                    ]
                )
            )
        return commands

