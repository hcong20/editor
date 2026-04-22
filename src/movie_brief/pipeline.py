from __future__ import annotations

from pathlib import Path
from typing import Any

from movie_brief.config import PipelineConfig
from movie_brief.models import (
    DeliveryResult,
    RenderPlan,
    Scene,
    ScriptSegment,
    Shot,
    StoryBeat,
    Utterance,
)
from movie_brief.stages.asr import build_asr_engine
from movie_brief.stages.delivery import build_delivery_orchestrator
from movie_brief.stages.render_plan import RenderPlanner
from movie_brief.stages.scene_understanding import build_scene_understanding_engine
from movie_brief.stages.script_generation import build_script_generator
from movie_brief.stages.selection import build_scene_selector
from movie_brief.stages.shot_detection import build_shot_detector
from movie_brief.stages.story_modeling import build_story_modeler
from movie_brief.utils import ensure_dir, format_seconds, write_json, write_text


class MovieCommentaryPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.shot_detector = build_shot_detector(config.providers.shot_detector)
        self.asr_engine = build_asr_engine(config.providers.asr)
        self.scene_engine = build_scene_understanding_engine(config.providers.vision)
        self.story_modeler = build_story_modeler(config.providers.story)
        self.script_generator = build_script_generator(config.providers.script)
        self.selector = build_scene_selector(config.providers.selector)
        self.render_planner = RenderPlanner()
        self.delivery = build_delivery_orchestrator(
            config.providers.video_editor,
            config.providers.tts,
            config.providers.subtitles,
        )

    def run(self, input_video: Path, output_dir: Path) -> None:
        if not input_video.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_video}")
        ensure_dir(output_dir)
        scene_frames_dir = ensure_dir(output_dir / "scene_frames")

        shots = self.shot_detector.detect(input_video, self.config)
        write_json(output_dir / "01_shots.json", shots)

        transcript = self.asr_engine.transcribe(input_video, shots, self.config)
        write_json(output_dir / "02_transcript.json", transcript)

        scenes = self.scene_engine.analyze(
            input_video,
            shots,
            transcript,
            self.config,
            scene_frames_dir,
        )
        write_json(output_dir / "03_scenes.json", [_scene_payload(scene) for scene in scenes])

        beats = self.story_modeler.build(scenes, self.config)
        write_json(output_dir / "04_story_beats.json", _story_beats_payload(beats, scenes))

        script_segments = self.script_generator.generate(beats, scenes, self.config)
        write_json(output_dir / "05_script.json", script_segments)

        selected_scene_ids = self.selector.select(scenes, beats, self.config)
        write_json(output_dir / "06_selected_scenes.json", selected_scene_ids)

        render_plan = self.render_planner.plan(
            input_video,
            scenes,
            script_segments,
            selected_scene_ids,
            self.config,
        )
        write_json(output_dir / "07_render_plan.json", render_plan)

        delivery_result: DeliveryResult | None = None
        if self.config.delivery.enabled:
            delivery_result = self.delivery.deliver(
                input_video=input_video,
                output_dir=output_dir,
                render_plan=render_plan,
                script_segments=script_segments,
                config=self.config,
            )
            write_json(output_dir / "08_delivery.json", delivery_result)

        summary = _build_summary(
            input_video=input_video,
            config=self.config,
            shots=shots,
            transcript=transcript,
            scenes=scenes,
            beats=beats,
            script_segments=script_segments,
            render_plan=render_plan,
            delivery_result=delivery_result,
        )
        write_text(output_dir / "summary.md", summary)


def _scene_payload(scene: Scene) -> dict:
    return {
        "scene_id": scene.scene_id,
        "shot_ids": scene.shot_ids,
        "window": {
            "start": scene.window.start,
            "end": scene.window.end,
            "duration": scene.window.duration,
        },
        "summary": scene.summary,
        "events": scene.events,
        "characters": scene.characters,
        "visual_cues": scene.visual_cues,
        "emotion_intensity": scene.emotion_intensity,
        "core_character_score": scene.core_character_score,
        "conflict_score": scene.conflict_score,
        "plot_progression_score": scene.plot_progression_score,
        "importance_score": scene.importance_score,
        "transcript_excerpt": scene.transcript_excerpt,
        "frame_paths": scene.frame_paths,
    }


def _build_summary(
    input_video: Path,
    config: PipelineConfig,
    shots: list[Shot],
    transcript: list[Utterance],
    scenes: list[Scene],
    beats: list[StoryBeat],
    script_segments: list[ScriptSegment],
    render_plan: RenderPlan,
    delivery_result: DeliveryResult | None,
) -> str:
    beat_lines = []
    for beat in beats:
        beat_lines.append(f"- {beat.title}: {beat.summary}")

    causal_overview_lines = _causal_overview_lines(beats, scenes)

    segment_lines = []
    for segment in script_segments:
        segment_lines.append(
            f"- {segment.title} ({segment.target_seconds}s): {segment.narration}"
        )

    delivery_lines = ["- 未启用自动交付"]
    if delivery_result is not None:
        delivery_lines = [
            f"- 交付版本: {delivery_result.variant}",
            f"- 导出 clip 数: {len(delivery_result.clip_paths)}",
            f"- 拼接视频: {delivery_result.concat_video_path or '暂无'}",
            f"- 配音音频: {delivery_result.narration_audio_path or '暂无'}",
            f"- 字幕文件: {delivery_result.subtitle_path or '暂无'}",
            f"- 最终视频: {delivery_result.final_video_path or '暂无'}",
        ]

    selected_ids = ", ".join(clip.scene_id for clip in render_plan.clips)
    return f"""# 运行摘要

## 输入

- 视频: {input_video}
- 目标时长: {config.project.target_minutes} 分钟
- 语言: {config.project.language}

## 统计

- Shot 数量: {len(shots)}
- Utterance 数量: {len(transcript)}
- Scene 数量: {len(scenes)}
- Story Beat 数量: {len(beats)}
- 选中片段数量: {len(render_plan.clips)}

## 故事结构

{chr(10).join(beat_lines) if beat_lines else "- 暂无"}

## 因果链总览

{chr(10).join(causal_overview_lines) if causal_overview_lines else "- 暂无"}

## 解说稿

{chr(10).join(segment_lines) if segment_lines else "- 暂无"}

## 剪辑计划

- 10 分钟版 clip 数: {len(render_plan.variants.get("commentary_10m", []))}
- 3 分钟版 clip 数: {len(render_plan.variants.get("short_3m", []))}
- 1 分钟版 clip 数: {len(render_plan.variants.get("teaser_1m", []))}
- 选中 scene: {selected_ids or "暂无"}

## 时间轴提示

- 首个 shot 起点: {format_seconds(shots[0].window.start) if shots else "00:00:00"}
- 最后 shot 终点: {format_seconds(shots[-1].window.end) if shots else "00:00:00"}

## 自动交付

{chr(10).join(delivery_lines)}
"""


def _story_beats_payload(beats: list[StoryBeat], scenes: list[Scene]) -> dict[str, Any]:
    return {
        "beats": beats,
        "cross_beat_causal_graph": _cross_beat_causal_graph(beats, scenes),
    }


def _cross_beat_causal_graph(
    beats: list[StoryBeat],
    scenes: list[Scene],
) -> dict[str, Any]:
    nodes = [
        {
            "beat_id": beat.beat_id,
            "title": beat.title,
            "scene_ids": beat.scene_ids,
            "importance": beat.importance,
            "climax_scene_id": beat.climax_scene_id,
        }
        for beat in beats
    ]
    beat_position = {beat.beat_id: index for index, beat in enumerate(beats)}
    beat_titles = {beat.beat_id: beat.title for beat in beats}
    scene_to_beat = {
        scene_id: beat.beat_id
        for beat in beats
        for scene_id in beat.scene_ids
    }
    scene_map = {scene.scene_id: scene for scene in scenes}

    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for beat in beats:
        for link in beat.causal_chain:
            parsed = _parse_causal_link(link)
            if parsed is None:
                continue
            cause_scene_id, effect_scene_id, reason = parsed
            source = scene_to_beat.get(cause_scene_id)
            target = scene_to_beat.get(effect_scene_id)
            if not source or not target or source == target:
                continue
            key = (source, target)
            edge = edge_map.setdefault(
                key,
                {
                    "from_beat_id": source,
                    "from_title": beat_titles.get(source, source),
                    "to_beat_id": target,
                    "to_title": beat_titles.get(target, target),
                    "reasons": [],
                    "evidence_links": [],
                    "strength_samples": [],
                },
            )
            if reason and reason not in edge["reasons"]:
                edge["reasons"].append(reason)
            if link not in edge["evidence_links"]:
                edge["evidence_links"].append(link)
            edge["strength_samples"].append(
                _scene_link_strength(
                    scene_map.get(cause_scene_id),
                    scene_map.get(effect_scene_id),
                )
            )

    if not edge_map and len(beats) > 1:
        for left, right in zip(beats, beats[1:]):
            key = (left.beat_id, right.beat_id)
            left_tail = left.scene_ids[-1] if left.scene_ids else None
            right_head = right.scene_ids[0] if right.scene_ids else None
            edge_map[key] = {
                "from_beat_id": left.beat_id,
                "from_title": left.title,
                "to_beat_id": right.beat_id,
                "to_title": right.title,
                "reasons": [f"{left.title}推动剧情进入{right.title}"],
                "evidence_links": [],
                "strength_samples": [
                    _scene_link_strength(
                        scene_map.get(left_tail) if left_tail else None,
                        scene_map.get(right_head) if right_head else None,
                    )
                ],
            }

    ordered_edges: list[dict[str, Any]] = []
    for key, edge in sorted(
        edge_map.items(),
        key=lambda item: (
            beat_position.get(item[0][0], 10_000),
            beat_position.get(item[0][1], 10_000),
        ),
    ):
        edge["weight"] = max(1, len(edge["evidence_links"]))
        strength_samples = edge.pop("strength_samples", [])
        if isinstance(strength_samples, list) and strength_samples:
            edge["causal_strength"] = round(
                sum(float(item) for item in strength_samples) / len(strength_samples),
                3,
            )
        else:
            edge["causal_strength"] = 0.35
        ordered_edges.append(edge)

    return {
        "nodes": nodes,
        "edges": ordered_edges,
    }


def _causal_overview_lines(beats: list[StoryBeat], scenes: list[Scene]) -> list[str]:
    graph = _cross_beat_causal_graph(beats, scenes)
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        return []

    lines: list[str] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("from_title") or edge.get("from_beat_id") or "上一段").strip()
        target = str(edge.get("to_title") or edge.get("to_beat_id") or "下一段").strip()
        reasons = edge.get("reasons")
        if isinstance(reasons, list) and reasons:
            reason = str(reasons[0]).strip()
        else:
            reason = "剧情因果继续推进"
        strength = edge.get("causal_strength")
        if isinstance(strength, (int, float)):
            lines.append(f"- {source} -> {target} (强度 {float(strength):.3f}): {reason}")
        else:
            lines.append(f"- {source} -> {target}: {reason}")

    climax_beat = next((beat for beat in beats if beat.climax_scene_id), None)
    if climax_beat:
        lines.append(
            f"- 高潮锚点位于 {climax_beat.title}（{climax_beat.climax_scene_id}）"
        )
    return lines


def _parse_causal_link(link: str) -> tuple[str, str, str] | None:
    if not isinstance(link, str):
        return None
    head, separator, tail = link.partition(":")
    if not separator:
        return None
    left, arrow, right = head.partition("->")
    if not arrow:
        return None
    cause_scene_id = left.strip()
    effect_scene_id = right.strip()
    reason = tail.strip()
    if not cause_scene_id or not effect_scene_id:
        return None
    return cause_scene_id, effect_scene_id, reason


def _scene_link_strength(cause: Scene | None, effect: Scene | None) -> float:
    if cause is None or effect is None:
        return 0.35

    conflict_gain = max(0.0, effect.conflict_score - cause.conflict_score)
    progression_gain = max(0.0, effect.plot_progression_score - cause.plot_progression_score)
    raw_score = conflict_gain * 0.65 + progression_gain * 0.35
    return round(min(1.0, 0.2 + raw_score * 1.9), 3)
