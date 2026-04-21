from __future__ import annotations

from movie_brief.config import PipelineConfig
from movie_brief.models import Scene, StoryBeat


class SceneSelector:
    def select(
        self,
        scenes: list[Scene],
        beats: list[StoryBeat],
        config: PipelineConfig,
    ) -> list[str]:
        raise NotImplementedError


class HeuristicSceneSelector(SceneSelector):
    def select(
        self,
        scenes: list[Scene],
        beats: list[StoryBeat],
        config: PipelineConfig,
    ) -> list[str]:
        if not scenes:
            return []

        scene_map = {scene.scene_id: scene for scene in scenes}
        causal_scores = _scene_causal_scores(scenes, beats)
        selected: dict[str, Scene] = {}
        target_count = max(
            len(beats) * config.compression.coverage_per_arc,
            int(len(scenes) * config.compression.selection_ratio),
        )

        for beat in beats:
            ranked = sorted(
                (scene_map[scene_id] for scene_id in beat.scene_ids if scene_id in scene_map),
                key=lambda scene: (
                    _scene_priority(scene, causal_scores),
                    scene.importance_score,
                ),
                reverse=True,
            )
            for scene in ranked[: config.compression.coverage_per_arc]:
                selected[scene.scene_id] = scene

        if len(selected) < target_count:
            for scene in sorted(
                scenes,
                key=lambda item: (
                    _scene_priority(item, causal_scores),
                    item.importance_score,
                ),
                reverse=True,
            ):
                selected[scene.scene_id] = scene
                if len(selected) >= target_count:
                    break

        return [
            scene.scene_id
            for scene in sorted(selected.values(), key=lambda item: item.window.start)
        ]


def _scene_priority(scene: Scene, causal_scores: dict[str, float]) -> float:
    importance_norm = max(0.0, min(1.0, scene.importance_score / 4.0))
    causal = max(0.0, min(1.0, causal_scores.get(scene.scene_id, 0.0)))
    return round(importance_norm * 0.68 + causal * 0.32, 6)


def _scene_causal_scores(
    scenes: list[Scene],
    beats: list[StoryBeat],
) -> dict[str, float]:
    scene_map = {scene.scene_id: scene for scene in scenes}
    raw_scores: dict[str, float] = {}

    for beat in beats:
        if beat.climax_scene_id and beat.climax_scene_id in scene_map:
            raw_scores[beat.climax_scene_id] = raw_scores.get(beat.climax_scene_id, 0.0) + 0.35

        for link in beat.causal_chain:
            parsed = _parse_causal_link(link)
            if parsed is None:
                continue
            cause_scene_id, effect_scene_id = parsed
            strength = _scene_link_strength(
                scene_map.get(cause_scene_id),
                scene_map.get(effect_scene_id),
            )
            raw_scores[cause_scene_id] = raw_scores.get(cause_scene_id, 0.0) + strength * 0.42
            raw_scores[effect_scene_id] = raw_scores.get(effect_scene_id, 0.0) + strength * 0.58

    if not raw_scores:
        return {}

    max_value = max(raw_scores.values())
    if max_value <= 0:
        return {}

    return {
        scene_id: round(min(1.0, value / max_value), 4)
        for scene_id, value in raw_scores.items()
    }


def _parse_causal_link(link: str) -> tuple[str, str] | None:
    if not isinstance(link, str):
        return None
    head, separator, _ = link.partition(":")
    if not separator:
        return None
    left, arrow, right = head.partition("->")
    if not arrow:
        return None
    cause_scene_id = left.strip()
    effect_scene_id = right.strip()
    if not cause_scene_id or not effect_scene_id:
        return None
    return cause_scene_id, effect_scene_id


def _scene_link_strength(cause: Scene | None, effect: Scene | None) -> float:
    if cause is None or effect is None:
        return 0.35

    conflict_gain = max(0.0, effect.conflict_score - cause.conflict_score)
    progression_gain = max(0.0, effect.plot_progression_score - cause.plot_progression_score)
    raw_score = conflict_gain * 0.65 + progression_gain * 0.35
    return round(min(1.0, 0.2 + raw_score * 1.9), 3)


def build_scene_selector(provider: str) -> SceneSelector:
    if provider in {"heuristic", "stub"}:
        return HeuristicSceneSelector()
    raise ValueError(
        f"Unsupported selector provider: {provider}. "
        "Please implement the provider in stages/selection.py."
    )

