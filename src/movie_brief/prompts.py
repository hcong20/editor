from __future__ import annotations

import json


def build_scene_analysis_prompt(
    scene_id: str,
    start_seconds: float,
    end_seconds: float,
    scene_index: int,
    total_scenes: int,
    transcript_excerpt: str,
) -> str:
    transcript_text = transcript_excerpt or "这一段没有可用对白，更多依赖画面叙事。"
    return f"""请分析这个电影场景，并严格输出 JSON。

场景信息：
- scene_id: {scene_id}
- 时间范围: {start_seconds:.2f}s 到 {end_seconds:.2f}s
- 所在位置: 第 {scene_index} / {total_scenes} 个 scene
- transcript_excerpt: {transcript_text}

请综合画面和对白，输出：
- summary: 1 到 2 句中文总结，突出人物处境和剧情作用
- events: 2 到 4 个关键事件
- characters: 0 到 5 个出现或明显被提及的人物
- visual_cues: 2 到 5 个重要视觉线索，例如场景、动作、构图、道具
- emotion_intensity: 0 到 1
- core_character_score: 0 到 1
- conflict_score: 0 到 1
- plot_progression_score: 0 到 1

评分规则：
- emotion_intensity：情绪波动和压迫感有多强
- core_character_score：这一段和主角线/关键角色线有多相关
- conflict_score：对抗、危险、误解、背叛、决断有多强
- plot_progression_score：这一段对剧情推进和因果链的贡献有多大

如果信息不足，也要给出谨慎判断，不要留空字段。"""


def scene_analysis_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "events": {
                "type": "array",
                "items": {"type": "string"},
            },
            "characters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "visual_cues": {
                "type": "array",
                "items": {"type": "string"},
            },
            "emotion_intensity": {"type": "number"},
            "core_character_score": {"type": "number"},
            "conflict_score": {"type": "number"},
            "plot_progression_score": {"type": "number"},
        },
        "required": [
            "summary",
            "events",
            "characters",
            "visual_cues",
            "emotion_intensity",
            "core_character_score",
            "conflict_score",
            "plot_progression_score",
        ],
    }



def build_story_beats_prompt(scene_payload: list[dict]) -> str:
    serialized = json.dumps(scene_payload, ensure_ascii=False, indent=2)
    return f"""你是专业电影编剧顾问，负责把 scene 列表重建为故事结构。

目标：
- 输出稳定可用的四段式结构：背景介绍 / 冲突爆发 / 高潮对抗 / 结局收束
- 明确识别一个高潮锚点 scene
- 提取关键因果链，说明“哪一段推动了哪一段”

硬性要求：
- 只输出 JSON
- scene_ids 必须来自输入数据，不要虚构
- summary 不是流水账，必须强调人物动机、冲突升级和因果关系
- causal_chain 里每条都要写成清晰的“原因 -> 结果”
- importance 范围是 0 到 4

可用 scene 数据（按时间顺序）：
{serialized}
"""


def story_beats_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "beat_id": {"type": "string"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "scene_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "importance": {"type": "number"},
                        "causal_chain": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "climax_scene_id": {"type": "string"},
                    },
                    "required": [
                        "beat_id",
                        "title",
                        "summary",
                        "scene_ids",
                        "importance",
                        "causal_chain",
                    ],
                },
            },
            "climax": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["scene_id", "reason"],
            },
            "causal_chain": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cause_scene_id": {"type": "string"},
                        "effect_scene_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["cause_scene_id", "effect_scene_id", "reason"],
                },
            },
        },
        "required": ["beats"],
    }


def build_bilibili_script_prompt(
    story_payload: list[dict],
    scene_payload: list[dict],
    max_chars: int,
    tone: str,
) -> str:
    story_serialized = json.dumps(story_payload, ensure_ascii=False, indent=2)
    scene_serialized = json.dumps(scene_payload, ensure_ascii=False, indent=2)
    return f"""你是一个成熟的 B 站电影解说博主，要写一篇适合中文口播的视频文案。

目标：
- 用 10 到 20 分钟能讲完的节奏，把电影主线讲清楚
- 风格要像优秀电影解说视频，而不是 AI 摘要
- 强调人物动机、因果链、冲突升级、转折和高潮
- 用口语化、有画面感的中文，但不要堆砌夸张空话
- 不要逐句复述对白，也不要机械罗列 scene
- 首段要迅速建立悬念或核心冲突
- 每一段都要让听众明白“为什么事情会发展到下一步”

硬性要求：
- 总字数不超过 {max_chars}
- 语言风格：{tone}
- 输出 4 段左右，对应“背景介绍 / 冲突爆发 / 高潮对抗 / 结局收束”
- narration 必须是可直接配音的自然中文
- scene_ids 要尽量引用最相关的 scene

故事结构数据：
{story_serialized}

可用场景数据：
{scene_serialized}
"""


def script_segments_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "narration": {"type": "string"},
                        "scene_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "target_seconds": {"type": "integer"},
                    },
                    "required": ["title", "narration", "scene_ids", "target_seconds"],
                },
            }
        },
        "required": ["segments"],
    }
