from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any
from urllib import error, request
import wave

from movie_brief.config import PipelineConfig
from movie_brief.models import ClipPlan, DeliveryResult, RenderPlan, ScriptSegment
from movie_brief.utils import ensure_dir, module_available, probe_duration_seconds, write_text


class VideoEditor:
    def export_variant(
        self,
        input_video: Path,
        clips: list[ClipPlan],
        output_dir: Path,
        config: PipelineConfig,
    ) -> tuple[list[str], str | None]:
        raise NotImplementedError


class FfmpegVideoEditor(VideoEditor):
    def export_variant(
        self,
        input_video: Path,
        clips: list[ClipPlan],
        output_dir: Path,
        config: PipelineConfig,
    ) -> tuple[list[str], str | None]:
        _require_command("ffmpeg")
        exports_dir = ensure_dir(output_dir / "exports")
        clip_paths: list[str] = []

        for clip in clips:
            clip_path = exports_dir / f"{clip.clip_id}.mp4"
            _run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{clip.start:.3f}",
                    "-to",
                    f"{clip.end:.3f}",
                    "-i",
                    str(input_video),
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(clip_path),
                ]
            )
            clip_paths.append(str(clip_path))

        if not clip_paths:
            return [], None

        concat_list_path = output_dir / "concat.txt"
        concat_list_path.write_text(
            "\n".join(_build_concat_line(Path(path)) for path in clip_paths),
            encoding="utf-8",
        )

        concat_path = output_dir / config.delivery.concat_filename
        copy_result = _run_command(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(concat_path),
            ],
            check=False,
        )
        if copy_result.returncode != 0:
            _run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list_path),
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(concat_path),
                ]
            )
        return clip_paths, str(concat_path)


class MoviePyVideoEditor(VideoEditor):
    def __init__(self) -> None:
        self.fallback = FfmpegVideoEditor()

    def export_variant(
        self,
        input_video: Path,
        clips: list[ClipPlan],
        output_dir: Path,
        config: PipelineConfig,
    ) -> tuple[list[str], str | None]:
        if not module_available("moviepy"):
            return self.fallback.export_variant(input_video, clips, output_dir, config)

        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips
        except ImportError:
            try:
                from moviepy import VideoFileClip, concatenate_videoclips
            except ImportError:
                return self.fallback.export_variant(input_video, clips, output_dir, config)

        if not clips:
            return [], None

        exports_dir = ensure_dir(output_dir / "exports")
        clip_paths: list[str] = []

        with VideoFileClip(str(input_video)) as source:
            rendered_clips = []
            for clip in clips:
                part = source.subclip(clip.start, clip.end)
                clip_path = exports_dir / f"{clip.clip_id}.mp4"
                part.write_videofile(
                    str(clip_path),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )
                rendered_clips.append(part)
                clip_paths.append(str(clip_path))

            concat_path = output_dir / config.delivery.concat_filename
            final_clip = concatenate_videoclips(rendered_clips, method="compose")
            final_clip.write_videofile(
                str(concat_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            final_clip.close()
            for item in rendered_clips:
                item.close()

        return clip_paths, str(concat_path)


class CapCutAPIVideoEditor(VideoEditor):
    def __init__(self) -> None:
        self.fallback = FfmpegVideoEditor()

    def export_variant(
        self,
        input_video: Path,
        clips: list[ClipPlan],
        output_dir: Path,
        config: PipelineConfig,
    ) -> tuple[list[str], str | None]:
        api_base = os.getenv("CAPCUT_API_BASE_URL")
        api_key = os.getenv("CAPCUT_API_KEY")
        if not api_base or not api_key:
            return self.fallback.export_variant(input_video, clips, output_dir, config)

        payload = {
            "input_video": str(input_video),
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "start": clip.start,
                    "end": clip.end,
                }
                for clip in clips
            ],
        }
        body = json_dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{api_base.rstrip('/')}/v1/edits",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                parsed = json_loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, ValueError):
            return self.fallback.export_variant(input_video, clips, output_dir, config)

        result_url = parsed.get("result_url") if isinstance(parsed, dict) else None
        if not isinstance(result_url, str) or not result_url.strip():
            return self.fallback.export_variant(input_video, clips, output_dir, config)

        concat_path = output_dir / config.delivery.concat_filename
        download_req = request.Request(result_url, method="GET")
        try:
            with request.urlopen(download_req, timeout=120) as response:
                concat_path.write_bytes(response.read())
        except (error.URLError, error.HTTPError):
            return self.fallback.export_variant(input_video, clips, output_dir, config)

        return [], str(concat_path)


class TTSEngine:
    provider_name = "unknown"

    def synthesize(self, text: str, output_path: Path, config: PipelineConfig) -> Path:
        raise NotImplementedError


class StubTTSEngine(TTSEngine):
    provider_name = "stub"

    def synthesize(self, text: str, output_path: Path, config: PipelineConfig) -> Path:
        target = output_path.with_suffix(".wav")
        duration_seconds = _estimate_narration_seconds(text)
        _write_silence_wav(target, duration_seconds)
        return target


class EdgeTTSEngine(TTSEngine):
    provider_name = "edge-tts"

    def synthesize(self, text: str, output_path: Path, config: PipelineConfig) -> Path:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'edge-tts' requires package 'edge-tts'."
            ) from exc

        target = output_path.with_suffix(".mp3")

        async def _save() -> None:
            communicate = edge_tts.Communicate(
                text=text,
                voice=config.tts.edge_voice,
                rate=config.tts.edge_rate,
            )
            await communicate.save(str(target))

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_save())
            finally:
                loop.close()
        else:
            asyncio.run(_save())
        return target


class ElevenLabsTTSEngine(TTSEngine):
    provider_name = "elevenlabs"

    def synthesize(self, text: str, output_path: Path, config: PipelineConfig) -> Path:
        api_key = os.getenv(config.tts.elevenlabs_api_key_env)
        voice_id = config.tts.elevenlabs_voice_id.strip()
        if not api_key:
            raise RuntimeError(
                f"Missing ElevenLabs API key in env {config.tts.elevenlabs_api_key_env}."
            )
        if not voice_id:
            raise RuntimeError("Config tts.elevenlabs_voice_id cannot be empty.")

        payload = {
            "text": text,
            "model_id": config.tts.elevenlabs_model_id,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.75,
            },
        }
        req = request.Request(
            url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            data=json_dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
                "xi-api-key": api_key,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                audio_bytes = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ElevenLabs request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"ElevenLabs network error: {exc.reason}") from exc

        target = output_path.with_suffix(".mp3")
        target.write_bytes(audio_bytes)
        return target


class AzureTTSEngine(TTSEngine):
    provider_name = "azure"

    def synthesize(self, text: str, output_path: Path, config: PipelineConfig) -> Path:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'azure' requires package 'azure-cognitiveservices-speech'."
            ) from exc

        key = os.getenv(config.tts.azure_key_env)
        region = os.getenv(config.tts.azure_region_env)
        if not key or not region:
            raise RuntimeError(
                f"Missing Azure Speech envs: {config.tts.azure_key_env}/{config.tts.azure_region_env}."
            )

        target = output_path.with_suffix(".mp3")
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.speech_synthesis_voice_name = config.tts.azure_voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(target))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        result = synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            cancel = speechsdk.CancellationDetails.from_result(result)
            raise RuntimeError(
                "Azure TTS failed: "
                f"reason={cancel.reason}, error_details={cancel.error_details}"
            )
        return target


class MacOSSayTTSEngine(TTSEngine):
    provider_name = "macos-say"

    def synthesize(self, text: str, output_path: Path, config: PipelineConfig) -> Path:
        if shutil.which("say") is None:
            raise RuntimeError("Provider 'macos-say' requires the macOS 'say' command.")

        normalized = re.sub(r"\s+", " ", text).strip() or "本片暂无可用解说内容。"
        aiff_path = output_path.with_suffix(".aiff")
        preferred_voice = os.getenv("MOVIE_BRIEF_SAY_VOICE", "Tingting").strip() or "Tingting"
        first_try = _run_command(
            ["say", "-v", preferred_voice, "-o", str(aiff_path), normalized],
            check=False,
        )
        if first_try.returncode != 0:
            _run_command(["say", "-o", str(aiff_path), normalized])

        if shutil.which("ffmpeg"):
            wav_path = output_path.with_suffix(".wav")
            _run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(aiff_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(wav_path),
                ]
            )
            try:
                aiff_path.unlink(missing_ok=True)
            except OSError:
                pass
            return wav_path

        return aiff_path


class AutoTTSEngine(TTSEngine):
    provider_name = "auto"

    def __init__(self) -> None:
        self.edge = EdgeTTSEngine()
        self.elevenlabs = ElevenLabsTTSEngine()
        self.azure = AzureTTSEngine()
        self.macos_say = MacOSSayTTSEngine()
        self.stub = StubTTSEngine()
        self.last_provider = self.stub.provider_name

    def synthesize(self, text: str, output_path: Path, config: PipelineConfig) -> Path:
        if module_available("edge_tts"):
            try:
                audio = self.edge.synthesize(text, output_path, config)
                self.last_provider = self.edge.provider_name
                return audio
            except RuntimeError:
                pass

        if os.getenv(config.tts.elevenlabs_api_key_env) and config.tts.elevenlabs_voice_id.strip():
            try:
                audio = self.elevenlabs.synthesize(text, output_path, config)
                self.last_provider = self.elevenlabs.provider_name
                return audio
            except RuntimeError:
                pass

        if (
            module_available("azure.cognitiveservices.speech")
            and os.getenv(config.tts.azure_key_env)
            and os.getenv(config.tts.azure_region_env)
        ):
            try:
                audio = self.azure.synthesize(text, output_path, config)
                self.last_provider = self.azure.provider_name
                return audio
            except RuntimeError:
                pass

        if shutil.which("say"):
            try:
                audio = self.macos_say.synthesize(text, output_path, config)
                self.last_provider = self.macos_say.provider_name
                return audio
            except RuntimeError:
                pass

        audio = self.stub.synthesize(text, output_path, config)
        self.last_provider = self.stub.provider_name
        return audio


class SubtitleEngine:
    def build(
        self,
        media_path: Path | None,
        narration_text: str,
        script_segments: list[ScriptSegment],
        output_path: Path,
        config: PipelineConfig,
    ) -> Path:
        raise NotImplementedError


class HeuristicSubtitleEngine(SubtitleEngine):
    def build(
        self,
        media_path: Path | None,
        narration_text: str,
        script_segments: list[ScriptSegment],
        output_path: Path,
        config: PipelineConfig,
    ) -> Path:
        chunks = _split_subtitle_chunks(
            narration_text,
            max_chars=config.subtitles.max_chars_per_line,
        )
        if not chunks:
            chunks = ["（无字幕内容）"]

        duration_hint = probe_duration_seconds(media_path) if media_path else None
        if duration_hint is None:
            duration_hint = _estimate_duration_from_segments(script_segments)
        if duration_hint is None:
            duration_hint = _estimate_narration_seconds(narration_text)

        line_duration = max(
            config.subtitles.min_seconds_per_line,
            duration_hint / max(1, len(chunks)),
        )
        records: list[tuple[float, float, str]] = []
        cursor = 0.0
        for text in chunks:
            start = cursor
            end = cursor + line_duration
            records.append((start, end, text))
            cursor = end

        _write_srt(output_path, records)
        return output_path


class WhisperSubtitleEngine(SubtitleEngine):
    def __init__(self) -> None:
        self.fallback = HeuristicSubtitleEngine()

    def build(
        self,
        media_path: Path | None,
        narration_text: str,
        script_segments: list[ScriptSegment],
        output_path: Path,
        config: PipelineConfig,
    ) -> Path:
        if media_path is None or not media_path.exists():
            return self.fallback.build(media_path, narration_text, script_segments, output_path, config)

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'whisper' requires package 'faster-whisper'."
            ) from exc

        model = WhisperModel(
            config.subtitles.whisper_model,
            device=config.subtitles.whisper_device,
            compute_type=config.subtitles.whisper_compute_type,
        )
        segments, _ = model.transcribe(
            str(media_path),
            language=config.project.language,
            vad_filter=True,
            condition_on_previous_text=True,
            beam_size=5,
        )

        records: list[tuple[float, float, str]] = []
        for segment in segments:
            text = str(segment.text).strip()
            if not text:
                continue
            start = float(segment.start)
            end = float(segment.end)
            if end <= start:
                end = start + max(config.subtitles.min_seconds_per_line, 0.3)
            records.append((start, end, text))

        if not records:
            return self.fallback.build(media_path, narration_text, script_segments, output_path, config)

        _write_srt(output_path, records)
        return output_path


class AutoSubtitleEngine(SubtitleEngine):
    def __init__(self) -> None:
        self.whisper = WhisperSubtitleEngine()
        self.heuristic = HeuristicSubtitleEngine()

    def build(
        self,
        media_path: Path | None,
        narration_text: str,
        script_segments: list[ScriptSegment],
        output_path: Path,
        config: PipelineConfig,
    ) -> Path:
        if module_available("faster_whisper"):
            try:
                return self.whisper.build(
                    media_path,
                    narration_text,
                    script_segments,
                    output_path,
                    config,
                )
            except RuntimeError:
                pass
        return self.heuristic.build(
            media_path,
            narration_text,
            script_segments,
            output_path,
            config,
        )


class DeliveryOrchestrator:
    def __init__(
        self,
        video_editor: VideoEditor,
        tts_engine: TTSEngine,
        subtitle_engine: SubtitleEngine,
    ) -> None:
        self.video_editor = video_editor
        self.tts_engine = tts_engine
        self.subtitle_engine = subtitle_engine

    def deliver(
        self,
        input_video: Path,
        output_dir: Path,
        render_plan: RenderPlan,
        script_segments: list[ScriptSegment],
        config: PipelineConfig,
    ) -> DeliveryResult:
        delivery_dir = ensure_dir(output_dir / "delivery")
        selected_clips = _resolve_variant_clips(render_plan, config.delivery.variant)
        clip_paths, concat_video_path = self.video_editor.export_variant(
            input_video=input_video,
            clips=selected_clips,
            output_dir=delivery_dir,
            config=config,
        )

        narration_text = _build_variant_narration(script_segments, selected_clips)
        write_text(delivery_dir / "narration.txt", narration_text)

        narration_audio = self.tts_engine.synthesize(
            narration_text,
            output_path=delivery_dir / "narration.mp3",
            config=config,
        )
        tts_provider = _resolved_tts_provider_name(self.tts_engine)
        if tts_provider == "stub":
            write_text(
                delivery_dir / "tts_warning.txt",
                "TTS provider fell back to 'stub' (silent audio). "
                "Install edge-tts or configure ElevenLabs/Azure credentials for real voice.\n",
            )

        subtitle_engine: SubtitleEngine = self.subtitle_engine
        force_text_subtitles = _should_force_text_subtitles(tts_provider)
        if force_text_subtitles:
            subtitle_engine = HeuristicSubtitleEngine()

        subtitle_media = narration_audio if narration_audio.exists() and not force_text_subtitles else None
        subtitle_path = subtitle_engine.build(
            media_path=subtitle_media,
            narration_text=narration_text,
            script_segments=script_segments,
            output_path=delivery_dir / "narration.srt",
            config=config,
        )

        final_video_path: Path | None = None
        if concat_video_path is not None:
            final_video_path = self._render_final_video(
                concat_video_path=Path(concat_video_path),
                narration_audio_path=narration_audio,
                subtitle_path=subtitle_path,
                output_path=output_dir / config.delivery.final_filename,
                burn_subtitles=config.delivery.burn_subtitles,
            )

        return DeliveryResult(
            variant=config.delivery.variant,
            clip_paths=clip_paths,
            concat_video_path=concat_video_path,
            narration_audio_path=str(narration_audio),
            tts_provider=tts_provider,
            subtitle_path=str(subtitle_path),
            final_video_path=str(final_video_path) if final_video_path else None,
        )

    def _render_final_video(
        self,
        concat_video_path: Path,
        narration_audio_path: Path,
        subtitle_path: Path | None,
        output_path: Path,
        burn_subtitles: bool,
    ) -> Path:
        _require_command("ffmpeg")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(concat_video_path),
            "-i",
            str(narration_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
        ]

        if burn_subtitles and subtitle_path and subtitle_path.exists():
            command.extend(
                [
                    "-vf",
                    f"subtitles={_escape_ffmpeg_filter_path(subtitle_path)}",
                ]
            )

        command.append(str(output_path))
        first_try = _run_command(command, check=False)

        if first_try.returncode != 0 and burn_subtitles and subtitle_path and subtitle_path.exists():
            fallback = [
                "ffmpeg",
                "-y",
                "-i",
                str(concat_video_path),
                "-i",
                str(narration_audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ]
            _run_command(fallback)
            return output_path

        if first_try.returncode != 0:
            detail = (first_try.stderr or first_try.stdout).strip()
            raise RuntimeError(f"Failed to compose final video: {detail}")

        return output_path


def _resolve_variant_clips(render_plan: RenderPlan, variant: str) -> list[ClipPlan]:
    clip_map = {clip.clip_id: clip for clip in render_plan.clips}

    variant_ids = render_plan.variants.get(variant)
    if not isinstance(variant_ids, list) or not variant_ids:
        variant_ids = render_plan.variants.get("commentary_10m")
    if not isinstance(variant_ids, list) or not variant_ids:
        variant_ids = list(clip_map.keys())

    clips = [clip_map[clip_id] for clip_id in variant_ids if clip_id in clip_map]
    if clips:
        return clips
    return render_plan.clips


def _build_variant_narration(
    script_segments: list[ScriptSegment],
    clips: list[ClipPlan],
) -> str:
    selected_scene_ids = {clip.scene_id for clip in clips}
    selected_segments = [
        segment
        for segment in script_segments
        if selected_scene_ids.intersection(segment.scene_ids)
    ]
    if not selected_segments:
        selected_segments = script_segments

    lines = []
    for segment in selected_segments:
        cleaned = _clean_narration_text(segment.narration)
        if cleaned:
            lines.append(cleaned)
    if not lines:
        return "本片暂无可用解说内容。"
    return "\n".join(lines)


def _clean_narration_text(text: str) -> str:
    raw = re.sub(r"\s+", " ", text).strip()
    if not raw:
        return ""

    blocked_patterns = [
        r"这部分要用更.*?方式讲",
        r"这一段的重要性分数达到\s*[0-9.]+",
        r"解说时不要陷进细碎对白.*$",
        r"因果链：[^。]*",
        r"高潮锚点：[^。]*",
        r"核心作用是[^。]*",
    ]

    sentences = re.split(r"(?<=[。！？!?])\s*", raw)
    kept: list[str] = []
    for sentence in sentences:
        trimmed = sentence.strip()
        if not trimmed:
            continue
        if any(re.search(pattern, trimmed) for pattern in blocked_patterns):
            continue
        kept.append(trimmed)

    if kept:
        return "".join(kept)
    return raw


def _resolved_tts_provider_name(engine: TTSEngine) -> str:
    last_provider = getattr(engine, "last_provider", None)
    if isinstance(last_provider, str) and last_provider.strip():
        return last_provider.strip()
    provider_name = getattr(engine, "provider_name", None)
    if isinstance(provider_name, str) and provider_name.strip():
        return provider_name.strip()
    return "unknown"


def _should_force_text_subtitles(tts_provider: str) -> bool:
    normalized = tts_provider.strip().lower()
    return normalized in {"stub", "macos-say"}


def _estimate_duration_from_segments(segments: list[ScriptSegment]) -> float | None:
    if not segments:
        return None
    seconds = sum(max(1, segment.target_seconds) for segment in segments)
    return float(max(1, seconds))


def _estimate_narration_seconds(text: str) -> float:
    characters = len(re.sub(r"\s+", "", text))
    if characters <= 0:
        return 6.0
    return max(6.0, characters / 4.2)


def _split_subtitle_chunks(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        max_chars = 24

    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []

    sentences = re.split(r"(?<=[。！？!?；;])\s*", compact)
    chunks: list[str] = []

    for sentence in sentences:
        pending = sentence.strip()
        if not pending:
            continue
        while len(pending) > max_chars:
            split_index = pending.rfind(" ", 0, max_chars + 1)
            if split_index <= max_chars // 2:
                split_index = max_chars
            head = pending[:split_index].strip()
            if head:
                chunks.append(head)
            pending = pending[split_index:].strip()
        if pending:
            chunks.append(pending)

    return chunks


def _write_srt(path: Path, records: list[tuple[float, float, str]]) -> None:
    lines: list[str] = []
    for index, (start, end, text) in enumerate(records, start=1):
        safe_end = max(end, start + 0.2)
        lines.extend(
            [
                str(index),
                f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(safe_end)}",
                text.strip(),
                "",
            ]
        )
    write_text(path, "\n".join(lines).strip() + "\n")


def _format_srt_timestamp(value: float) -> str:
    millis = max(0, int(round(value * 1000)))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milli = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milli:03d}"


def _write_silence_wav(path: Path, duration_seconds: float) -> None:
    sample_rate = 16_000
    frame_count = max(1, int(sample_rate * max(0.2, duration_seconds)))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def _build_concat_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def _escape_ffmpeg_filter_path(path: Path) -> str:
    return (
        str(path)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )


def _require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(f"Command not found: {command_name}")


def _run_command(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Command failed: {shlex.join(command)}\n{detail}")
    return result


def build_video_editor(provider: str) -> VideoEditor:
    normalized = provider.strip().lower()
    if normalized in {"ffmpeg", "default", "stub"}:
        return FfmpegVideoEditor()
    if normalized in {"moviepy", "movie-py"}:
        return MoviePyVideoEditor()
    if normalized in {"capcut", "capcut-api", "capcut_api"}:
        return CapCutAPIVideoEditor()
    if normalized == "auto":
        if shutil.which("ffmpeg"):
            return FfmpegVideoEditor()
        if module_available("moviepy"):
            return MoviePyVideoEditor()
        return CapCutAPIVideoEditor()
    raise ValueError(
        f"Unsupported video editor provider: {provider}. "
        "Please implement it in stages/delivery.py."
    )


def build_tts_engine(provider: str) -> TTSEngine:
    normalized = provider.strip().lower()
    if normalized in {"stub", "template"}:
        return StubTTSEngine()
    if normalized in {"macos-say", "macos_say", "say"}:
        return MacOSSayTTSEngine()
    if normalized in {"edge", "edge-tts", "edge_tts"}:
        return EdgeTTSEngine()
    if normalized in {"elevenlabs", "11labs"}:
        return ElevenLabsTTSEngine()
    if normalized in {"azure", "azure-tts", "azure_tts"}:
        return AzureTTSEngine()
    if normalized in {"auto", "default"}:
        return AutoTTSEngine()
    raise ValueError(
        f"Unsupported TTS provider: {provider}. "
        "Please implement it in stages/delivery.py."
    )


def build_subtitle_engine(provider: str) -> SubtitleEngine:
    normalized = provider.strip().lower()
    if normalized in {"heuristic", "stub", "template"}:
        return HeuristicSubtitleEngine()
    if normalized in {"whisper", "faster-whisper", "faster_whisper"}:
        return WhisperSubtitleEngine()
    if normalized in {"auto", "default"}:
        return AutoSubtitleEngine()
    raise ValueError(
        f"Unsupported subtitle provider: {provider}. "
        "Please implement it in stages/delivery.py."
    )


def build_delivery_orchestrator(
    video_editor_provider: str,
    tts_provider: str,
    subtitle_provider: str,
) -> DeliveryOrchestrator:
    return DeliveryOrchestrator(
        video_editor=build_video_editor(video_editor_provider),
        tts_engine=build_tts_engine(tts_provider),
        subtitle_engine=build_subtitle_engine(subtitle_provider),
    )


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def json_loads(raw: str) -> Any:
    import json

    return json.loads(raw)
