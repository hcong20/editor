from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib.util
from pathlib import Path
import json
import re
import subprocess
import sys
from typing import Any, Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def probe_duration_seconds(video_path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    return [items[index : index + size] for index in range(0, len(items), size)]


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def discover_sidecar_transcript(video_path: Path) -> Path | None:
    for suffix in (".srt", ".vtt", ".txt"):
        candidate = video_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def parse_sidecar_transcript(path: Path, fallback_duration: float) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return parse_srt(text)
    if suffix == ".vtt":
        cleaned = "\n".join(line for line in text.splitlines() if line.strip() != "WEBVTT")
        return parse_srt(cleaned.replace(".", ","))
    if suffix == ".txt":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []
        step = fallback_duration / len(lines)
        output = []
        for index, line in enumerate(lines):
            start = index * step
            end = min(fallback_duration, start + step)
            output.append({"start": start, "end": end, "text": line, "speaker": None})
        return output
    return []


def parse_srt(text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", text.strip())
    output: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timestamp_line = lines[1] if "-->" in lines[1] else lines[0]
        if "-->" not in timestamp_line:
            continue
        start_text, end_text = [part.strip() for part in timestamp_line.split("-->")]
        start = parse_timestamp(start_text)
        end = parse_timestamp(end_text)
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        payload = " ".join(text_lines).strip()
        if not payload:
            continue
        speaker = None
        if ":" in payload[:24]:
            possible_speaker, maybe_text = payload.split(":", 1)
            if len(possible_speaker) <= 20:
                speaker = possible_speaker.strip()
                payload = maybe_text.strip()
        output.append({"start": start, "end": end, "text": payload, "speaker": speaker})
    return output


def parse_timestamp(value: str) -> float:
    normalized = value.replace(",", ".")
    hours, minutes, seconds = normalized.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_seconds(value: float) -> str:
    total = int(max(0, round(value)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def first_non_empty(items: Iterable[str], fallback: str) -> str:
    for item in items:
        if item.strip():
            return item.strip()
    return fallback


def trim_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
