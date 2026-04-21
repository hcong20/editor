from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
import subprocess

from movie_brief.utils import ensure_dir


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/jpeg"


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def image_to_data_url(path: Path) -> str:
    return f"data:{guess_mime_type(path)};base64,{image_to_base64(path)}"


def evenly_spaced_timestamps(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    duration = max(0.0, end - start)
    if duration <= 0.0:
        return [round(max(0.0, start), 3)]
    return [
        round(start + duration * ((index + 1) / (count + 1)), 3)
        for index in range(count)
    ]


def extract_representative_frames(
    video_path: Path,
    start: float,
    end: float,
    output_dir: Path,
    scene_id: str,
    frame_count: int,
    image_max_width: int,
    image_quality: int,
) -> list[Path]:
    ensure_dir(output_dir)
    paths: list[Path] = []

    for index, timestamp in enumerate(
        evenly_spaced_timestamps(start, end, frame_count),
        start=1,
    ):
        output_path = output_dir / f"{scene_id}_{index:02d}.jpg"
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
        ]
        if image_max_width > 0:
            command.extend(["-vf", f"scale=min({image_max_width}\\,iw):-2"])
        command.extend(["-q:v", str(max(1, image_quality)), str(output_path)])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return []

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            paths.append(output_path)

    return paths
