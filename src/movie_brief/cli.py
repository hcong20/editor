from __future__ import annotations

import argparse
from pathlib import Path

from movie_brief.config import PipelineConfig
from movie_brief.pipeline import MovieCommentaryPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the automatic movie commentary MVP pipeline."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the movie file.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional TOML config path.",
    )
    parser.add_argument(
        "--deliver",
        action="store_true",
        help="Enable automatic delivery (cut + TTS + subtitles).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Delivery variant to render, e.g. commentary_10m.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig.load(args.config)
    if args.deliver:
        config.delivery.enabled = True
    if args.variant:
        config.delivery.variant = args.variant
    pipeline = MovieCommentaryPipeline(config)
    pipeline.run(args.input, args.output)

    print(f"Pipeline completed. Results are in: {args.output}")
    return 0

