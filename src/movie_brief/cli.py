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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig.load(args.config)
    pipeline = MovieCommentaryPipeline(config)
    pipeline.run(args.input, args.output)

    print(f"Pipeline completed. Results are in: {args.output}")
    return 0

