"""Command-line interface: `redlight play|benchmark|make-audio|build-site`.

Every flag lives on its subparser, never on the top-level `redlight` parser
— `redlight play --threshold 0.2` and a hypothetical `redlight --threshold
0.2 play` would otherwise silently mean different things depending on
argparse's own precedence rules. Keeping the top-level parser flag-free
(besides the built-in `-h`/`--help`) makes that ambiguity impossible by
construction; `tests/test_cli.py` introspects the built parser to enforce
it, so a future flag added in the wrong place fails loudly instead of
quietly shadowing a subcommand's own.

`benchmark` and `build-site` import their backing modules lazily, inside
their handlers rather than at the top of this file: `benchmark.py` and
`export.py` land in later work, and importing either at module load time
would break every other subcommand (including `--help`) until then. The
lazy import also means a user who never runs those two subcommands is
never asked to pay for whatever those modules pull in (matplotlib,
scikit-learn, ...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from redlight import app, synth
from redlight.app import MatchReport
from redlight.config import ConfigError, GameConfig
from redlight.sources import SourceError

DEFAULT_BENCHMARK_OUT = "reports/benchmark_results.json"
DEFAULT_SITE_OUT = "docs/index.html"
DEFAULT_AUDIO_OUT = "assets/doll_song.wav"


def _parse_source(raw: str) -> str | int:
    """A CLI `--source` value: a plain integer string is a webcam index,
    anything else is passed through as a file path (or stream URL).
    """
    try:
        return int(raw)
    except ValueError:
        return raw


def _format_summary(report: MatchReport) -> str:
    """One line covering how a match ended, for `redlight play`'s stdout."""
    eliminated = len(report.eliminations)
    return (
        f"{report.outcome} - {report.players} player(s), "
        f"{report.survivors} survived, {eliminated} eliminated"
    )


def _add_play_arguments(parser: argparse.ArgumentParser) -> None:
    """Register every `GameConfig` knob plus the run-time flags `play` needs."""
    defaults = GameConfig()

    parser.add_argument(
        "--source",
        default="0",
        help="Video file path, stream URL, or webcam index (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.seed, help="Random seed; omit for a non-deterministic match."
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=defaults.countdown_s,
        dest="countdown_s",
        metavar="S",
        help="Countdown length in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=defaults.duration_s,
        dest="duration_s",
        metavar="S",
        help="Total match length in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--phase-min",
        type=float,
        default=defaults.phase_min_s,
        dest="phase_min_s",
        metavar="S",
        help="Shortest red/green phase, in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--phase-max",
        type=float,
        default=defaults.phase_max_s,
        dest="phase_max_s",
        metavar="S",
        help="Longest red/green phase, in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=defaults.grace_s,
        dest="grace_s",
        metavar="S",
        help="Grace window after red light starts, in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--metric",
        choices=["flow", "diff"],
        default=defaults.metric,
        help="Motion metric: optical flow or frame difference (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=defaults.threshold,
        help="Flow threshold, in body-fractions per second (default: %(default)s).",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=defaults.diff_threshold,
        dest="diff_threshold",
        help="Frame-difference threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=defaults.confirm_frames,
        dest="confirm_frames",
        help="Consecutive frames of motion required before an elimination (default: %(default)s).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=defaults.smoothing,
        help="Temporal smoothing factor, in (0, 1] (default: %(default)s).",
    )
    parser.add_argument(
        "--detector",
        choices=["yolo", "hog"],
        default=defaults.detector,
        help="Person detector: yolo (accurate) or hog (fast, no weights) (default: %(default)s).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=defaults.conf,
        help="Detector confidence threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--max-misses",
        type=int,
        default=defaults.max_misses,
        dest="max_misses",
        help="Frames a player can go undetected before their track is dropped (default: %(default)s).",
    )
    parser.add_argument("--headless", action="store_true", help="Run with no display window.")
    parser.add_argument(
        "--record", default=None, metavar="PATH", help="Write the annotated match to an mp4 file."
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        metavar="N",
        help="Discard this many frames before the match clock starts (default: %(default)s).",
    )
    parser.add_argument(
        "--auto-start",
        type=int,
        default=None,
        dest="auto_start",
        metavar="N",
        help="Start once N consecutive frames see a player; required for --headless.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        dest="max_seconds",
        metavar="S",
        help="Abort the run if the match hasn't finished within this many seconds.",
    )
    parser.add_argument("--mute", action="store_true", help="Disable the chant and buzzer.")


def _cmd_play(args: argparse.Namespace) -> int:
    config = GameConfig(
        seed=args.seed,
        countdown_s=args.countdown_s,
        duration_s=args.duration_s,
        phase_min_s=args.phase_min_s,
        phase_max_s=args.phase_max_s,
        grace_s=args.grace_s,
        metric=args.metric,
        threshold=args.threshold,
        diff_threshold=args.diff_threshold,
        confirm_frames=args.confirm_frames,
        smoothing=args.smoothing,
        detector=args.detector,
        conf=args.conf,
        max_misses=args.max_misses,
    )
    try:
        config.validate()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    source = _parse_source(args.source)

    try:
        report = app.run(
            config,
            source,
            headless=args.headless,
            record=args.record,
            auto_start_frames=args.auto_start,
            skip=args.skip,
            max_seconds=args.max_seconds,
            mute=args.mute,
        )
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_format_summary(report))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    try:
        from . import benchmark
    except ImportError as exc:
        print(
            "error: benchmark module not available yet "
            f"(redlight.benchmark: {exc})",
            file=sys.stderr,
        )
        return 1

    benchmark.run_benchmark(args.out, fast=args.fast)
    return 0


def _cmd_build_site(args: argparse.Namespace) -> int:
    try:
        from . import export
    except ImportError as exc:
        print(
            "error: build-site module not available yet "
            f"(redlight.export: {exc})",
            file=sys.stderr,
        )
        return 1

    export.build_site(args.out)
    return 0


def _cmd_make_audio(args: argparse.Namespace) -> int:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    duration_s = synth.render_chant(args.out)
    print(f"wrote {args.out} ({duration_s:.2f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the `redlight` argument parser.

    No flags live on the top-level parser besides the built-in
    `-h`/`--help` — every real flag belongs to exactly one subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="redlight",
        description="A computer-vision referee for Red Light, Green Light.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser("play", help="Run a match against a webcam or video file.")
    _add_play_arguments(play_parser)
    play_parser.set_defaults(func=_cmd_play)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Score the judge and the baseline designs on ground-truth footage."
    )
    benchmark_parser.add_argument(
        "--out",
        default=DEFAULT_BENCHMARK_OUT,
        help="Where to write the results JSON (default: %(default)s).",
    )
    benchmark_parser.add_argument(
        "--fast", action="store_true", help="Run on a small frame subset for a quick sanity check."
    )
    benchmark_parser.set_defaults(func=_cmd_benchmark)

    make_audio_parser = subparsers.add_parser("make-audio", help="Render the chant to a WAV file.")
    make_audio_parser.add_argument(
        "--out",
        default=DEFAULT_AUDIO_OUT,
        help="Where to write the WAV file (default: %(default)s).",
    )
    make_audio_parser.set_defaults(func=_cmd_make_audio)

    build_site_parser = subparsers.add_parser(
        "build-site", help="Build the browser arena as a single static HTML file."
    )
    build_site_parser.add_argument(
        "--out",
        default=DEFAULT_SITE_OUT,
        help="Where to write the site HTML (default: %(default)s).",
    )
    build_site_parser.set_defaults(func=_cmd_build_site)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.func
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
