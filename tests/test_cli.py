"""Tests for the `redlight` command-line interface.

`play` tests that actually run a match use `data/vtest.avi` with the fast
HOG detector and tiny durations (mirroring `tests/test_app.py`'s
`make_config`/`RUN_KWARGS`), so this file stays fast even though it drives
the real CLI -> `app.run` path end to end.
"""

from __future__ import annotations

import argparse

import pytest

from redlight import cli

VIDEO = "data/vtest.avi"

# A config tuned to run fast on vtest.avi, mirroring test_app.py's make_config.
FAST_PLAY_ARGS = [
    "--source", VIDEO,
    "--headless",
    "--detector", "hog",
    "--conf", "0.3",
    "--seed", "7",
    "--countdown", "0.2",
    "--duration", "0.8",
    "--phase-min", "0.2",
    "--phase-max", "0.3",
    "--grace", "0.0",
    "--confirm-frames", "2",
    "--max-misses", "8",
    "--skip", "3",
    "--auto-start", "2",
    "--max-seconds", "15",
    "--mute",
]


class TestHelp:
    def test_help_exits_zero_and_lists_four_subcommands(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--help"])
        assert exc_info.value.code == 0

        out = capsys.readouterr().out
        for name in ("play", "benchmark", "make-audio", "build-site"):
            assert name in out

    def test_no_subcommand_exits_nonzero_with_no_traceback(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main([])
        assert exc_info.value.code != 0
        assert "Traceback" not in capsys.readouterr().err


class TestParserHasNoFlagCollisions:
    """Playbook 7g: no top-level flag may shadow a subcommand's own flag."""

    def test_top_level_parser_defines_no_custom_flags(self):
        parser = cli.build_parser()
        top_level_flags = {
            opt
            for action in parser._actions
            if not isinstance(action, argparse._SubParsersAction)
            for opt in action.option_strings
        } - {"-h", "--help"}

        assert top_level_flags == set()

    def test_no_option_string_collides_between_parent_and_any_subcommand(self):
        parser = cli.build_parser()
        subparsers_action = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        )
        top_level_flags = {
            opt
            for action in parser._actions
            if not isinstance(action, argparse._SubParsersAction)
            for opt in action.option_strings
        } - {"-h", "--help"}

        for name, subparser in subparsers_action.choices.items():
            sub_flags = {
                opt for action in subparser._actions for opt in action.option_strings
            } - {"-h", "--help"}
            collisions = top_level_flags & sub_flags
            assert not collisions, f"subcommand {name!r} collides with top-level flags: {collisions}"

    def test_exactly_four_subcommands_registered(self):
        parser = cli.build_parser()
        subparsers_action = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        )
        assert set(subparsers_action.choices) == {"play", "benchmark", "make-audio", "build-site"}


class TestPlayValidation:
    def test_bad_threshold_exits_two_with_config_error_on_stderr_no_traceback(self, capsys):
        code = cli.main(["play", "--threshold", "-1", "--headless", "--auto-start", "1"])

        assert code == 2
        err = capsys.readouterr().err
        assert "threshold" in err
        assert "Traceback" not in err

    def test_missing_source_prints_clean_error_no_traceback(self, capsys):
        # --detector hog: the default "yolo" detector is constructed before
        # the source is ever opened (see app.run), so leaving it at the
        # default would pay YOLO's model-load cost just to test an error
        # path that never gets that far in spirit.
        bad_path = "data/does_not_exist_at_all.avi"
        code = cli.main(
            ["play", "--source", bad_path, "--headless", "--auto-start", "1", "--detector", "hog"]
        )

        assert code != 0
        err = capsys.readouterr().err
        assert bad_path in err
        assert "Traceback" not in err

    def test_headless_without_auto_start_is_a_clean_config_error(self, capsys):
        code = cli.main(["play", "--source", VIDEO, "--headless"])

        assert code == 2
        err = capsys.readouterr().err
        assert "auto_start" in err or "headless" in err
        assert "Traceback" not in err


class TestPlaySourceParsing:
    def test_int_like_source_becomes_webcam_index(self):
        assert cli._parse_source("0") == 0
        assert cli._parse_source("2") == 2

    def test_non_int_source_stays_a_path(self):
        assert cli._parse_source(VIDEO) == VIDEO


class TestPlaySmoke:
    """One real run through the CLI, exercised with the fast HOG detector
    so it stays well under a second even though it drives the full
    detect -> track -> judge -> game -> HUD pipeline (marked slow only out
    of caution for slower machines; --skip/--duration keep it small).
    """

    def test_play_on_vtest_headless_exits_zero_and_prints_summary(self, capsys):
        code = cli.main(["play", *FAST_PLAY_ARGS])

        assert code == 0
        out = capsys.readouterr().out.strip()
        assert out
        assert any(word in out for word in ("victory", "wipeout", "aborted"))
        assert "player" in out
        assert "survived" in out
        assert "eliminated" in out


class TestMakeAudio:
    def test_renders_to_out_path_and_prints_duration(self, tmp_path, capsys):
        out_path = tmp_path / "chant.wav"

        code = cli.main(["make-audio", "--out", str(out_path)])

        assert code == 0
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        out = capsys.readouterr().out
        assert str(out_path) in out
        assert "s)" in out  # duration printed, e.g. "(3.45s)"


class TestLazyModulesNotYetAvailable:
    """`benchmark.py` (Task 9) and `export.py` (Task 11) don't exist yet.
    These pin today's graceful-degradation behavior; once those modules
    land, these tests should be replaced with real ones exercising them.
    """

    def test_benchmark_reports_actionable_error_and_exits_one(self, capsys):
        code = cli.main(["benchmark", "--fast"])

        assert code == 1
        err = capsys.readouterr().err
        assert "benchmark" in err.lower()
        assert "not available" in err
        assert "Traceback" not in err

    def test_build_site_reports_actionable_error_and_exits_one(self, capsys):
        code = cli.main(["build-site"])

        assert code == 1
        err = capsys.readouterr().err
        assert "not available" in err
        assert "Traceback" not in err


class TestMainReturnsInt:
    def test_make_audio_default_out_returns_plain_int(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        result = cli.main(["make-audio"])

        assert result == 0
        assert isinstance(result, int)
        assert (tmp_path / "assets" / "doll_song.wav").exists() or (
            tmp_path / cli.DEFAULT_AUDIO_OUT
        ).exists()
