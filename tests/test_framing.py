"""Tracked text contains no banned phrases and no absolute local paths.

The scan covers every text file git tracks — sources, tests, docs, the built
`docs/index.html`, the packaging metadata and this file too — so what is
checked is exactly what is published. Build output, virtual environments and
scratch files are not tracked, so they are not read.

Each banned phrase is assembled from fragments, so the list below never
contains a literal copy of what it guards against, and this file needs no
exemption from its own scan.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

BANNED = (
    r"\b20" + r"23\b",
    "col" + "lege",
    "univer" + "sity",
    "course" + "work",
    "re" + "built",
    "re" + "build",
    "leg" + "acy",
    "origin" + "ally",
    "the orig" + "inal",
    "cla" + "ude",
    "anthro" + "pic",
    "co-" + "authored",
)
"""Every pattern, matched case-insensitively, against every line of every
tracked text file."""

ABSOLUTE_PATH = "/Us" + "ers/"
"""A home-directory path: it names a machine, and it resolves on no other."""

BINARY_SUFFIXES = frozenset(
    {
        ".avi",
        ".gif",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mp4",
        ".pdf",
        ".png",
        ".pt",
        ".ttf",
        ".wav",
        ".webp",
        ".woff",
        ".woff2",
        ".zip",
    }
)
"""Extensions skipped outright. Anything else is read, and a file that will
not decode as UTF-8 is skipped as binary rather than guessed at."""


def tracked_files() -> list[Path]:
    """Every file git knows about, as absolute paths."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / name for name in result.stdout.split("\0") if name]


def text_files() -> list[Path]:
    files = []
    for path in tracked_files():
        if not path.exists() or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        files.append(path)
    return files


def matching_lines(pattern: str, paths: list[Path]) -> list[str]:
    """`path:line: text` for every line matching `pattern`, case-insensitively."""
    compiled = re.compile(pattern, re.IGNORECASE)
    hits = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if compiled.search(line):
                try:
                    shown = path.relative_to(REPO_ROOT)
                except ValueError:
                    shown = path
                # Built pages are one enormous line; a slice keeps the failure
                # readable instead of dumping half a megabyte into the report.
                hits.append(f"{shown}:{number}: {line.strip()[:160]}")
    return hits


if not (REPO_ROOT / ".git").exists():
    # A ZIP download or a `git archive` copy has no history, so there is no
    # list of tracked files to scan. Skip loudly rather than fail collection,
    # which would stop every other test in the suite from running too.
    pytest.skip(
        "not a git checkout: this scan reads `git ls-files`, so it only runs "
        "in a clone of the repository",
        allow_module_level=True,
    )

TEXT_FILES = text_files()


def test_the_repository_has_text_files_to_check():
    """A scan that found nothing to scan would pass every test below."""
    names = {path.name for path in TEXT_FILES}
    for expected in ("README.md", "pyproject.toml", "index.html", "judge.py", "test_framing.py"):
        assert expected in names, f"{expected} should be tracked and readable as text"
    assert len(TEXT_FILES) > 20, f"only {len(TEXT_FILES)} text files found; is git ls-files working?"


def test_the_built_page_is_covered_by_this_scan():
    """The published page is generated, and generated text still ships."""
    assert (REPO_ROOT / "docs" / "index.html") in TEXT_FILES


@pytest.mark.parametrize("pattern", BANNED, ids=[f"phrase{i:02d}" for i in range(len(BANNED))])
def test_no_tracked_file_carries_a_banned_phrase(pattern):
    hits = matching_lines(pattern, TEXT_FILES)
    assert hits == [], "banned phrase found:\n  " + "\n  ".join(hits[:20])


def test_no_tracked_file_carries_an_absolute_local_path():
    hits = matching_lines(re.escape(ABSOLUTE_PATH), TEXT_FILES)
    assert hits == [], "absolute local path found:\n  " + "\n  ".join(hits[:20])


def test_every_pattern_catches_its_phrase_in_any_case(tmp_path):
    """A pattern that could never match would pass the scan above forever.

    Each phrase is planted, upper-cased, in a scratch file and must be found
    by the same `matching_lines` the scan uses; an ordinary sentence must not.
    """
    phrases = ["shipped in 20" + "23.", *BANNED[1:]]
    patterns = [*BANNED, re.escape(ABSOLUTE_PATH)]
    phrases.append("see " + ABSOLUTE_PATH + "someone/file")

    clean = tmp_path / "clean.txt"
    clean.write_text("A referee for a party game, measured on real footage.\n", encoding="utf-8")
    planted = tmp_path / "planted.txt"
    for index, (pattern, phrase) in enumerate(zip(patterns, phrases)):
        planted.write_text("Some text, " + phrase.upper() + ", more text.\n", encoding="utf-8")
        assert matching_lines(pattern, [planted]), f"pattern {index} missed its own phrase"
        assert matching_lines(pattern, [clean]) == [], f"pattern {index} flagged a clean line"
