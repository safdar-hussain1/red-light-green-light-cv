"""The repository has to read as a product, everywhere, all the time.

This is a copy test, not a code test, and it is here because framing rots in
exactly the places nobody re-reads: a docstring written during a spike, a
sentence in a README that made sense when it was written, a line in a
generated page that came along for the ride. One of those is enough to
change what the whole thing looks like to somebody arriving cold.

So the check is mechanical and it covers everything tracked in git —
sources, tests, docs, the built `docs/index.html`, the packaging metadata.
The only file exempt is this one, which has to spell the phrases out to look
for them, and a test below holds that exemption to exactly one file. A
pattern that fires anywhere else is either a genuine slip or a sign the
pattern is wrong, and the second case is a conversation rather than a quiet
addition to an allowlist.

Three groups of patterns, for three different failure modes:

* **Borrowed identity.** Naming somebody else's property, or a year, ties
  this project to something it is not and dates it to a moment it is not
  from.
* **Coursework framing.** "College", "university", "coursework" describe why
  something was made rather than what it does, and a reader who wanted to
  know what it does now has to look past that.
* **Second-system framing.** "Rebuild", "legacy", "originally", "the
  original" all describe this work as a version of some earlier work. It
  isn't one, and writing as though it were invites the reader to evaluate it
  against something they cannot see.

The last group also covers the tooling that helped write it. Whose keyboard
produced a line is not part of what the software does.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SCANNER = Path(__file__).resolve()
"""This file, excluded from its own scan.

The only exemption, and an unavoidable one: a list of phrases that must not
appear anywhere is itself a file containing every one of them. Excluding the
scanner is the whole of the exemption — there is no allowlist beside it, and
`test_only_the_scanner_is_exempt` holds it to exactly one file.
"""

BANNED = (
    r"squid game",
    r"netflix",
    r"\b2023\b",
    r"college",
    r"university",
    r"coursework",
    r"rebuilt",
    r"rebuild",
    r"legacy",
    r"originally",
    r"the original",
    r"claude",
    r"anthropic",
    r"co-authored",
)
"""Every pattern, matched case-insensitively, against every tracked text file."""

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
    """Every file git knows about, as absolute paths.

    Deliberately `git ls-files` rather than a directory walk: the point is to
    check what is published, and a walk would sweep up build output, virtual
    environments and scratch files that are not part of the repository.
    """
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
        if path.resolve() == SCANNER:
            continue
        if not path.exists() or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        files.append(path)
    return files


TEXT_FILES = text_files()


def test_the_repository_has_text_files_to_check():
    """A scan that found nothing to scan would pass every test below."""
    names = {path.name for path in TEXT_FILES}
    for expected in ("README.md", "pyproject.toml", "index.html", "judge.py"):
        assert expected in names, f"{expected} should be tracked and readable as text"
    assert len(TEXT_FILES) > 20, f"only {len(TEXT_FILES)} text files found; is git ls-files working?"


@pytest.mark.parametrize("pattern", BANNED)
def test_no_tracked_file_carries_a_banned_phrase(pattern):
    """No tracked text file mentions the phrase, in any case."""
    compiled = re.compile(pattern, re.IGNORECASE)
    hits = []

    for path in TEXT_FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if compiled.search(line):
                relative = path.relative_to(REPO_ROOT)
                # Built pages are one enormous line; a slice keeps the failure
                # readable instead of dumping half a megabyte into the report.
                hits.append(f"{relative}:{number}: {line.strip()[:160]}")

    assert hits == [], "banned phrase " + repr(pattern) + " found:\n  " + "\n  ".join(hits[:20])


def test_only_the_scanner_is_exempt():
    """Exactly one tracked text file is skipped, and it is this one.

    If a second exemption ever appears it will be because something was
    easier to skip than to reword, which is the failure this whole module
    exists to prevent.
    """
    scanned = {path.resolve() for path in TEXT_FILES}
    tracked_text = {
        path.resolve()
        for path in tracked_files()
        if path.exists() and path.suffix.lower() not in BINARY_SUFFIXES
    }
    decodable = set()
    for path in tracked_text:
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        decodable.add(path)

    assert decodable - scanned == {SCANNER}


def test_the_built_page_is_covered_by_this_scan():
    """The published page is generated, and generated text still ships.

    A scan that quietly skipped `docs/index.html` would let anything inlined
    from `site/` through, which is precisely the text a visitor reads first.
    """
    assert (REPO_ROOT / "docs" / "index.html") in TEXT_FILES
