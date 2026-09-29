#!/usr/bin/env python3
"""Fail closed if the Streamlit deployment package is incomplete or unsafe."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mad.public_viewer_data import (  # noqa: E402
    DEFAULT_PUBLIC_BUNDLE,
    PUBLIC_QUESTION_IDS,
    load_public_experiment_overview,
    load_public_question_replay,
)


REQUIRED_FILES = (
    REPOSITORY_ROOT / "app" / "viewer.py",
    REPOSITORY_ROOT / ".streamlit" / "config.toml",
    REPOSITORY_ROOT / "requirements.txt",
    DEFAULT_PUBLIC_BUNDLE,
)
FORBIDDEN_TRACKED_FILES = (
    ".env",
    "storage/results.sqlite",
    "storage/cache.sqlite",
)


def _tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def main() -> None:
    missing = [str(path) for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise SystemExit("deployment files are missing:\n- " + "\n- ".join(missing))

    tracked = _tracked_files()
    exposed = [path for path in FORBIDDEN_TRACKED_FILES if path in tracked]
    if exposed:
        raise SystemExit(
            "private files must not be tracked:\n- " + "\n- ".join(exposed)
        )

    overview = load_public_experiment_overview()
    replays = [load_public_question_replay(question_id) for question_id in PUBLIC_QUESTION_IDS]
    if overview.question_count != 300:
        raise SystemExit("public overview is not the accepted 300-question result")
    if any(replay.run_id != overview.run_id for replay in replays):
        raise SystemExit("a public replay belongs to a different run")

    print("public deployment preflight passed")
    print(f"run: {overview.run_id}")
    print(f"curated replays: {len(replays)}")
    print(f"bundle: {DEFAULT_PUBLIC_BUNDLE}")
    print("private databases tracked: no")
    print("API secrets required: no")


if __name__ == "__main__":
    main()
