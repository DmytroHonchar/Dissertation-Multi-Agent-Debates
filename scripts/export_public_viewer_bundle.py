#!/usr/bin/env python3
"""Create the sanitized, read-only data artifact used by the public viewer.

The source databases are opened through ``mad.viewer_data`` in read-only mode.
Only the accepted overview and six approved replay questions are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mad.public_viewer_data import (  # noqa: E402
    DEFAULT_PUBLIC_BUNDLE,
    PUBLIC_BUNDLE_SCHEMA_VERSION,
    PUBLIC_QUESTION_IDS,
    find_forbidden_public_content,
    sanitize_public_value,
)
from mad.viewer_data import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    DEFAULT_RESULTS_PATH,
    MAIN_RUN_ID,
    load_experiment_overview,
    load_question_replay,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundle() -> dict[str, object]:
    results_path = Path(DEFAULT_RESULTS_PATH)
    cache_path = Path(DEFAULT_CACHE_PATH)
    before = {
        "results_sha256": _sha256(results_path),
        "cache_sha256": _sha256(cache_path),
    }

    overview = load_experiment_overview()
    questions = {
        question_id: asdict(load_question_replay(question_id))
        for question_id in PUBLIC_QUESTION_IDS
    }

    after = {
        "results_sha256": _sha256(results_path),
        "cache_sha256": _sha256(cache_path),
    }
    if before != after:
        raise RuntimeError("source databases changed while exporting the public bundle")

    return {
        "schema_version": PUBLIC_BUNDLE_SCHEMA_VERSION,
        "provenance": {
            "run_id": MAIN_RUN_ID,
            "settings_version": overview.settings_version,
            "evaluation_version": overview.evaluation_version,
            **before,
            "question_ids": list(PUBLIC_QUESTION_IDS),
        },
        "overview": asdict(overview),
        "questions": questions,
    }


def write_bundle(output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --force after verification"
        )

    payload = sanitize_public_value(build_bundle())
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    found = find_forbidden_public_content(text)
    if found:
        raise RuntimeError(
            "refusing unsafe public bundle containing: " + ", ".join(found)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PUBLIC_BUNDLE,
        help="destination JSON path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing bundle after all safety checks pass",
    )
    arguments = parser.parse_args()
    write_bundle(arguments.output, force=arguments.force)
    print(f"wrote sanitized public bundle: {arguments.output}")


if __name__ == "__main__":
    main()
