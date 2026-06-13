#!/usr/bin/env python3
"""Create the one-time frozen VEPRAD test split.

The script physically separates the frozen test data from the train/validation
pool. It reads the clean regular-speaker VEPRAD source manifest, selects:

- all utterances from N held-out speakers, and
- ceil(ratio * speaker_utterances) from every other speaker,

then moves referenced audio/transcript files into:

- data/test/raw/veprad/
- data/cross_validation_splits/raw_train_val/raw/veprad/
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_MANIFEST = Path("data/manifests/veprad/source.jsonl")
DEFAULT_TEST_DIR = Path("data/test")
DEFAULT_TRAIN_VAL_DIR = Path("data/cross_validation_splits/raw_train_val")
DEFAULT_LEFTOVER_QUARANTINE = Path("data/quarantine/raw_leftovers_after_frozen_test_split")
RAW_PREFIX = Path("data/raw/veprad")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_text(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath_for(row: dict, key: str) -> Path:
    value = row.get(key)
    if not value:
        raise SystemExit(f"Missing {key} for utterance {row.get('utterance_id')}")
    path = Path(value)
    if path.is_absolute():
        raise SystemExit(
            f"{key} must be repository-relative, got absolute path for "
            f"{row.get('utterance_id')}: {value}"
        )
    try:
        path.relative_to(RAW_PREFIX)
    except ValueError as exc:
        raise SystemExit(
            f"{key} must start with {RAW_PREFIX}, got {value} for "
            f"{row.get('utterance_id')}"
        ) from exc
    return path


def target_relpath(original_relpath: Path, output_raw_root: Path) -> Path:
    suffix = original_relpath.relative_to(RAW_PREFIX)
    return output_raw_root / suffix


def rewrite_row(row: dict, output_raw_root: Path) -> dict:
    new_row = dict(row)
    audio_rel = target_relpath(relpath_for(row, "audio_relpath"), output_raw_root)
    transcript_rel = target_relpath(
        relpath_for(row, "transcript_relpath"), output_raw_root
    )

    new_row["audio_relpath"] = audio_rel.as_posix()
    new_row["transcript_relpath"] = transcript_rel.as_posix()
    new_row["audio_filepath"] = audio_rel.as_posix()
    new_row["transcript_filepath"] = transcript_rel.as_posix()
    new_row["raw_root"] = output_raw_root.as_posix()
    return new_row


def summarize(rows: list[dict]) -> dict:
    speakers = Counter(row["speaker_id"] for row in rows)
    corpora = Counter(row.get("corpus", "unknown") for row in rows)
    genders = Counter(row.get("gender", "unknown") for row in rows)
    seconds = sum(float(row.get("duration") or 0.0) for row in rows)
    return {
        "utterances": len(rows),
        "total_seconds": round(seconds, 6),
        "total_hours": round(seconds / 3600.0, 6),
        "speaker_count": len(speakers),
        "speakers": dict(sorted(speakers.items())),
        "corpora": dict(sorted(corpora.items())),
        "genders": dict(sorted(genders.items())),
    }


def ensure_unique(rows: list[dict], key: str) -> None:
    values = [row.get(key) for row in rows]
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        sample = ", ".join(map(str, sorted(duplicates)[:10]))
        raise SystemExit(f"Duplicate {key} values: {sample}")


def ensure_clean_output(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if force:
        shutil.rmtree(path)
        return
    raise SystemExit(f"Refusing to overwrite existing output: {path}")


def move_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Missing source file before move: {src}")
    if dst.exists():
        raise SystemExit(f"Refusing to overwrite target file: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for current, dirs, files in os.walk(root, topdown=False):
        path = Path(current)
        if not dirs and not files:
            path.rmdir()


def nonempty_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def select_split(
    rows: list[dict],
    heldout_speakers: int,
    per_speaker_ratio: float,
    seed: int | None,
) -> tuple[set[str], set[str], dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)

    speakers = sorted(by_speaker)
    if heldout_speakers >= len(speakers):
        raise SystemExit(
            f"Need fewer held-out speakers than total speakers: "
            f"{heldout_speakers} >= {len(speakers)}"
        )
    if not (0.0 < per_speaker_ratio < 1.0):
        raise SystemExit(f"per-speaker ratio must be in (0, 1), got {per_speaker_ratio}")

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    full_test_speakers = set(rng.sample(speakers, heldout_speakers))
    test_ids: set[str] = set()
    sampled_counts: dict[str, int] = {}

    for speaker in speakers:
        speaker_rows = sorted(by_speaker[speaker], key=lambda row: row["utterance_id"])
        if speaker in full_test_speakers:
            sampled = speaker_rows
        else:
            sample_count = math.ceil(len(speaker_rows) * per_speaker_ratio)
            sampled = rng.sample(speaker_rows, sample_count)
        sampled_counts[speaker] = len(sampled)
        test_ids.update(row["utterance_id"] for row in sampled)

    selection = {
        "heldout_speakers": sorted(full_test_speakers),
        "same_speaker_test_ratio": per_speaker_ratio,
        "same_speaker_test_counts": dict(sorted(sampled_counts.items())),
        "random_seed": seed,
        "random_source": "python_random_seed" if seed is not None else "system_random",
    }
    return full_test_speakers, test_ids, selection


def write_readmes(test_dir: Path, cv_root: Path) -> None:
    test_readme = """# Frozen Test Split

This folder contains the one-time held-out VEPRAD test split.

The JSONL manifest contains transcript text and is intentionally ignored by Git.
The tracked `test.meta.json` file is the public audit record for the split.

Do not use this folder for model selection or cross-validation.
"""
    cv_readme = """# Cross-Validation Splits

`raw_train_val/` contains the non-test VEPRAD source pool used to create
train/validation splits.

Future split folders should contain only JSONL manifests and metadata. They
should reference files under `raw_train_val/raw/veprad/` and must not reference
`data/test/`.
"""
    test_dir.mkdir(parents=True, exist_ok=True)
    cv_root.mkdir(parents=True, exist_ok=True)
    (test_dir / "README.md").write_text(test_readme, encoding="utf-8")
    (cv_root / "README.md").write_text(cv_readme, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    source_manifest = project_root / args.source_manifest
    raw_root = project_root / RAW_PREFIX
    test_dir = project_root / args.test_dir
    train_val_dir = project_root / args.train_val_dir
    leftover_quarantine = project_root / args.leftover_quarantine

    if not source_manifest.exists():
        raise SystemExit(f"Missing source manifest: {source_manifest}")
    if not raw_root.exists():
        raise SystemExit(f"Missing raw root: {raw_root}")

    rows = read_jsonl(source_manifest)
    if not rows:
        raise SystemExit("Source manifest is empty")

    ensure_unique(rows, "utterance_id")
    ensure_unique(rows, "audio_relpath")
    ensure_unique(rows, "transcript_relpath")

    forbidden_rows = [
        row
        for row in rows
        if row.get("speaker_id") == "sm04" or row.get("corpus") == "sm04"
    ]
    if forbidden_rows:
        raise SystemExit(
            f"Source manifest still contains sm04 rows: {len(forbidden_rows)}"
        )

    missing_files: list[str] = []
    for row in rows:
        for key in ("audio_relpath", "transcript_relpath"):
            rel = relpath_for(row, key)
            if not (project_root / rel).exists():
                missing_files.append(str(rel))
    if missing_files:
        sample = "\n".join(missing_files[:20])
        raise SystemExit(f"Missing referenced source files:\n{sample}")

    _, test_ids, selection = select_split(
        rows=rows,
        heldout_speakers=args.heldout_speakers,
        per_speaker_ratio=args.per_speaker_test_ratio,
        seed=args.seed,
    )

    test_rows_original = [row for row in rows if row["utterance_id"] in test_ids]
    train_val_rows_original = [row for row in rows if row["utterance_id"] not in test_ids]

    if set(row["utterance_id"] for row in test_rows_original) & set(
        row["utterance_id"] for row in train_val_rows_original
    ):
        raise SystemExit("Internal error: test/train-val overlap")
    if len(test_rows_original) + len(train_val_rows_original) != len(rows):
        raise SystemExit("Internal error: split row counts do not add up")

    test_raw_rel = args.test_dir / "raw" / "veprad"
    train_val_raw_rel = args.train_val_dir / "raw" / "veprad"
    test_rows = [rewrite_row(row, test_raw_rel) for row in test_rows_original]
    train_val_rows = [
        rewrite_row(row, train_val_raw_rel) for row in train_val_rows_original
    ]

    source_id_hash = sha256_text(sorted(row["utterance_id"] for row in rows))
    test_id_hash = sha256_text(sorted(row["utterance_id"] for row in test_rows))
    train_val_id_hash = sha256_text(
        sorted(row["utterance_id"] for row in train_val_rows)
    )

    source_manifest_hash = sha256_file(source_manifest)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    meta_common = {
        "dataset": "veprad",
        "created_at_utc": created_at,
        "input_manifest": args.source_manifest.as_posix(),
        "input_manifest_role": "one_time_pre_split_source",
        "input_manifest_sha256": source_manifest_hash,
        "source_utterance_id_sha256": source_id_hash,
        "source_summary": summarize(rows),
        "selection_policy": {
            "description": (
                "Permanent test split: all utterances from randomly selected "
                "held-out speakers plus ceil(ratio * utterances) from every "
                "remaining speaker."
            ),
            "heldout_speaker_count": args.heldout_speakers,
            "same_speaker_test_ratio": args.per_speaker_test_ratio,
            "rounding": "ceil_per_speaker",
        },
        "selection": selection,
        "excluded_data": {
            "sm04": (
                "Excluded/quarantined before splitting because it is a "
                "segmented speech-synthesis subset of a male VEPRAD speaker "
                "and may leak speaker/content information relative to m04."
            )
        },
    }

    test_meta = {
        **meta_common,
        "logical_split": "frozen_test",
        "manifest": (args.test_dir / "test.jsonl").as_posix(),
        "raw_root": test_raw_rel.as_posix(),
        "utterance_id_sha256": test_id_hash,
        "summary": summarize(test_rows),
    }
    train_val_meta = {
        **meta_common,
        "logical_split": "raw_train_val_source",
        "manifest": (args.train_val_dir / "source.jsonl").as_posix(),
        "raw_root": train_val_raw_rel.as_posix(),
        "utterance_id_sha256": train_val_id_hash,
        "summary": summarize(train_val_rows),
    }

    print("Source utterances:", len(rows))
    print("Source speakers:", len(set(row["speaker_id"] for row in rows)))
    print("Held-out speakers:", ", ".join(selection["heldout_speakers"]))
    print("Frozen test utterances:", len(test_rows))
    print("Train/val source utterances:", len(train_val_rows))
    print("Source ID hash:", source_id_hash)
    print("Test ID hash:", test_id_hash)
    print("Train/val ID hash:", train_val_id_hash)

    if args.dry_run:
        print("Dry run only; no files moved or written.")
        return 0

    ensure_clean_output(test_dir, args.force)
    ensure_clean_output(train_val_dir, args.force)
    if leftover_quarantine.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing quarantine: {leftover_quarantine}")
    if leftover_quarantine.exists() and args.force:
        shutil.rmtree(leftover_quarantine)

    moved_sources: set[Path] = set()
    for original_rows, output_rows in (
        (test_rows_original, test_rows),
        (train_val_rows_original, train_val_rows),
    ):
        for original, output in zip(original_rows, output_rows):
            for key in ("audio_relpath", "transcript_relpath"):
                src = project_root / relpath_for(original, key)
                dst = project_root / Path(output[key])
                move_file(src, dst)
                moved_sources.add(src)

    write_jsonl(test_dir / "test.jsonl", test_rows)
    write_json(test_dir / "test.meta.json", test_meta)
    write_jsonl(train_val_dir / "source.jsonl", train_val_rows)
    write_json(train_val_dir / "source.meta.json", train_val_meta)
    write_readmes(test_dir, train_val_dir.parent)

    remove_empty_dirs(raw_root)
    leftovers = nonempty_paths(raw_root)
    if leftovers:
        target = leftover_quarantine / "veprad"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(raw_root), str(target))
        remove_empty_dirs(project_root / "data" / "raw")
        print(f"Moved leftover non-active raw files to {leftover_quarantine}")
    else:
        remove_empty_dirs(project_root / "data" / "raw")

    verify_outputs(test_rows, train_val_rows, project_root, test_dir, train_val_dir)
    print("Frozen test split created and verified.")
    return 0


def verify_outputs(
    test_rows: list[dict],
    train_val_rows: list[dict],
    project_root: Path,
    test_dir: Path,
    train_val_dir: Path,
) -> None:
    test_ids = {row["utterance_id"] for row in test_rows}
    train_val_ids = {row["utterance_id"] for row in train_val_rows}
    if test_ids & train_val_ids:
        raise SystemExit("Verification failed: overlapping utterance ids")

    test_paths: set[str] = set()
    train_val_paths: set[str] = set()
    for label, rows, expected_prefix, collector in (
        ("test", test_rows, "data/test/raw/veprad/", test_paths),
        (
            "train_val",
            train_val_rows,
            "data/cross_validation_splits/raw_train_val/raw/veprad/",
            train_val_paths,
        ),
    ):
        for row in rows:
            if row.get("speaker_id") == "sm04" or row.get("corpus") == "sm04":
                raise SystemExit(f"Verification failed: sm04 found in {label}")
            for key in ("audio_relpath", "transcript_relpath"):
                value = row[key]
                if not value.startswith(expected_prefix):
                    raise SystemExit(
                        f"Verification failed: {label} {key} outside expected "
                        f"prefix: {value}"
                    )
                if not (project_root / value).exists():
                    raise SystemExit(
                        f"Verification failed: missing output file {value}"
                    )
                collector.add(value)

    if test_paths & train_val_paths:
        raise SystemExit("Verification failed: overlapping file paths")
    if not (test_dir / "test.jsonl").exists():
        raise SystemExit("Verification failed: missing test manifest")
    if not (train_val_dir / "source.jsonl").exists():
        raise SystemExit("Verification failed: missing train/val source manifest")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--train-val-dir", type=Path, default=DEFAULT_TRAIN_VAL_DIR)
    parser.add_argument(
        "--leftover-quarantine",
        type=Path,
        default=DEFAULT_LEFTOVER_QUARANTINE,
        help="Where old raw files not referenced by the clean manifest are moved.",
    )
    parser.add_argument("--heldout-speakers", type=int, default=3)
    parser.add_argument("--per-speaker-test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional one-time seed. If omitted, system randomness is used.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing output/quarantine directories before writing.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
