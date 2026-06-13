#!/usr/bin/env python3
"""Create one VEPRAD train/validation split from the non-test source pool."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_MANIFEST = Path("data/cross_validation_splits/raw_train_val/source.jsonl")
DEFAULT_OUTPUT_ROOT = Path("data/cross_validation_splits")
SOURCE_RAW_PREFIX = Path("data/cross_validation_splits/raw_train_val/raw/veprad")


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
    counts = Counter(row.get(key) for row in rows)
    duplicates = [value for value, count in counts.items() if count > 1]
    if duplicates:
        sample = ", ".join(map(str, sorted(duplicates)[:10]))
        raise SystemExit(f"Duplicate {key} values: {sample}")


def ensure_relative_path(value: str, key: str, utterance_id: str) -> Path:
    if not value:
        raise SystemExit(f"Missing {key} for utterance {utterance_id}")
    path = Path(value)
    if path.is_absolute():
        raise SystemExit(f"{key} must be relative for {utterance_id}: {value}")
    return path


def validate_source_rows(project_root: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("Source manifest is empty")
    ensure_unique(rows, "utterance_id")
    ensure_unique(rows, "audio_relpath")
    ensure_unique(rows, "transcript_relpath")

    missing: list[str] = []
    bad_prefix: list[str] = []
    for row in rows:
        utterance_id = row.get("utterance_id", "<unknown>")
        if row.get("speaker_id") == "sm04" or row.get("corpus") == "sm04":
            raise SystemExit(f"Source manifest contains quarantined sm04 row: {utterance_id}")
        for key in ("audio_filepath", "audio_relpath", "transcript_filepath", "transcript_relpath"):
            value = row.get(key)
            path = ensure_relative_path(value, key, utterance_id)
            if key.endswith("_relpath") or key.endswith("_filepath"):
                if key.startswith("audio") and path != Path(row["audio_relpath"]):
                    raise SystemExit(f"audio path mismatch for {utterance_id}: {key}")
                if key.startswith("transcript") and path != Path(row["transcript_relpath"]):
                    raise SystemExit(f"transcript path mismatch for {utterance_id}: {key}")
            if key in {"audio_relpath", "transcript_relpath"}:
                try:
                    path.relative_to(SOURCE_RAW_PREFIX)
                except ValueError:
                    bad_prefix.append(f"{utterance_id}:{key}:{value}")
                if not (project_root / path).exists():
                    missing.append(value)
        raw_root = ensure_relative_path(row.get("raw_root"), "raw_root", utterance_id)
        if raw_root != SOURCE_RAW_PREFIX:
            raise SystemExit(f"Unexpected raw_root for {utterance_id}: {raw_root}")

    if bad_prefix:
        raise SystemExit("Paths outside raw_train_val source:\n" + "\n".join(bad_prefix[:20]))
    if missing:
        raise SystemExit("Missing referenced files:\n" + "\n".join(missing[:20]))


def select_split(
    rows: list[dict],
    val_speaker_count: int,
    same_speaker_val_ratio: float,
    seed: int | None,
) -> dict[str, list[dict] | dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)

    speakers = sorted(by_speaker)
    if val_speaker_count >= len(speakers):
        raise SystemExit(
            f"Need fewer validation speakers than total speakers: "
            f"{val_speaker_count} >= {len(speakers)}"
        )
    if not (0.0 < same_speaker_val_ratio < 1.0):
        raise SystemExit(
            f"same-speaker validation ratio must be in (0, 1), got "
            f"{same_speaker_val_ratio}"
        )

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    unseen_speakers = set(rng.sample(speakers, val_speaker_count))

    val_unseen: list[dict] = []
    val_seen: list[dict] = []
    train_seen: list[dict] = []
    same_speaker_val_counts: dict[str, int] = {}

    for speaker in speakers:
        speaker_rows = sorted(by_speaker[speaker], key=lambda row: row["utterance_id"])
        if speaker in unseen_speakers:
            val_unseen.extend(speaker_rows)
            same_speaker_val_counts[speaker] = len(speaker_rows)
            continue

        sample_count = math.ceil(len(speaker_rows) * same_speaker_val_ratio)
        sampled_ids = {
            row["utterance_id"] for row in rng.sample(speaker_rows, sample_count)
        }
        same_speaker_val_counts[speaker] = sample_count
        for row in speaker_rows:
            if row["utterance_id"] in sampled_ids:
                val_seen.append(row)
            else:
                train_seen.append(row)

    train_unseen: list[dict] = []
    train = sorted(train_seen + train_unseen, key=lambda row: row["utterance_id"])
    val = sorted(val_seen + val_unseen, key=lambda row: row["utterance_id"])

    return {
        "train": train,
        "train_seen_speakers": sorted(train_seen, key=lambda row: row["utterance_id"]),
        "train_unseen_speakers": train_unseen,
        "val": val,
        "val_seen_speakers": sorted(val_seen, key=lambda row: row["utterance_id"]),
        "val_unseen_speakers": sorted(val_unseen, key=lambda row: row["utterance_id"]),
        "selection": {
            "validation_unseen_speakers": sorted(unseen_speakers),
            "same_speaker_validation_ratio": same_speaker_val_ratio,
            "same_speaker_validation_counts": dict(sorted(same_speaker_val_counts.items())),
            "random_seed": seed,
            "random_source": "python_random_seed" if seed is not None else "system_random",
        },
    }


def validate_split(source_rows: list[dict], split: dict[str, list[dict] | dict]) -> None:
    source_ids = {row["utterance_id"] for row in source_rows}
    train = split["train"]
    val = split["val"]
    train_seen = split["train_seen_speakers"]
    train_unseen = split["train_unseen_speakers"]
    val_seen = split["val_seen_speakers"]
    val_unseen = split["val_unseen_speakers"]

    train_ids = {row["utterance_id"] for row in train}
    val_ids = {row["utterance_id"] for row in val}
    if train_ids & val_ids:
        raise SystemExit("Train/validation utterance overlap")
    if train_ids | val_ids != source_ids:
        raise SystemExit("Train + validation does not equal source pool")

    component_ids = {
        "train_seen_speakers": {row["utterance_id"] for row in train_seen},
        "train_unseen_speakers": {row["utterance_id"] for row in train_unseen},
        "val_seen_speakers": {row["utterance_id"] for row in val_seen},
        "val_unseen_speakers": {row["utterance_id"] for row in val_unseen},
    }
    if component_ids["train_seen_speakers"] | component_ids["train_unseen_speakers"] != train_ids:
        raise SystemExit("Train components do not add up to train")
    if component_ids["val_seen_speakers"] | component_ids["val_unseen_speakers"] != val_ids:
        raise SystemExit("Validation components do not add up to validation")

    all_component_ids: set[str] = set()
    for name, ids in component_ids.items():
        if all_component_ids & ids:
            raise SystemExit(f"Component overlap involving {name}")
        all_component_ids.update(ids)
    if all_component_ids != source_ids:
        raise SystemExit("Components do not add up to source pool")

    unseen_speakers = set(split["selection"]["validation_unseen_speakers"])
    train_speakers = {row["speaker_id"] for row in train}
    if train_speakers & unseen_speakers:
        raise SystemExit("Full held-out validation speakers appear in train")
    val_unseen_speakers = {row["speaker_id"] for row in val_unseen}
    if val_unseen_speakers != unseen_speakers:
        raise SystemExit("val_unseen_speakers does not match selected unseen speakers")
    if train_unseen:
        raise SystemExit("train_unseen_speakers must be empty by construction")

    for label, rows in (
        ("train", train),
        ("val", val),
        ("val_seen_speakers", val_seen),
        ("val_unseen_speakers", val_unseen),
    ):
        for row in rows:
            for key in ("audio_relpath", "transcript_relpath"):
                if str(row[key]).startswith("data/test/"):
                    raise SystemExit(f"{label} references frozen test path: {row[key]}")


def split_meta(
    split_id: str,
    source_manifest: Path,
    source_rows: list[dict],
    split: dict[str, list[dict] | dict],
) -> dict:
    source_ids = sorted(row["utterance_id"] for row in source_rows)
    component_files = {
        "train": "train.jsonl",
        "train_seen_speakers": "train_seen_speakers.jsonl",
        "train_unseen_speakers": "train_unseen_speakers.jsonl",
        "val": "val.jsonl",
        "val_seen_speakers": "val_seen_speakers.jsonl",
        "val_unseen_speakers": "val_unseen_speakers.jsonl",
    }
    components = {}
    for name, filename in component_files.items():
        rows = split[name]
        components[name] = {
            "manifest": filename,
            "utterance_id_sha256": sha256_text(sorted(row["utterance_id"] for row in rows)),
            "summary": summarize(rows),
        }

    return {
        "dataset": "veprad",
        "split_id": split_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_manifest": source_manifest.as_posix(),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_utterance_id_sha256": sha256_text(source_ids),
        "source_summary": summarize(source_rows),
        "selection_policy": {
            "description": (
                "Cross-validation split: all utterances from randomly selected "
                "validation speakers plus ceil(ratio * utterances) from every "
                "remaining speaker."
            ),
            "validation_unseen_speaker_count": len(split["selection"]["validation_unseen_speakers"]),
            "same_speaker_validation_ratio": split["selection"]["same_speaker_validation_ratio"],
            "rounding": "ceil_per_speaker",
        },
        "selection": split["selection"],
        "component_definitions": {
            "seen_speakers": (
                "Speakers represented in both train and validation. Their "
                "validation rows are the per-speaker 10% utterance holdout."
            ),
            "unseen_speakers": (
                "Validation speakers held out completely from train. "
                "train_unseen_speakers.jsonl is intentionally empty."
            ),
        },
        "components": components,
    }


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    source_manifest = project_root / args.source_manifest
    output_root = project_root / args.output_root
    split_dir = output_root / args.split_id

    if not source_manifest.exists():
        raise SystemExit(f"Missing source manifest: {source_manifest}")
    if split_dir.exists():
        if args.force and not args.dry_run:
            shutil.rmtree(split_dir)
        else:
            raise SystemExit(f"Refusing to overwrite existing split: {split_dir}")

    source_rows = read_jsonl(source_manifest)
    validate_source_rows(project_root, source_rows)
    split = select_split(
        rows=source_rows,
        val_speaker_count=args.val_speakers,
        same_speaker_val_ratio=args.same_speaker_val_ratio,
        seed=args.seed,
    )
    validate_split(source_rows, split)
    meta = split_meta(args.split_id, args.source_manifest, source_rows, split)

    print("Split ID:", args.split_id)
    print("Source utterances:", len(source_rows))
    print("Validation unseen speakers:", ", ".join(split["selection"]["validation_unseen_speakers"]))
    print("Train utterances:", len(split["train"]))
    print("Validation utterances:", len(split["val"]))
    print("Validation unseen-speaker utterances:", len(split["val_unseen_speakers"]))
    print("Validation seen-speaker utterances:", len(split["val_seen_speakers"]))
    print("Train unseen-speaker utterances:", len(split["train_unseen_speakers"]))
    print("Train seen-speaker utterances:", len(split["train_seen_speakers"]))

    if args.dry_run:
        print("Dry run only; no files written.")
        return 0

    split_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "train",
        "train_seen_speakers",
        "train_unseen_speakers",
        "val",
        "val_seen_speakers",
        "val_unseen_speakers",
    ):
        write_jsonl(split_dir / f"{name}.jsonl", split[name])
    write_json(split_dir / "split.meta.json", meta)
    print(f"Wrote {split_dir}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--val-speakers", type=int, default=3)
    parser.add_argument("--same-speaker-val-ratio", type=float, default=0.10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed. Omit for a random CV split.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
