#!/usr/bin/env python3
"""Convert this project's VEPRAD JSONL manifests to PaddleSpeech raw JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def convert_split(
    split: str,
    project_root: Path,
    output_root: Path,
    check_audio: bool,
) -> tuple[int, int]:
    src = project_root / "data" / "manifests" / "veprad" / f"{split}.jsonl"
    dst = output_root / "manifests" / f"{split}.raw.jsonl"
    if not src.exists():
        raise SystemExit(f"Missing input manifest: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    missing_audio = 0

    with dst.open("w", encoding="utf-8") as out:
        for row in read_jsonl(src):
            audio_relpath = row.get("audio_relpath")
            if not audio_relpath:
                raise SystemExit(f"{src}: row is missing audio_relpath")

            audio_path = project_root / audio_relpath
            if check_audio and not audio_path.exists():
                missing_audio += 1

            out_row = {
                "utt": row.get("utterance_id") or Path(audio_relpath).stem,
                "utt2spk": row.get("speaker_id") or "unknown",
                "feat": audio_relpath,
                "feat_shape": [float(row["duration"])],
                "text": row["text"],
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            count += 1

    return count, missing_audio


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert VEPRAD project manifests into PaddleSpeech raw manifests."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Root containing data/manifests/veprad and data/raw/veprad.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output root. Writes manifests/{train,dev,test}.raw.jsonl.",
    )
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Verify that referenced audio files exist under --project-root.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    output_root = args.output_root

    total = 0
    total_missing = 0
    for split in SPLITS:
        count, missing_audio = convert_split(
            split=split,
            project_root=project_root,
            output_root=output_root,
            check_audio=args.check_audio,
        )
        total += count
        total_missing += missing_audio
        print(f"{split}: rows={count} missing_audio={missing_audio}")

    print(f"total: rows={total} missing_audio={total_missing}")
    return 1 if total_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
