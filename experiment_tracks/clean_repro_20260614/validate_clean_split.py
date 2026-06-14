#!/usr/bin/env python3
"""Validate clean VEPRAD split isolation and optional KenLM text filtering."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DIR = Path("data/test")
MANIFEST_NAMES = (
    "train",
    "train_seen_speakers",
    "train_unseen_speakers",
    "val",
    "val_seen_speakers",
    "val_unseen_speakers",
)
TEST_NAMES = ("test", "test_seen_speakers", "test_unseen_speakers")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").split())


def rel_to_project(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_manifests(project_root: Path, split_dir: Path, test_dir: Path) -> dict[str, list[dict]]:
    manifests: dict[str, list[dict]] = {}
    for name in MANIFEST_NAMES:
        path = split_dir / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing split manifest: {path}")
        manifests[name] = read_jsonl(path)
    for name in TEST_NAMES:
        path = test_dir / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing frozen test manifest: {path}")
        manifests[name] = read_jsonl(path)
    return manifests


def summarize(rows: list[dict]) -> dict:
    speakers = Counter(row.get("speaker_id", "unknown") for row in rows)
    seconds = sum(float(row.get("duration") or 0.0) for row in rows)
    return {
        "utterances": len(rows),
        "speaker_count": len(speakers),
        "total_hours": round(seconds / 3600.0, 6),
        "speakers": dict(sorted(speakers.items())),
    }


def ids(rows: list[dict]) -> set[str]:
    return {str(row.get("utterance_id")) for row in rows}


def speakers(rows: list[dict]) -> set[str]:
    return {str(row.get("speaker_id")) for row in rows}


def check_no_sm04(manifests: dict[str, list[dict]], errors: list[str]) -> None:
    for name, rows in manifests.items():
        bad = [
            row.get("utterance_id")
            for row in rows
            if row.get("speaker_id") == "sm04" or row.get("corpus") == "sm04"
        ]
        if bad:
            errors.append(f"{name} contains sm04 rows: {bad[:10]}")


def check_paths(project_root: Path, manifests: dict[str, list[dict]], errors: list[str]) -> None:
    for name, rows in manifests.items():
        expected_prefix = (
            "data/test/"
            if name.startswith("test")
            else "data/cross_validation_splits/raw_train_val/"
        )
        for row in rows:
            utt = row.get("utterance_id", "<unknown>")
            for key in ("audio_relpath", "audio_filepath", "transcript_relpath", "transcript_filepath"):
                value = row.get(key)
                if not value:
                    errors.append(f"{name}:{utt} missing {key}")
                    continue
                path = Path(value)
                if path.is_absolute():
                    errors.append(f"{name}:{utt} has absolute {key}: {value}")
                if key in {"audio_relpath", "transcript_relpath"}:
                    if not value.startswith(expected_prefix):
                        errors.append(
                            f"{name}:{utt} {key} outside expected prefix {expected_prefix}: {value}"
                        )
                    if not (project_root / value).exists():
                        errors.append(f"{name}:{utt} missing referenced file: {value}")


def check_components(manifests: dict[str, list[dict]], errors: list[str]) -> None:
    train = ids(manifests["train"])
    val = ids(manifests["val"])
    test = ids(manifests["test"])

    for left_name, left_ids, right_name, right_ids in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        overlap = sorted(left_ids & right_ids)
        if overlap:
            errors.append(f"{left_name}/{right_name} utterance overlap: {overlap[:10]}")

    train_parts = ids(manifests["train_seen_speakers"]) | ids(manifests["train_unseen_speakers"])
    val_parts = ids(manifests["val_seen_speakers"]) | ids(manifests["val_unseen_speakers"])
    test_parts = ids(manifests["test_seen_speakers"]) | ids(manifests["test_unseen_speakers"])
    if train_parts != train:
        errors.append("train components do not exactly equal train")
    if val_parts != val:
        errors.append("validation components do not exactly equal validation")
    if test_parts != test:
        errors.append("test components do not exactly equal frozen test")
    if ids(manifests["train_seen_speakers"]) & ids(manifests["train_unseen_speakers"]):
        errors.append("train seen/unseen components overlap")
    if ids(manifests["val_seen_speakers"]) & ids(manifests["val_unseen_speakers"]):
        errors.append("val seen/unseen components overlap")
    if ids(manifests["test_seen_speakers"]) & ids(manifests["test_unseen_speakers"]):
        errors.append("test seen/unseen components overlap")


def check_speaker_policy(manifests: dict[str, list[dict]], errors: list[str]) -> dict:
    train_speakers = speakers(manifests["train"])
    val_seen = speakers(manifests["val_seen_speakers"])
    val_unseen = speakers(manifests["val_unseen_speakers"])
    test_unseen = speakers(manifests["test_unseen_speakers"])

    missing_val_seen = sorted(val_seen - train_speakers)
    if missing_val_seen:
        errors.append(f"val_seen speakers missing from train: {missing_val_seen}")

    val_unseen_leak = sorted(val_unseen & train_speakers)
    if val_unseen_leak:
        errors.append(f"val_unseen speakers appear in train: {val_unseen_leak}")

    test_unseen_leak = sorted(test_unseen & (train_speakers | speakers(manifests["val"])))
    if test_unseen_leak:
        errors.append(f"test_unseen speakers appear in CV train/val: {test_unseen_leak}")

    return {
        "train_speakers": sorted(train_speakers),
        "val_seen_speakers": sorted(val_seen),
        "val_unseen_speakers": sorted(val_unseen),
        "test_seen_speakers": sorted(speakers(manifests["test_seen_speakers"])),
        "test_unseen_speakers": sorted(test_unseen),
    }


def text_overlap_report(
    train_rows: list[dict],
    heldout_rows: list[dict],
    threshold: float,
    length_ratio_min: float,
) -> dict:
    heldout_texts = [normalize_text(row.get("text", "")) for row in heldout_rows]
    heldout_counts = Counter(text for text in heldout_texts if text)
    heldout_unique = list(heldout_counts)
    exact = 0
    fuzzy = 0
    fuzzy_examples = []

    for row in train_rows:
        text = normalize_text(row.get("text", ""))
        if not text:
            continue
        if text in heldout_counts:
            exact += 1
            continue
        best_ratio = 0.0
        best_text = None
        for candidate in heldout_unique:
            if not candidate:
                continue
            length_ratio = min(len(text), len(candidate)) / max(len(text), len(candidate))
            if length_ratio < length_ratio_min:
                continue
            matcher = SequenceMatcher(None, text, candidate, autojunk=False)
            if matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_text = candidate
        if best_text is not None and best_ratio >= threshold:
            fuzzy += 1
            if len(fuzzy_examples) < 10:
                fuzzy_examples.append(
                    {
                        "utterance_id": row.get("utterance_id"),
                        "ratio": round(best_ratio, 6),
                        "text": text,
                        "matched_heldout_text": best_text,
                    }
                )
    return {
        "threshold": threshold,
        "length_ratio_min": length_ratio_min,
        "exact_train_rows_matching_heldout_text": exact,
        "fuzzy_train_rows_matching_heldout_text": fuzzy,
        "fuzzy_examples": fuzzy_examples,
    }


def load_lm_texts(project_root: Path, lm_dir: Path) -> tuple[list[str], dict]:
    meta_path = lm_dir / "lm.meta.json"
    text_path = lm_dir / "train_text.txt"
    if not meta_path.exists():
        raise SystemExit(f"Missing KenLM meta: {meta_path}")
    if not text_path.exists():
        raise SystemExit(f"Missing KenLM train text: {text_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    texts = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts, meta


def lm_filter_report(
    project_root: Path,
    lm_dir: Path,
    heldout_rows: list[dict],
    threshold: float,
    length_ratio_min: float,
) -> dict:
    lm_texts, meta = load_lm_texts(project_root, lm_dir)
    pseudo_rows = [{"utterance_id": f"lm_text_{i}", "text": text} for i, text in enumerate(lm_texts)]
    overlap = text_overlap_report(
        pseudo_rows,
        heldout_rows,
        threshold=threshold,
        length_ratio_min=length_ratio_min,
    )
    passed = (
        overlap["exact_train_rows_matching_heldout_text"] == 0
        and overlap["fuzzy_train_rows_matching_heldout_text"] == 0
    )
    return {
        "lm_dir": rel_to_project(project_root, lm_dir),
        "passed": passed,
        "lm_meta_filter_summary": meta.get("lm_text_filter", {}),
        "post_filter_overlap": overlap,
    }


def validate(args: argparse.Namespace) -> dict:
    project_root = args.project_root.resolve()
    split_dir = args.split_dir if args.split_dir.is_absolute() else project_root / args.split_dir
    test_dir = args.test_dir if args.test_dir.is_absolute() else project_root / args.test_dir
    manifests = load_manifests(project_root, split_dir, test_dir)

    errors: list[str] = []
    check_no_sm04(manifests, errors)
    check_paths(project_root, manifests, errors)
    check_components(manifests, errors)
    speaker_report = check_speaker_policy(manifests, errors)

    heldout_rows = manifests["val"] + manifests["test"]
    text_overlap = text_overlap_report(
        manifests["train"],
        heldout_rows,
        threshold=args.fuzzy_threshold,
        length_ratio_min=args.fuzzy_length_ratio_min,
    )

    lm_report = None
    if args.lm_dir:
        lm_dir = args.lm_dir if args.lm_dir.is_absolute() else project_root / args.lm_dir
        lm_report = lm_filter_report(
            project_root,
            lm_dir,
            heldout_rows,
            threshold=args.fuzzy_threshold,
            length_ratio_min=args.fuzzy_length_ratio_min,
        )
        if not lm_report["passed"]:
            errors.append("KenLM post-filter corpus still matches held-out text")

    report = {
        "schema_version": 1,
        "split_dir": rel_to_project(project_root, split_dir),
        "test_dir": rel_to_project(project_root, test_dir),
        "passed": not errors,
        "errors": errors,
        "summaries": {name: summarize(rows) for name, rows in manifests.items()},
        "speaker_policy": speaker_report,
        "pre_lm_train_text_overlap_report": text_overlap,
    }
    if lm_report is not None:
        report["kenlm_filter_validation"] = lm_report
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--lm-dir", type=Path)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.90)
    parser.add_argument("--fuzzy-length-ratio-min", type=float, default=0.80)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = args.out if args.out.is_absolute() else args.project_root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
