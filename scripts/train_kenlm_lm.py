#!/usr/bin/env python3
"""Train a KenLM n-gram language model from one train-only JSONL manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def rel_to_project(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").split())


def resolve_tool(name: str, override: str | None) -> str:
    if override:
        path = shutil.which(override) or override
        if Path(path).exists():
            return path
        raise SystemExit(f"Missing KenLM tool override for {name}: {override}")
    path = shutil.which(name)
    if path:
        return path
    raise SystemExit(
        f"Missing KenLM tool `{name}` on PATH. Install/build KenLM and rerun."
    )


def default_exclude_manifests(project_root: Path, train_manifest: Path) -> list[Path]:
    manifests: list[Path] = []

    split_val = train_manifest.parent / "val.jsonl"
    if train_manifest.name == "train.jsonl" and split_val.exists():
        manifests.append(split_val)

    frozen_test = project_root / "data" / "test" / "test.jsonl"
    if frozen_test.exists():
        manifests.append(frozen_test)

    return manifests


def resolve_exclude_manifests(
    project_root: Path,
    train_manifest: Path,
    explicit_manifests: list[Path],
    disable_defaults: bool,
) -> list[Path]:
    resolved: list[Path] = []
    if not disable_defaults:
        resolved.extend(default_exclude_manifests(project_root, train_manifest))
    resolved.extend(resolve_project_path(project_root, path) for path in explicit_manifests)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            raise SystemExit(f"Missing exclude manifest: {path}")
        unique.append(path)
    return unique


def summarize_exclude_manifest(
    project_root: Path, path: Path, rows: list[dict], text_field: str
) -> dict:
    texts = [normalize_text(row.get(text_field, "")) for row in rows]
    nonempty = [text for text in texts if text]
    return {
        "path": rel_to_project(project_root, path),
        "sha256": sha256_file(path),
        "utterances": len(rows),
        "nonempty_text_rows": len(nonempty),
        "unique_texts": len(set(nonempty)),
    }


def load_exclusion_text_counts(
    project_root: Path,
    exclude_manifests: list[Path],
    text_field: str,
) -> tuple[Counter, list[dict]]:
    text_counts = Counter()
    summaries: list[dict] = []
    for manifest in exclude_manifests:
        rows = list(read_jsonl(manifest))
        summaries.append(summarize_exclude_manifest(project_root, manifest, rows, text_field))
        for row in rows:
            text = normalize_text(row.get(text_field, ""))
            if text:
                text_counts[text] += 1
    return text_counts, summaries


def is_potential_match_length(text_len: int, candidate_len: int, min_ratio: float) -> bool:
    if text_len == 0 or candidate_len == 0:
        return False
    return min(text_len, candidate_len) / max(text_len, candidate_len) >= min_ratio


def best_text_match(
    text: str,
    candidates: list[str],
    candidate_counts: Counter,
    threshold: float,
    length_ratio_min: float,
) -> tuple[float, str | None, int, str]:
    if text in candidate_counts:
        return 1.0, text, candidate_counts[text], "exact"
    if threshold >= 1.0:
        return 0.0, None, 0, "none"

    best_ratio = 0.0
    best_candidate: str | None = None
    text_len = len(text)
    for candidate in candidates:
        if not is_potential_match_length(text_len, len(candidate), length_ratio_min):
            continue
        matcher = SequenceMatcher(None, text, candidate, autojunk=False)
        if matcher.quick_ratio() < threshold:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_candidate = candidate
            if ratio >= 0.999999:
                break

    if best_candidate is None or best_ratio < threshold:
        return best_ratio, None, 0, "none"
    return best_ratio, best_candidate, candidate_counts[best_candidate], "fuzzy"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def filter_rows_for_lm(
    project_root: Path,
    train_manifest: Path,
    rows: list[dict],
    text_field: str,
    exclude_manifests: list[Path],
    threshold: float,
    length_ratio_min: float,
) -> tuple[list[dict], list[dict], dict]:
    if not (0.0 < threshold <= 1.0):
        raise SystemExit(f"fuzzy exclude threshold must be in (0, 1], got {threshold}")
    if not (0.0 < length_ratio_min <= 1.0):
        raise SystemExit(
            f"fuzzy length ratio minimum must be in (0, 1], got {length_ratio_min}"
        )

    exclude_counts, exclude_summaries = load_exclusion_text_counts(
        project_root, exclude_manifests, text_field
    )
    exclude_texts = list(exclude_counts)

    kept: list[dict] = []
    excluded: list[dict] = []
    exact = 0
    fuzzy = 0
    empty = 0

    for row in rows:
        text = normalize_text(row.get(text_field, ""))
        if not text:
            empty += 1
            kept.append(row)
            continue

        ratio, match_text, match_count, match_type = best_text_match(
            text,
            exclude_texts,
            exclude_counts,
            threshold=threshold,
            length_ratio_min=length_ratio_min,
        )
        if match_text is None:
            kept.append(row)
            continue

        if match_type == "exact":
            exact += 1
        else:
            fuzzy += 1
        excluded.append(
            {
                "utterance_id": row.get("utterance_id"),
                "speaker_id": row.get("speaker_id"),
                "match_type": match_type,
                "match_ratio": round(ratio, 6),
                "heldout_text_count": match_count,
                "text": text,
                "matched_heldout_text": match_text,
            }
        )

    summary = {
        "enabled": bool(exclude_manifests),
        "description": (
            "KenLM corpus excludes train rows whose normalized transcript exactly "
            "or fuzzily matches validation/test text."
        ),
        "train_manifest": rel_to_project(project_root, train_manifest),
        "exclude_manifests": exclude_summaries,
        "text_field": text_field,
        "fuzzy_threshold": threshold,
        "length_ratio_min": length_ratio_min,
        "train_rows_before_filter": len(rows),
        "train_rows_after_filter": len(kept),
        "excluded_rows": len(excluded),
        "excluded_exact_rows": exact,
        "excluded_fuzzy_rows": fuzzy,
        "empty_text_rows_kept": empty,
        "heldout_unique_texts": len(exclude_counts),
        "heldout_text_rows": sum(exclude_counts.values()),
    }
    return kept, excluded, summary


def extract_text(rows: list[dict], text_field: str) -> tuple[list[str], dict]:
    lines: list[str] = []
    speakers = Counter()
    word_counter = Counter()
    char_counter = Counter()
    empty = 0

    for row in rows:
        text = normalize_text(row.get(text_field, ""))
        if not text:
            empty += 1
            continue
        lines.append(text)
        speakers[row.get("speaker_id", "unknown")] += 1
        words = text.split()
        word_counter.update(words)
        char_counter.update(text)

    summary = {
        "utterances": len(rows),
        "nonempty_text_lines": len(lines),
        "empty_text_rows": empty,
        "speaker_count": len(speakers),
        "speakers": dict(sorted(speakers.items())),
        "word_tokens": sum(word_counter.values()),
        "word_types": len(word_counter),
        "character_tokens_including_spaces": sum(char_counter.values()),
        "character_types_including_space": len(char_counter),
    }
    return lines, summary


def write_text_corpus(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def run_command(cmd: list[str], log_path: Path) -> float:
    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"See log: {log_path}"
        )
    return elapsed


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_lm(args: argparse.Namespace) -> dict:
    project_root = args.project_root.resolve()
    train_manifest = resolve_project_path(project_root, args.train_manifest).resolve()
    output_dir = resolve_project_path(project_root, args.output_dir)
    exclude_manifests = resolve_exclude_manifests(
        project_root=project_root,
        train_manifest=train_manifest,
        explicit_manifests=args.exclude_manifest,
        disable_defaults=args.no_default_exclude_manifests,
    )

    if not train_manifest.exists():
        raise SystemExit(f"Missing train manifest: {train_manifest}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.force:
            shutil.rmtree(output_dir)
        else:
            raise SystemExit(f"Refusing to overwrite non-empty LM output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(train_manifest))
    lm_rows, excluded_rows, filter_summary = filter_rows_for_lm(
        project_root=project_root,
        train_manifest=train_manifest,
        rows=rows,
        text_field=args.text_field,
        exclude_manifests=exclude_manifests,
        threshold=args.fuzzy_exclude_threshold,
        length_ratio_min=args.fuzzy_length_ratio_min,
    )
    if excluded_rows:
        write_jsonl(output_dir / "excluded_text_matches.jsonl", excluded_rows)

    text_lines, text_summary = extract_text(lm_rows, args.text_field)
    if not text_lines:
        raise SystemExit(
            f"No non-empty `{args.text_field}` rows left for KenLM after text filtering"
        )

    lmplz = resolve_tool("lmplz", args.lmplz)
    build_binary = resolve_tool("build_binary", args.build_binary)

    text_path = output_dir / "train_text.txt"
    arpa_path = output_dir / "lm.arpa"
    binary_path = output_dir / "lm.klm"
    write_text_corpus(text_path, text_lines)

    lmplz_cmd = [
        lmplz,
        "-o",
        str(args.order),
        "-S",
        args.memory,
    ]
    if args.prune:
        lmplz_cmd.extend(["--prune", *args.prune])
    lmplz_cmd.extend(["--text", str(text_path), "--arpa", str(arpa_path)])

    build_cmd = [
        build_binary,
        "-a",
        str(args.pointer_bits),
        "-q",
        str(args.prob_bits),
        "-b",
        str(args.backoff_bits),
        args.model_type,
        str(arpa_path),
        str(binary_path),
    ]

    started = dt.datetime.now(dt.timezone.utc)
    lmplz_seconds = run_command(lmplz_cmd, output_dir / "lmplz.log")
    build_seconds = run_command(build_cmd, output_dir / "build_binary.log")
    finished = dt.datetime.now(dt.timezone.utc)

    meta = {
        "schema_version": 1,
        "created_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "elapsed_seconds": round(lmplz_seconds + build_seconds, 6),
        "lmplz_seconds": round(lmplz_seconds, 6),
        "build_binary_seconds": round(build_seconds, 6),
        "train_manifest": rel_to_project(project_root, train_manifest),
        "train_manifest_sha256": sha256_file(train_manifest),
        "text_field": args.text_field,
        "text_sha256": sha256_text_lines(text_lines),
        "text_summary": text_summary,
        "lm_text_filter": filter_summary,
        "kenlm": {
            "order": args.order,
            "memory": args.memory,
            "prune": args.prune,
            "model_type": args.model_type,
            "pointer_bits": args.pointer_bits,
            "prob_bits": args.prob_bits,
            "backoff_bits": args.backoff_bits,
            "lmplz": lmplz,
            "build_binary": build_binary,
        },
        "artifacts": {
            "text": {
                "path": rel_to_project(project_root, text_path),
                "sha256": sha256_file(text_path),
                "bytes": text_path.stat().st_size,
            },
            "arpa": {
                "path": rel_to_project(project_root, arpa_path),
                "sha256": sha256_file(arpa_path),
                "bytes": arpa_path.stat().st_size,
            },
            "binary": {
                "path": rel_to_project(project_root, binary_path),
                "sha256": sha256_file(binary_path),
                "bytes": binary_path.stat().st_size,
            },
        },
    }
    if excluded_rows:
        excluded_path = output_dir / "excluded_text_matches.jsonl"
        meta["artifacts"]["excluded_text_matches"] = {
            "path": rel_to_project(project_root, excluded_path),
            "sha256": sha256_file(excluded_path),
            "bytes": excluded_path.stat().st_size,
        }
    write_json(output_dir / "lm.meta.json", meta)
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Heldout manifest whose transcript text must be excluded from the "
            "KenLM corpus. May be passed multiple times. By default, a CV "
            "train.jsonl also excludes sibling val.jsonl and data/test/test.jsonl."
        ),
    )
    parser.add_argument(
        "--no-default-exclude-manifests",
        action="store_true",
        help="Disable automatic exclusion of sibling val.jsonl and frozen test.jsonl.",
    )
    parser.add_argument(
        "--fuzzy-exclude-threshold",
        type=float,
        default=0.90,
        help="Exclude train text with similarity >= this value to heldout text.",
    )
    parser.add_argument(
        "--fuzzy-length-ratio-min",
        type=float,
        default=0.80,
        help="Only fuzzy-compare texts whose shorter/longer character length ratio meets this value.",
    )
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--memory", default="80%")
    parser.add_argument("--prune", nargs="*", default=["0"])
    parser.add_argument("--model-type", choices=("trie", "probing"), default="trie")
    parser.add_argument("--pointer-bits", type=int, default=22)
    parser.add_argument("--prob-bits", type=int, default=8)
    parser.add_argument("--backoff-bits", type=int, default=8)
    parser.add_argument("--lmplz")
    parser.add_argument("--build-binary")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    meta = train_lm(parse_args(argv))
    filter_meta = meta["lm_text_filter"]
    print("KenLM training finished")
    print(f"- binary: {meta['artifacts']['binary']['path']}")
    print(f"- meta: {meta['artifacts']['binary']['path'].rsplit('/', 1)[0]}/lm.meta.json")
    print(
        "- lm_text_rows: "
        f"{filter_meta['train_rows_after_filter']}/"
        f"{filter_meta['train_rows_before_filter']} kept, "
        f"{filter_meta['excluded_rows']} excluded"
    )
    print(f"- elapsed_seconds: {meta['elapsed_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
