#!/usr/bin/env python3
"""Recompute WER and CER from a PaddleSpeech DeepSpeech2 result JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def edit_distance(left, right) -> int:
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def load_rows(result_file: Path):
    with result_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{result_file}:{line_number}: invalid JSON: {exc}") from exc

            try:
                ref = row["refs"][0]
                hyp = row["hyps"][0]
            except (KeyError, IndexError) as exc:
                raise SystemExit(f"{result_file}:{line_number}: missing refs/hyps") from exc

            yield row.get("utt", f"row_{line_number}"), ref, hyp


def compute(result_file: Path):
    rows = []
    word_edits = 0
    word_total = 0
    char_edits = 0
    char_total = 0
    char_ns_edits = 0
    char_ns_total = 0
    exact = 0

    for utt, ref, hyp in load_rows(result_file):
        ref_words = ref.split()
        hyp_words = hyp.split()
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        ref_chars_no_space = list(ref.replace(" ", ""))
        hyp_chars_no_space = list(hyp.replace(" ", ""))

        we = edit_distance(ref_words, hyp_words)
        ce = edit_distance(ref_chars, hyp_chars)
        cne = edit_distance(ref_chars_no_space, hyp_chars_no_space)

        word_edits += we
        word_total += len(ref_words)
        char_edits += ce
        char_total += len(ref_chars)
        char_ns_edits += cne
        char_ns_total += len(ref_chars_no_space)
        exact += ref == hyp

        rows.append(
            {
                "utt": utt,
                "wer": we / len(ref_words) if ref_words else 0.0,
                "wer_edits": we,
                "ref_words": len(ref_words),
                "cer": ce / len(ref_chars) if ref_chars else 0.0,
                "cer_edits": ce,
                "ref_chars": len(ref_chars),
                "cer_no_space": cne / len(ref_chars_no_space)
                if ref_chars_no_space
                else 0.0,
                "cer_no_space_edits": cne,
                "ref_chars_no_space": len(ref_chars_no_space),
                "exact": ref == hyp,
                "ref": ref,
                "hyp": hyp,
            }
        )

    metrics = {
        "num_utterances": len(rows),
        "exact_match_utterances": exact,
        "exact_match_rate": exact / len(rows) if rows else 0.0,
        "word_edits": word_edits,
        "reference_words": word_total,
        "wer": word_edits / word_total if word_total else 0.0,
        "char_edits_including_spaces": char_edits,
        "reference_chars_including_spaces": char_total,
        "cer_including_spaces": char_edits / char_total if char_total else 0.0,
        "char_edits_no_spaces": char_ns_edits,
        "reference_chars_no_spaces": char_ns_total,
        "cer_no_spaces": char_ns_edits / char_ns_total if char_ns_total else 0.0,
        "macro_wer": statistics.mean(r["wer"] for r in rows) if rows else 0.0,
        "macro_cer_including_spaces": statistics.mean(r["cer"] for r in rows)
        if rows
        else 0.0,
        "macro_cer_no_spaces": statistics.mean(r["cer_no_space"] for r in rows)
        if rows
        else 0.0,
    }
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--utterance-out", type=Path)
    args = parser.parse_args()

    metrics, rows = compute(args.result_file)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.utterance_out:
        args.utterance_out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "utt",
            "wer",
            "wer_edits",
            "ref_words",
            "cer",
            "cer_edits",
            "ref_chars",
            "cer_no_space",
            "cer_no_space_edits",
            "ref_chars_no_space",
            "exact",
            "ref",
            "hyp",
        ]
        with args.utterance_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in sorted(rows, key=lambda item: (item["wer"], item["cer"]), reverse=True):
                writer.writerow({field: row[field] for field in fields})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
