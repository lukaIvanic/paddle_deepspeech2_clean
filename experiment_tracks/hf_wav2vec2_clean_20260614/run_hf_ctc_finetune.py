#!/usr/bin/env python3
"""Fine-tune a Hugging Face CTC+LM ASR model on clean VEPRAD splits."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_tracks.clean_repro_20260614.validate_clean_split import validate as validate_split
from scripts import create_cv_split, train_kenlm_lm

TRACK_ID = "hf_wav2vec2_clean_20260614"
DEFAULT_SPLIT_ID = "cv_hf_wav2vec2_clean_20260614_001"
DEFAULT_MODEL = "classla/wav2vec2-large-slavic-parlaspeech-hr-lm"
EVAL_COMPONENTS = (
    "val",
    "val_seen_speakers",
    "val_unseen_speakers",
    "test",
    "test_seen_speakers",
    "test_unseen_speakers",
)


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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def rel_to_project(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def maybe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").lower().replace("|", " ").split())


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


def compute_metrics(pairs: list[dict]) -> dict:
    word_edits = word_total = 0
    char_edits = char_total = 0
    char_ns_edits = char_ns_total = 0
    exact = 0
    per_utt = []

    for item in pairs:
        ref = normalize_text(item["ref"])
        hyp = normalize_text(item["hyp"])
        ref_words = ref.split()
        hyp_words = hyp.split()
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        ref_chars_ns = list(ref.replace(" ", ""))
        hyp_chars_ns = list(hyp.replace(" ", ""))

        we = edit_distance(ref_words, hyp_words)
        ce = edit_distance(ref_chars, hyp_chars)
        cne = edit_distance(ref_chars_ns, hyp_chars_ns)
        word_edits += we
        word_total += len(ref_words)
        char_edits += ce
        char_total += len(ref_chars)
        char_ns_edits += cne
        char_ns_total += len(ref_chars_ns)
        exact += ref == hyp
        per_utt.append(
            {
                "utterance_id": item["utterance_id"],
                "wer": we / len(ref_words) if ref_words else 0.0,
                "cer_including_spaces": ce / len(ref_chars) if ref_chars else 0.0,
                "cer_no_spaces": cne / len(ref_chars_ns) if ref_chars_ns else 0.0,
            }
        )

    return {
        "num_utterances": len(pairs),
        "exact_match_utterances": exact,
        "exact_match_rate": exact / len(pairs) if pairs else 0.0,
        "word_edits": word_edits,
        "reference_words": word_total,
        "wer": word_edits / word_total if word_total else 0.0,
        "char_edits_including_spaces": char_edits,
        "reference_chars_including_spaces": char_total,
        "cer_including_spaces": char_edits / char_total if char_total else 0.0,
        "char_edits_no_spaces": char_ns_edits,
        "reference_chars_no_spaces": char_ns_total,
        "cer_no_spaces": char_ns_edits / char_ns_total if char_ns_total else 0.0,
        "macro_wer": statistics.mean([row["wer"] for row in per_utt]) if per_utt else 0.0,
        "macro_cer_including_spaces": statistics.mean(
            [row["cer_including_spaces"] for row in per_utt]
        )
        if per_utt
        else 0.0,
        "macro_cer_no_spaces": statistics.mean([row["cer_no_spaces"] for row in per_utt])
        if per_utt
        else 0.0,
    }


def ensure_split(args: argparse.Namespace) -> Path:
    split_dir = args.project_root / "data" / "cross_validation_splits" / args.split_id
    required = ["train.jsonl", "val.jsonl", "val_seen_speakers.jsonl", "val_unseen_speakers.jsonl"]
    if split_dir.exists():
        missing = [name for name in required if not (split_dir / name).exists()]
        if args.force_split:
            shutil.rmtree(split_dir)
        elif missing:
            raise SystemExit(
                f"Split directory exists but is missing manifest files: {split_dir} "
                f"({', '.join(missing)}). Use --force-split or a new --split-id."
            )
        else:
            return split_dir

    ns = SimpleNamespace(
        project_root=args.project_root,
        source_manifest=Path("data/cross_validation_splits/raw_train_val/source.jsonl"),
        output_root=Path("data/cross_validation_splits"),
        split_id=args.split_id,
        val_speakers=3,
        same_speaker_val_ratio=0.10,
        seed=args.split_seed,
        dry_run=False,
        force=False,
    )
    create_cv_split.run(ns)
    return split_dir


def validate_to_file(
    args: argparse.Namespace, split_dir: Path, out: Path, lm_dir: Path | None = None
) -> dict:
    report = validate_split(
        SimpleNamespace(
            project_root=args.project_root,
            split_dir=split_dir,
            test_dir=Path("data/test"),
            lm_dir=lm_dir,
            fuzzy_threshold=args.fuzzy_threshold,
            fuzzy_length_ratio_min=args.fuzzy_length_ratio_min,
        )
    )
    write_json(out, report)
    if not report["passed"]:
        raise SystemExit(f"Validation failed; see {out}")
    return report


def manifest_paths(project_root: Path, split_dir: Path) -> dict[str, Path]:
    return {
        "train": split_dir / "train.jsonl",
        "val": split_dir / "val.jsonl",
        "val_seen_speakers": split_dir / "val_seen_speakers.jsonl",
        "val_unseen_speakers": split_dir / "val_unseen_speakers.jsonl",
        "test": project_root / "data" / "test" / "test.jsonl",
        "test_seen_speakers": project_root / "data" / "test" / "test_seen_speakers.jsonl",
        "test_unseen_speakers": project_root / "data" / "test" / "test_unseen_speakers.jsonl",
    }


def train_lm(args: argparse.Namespace, split_dir: Path, lm_dir: Path) -> dict:
    argv = [
        "--project-root",
        str(args.project_root),
        "--train-manifest",
        str(split_dir / "train.jsonl"),
        "--output-dir",
        str(lm_dir),
        "--order",
        str(args.kenlm_order),
        "--memory",
        args.kenlm_memory,
        "--fuzzy-exclude-threshold",
        str(args.fuzzy_threshold),
        "--fuzzy-length-ratio-min",
        str(args.fuzzy_length_ratio_min),
    ]
    if args.force_run:
        argv.append("--force")
    train_kenlm_lm.main(argv)
    return json.loads((lm_dir / "lm.meta.json").read_text(encoding="utf-8"))


class CTCManifestDataset:
    def __init__(self, rows: list[dict], project_root: Path, processor, with_labels: bool = True):
        self.rows = rows
        self.project_root = project_root
        self.processor = processor
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        import soundfile as sf

        row = self.rows[idx]
        audio_path = self.project_root / row["audio_filepath"]
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise RuntimeError(f"Expected 16 kHz audio, got {sample_rate}: {audio_path}")

        item = {
            "input_values": self.processor(
                audio, sampling_rate=sample_rate
            ).input_values[0],
            "utterance_id": row["utterance_id"],
            "speaker_id": row.get("speaker_id"),
            "text": row.get("text", ""),
        }
        if self.with_labels:
            item["labels"] = self.processor.tokenizer(row.get("text", "")).input_ids
        return item


class DataCollatorCTCWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")

        if "labels" in features[0]:
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, padding=True, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
        return batch


def eval_collate(processor, features: list[dict]) -> dict:
    batch = processor.pad(
        [{"input_values": feature["input_values"]} for feature in features],
        padding=True,
        return_tensors="pt",
    )
    batch["meta"] = [
        {
            "utterance_id": feature["utterance_id"],
            "speaker_id": feature.get("speaker_id"),
            "text": feature.get("text", ""),
        }
        for feature in features
    ]
    return batch


def tokenizer_coverage(processor, rows_by_name: dict[str, list[dict]]) -> dict:
    vocab = set(processor.tokenizer.get_vocab())
    delimiter = getattr(processor.tokenizer, "word_delimiter_token", "|")
    by_subset = {}
    for name, rows in rows_by_name.items():
        chars = Counter()
        for row in rows:
            chars.update(row.get("text", ""))
        missing = []
        for char in sorted(chars):
            token = delimiter if char == " " else char
            if token not in vocab:
                missing.append(char)
        by_subset[name] = {
            "characters": "".join(sorted(chars)),
            "missing_characters": missing,
            "missing_count": len(missing),
        }
    return {
        "tokenizer_vocab_size": len(vocab),
        "word_delimiter_token": delimiter,
        "subsets": by_subset,
    }


def pyctcdecode_labels(processor) -> list[str]:
    tokenizer = processor.tokenizer
    vocab_size = len(tokenizer)
    specials = set(tokenizer.all_special_tokens)
    labels = []
    for idx in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(idx)
        if idx == tokenizer.pad_token_id:
            labels.append("")
        elif token == getattr(tokenizer, "word_delimiter_token", "|"):
            labels.append(" ")
        elif token in specials:
            labels.append("")
        else:
            labels.append(token)
    return labels


def build_lm_decoder(processor, lm_path: Path, alpha: float, beta: float):
    from pyctcdecode import build_ctcdecoder

    return build_ctcdecoder(
        pyctcdecode_labels(processor),
        kenlm_model_path=str(lm_path),
        alpha=alpha,
        beta=beta,
    )


def evaluate_subset(
    args: argparse.Namespace,
    model,
    processor,
    rows: list[dict],
    subset: str,
    out_dir: Path,
    decoder_name: str = "greedy",
    decoder_mode: str = "greedy",
    local_decoder=None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    device = torch.device(args.device)
    model.eval()
    dataset = CTCManifestDataset(rows, args.project_root, processor, with_labels=False)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda features: eval_collate(processor, features),
    )

    predictions = []
    progress = tqdm(loader, desc=f"eval {decoder_name}:{subset}", leave=False)
    with torch.no_grad():
        for batch in progress:
            meta = batch.pop("meta")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16 and args.device == "cuda"):
                logits = model(**batch).logits

            if decoder_mode == "greedy":
                pred_ids = torch.argmax(logits, dim=-1)
                hyps = processor.tokenizer.batch_decode(pred_ids)
            elif decoder_mode == "bundled_lm":
                decoded = processor.batch_decode(
                    logits.detach().cpu().numpy(),
                    beam_width=args.bundled_lm_beam,
                )
                hyps = decoded.text if hasattr(decoded, "text") else decoded
            elif decoder_mode == "local_kenlm":
                if local_decoder is None:
                    raise RuntimeError("local_kenlm decoder requested without decoder")
                logits_np = logits.detach().cpu().numpy()
                hyps = [
                    local_decoder.decode(logit, beam_width=args.kenlm_beam)
                    for logit in logits_np
                ]
            else:
                raise RuntimeError(f"Unknown decoder mode: {decoder_mode}")

            for item, hyp in zip(meta, hyps):
                predictions.append(
                    {
                        "utt": item["utterance_id"],
                        "utterance_id": item["utterance_id"],
                        "speaker_id": item.get("speaker_id"),
                        "refs": [normalize_text(item["text"])],
                        "hyps": [normalize_text(hyp)],
                        "ref": normalize_text(item["text"]),
                        "hyp": normalize_text(hyp),
                    }
                )

    metrics = compute_metrics(predictions)
    metrics.update({"subset": subset, "decoder": decoder_name})
    write_json(out_dir / "metrics" / decoder_name / f"{subset}.json", metrics)
    write_jsonl(out_dir / "details" / "predictions" / decoder_name / f"{subset}.jsonl", predictions)
    return metrics


def training_arguments(args: argparse.Namespace, output_dir: Path):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "fp16": args.fp16 and args.device == "cuda",
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "report_to": [],
        "remove_unused_columns": False,
        "group_by_length": True,
        "dataloader_num_workers": args.num_workers,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if args.warmup_steps is not None:
        kwargs["warmup_steps"] = args.warmup_steps
    else:
        kwargs["warmup_ratio"] = args.warmup_ratio
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    return TrainingArguments(**kwargs)


def load_model_and_processor(args: argparse.Namespace, rows_by_name: dict[str, list[dict]]):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor, set_seed

    if args.seed is not None:
        set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCTC.from_pretrained(args.model_name)
    model.config.ctc_loss_reduction = "mean"
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    if args.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    coverage = tokenizer_coverage(processor, rows_by_name)
    write_json(args.results_root / "details" / "tokenizer_coverage.json", coverage)
    missing = {
        name: report["missing_characters"]
        for name, report in coverage["subsets"].items()
        if report["missing_characters"]
    }
    if missing and not args.allow_missing_tokenizer_chars:
        raise SystemExit(f"Tokenizer is missing VEPRAD characters: {missing}")

    has_bundled_lm = hasattr(processor, "decoder")
    if not has_bundled_lm and not args.allow_no_bundled_lm:
        raise SystemExit(
            f"Processor for {args.model_name} does not expose a bundled LM decoder. "
            "Use an *-lm checkpoint or pass --allow-no-bundled-lm."
        )
    return model.to(args.device), processor, {
        "tokenizer_coverage": coverage,
        "processor_class": processor.__class__.__name__,
        "model_class": model.__class__.__name__,
        "has_bundled_lm": has_bundled_lm,
        "local_tokenizer_fix": maybe_read_json(Path(args.model_name) / "LOCAL_TOKENIZER_FIX.json")
        if Path(args.model_name).exists()
        else None,
    }


def evaluate_all(
    args: argparse.Namespace,
    model,
    processor,
    rows_by_name: dict[str, list[dict]],
    phase: str,
    local_decoder=None,
) -> dict:
    metrics = {}
    decoder_specs = [("greedy", "greedy")]
    if hasattr(processor, "decoder"):
        decoder_specs.append(("bundled_lm", "bundled_lm"))
    if local_decoder is not None:
        decoder_specs.append(("local_veprad_kenlm", "local_kenlm"))

    for short_name, mode in decoder_specs:
        report_name = f"{phase}_{short_name}"
        metrics[report_name] = {}
        for subset in EVAL_COMPONENTS:
            metrics[report_name][subset] = evaluate_subset(
                args,
                model,
                processor,
                rows_by_name[subset],
                subset,
                args.results_root,
                decoder_name=report_name,
                decoder_mode=mode,
                local_decoder=local_decoder,
            )
    return metrics


def fine_tune(args: argparse.Namespace, model, processor, rows_by_name: dict[str, list[dict]]) -> dict:
    from transformers import Trainer

    train_dataset = CTCManifestDataset(rows_by_name["train"], args.project_root, processor)
    collator = DataCollatorCTCWithPadding(processor)
    trainer = Trainer(
        model=model,
        args=training_arguments(args, args.results_root / "details" / "model"),
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=processor.feature_extractor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.results_root / "details" / "model" / "final"))
    processor.save_pretrained(str(args.results_root / "details" / "model" / "final"))

    train_summary = {
        "global_step": trainer.state.global_step,
        "epochs": args.epochs,
        "train_utterances": len(rows_by_name["train"]),
        "model_name": args.model_name,
        "freeze_feature_encoder": args.freeze_feature_encoder,
        "gradient_checkpointing": args.gradient_checkpointing,
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio if args.warmup_steps is None else None,
    }
    write_json(args.results_root / "training_summary.json", train_summary)
    return train_summary


def write_summary_markdown(path: Path, report: dict) -> None:
    lines = [
        f"# {report['track_id']}",
        "",
        f"- Model: `{report['model_name']}`",
        f"- Split: `{report['split_id']}`",
        f"- Train utterances: `{report['training']['train_utterances']}`",
        f"- Epochs: `{report['training']['epochs']}`",
        f"- Bundled LM: `{report['external_lm']['has_bundled_lm']}`",
        "",
        "## Metrics",
        "",
        "| Decoder | Subset | WER | CER spaces | CER no spaces |",
        "|---|---:|---:|---:|---:|",
    ]
    for decoder_name in report["metrics"]:
        for subset in EVAL_COMPONENTS:
            metrics = report["metrics"][decoder_name][subset]
            lines.append(
                f"| {decoder_name} | {subset} | {metrics['wer']:.6f} | "
                f"{metrics['cer_including_spaces']:.6f} | {metrics['cer_no_spaces']:.6f} |"
            )
    local_lm = report.get("local_veprad_kenlm") or {}
    if local_lm.get("enabled") and local_lm.get("meta"):
        filter_meta = local_lm["meta"]["lm_text_filter"]
        lines.extend(
            [
                "",
                "## Local VEPRAD KenLM",
                "",
                f"- Order: `{local_lm['meta']['kenlm']['order']}`",
                f"- Kept train rows: `{filter_meta['train_rows_after_filter']}` / "
                f"`{filter_meta['train_rows_before_filter']}`",
                f"- Excluded exact/fuzzy rows: `{filter_meta['excluded_exact_rows']}` / "
                f"`{filter_meta['excluded_fuzzy_rows']}`",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--track-id", default=TRACK_ID)
    parser.add_argument("--split-id", default=DEFAULT_SPLIT_ID)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-feature-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing-tokenizer-chars", action="store_true")
    parser.add_argument("--allow-no-bundled-lm", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--skip-pre-finetune-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--bundled-lm-beam", type=int, default=100)
    parser.add_argument("--train-local-veprad-kenlm", action="store_true")
    parser.add_argument("--kenlm-order", type=int, default=5)
    parser.add_argument("--kenlm-memory", default="80%")
    parser.add_argument("--kenlm-beam", type=int, default=40)
    parser.add_argument("--kenlm-alpha", type=float, default=2.5)
    parser.add_argument("--kenlm-beta", type=float, default=0.3)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.90)
    parser.add_argument("--fuzzy-length-ratio-min", type=float, default=0.80)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.results_root = args.project_root / "results" / args.track_id
    return args


def main() -> int:
    args = parse_args()
    if args.results_root.exists() and args.force_run:
        shutil.rmtree(args.results_root)
    if args.results_root.exists() and any(args.results_root.iterdir()) and not args.force_run:
        raise SystemExit(f"Refusing to overwrite existing results root: {args.results_root}")
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "details").mkdir(exist_ok=True)

    split_dir = ensure_split(args)
    split_validation = validate_to_file(
        args,
        split_dir,
        args.results_root / "details" / "validation" / "split_validation.json",
    )
    paths = manifest_paths(args.project_root, split_dir)
    rows_by_name = {name: read_jsonl(path) for name, path in paths.items()}

    model, processor, load_report = load_model_and_processor(args, rows_by_name)

    metrics = {}
    if not args.skip_pre_finetune_eval:
        metrics.update(evaluate_all(args, model, processor, rows_by_name, "pre_finetune"))

    if args.skip_train:
        training_summary = {
            "global_step": 0,
            "epochs": 0,
            "train_utterances": len(rows_by_name["train"]),
            "model_name": args.model_name,
            "skipped": True,
        }
    else:
        training_summary = fine_tune(args, model, processor, rows_by_name)
        metrics.update(evaluate_all(args, model, processor, rows_by_name, "finetuned"))

    kenlm_meta = None
    lm_validation = None
    if args.train_local_veprad_kenlm:
        lm_dir = args.results_root / "details" / "lm"
        kenlm_meta = train_lm(args, split_dir, lm_dir)
        lm_validation = validate_to_file(
            args,
            split_dir,
            args.results_root / "details" / "validation" / "local_veprad_kenlm_validation.json",
            lm_dir=lm_dir,
        )
        decoder = build_lm_decoder(
            processor,
            lm_dir / "lm.klm",
            alpha=args.kenlm_alpha,
            beta=args.kenlm_beta,
        )
        write_json(
            lm_dir / "decode_config.json",
            {
                "decoder": "pyctcdecode",
                "beam": args.kenlm_beam,
                "alpha": args.kenlm_alpha,
                "beta": args.kenlm_beta,
                "lm_path": rel_to_project(args.project_root, lm_dir / "lm.klm"),
            },
        )
        metrics.update(evaluate_all(args, model, processor, rows_by_name, "finetuned", local_decoder=decoder))

    report = {
        "schema_version": 1,
        "track_id": args.track_id,
        "model_name": args.model_name,
        "split_id": args.split_id,
        "split_dir": rel_to_project(args.project_root, split_dir),
        "results_root": rel_to_project(args.project_root, args.results_root),
        "hyperparameters": {
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_train_batch_size": args.train_batch_size
            * args.gradient_accumulation_steps,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio if args.warmup_steps is None else None,
            "fp16": args.fp16 and args.device == "cuda",
            "seed": args.seed,
            "bundled_lm_beam": args.bundled_lm_beam,
        },
        "split_validation": split_validation,
        "external_lm": {
            "source": args.model_name,
            "description": (
                "Bundled checkpoint LM is fixed external ParlaMint LM from the "
                "Hugging Face model; this script does not train it on VEPRAD."
            ),
            "has_bundled_lm": load_report["has_bundled_lm"],
            "local_tokenizer_fix": load_report["local_tokenizer_fix"],
        },
        "local_veprad_kenlm": {
            "enabled": args.train_local_veprad_kenlm,
            "validation": lm_validation,
            "meta": kenlm_meta,
        },
        "training": training_summary,
        "processor": {
            "processor_class": load_report["processor_class"],
            "model_class": load_report["model_class"],
        },
        "tokenizer_coverage": load_report["tokenizer_coverage"],
        "metrics": metrics,
    }
    write_json(args.results_root / "track_report.json", report)
    write_summary_markdown(args.results_root / "summary.md", report)
    print(f"Wrote {rel_to_project(args.project_root, args.results_root / 'track_report.json')}")
    print(f"Wrote {rel_to_project(args.project_root, args.results_root / 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
