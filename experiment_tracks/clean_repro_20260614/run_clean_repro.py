#!/usr/bin/env python3
"""Run the clean VEPRAD reproduction track with split and KenLM validation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import create_cv_split, run_ds2_pipeline, train_kenlm_lm
from experiment_tracks.clean_repro_20260614.validate_clean_split import validate as validate_split


TRACK_ID = "clean_repro_20260614"
DEFAULT_SPLIT_ID = "cv_clean_repro_20260614_001"
MODEL_CONFIGS = {
    "paper_small": Path("conf/deepspeech2_paper_small.yaml"),
    "paper_7gru": Path("conf/deepspeech2_paper_7gru.yaml"),
    "paper_7lstm": Path("conf/deepspeech2_paper_7lstm.yaml"),
}
EVAL_COMPONENTS = (
    "val",
    "val_seen_speakers",
    "val_unseen_speakers",
    "test",
    "test_seen_speakers",
    "test_unseen_speakers",
)


def rel_to_project(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ensure_split(args: argparse.Namespace) -> Path:
    split_dir = args.project_root / "data" / "cross_validation_splits" / args.split_id
    if split_dir.exists():
        required = ["train.jsonl", "val.jsonl", "val_seen_speakers.jsonl", "val_unseen_speakers.jsonl"]
        missing = [name for name in required if not (split_dir / name).exists()]
        if args.force_split:
            shutil.rmtree(split_dir)
        elif missing:
            raise SystemExit(
                f"Split directory exists but is missing ignored manifest files: {split_dir} "
                f"({', '.join(missing)}). Rerun with --force-split or choose a new --split-id."
            )
        else:
            print(f"Using existing split: {rel_to_project(args.project_root, split_dir)}")
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


def validate_to_file(args: argparse.Namespace, split_dir: Path, out: Path, lm_dir: Path | None = None) -> dict:
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


def make_decode_config(
    project_root: Path,
    output_path: Path,
    lm_path: Path,
    beam_size: int,
    alpha: float,
    beta: float,
) -> Path:
    payload = yaml.safe_load((project_root / "conf" / "tuning" / "decode.yaml").read_text())
    payload["lang_model_path"] = rel_to_project(project_root, lm_path)
    payload["beam_size"] = beam_size
    payload["alpha"] = alpha
    payload["beta"] = beta
    payload["decoding_method"] = "ctc_beam_search"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def link_checkpoint(source_run: Path, target_run: Path) -> None:
    source = source_run / "details" / "model" / "checkpoints" / "avg_1.pdparams"
    if not source.exists():
        raise SystemExit(f"Missing trained avg checkpoint: {source}")
    target_dir = target_run / "details" / "model" / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "avg_1.pdparams"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def run_pipeline(args: argparse.Namespace, run_name: str, config: Path, decode_config: Path, skip_train: bool, reuse_run_dir: bool) -> None:
    argv = [
        "--project-root",
        str(args.project_root),
        "--results-root",
        str(args.results_root),
        "--split-dir",
        str(args.split_dir),
        "--run-name",
        run_name,
        "--config",
        str(config),
        "--decode-config",
        str(decode_config),
        "--device",
        args.device,
        "--gpu-id",
        args.gpu_id,
        "--avg-num",
        "1",
        "--write-utterance-metrics",
    ]
    if args.epochs is not None:
        argv.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        argv.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        argv.extend(["--num-workers", str(args.num_workers)])
    if skip_train:
        argv.append("--skip-train")
    if reuse_run_dir:
        argv.append("--reuse-run-dir")
    elif args.force_run:
        argv.append("--force-run")
    run_ds2_pipeline.main(argv)


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


def load_metrics(run_dir: Path) -> dict:
    metrics = {}
    for subset in EVAL_COMPONENTS:
        path = run_dir / "details" / "metrics" / f"{subset}.json"
        if path.exists():
            metrics[subset] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def run_model(args: argparse.Namespace, model_key: str, config: Path) -> dict:
    nolm_name = f"{model_key}_nolm_beam1"
    kenlm_name = f"{model_key}_kenlm_b{args.kenlm_beam}_a{str(args.kenlm_alpha).replace('.', 'p')}_b{str(args.kenlm_beta).replace('.', 'p')}"
    nolm_run = args.results_root / nolm_name
    kenlm_run = args.results_root / kenlm_name

    run_pipeline(
        args=args,
        run_name=nolm_name,
        config=config,
        decode_config=Path("conf/tuning/decode.yaml"),
        skip_train=False,
        reuse_run_dir=False,
    )

    if kenlm_run.exists() and args.force_run:
        shutil.rmtree(kenlm_run)
    kenlm_run.mkdir(parents=True, exist_ok=not args.force_run)
    link_checkpoint(nolm_run, kenlm_run)

    lm_dir = kenlm_run / "details" / "lm"
    lm_meta = train_lm(args, args.split_dir, lm_dir)
    lm_validation = validate_to_file(
        args,
        args.split_dir,
        kenlm_run / "details" / "lm_validation.json",
        lm_dir=lm_dir,
    )
    decode_config = make_decode_config(
        project_root=args.project_root,
        output_path=lm_dir / "decode_kenlm.yaml",
        lm_path=lm_dir / "lm.klm",
        beam_size=args.kenlm_beam,
        alpha=args.kenlm_alpha,
        beta=args.kenlm_beta,
    )
    run_pipeline(
        args=args,
        run_name=kenlm_name,
        config=config,
        decode_config=decode_config,
        skip_train=True,
        reuse_run_dir=True,
    )

    return {
        "model_key": model_key,
        "config": rel_to_project(args.project_root, args.project_root / config),
        "nolm_run": rel_to_project(args.project_root, nolm_run),
        "kenlm_run": rel_to_project(args.project_root, kenlm_run),
        "kenlm_meta": lm_meta,
        "kenlm_validation": lm_validation,
        "nolm_metrics": load_metrics(nolm_run),
        "kenlm_metrics": load_metrics(kenlm_run),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--track-id", default=TRACK_ID)
    parser.add_argument("--split-id", default=DEFAULT_SPLIT_ID)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument(
        "--models",
        default="paper_small,paper_7gru,paper_7lstm",
        help="Comma-separated model keys: " + ",".join(MODEL_CONFIGS),
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--kenlm-order", type=int, default=5)
    parser.add_argument("--kenlm-memory", default="80%")
    parser.add_argument("--kenlm-beam", type=int, default=40)
    parser.add_argument("--kenlm-alpha", type=float, default=2.5)
    parser.add_argument("--kenlm-beta", type=float, default=0.3)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.90)
    parser.add_argument("--fuzzy-length-ratio-min", type=float, default=0.80)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.results_root = Path("results") / args.track_id
    args.split_dir = args.project_root / "data" / "cross_validation_splits" / args.split_id
    args.model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(args.model_keys) - set(MODEL_CONFIGS))
    if unknown:
        raise SystemExit(f"Unknown model key(s): {', '.join(unknown)}")
    return args


def main() -> int:
    args = parse_args()
    split_dir = ensure_split(args)
    args.split_dir = split_dir
    args.results_root = args.project_root / args.results_root
    args.results_root.mkdir(parents=True, exist_ok=True)

    split_validation = validate_to_file(
        args,
        split_dir,
        args.results_root / "split_validation.json",
    )

    model_reports = []
    for model_key in args.model_keys:
        model_reports.append(run_model(args, model_key, MODEL_CONFIGS[model_key]))

    report = {
        "schema_version": 1,
        "track_id": args.track_id,
        "split_id": args.split_id,
        "split_dir": rel_to_project(args.project_root, split_dir),
        "results_root": rel_to_project(args.project_root, args.results_root),
        "split_validation": split_validation,
        "models": model_reports,
    }
    write_json(args.results_root / "track_report.json", report)
    print(f"Wrote {rel_to_project(args.project_root, args.results_root / 'track_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
