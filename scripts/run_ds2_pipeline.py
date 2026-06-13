#!/usr/bin/env python3
"""Run a VEPRAD DeepSpeech2 split/train/eval pipeline without shell subprocesses."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CV_ROOT = Path("data/cross_validation_splits")
DEFAULT_TEST_DIR = Path("data/test")
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_CONFIG = Path("conf/deepspeech2.yaml")
DEFAULT_DECODE_CONFIG = Path("conf/tuning/decode.yaml")
DEFAULT_PREPROCESS_CONFIG = Path("conf/preprocess.yaml")

EVAL_COMPONENTS = (
    "val",
    "val_seen_speakers",
    "val_unseen_speakers",
    "test",
    "test_seen_speakers",
    "test_unseen_speakers",
)
SPECIAL_VOCAB_TOKENS = {"<blank>", "<unk>", "<sos/eos>"}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = Tee(old_stdout, log_file)
        sys.stderr = Tee(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@contextlib.contextmanager
def loguru_file(path: Path):
    try:
        from loguru import logger
    except Exception:
        yield
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(str(path), level="DEBUG", enqueue=True, rotation="500 MB")
    try:
        yield
    finally:
        logger.remove(sink_id)


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        raise SystemExit("Run/split name cannot be empty after sanitizing")
    return value


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def as_project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def rel_to_project(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def summarize_rows(rows: list[dict]) -> dict:
    speakers = Counter(row.get("speaker_id", "unknown") for row in rows)
    seconds = sum(float(row.get("duration") or 0.0) for row in rows)
    return {
        "utterances": len(rows),
        "total_seconds": round(seconds, 6),
        "total_hours": round(seconds / 3600.0, 6),
        "speaker_count": len(speakers),
        "speakers": dict(sorted(speakers.items())),
    }


def configure_environment(project_root: Path, device: str, gpu_id: str) -> int:
    paddlespeech_root = project_root / "PaddleSpeech"
    if not paddlespeech_root.exists():
        raise SystemExit(
            f"Missing PaddleSpeech checkout: {paddlespeech_root}\n"
            "Clone/sync PaddleSpeech on the training machine before running this."
        )

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(paddlespeech_root))
    os.environ["PYTHONPATH"] = (
        f"{paddlespeech_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )
    os.environ["PATH"] = (
        f"{paddlespeech_root}{os.pathsep}{paddlespeech_root / 'utils'}"
        f"{os.pathsep}{os.environ.get('PATH', '')}"
    )
    os.environ.setdefault("PYTHONIOENCODING", "UTF-8")
    os.environ.setdefault("LC_ALL", "C")
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")

    if device == "cpu":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return 0

    if "," in gpu_id:
        raise SystemExit(
            "This direct Python pipeline currently supports one GPU only. "
            "Use a single --gpu-id, for example --gpu-id 0."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return 1


def ensure_split(args: argparse.Namespace, split_id: str) -> Path:
    from scripts import create_cv_split

    project_root = args.project_root
    cv_root = as_project_path(project_root, args.cv_root)
    split_dir = cv_root / split_id
    if split_dir.exists() and not args.force_split:
        print(f"Using existing CV split: {rel_to_project(project_root, split_dir)}")
        return split_dir

    ns = SimpleNamespace(
        project_root=project_root,
        source_manifest=args.source_manifest,
        output_root=args.cv_root,
        split_id=split_id,
        val_speakers=args.val_speakers,
        same_speaker_val_ratio=args.same_speaker_val_ratio,
        seed=args.split_seed,
        dry_run=False,
        force=args.force_split,
    )
    create_cv_split.run(ns)
    return split_dir


def resolve_split_dirs(args: argparse.Namespace) -> list[Path]:
    project_root = args.project_root
    if args.split_dir:
        return [as_project_path(project_root, path) for path in args.split_dir]

    split_id = sanitize_name(args.split_id or f"cv_{dt.datetime.now():%Y%m%d_%H%M%S}")
    return [ensure_split(args, split_id)]


def require_split_files(split_dir: Path) -> None:
    required = [
        "train.jsonl",
        "val.jsonl",
        "val_seen_speakers.jsonl",
        "val_unseen_speakers.jsonl",
        "split.meta.json",
    ]
    missing = [name for name in required if not (split_dir / name).exists()]
    if missing:
        raise SystemExit(f"{split_dir} is missing required files: {', '.join(missing)}")


def split_id_from_dir(split_dir: Path) -> str:
    return sanitize_name(split_dir.name)


def component_project_manifests(project_root: Path, split_dir: Path) -> dict[str, Path]:
    return {
        "train": split_dir / "train.jsonl",
        "val": split_dir / "val.jsonl",
        "val_seen_speakers": split_dir / "val_seen_speakers.jsonl",
        "val_unseen_speakers": split_dir / "val_unseen_speakers.jsonl",
        "test": project_root / DEFAULT_TEST_DIR / "test.jsonl",
        "test_seen_speakers": project_root
        / DEFAULT_TEST_DIR
        / "test_seen_speakers.jsonl",
        "test_unseen_speakers": project_root
        / DEFAULT_TEST_DIR
        / "test_unseen_speakers.jsonl",
    }


def project_to_paddlespeech_raw(
    project_root: Path,
    source_manifest: Path,
    output_manifest: Path,
    check_audio: bool,
) -> dict:
    rows = read_jsonl(source_manifest)
    output_rows: list[dict] = []
    missing_audio: list[str] = []
    absolute_paths: list[str] = []
    sm04_rows: list[str] = []

    for row in rows:
        utt = row.get("utterance_id")
        audio_relpath = row.get("audio_relpath") or row.get("audio_filepath")
        if not utt:
            raise SystemExit(f"{source_manifest}: row missing utterance_id")
        if not audio_relpath:
            raise SystemExit(f"{source_manifest}: {utt} missing audio path")
        if Path(audio_relpath).is_absolute():
            absolute_paths.append(f"{utt}:{audio_relpath}")
        if row.get("speaker_id") == "sm04" or row.get("corpus") == "sm04":
            sm04_rows.append(utt)
        if check_audio and not (project_root / audio_relpath).exists():
            missing_audio.append(audio_relpath)

        output_rows.append(
            {
                "utt": utt,
                "utt2spk": row.get("speaker_id") or "unknown",
                "feat": audio_relpath,
                "feat_shape": [float(row["duration"])],
                "text": row["text"],
            }
        )

    if absolute_paths:
        raise SystemExit(
            "Project manifests must use relative audio paths. Sample:\n"
            + "\n".join(absolute_paths[:10])
        )
    if sm04_rows:
        raise SystemExit(
            f"{source_manifest} contains quarantined sm04 rows. Sample: "
            + ", ".join(sm04_rows[:10])
        )
    if missing_audio:
        raise SystemExit(
            f"{source_manifest} references missing audio. Sample:\n"
            + "\n".join(missing_audio[:10])
        )

    write_jsonl(output_manifest, output_rows)
    return {
        "source_manifest": rel_to_project(project_root, source_manifest),
        "raw_manifest": rel_to_project(project_root, output_manifest),
        "summary": summarize_rows(rows),
    }


def rewrite_preprocess_config(
    source_path: Path,
    output_path: Path,
    cmvn_path: Path,
    project_root: Path,
) -> None:
    cmvn_value = rel_to_project(project_root, cmvn_path)
    try:
        import yaml

        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        for step in payload.get("process", []):
            if step.get("type") == "cmvn_json":
                step["cmvn_path"] = cmvn_value
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return
    except Exception:
        pass

    lines = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("cmvn_path:"):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}cmvn_path: {cmvn_value}")
        else:
            lines.append(line)
    write_text(output_path, "\n".join(lines) + "\n")


def read_vocab_chars(vocab_path: Path) -> set[str]:
    chars: set[str] = set()
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        token = line.strip("\n")
        if token in SPECIAL_VOCAB_TOKENS:
            continue
        chars.add(" " if token == "<space>" else token)
    return chars


def vocab_coverage(vocab_path: Path, manifests: dict[str, Path]) -> dict:
    known = read_vocab_chars(vocab_path)
    coverage = {}
    for name, manifest in manifests.items():
        rows = read_jsonl(manifest)
        counter: Counter[str] = Counter()
        for row in rows:
            for char in row.get("text", ""):
                if char not in known:
                    counter[char] += 1
        coverage[name] = {
            "unknown_character_total": sum(counter.values()),
            "unknown_characters": dict(sorted(counter.items())),
        }
    return coverage


def prepare_paddlespeech_data(
    project_root: Path,
    split_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict:
    from paddlespeech.dataset.s2t.build_vocab import build_vocab
    from paddlespeech.dataset.s2t.compute_mean_std import compute_cmvn
    from paddlespeech.dataset.s2t.format_data import format_data

    generated_dir = run_dir / "details" / "generated"
    data_dir = generated_dir / "paddlespeech_data"
    raw_dir = data_dir / "raw_manifests"
    formatted_dir = data_dir / "formatted_manifests"
    vocab_path = data_dir / "lang_char" / "vocab.txt"
    cmvn_path = data_dir / "mean_std.json"
    preprocess_path = data_dir / "preprocess.yaml"

    raw_dir.mkdir(parents=True, exist_ok=True)
    formatted_dir.mkdir(parents=True, exist_ok=True)

    project_manifests = component_project_manifests(project_root, split_dir)
    for name, path in project_manifests.items():
        if not path.exists():
            raise SystemExit(f"Missing {name} manifest: {path}")

    raw_manifests: dict[str, Path] = {}
    formatted_manifests: dict[str, Path] = {}
    component_meta = {}
    for name, source in project_manifests.items():
        raw_path = raw_dir / f"{name}.raw.jsonl"
        component_meta[name] = project_to_paddlespeech_raw(
            project_root=project_root,
            source_manifest=source,
            output_manifest=raw_path,
            check_audio=args.check_audio,
        )
        raw_manifests[name] = raw_path
        formatted_manifests[name] = formatted_dir / f"{name}.jsonl"

    print("Computing CMVN from training manifest...")
    compute_cmvn(
        manifest_path=rel_to_project(project_root, raw_manifests["train"]),
        output_path=rel_to_project(project_root, cmvn_path),
        num_samples=args.cmvn_num_samples,
        num_workers=args.cmvn_num_workers,
        spectrum_type="fbank",
        feat_dim=161,
        delta_delta=False,
        stride_ms=10,
        window_ms=25,
        sample_rate=16000,
        use_dB_normalization=False,
    )

    print("Building character vocabulary from training manifest...")
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    build_vocab(
        manifest_paths=[rel_to_project(project_root, raw_manifests["train"])],
        vocab_path=rel_to_project(project_root, vocab_path),
        unit_type="char",
        count_threshold=0,
    )

    rewrite_preprocess_config(
        source_path=as_project_path(project_root, args.preprocess_config),
        output_path=preprocess_path,
        cmvn_path=cmvn_path,
        project_root=project_root,
    )

    for name, raw_path in raw_manifests.items():
        if component_meta[name]["summary"]["utterances"] == 0:
            write_text(formatted_manifests[name], "")
            continue
        print(f"Formatting {name} manifest...")
        formatted_manifests[name].parent.mkdir(parents=True, exist_ok=True)
        format_data(
            manifest_paths=[rel_to_project(project_root, raw_path)],
            output_path=rel_to_project(project_root, formatted_manifests[name]),
            cmvn_path=rel_to_project(project_root, cmvn_path),
            unit_type="char",
            vocab_path=rel_to_project(project_root, vocab_path),
            spm_model_prefix="",
        )

    coverage = vocab_coverage(vocab_path, project_manifests)
    write_json(data_dir / "vocab_coverage.json", coverage)

    return {
        "data_dir": data_dir,
        "cmvn_path": cmvn_path,
        "vocab_path": vocab_path,
        "preprocess_path": preprocess_path,
        "raw_manifests": raw_manifests,
        "formatted_manifests": formatted_manifests,
        "components": component_meta,
        "vocab_coverage": coverage,
    }


def import_training_helpers():
    from yacs.config import CfgNode

    from paddlespeech.dataset.s2t.avg_model import average_checkpoints
    from paddlespeech.s2t.exps.deepspeech2.bin import test as ds2_test
    from paddlespeech.s2t.exps.deepspeech2.bin import train as ds2_train
    from paddlespeech.s2t.training.cli import default_argument_parser

    return CfgNode, average_checkpoints, ds2_test, ds2_train, default_argument_parser


def make_train_args(default_argument_parser, args: argparse.Namespace, model_dir: Path):
    argv = [
        "--ngpu",
        str(args.ngpu),
        "--config",
        rel_to_project(args.project_root, as_project_path(args.project_root, args.config)),
        "--output",
        rel_to_project(args.project_root, model_dir),
        "--profiler-options",
        "",
        "--seed",
        str(args.train_seed),
    ]
    return default_argument_parser().parse_args(argv)


def make_test_args(
    default_argument_parser,
    args: argparse.Namespace,
    checkpoint_prefix: Path,
    result_file: Path,
):
    argv = [
        "--ngpu",
        str(args.ngpu),
        "--config",
        rel_to_project(args.project_root, as_project_path(args.project_root, args.config)),
        "--decode_cfg",
        rel_to_project(
            args.project_root, as_project_path(args.project_root, args.decode_config)
        ),
        "--result_file",
        rel_to_project(args.project_root, result_file),
        "--checkpoint_path",
        rel_to_project(args.project_root, checkpoint_prefix),
    ]
    return default_argument_parser().parse_args(argv)


def load_base_config(CfgNode, config_path: Path):
    config = CfgNode(new_allowed=True)
    config.merge_from_file(str(config_path))
    return config


def load_decode_config(CfgNode, decode_path: Path):
    decode = CfgNode(new_allowed=True)
    decode.merge_from_file(str(decode_path))
    return decode


def build_train_config(CfgNode, project_root: Path, args, prepared: dict):
    config = load_base_config(CfgNode, as_project_path(project_root, args.config))
    config.defrost()
    config.train_manifest = rel_to_project(project_root, prepared["formatted_manifests"]["train"])
    config.dev_manifest = rel_to_project(project_root, prepared["formatted_manifests"]["val"])
    config.test_manifest = rel_to_project(project_root, prepared["formatted_manifests"]["val"])
    config.vocab_filepath = rel_to_project(project_root, prepared["vocab_path"])
    config.preprocess_config = rel_to_project(project_root, prepared["preprocess_path"])
    if args.epochs is not None:
        config.n_epoch = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.freeze()
    return config


def build_eval_config(CfgNode, project_root: Path, args, prepared: dict, eval_name: str):
    config = build_train_config(CfgNode, project_root, args, prepared)
    config.defrost()
    config.test_manifest = rel_to_project(project_root, prepared["formatted_manifests"][eval_name])
    config.decode = load_decode_config(CfgNode, as_project_path(project_root, args.decode_config))
    config.freeze()
    return config


def train_and_eval(
    project_root: Path,
    split_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
    prepared: dict,
) -> dict:
    CfgNode, average_checkpoints, ds2_test, ds2_train, default_argument_parser = (
        import_training_helpers()
    )

    model_dir = run_dir / "details" / "model"
    checkpoint_dir = model_dir / "checkpoints"
    eval_dir = run_dir / "details" / "eval"
    metrics_dir = run_dir / "details" / "metrics"
    analysis_dir = run_dir / "details" / "analysis"
    logs_dir = run_dir / "details" / "logs"

    summary = {
        "model_dir": rel_to_project(project_root, model_dir),
        "avg_checkpoint_prefix": None,
        "evals": {},
    }

    if not args.skip_train:
        train_config = build_train_config(CfgNode, project_root, args, prepared)
        train_args = make_train_args(default_argument_parser, args, model_dir)
        with tee_output(logs_dir / "train.stdout.log"), loguru_file(
            logs_dir / "train.log"
        ):
            ds2_train.main(train_config, train_args)
    else:
        print("Skipping training by request.")

    checkpoint_prefix = checkpoint_dir / f"avg_{args.avg_num}"
    if not args.skip_train:
        print(f"Averaging best {args.avg_num} checkpoint(s)...")
        with tee_output(logs_dir / "avg.stdout.log"), loguru_file(logs_dir / "avg.log"):
            average_checkpoints(
                dst_model=rel_to_project(project_root, checkpoint_prefix) + ".pdparams",
                ckpt_dir=rel_to_project(project_root, checkpoint_dir),
                val_best=True,
                num=args.avg_num,
            )
    elif not (checkpoint_prefix.with_suffix(".pdparams")).exists():
        raise SystemExit(
            f"--skip-train was set, but averaged checkpoint is missing: "
            f"{checkpoint_prefix}.pdparams"
        )

    summary["avg_checkpoint_prefix"] = rel_to_project(project_root, checkpoint_prefix)

    if args.skip_eval:
        print("Skipping evaluation by request.")
        return summary

    from scripts.recompute_asr_metrics import compute as compute_asr_metrics

    for eval_name in EVAL_COMPONENTS:
        utterances = prepared["components"][eval_name]["summary"]["utterances"]
        if utterances == 0:
            summary["evals"][eval_name] = {"skipped": True, "reason": "empty_manifest"}
            continue

        result_file = eval_dir / f"{eval_name}.rsl"
        metrics_file = metrics_dir / f"{eval_name}.json"
        utterance_file = analysis_dir / f"{eval_name}.utterance_metrics.tsv"
        eval_config = build_eval_config(CfgNode, project_root, args, prepared, eval_name)
        test_args = make_test_args(
            default_argument_parser=default_argument_parser,
            args=args,
            checkpoint_prefix=checkpoint_prefix,
            result_file=result_file,
        )
        print(f"Evaluating {eval_name} ({utterances} utterances)...")
        with tee_output(logs_dir / f"eval_{eval_name}.stdout.log"), loguru_file(
            logs_dir / f"eval_{eval_name}.log"
        ):
            ds2_test.main(eval_config, test_args)

        metrics, rows = compute_asr_metrics(result_file)
        write_json(metrics_file, metrics)
        if args.write_utterance_metrics:
            write_utterance_metrics(utterance_file, rows)
        summary["evals"][eval_name] = {
            "skipped": False,
            "result_file": rel_to_project(project_root, result_file),
            "metrics_file": rel_to_project(project_root, metrics_file),
            "metrics": metrics,
        }

    return summary


def write_utterance_metrics(path: Path, rows: list[dict]) -> None:
    import csv

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["wer"], item["cer"]), reverse=True):
            writer.writerow({field: row[field] for field in fields})


def copy_public_configs(project_root: Path, run_dir: Path, args: argparse.Namespace) -> None:
    config_dir = run_dir / "details" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.config, args.decode_config, args.preprocess_config):
        source = as_project_path(project_root, path)
        if source.exists():
            shutil.copy2(source, config_dir / source.name)


def write_run_summary(run_dir: Path, run_meta: dict, eval_summary: dict) -> None:
    lines = [
        f"# {run_meta['run_name']}",
        "",
        "DeepSpeech2 VEPRAD pipeline run.",
        "",
        "## Split",
        "",
        f"- CV split: `{run_meta['split_id']}`",
        f"- Split folder: `{run_meta['split_dir']}`",
        f"- Created at UTC: `{run_meta['created_at_utc']}`",
        "",
        "## Data",
        "",
    ]
    for name in ("train", "val", "val_seen_speakers", "val_unseen_speakers", "test", "test_seen_speakers", "test_unseen_speakers"):
        summary = run_meta["components"][name]["summary"]
        lines.append(
            f"- `{name}`: {summary['utterances']} utterances, "
            f"{summary['speaker_count']} speakers, {summary['total_hours']} h"
        )

    lines.extend(
        [
            "",
            "## Model",
            "",
            f"- Output: `{eval_summary['model_dir']}`",
            f"- Averaged checkpoint prefix: `{eval_summary['avg_checkpoint_prefix']}`",
            f"- Avg num: `{run_meta['avg_num']}`",
            f"- Device: `{run_meta['device']}`",
            "",
            "## Evaluation",
            "",
        ]
    )

    if not eval_summary["evals"]:
        lines.append("Evaluation was skipped.")
    else:
        lines.append("| Subset | Utterances | WER | CER spaces | CER no spaces | Exact |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name in EVAL_COMPONENTS:
            item = eval_summary["evals"].get(name)
            if not item or item.get("skipped"):
                lines.append(f"| `{name}` | 0 | skipped | skipped | skipped | skipped |")
                continue
            metrics = item["metrics"]
            lines.append(
                f"| `{name}` | {metrics['num_utterances']} | "
                f"{metrics['wer']:.6f} | "
                f"{metrics['cer_including_spaces']:.6f} | "
                f"{metrics['cer_no_spaces']:.6f} | "
                f"{metrics['exact_match_rate']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The pipeline calls PaddleSpeech Python functions directly; it does not shell out through the old recipe scripts.",
            "- Per-utterance decode files and generated PaddleSpeech manifests are local artifacts and are ignored by Git.",
            "",
        ]
    )
    write_text(run_dir / "summary.md", "\n".join(lines))


def run_one_split(
    project_root: Path,
    split_dir: Path,
    base_run_name: str,
    args: argparse.Namespace,
    total_splits: int,
) -> Path:
    require_split_files(split_dir)
    split_id = split_id_from_dir(split_dir)
    run_name = base_run_name if total_splits == 1 else f"{base_run_name}_{split_id}"
    run_dir = as_project_path(project_root, args.results_root) / run_name
    if run_dir.exists() and not args.force_run:
        raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    copy_public_configs(project_root, run_dir, args)
    prepared = prepare_paddlespeech_data(project_root, split_dir, run_dir, args)

    run_meta = {
        "run_name": run_name,
        "split_id": split_id,
        "split_dir": rel_to_project(project_root, split_dir),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": "cpu" if args.ngpu == 0 else f"gpu:{args.gpu_id}",
        "ngpu": args.ngpu,
        "avg_num": args.avg_num,
        "config": rel_to_project(project_root, as_project_path(project_root, args.config)),
        "decode_config": rel_to_project(
            project_root, as_project_path(project_root, args.decode_config)
        ),
        "preprocess_config": rel_to_project(project_root, prepared["preprocess_path"]),
        "cmvn_path": rel_to_project(project_root, prepared["cmvn_path"]),
        "vocab_path": rel_to_project(project_root, prepared["vocab_path"]),
        "components": prepared["components"],
        "vocab_coverage": prepared["vocab_coverage"],
    }
    write_json(run_dir / "details" / "run.meta.json", run_meta)

    eval_summary = train_and_eval(project_root, split_dir, run_dir, args, prepared)
    write_json(run_dir / "details" / "run.eval.json", eval_summary)
    write_run_summary(run_dir, run_meta, eval_summary)
    return run_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", help="Recognizable results/<run-name> folder.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        action="append",
        help="Existing CV split folder. May be passed multiple times.",
    )
    parser.add_argument("--split-id", help="CV split id to create or reuse.")
    parser.add_argument("--cv-root", type=Path, default=DEFAULT_CV_ROOT)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_CV_ROOT / "raw_train_val" / "source.jsonl",
    )
    parser.add_argument("--val-speakers", type=int, default=3)
    parser.add_argument("--same-speaker-val-ratio", type=float, default=0.10)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional CV split seed. Omit for system-random CV split creation.",
    )
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--force-run", action="store_true")

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--decode-config", type=Path, default=DEFAULT_DECODE_CONFIG)
    parser.add_argument("--preprocess-config", type=Path, default=DEFAULT_PREPROCESS_CONFIG)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cmvn-num-samples", type=int, default=2000)
    parser.add_argument("--cmvn-num-workers", type=int, default=8)
    parser.add_argument("--avg-num", type=int, default=1)
    parser.add_argument("--check-audio", action="store_true", default=True)
    parser.add_argument("--no-check-audio", action="store_false", dest="check_audio")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--write-utterance-metrics",
        action="store_true",
        help="Write per-utterance TSV files with refs/hyps. These are ignored by Git.",
    )
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    os.chdir(args.project_root)
    args.ngpu = configure_environment(args.project_root, args.device, args.gpu_id)

    split_dirs = resolve_split_dirs(args)
    base_run_name = sanitize_name(
        args.run_name
        or f"{now_stamp()}_ds2_{split_id_from_dir(split_dirs[0])}"
    )

    run_dirs = []
    for split_dir in split_dirs:
        run_dir = run_one_split(
            project_root=args.project_root,
            split_dir=split_dir,
            base_run_name=base_run_name,
            args=args,
            total_splits=len(split_dirs),
        )
        run_dirs.append(run_dir)
        print(f"Completed run: {rel_to_project(args.project_root, run_dir)}")

    print("Pipeline finished.")
    for run_dir in run_dirs:
        print(f"- {rel_to_project(args.project_root, run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
