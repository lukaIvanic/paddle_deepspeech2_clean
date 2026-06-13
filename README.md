# PaddleSpeech DeepSpeech2 VEPRAD

Self-contained folder for the official PaddleSpeech DeepSpeech2 VEPRAD run.

This folder replaces the old toy/local DeepSpeech2 experiments. It contains the
VEPRAD split metadata, PaddleSpeech source checkout, recipe files, helper
scripts, and archived completed-run outputs needed to reproduce or audit the
from-scratch PaddleSpeech DeepSpeech2 baseline.

GitHub note: the public repository intentionally excludes the official
`PaddleSpeech/` checkout, raw VEPRAD corpus, JSONL transcript manifests,
generated PaddleSpeech data, checkpoint/model files, and per-utterance decoded
transcript outputs. A local/private full bundle is needed to rerun the recipe
end to end.

## What Is Here

- `PaddleSpeech/`: pinned PaddleSpeech repository checkout at commit
  `6b25a400008d393f9c3af837b3c692b17f29ee1a`.
- `data/test/`: frozen held-out test split metadata. The local JSONL manifest
  and raw audio/text files are ignored by Git.
- `data/cross_validation_splits/raw_train_val/`: non-test source pool for
  future train/validation splits. The local JSONL manifest and raw audio/text
  files are ignored by Git.
- `data/cross_validation_splits/`: parent folder for future CV split folders.
- `data/paddlespeech/`: generated PaddleSpeech raw manifests. This directory is
  created by `scripts/data.sh` during stage `-1` and is not required before the
  first run.
- `conf/`: training, preprocessing, and decode configs.
- `scripts/`: adapted PaddleSpeech recipe scripts for VEPRAD plus helper
  utilities. `scripts/run.sh` is the main recipe entrypoint.
- `scripts/create_frozen_test_split.py`: one-time script that physically
  separated the frozen test split from the train/validation source pool.
- `scripts/create_cv_split.py`: creates one train/validation split from
  `data/cross_validation_splits/raw_train_val/source.jsonl`.
- `scripts/convert_project_manifests_to_paddlespeech.py`: converts bundled
  VEPRAD manifests into PaddleSpeech raw JSONL.
- `scripts/recompute_asr_metrics.py`: independently recomputes WER/CER from a
  PaddleSpeech result file.
- `results/2026-06-13_ds2_baseline_veprad/`: completed baseline run summary
  and tracked supporting details.

The checkpoint weights are not stored here. Public result folders keep summaries,
configs, metrics, and logs; local ignored details may contain decoded outputs.
The bundled recipe/data can regenerate checkpoints.

## Environment Setup

Use a Linux GPU machine with CUDA-compatible PaddlePaddle support. The successful
run used Python `3.10.14`, PaddlePaddle `2.6.2`, PaddleSpeech `1.5.0`, NumPy
`1.26.4`, and an RTX 3090.

## Current Remote Server

The current GPU box and Python environment notes are in `REMOTE_SERVER.md`.
Code updates should go through the public Git repository.

From this folder:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install paddlepaddle-gpu==2.6.2 paddlespeech==1.5.0 numpy==1.26.4
```

Verify the bundled PaddleSpeech checkout:

```bash
git -C PaddleSpeech rev-parse HEAD
```

Expected commit:

```text
6b25a400008d393f9c3af837b3c692b17f29ee1a
```

After a train/validation split has been generated and converted to the
PaddleSpeech `data/manifest.*` files, run the recipe:

```bash
bash scripts/run.sh --stage 0 --stop_stage 0 --gpus 0
bash scripts/run.sh --stage 1 --stop_stage 1 --gpus 0
bash scripts/run.sh --stage 2 --stop_stage 3 --gpus 0 --avg_num 1
```

## Dataset Split Pipeline

The archived baseline below used the existing VEPRAD manifests copied from the
earlier project. Future experiment runs use a stricter two-dimensional split
protocol that separates both speakers and utterances.

The ambiguous `sm04` subset was excluded before creating new splits. It appears
to be a segmented speech-synthesis subset from a male VEPRAD speaker and can
overlap with regular `m04` material, so keeping it in the ASR pool would risk
speaker/content leakage.

The frozen test split has been created once and should not be regenerated during
model selection. It contains all utterances from 3 randomly selected speakers
and 10% of each remaining speaker's utterances, rounded up so speakers with very
few utterances still contribute at least one test example. The selected fully
held-out speakers are `m04`, `z06`, and `z14`.

The test data lives under `data/test/`. The remaining non-test pool lives under
`data/cross_validation_splits/raw_train_val/`. Cross-validation runs should be
sampled only from that non-test pool. For each validation run, randomly select 3
validation speakers and use all of their remaining utterances for validation.
From all other available speakers, also add 10% of each speaker's remaining
utterances, rounded up, to validation. All remaining utterances form the
training split for that run. These validation folds are intentionally random per
run; the frozen manifests written for a run are the reproducibility record
rather than a hard-coded seed.

Each generated CV split should keep component manifests for separate scoring:
`val_unseen_speakers.jsonl`, `val_seen_speakers.jsonl`,
`train_seen_speakers.jsonl`, and an intentionally empty
`train_unseen_speakers.jsonl` that verifies full validation speakers do not
appear in training.

The frozen test split also keeps component manifests for separate final scoring:
`test_unseen_speakers.jsonl` for the fully held-out speakers and
`test_seen_speakers.jsonl` for the same-speaker utterance holdout.

| Stage | Pool | Speaker-held-out component | Same-speaker utterance component | Output |
| --- | --- | --- | --- | --- |
| Source corpus | Regular VEPRAD `mXX/zXX` utterances, excluding `sm04` | None | None | One-time pre-split source manifest |
| Frozen test split | Full regular-speaker source manifest | All utterances from `m04`, `z06`, and `z14` | 10% of utterances from all remaining speakers, rounded up per speaker | `data/test/` |
| CV input pool | Full source minus frozen test | Test speakers and test utterances removed | Test utterances removed from remaining speakers | `data/cross_validation_splits/raw_train_val/` |
| One CV validation split | CV input pool | All remaining utterances from 3 random validation speakers | 10% of remaining utterances from all other available speakers, rounded up per speaker | Validation manifest for this run |
| One CV training split | CV input pool minus validation split | Validation speakers removed for that run | Validation utterances removed from remaining speakers | Training manifest for this run |

This protocol is meant to evaluate two kinds of generalization at the same
time: performance on unseen speakers and performance on held-out utterances
from speakers that are still represented in training.

## Completed Run

- Date: 2026-06-13
- Original remote workdir: `/workspace/rogj_paddlespeech_ds2`
- PaddleSpeech commit: `6b25a400008d393f9c3af837b3c692b17f29ee1a`
- PaddlePaddle: `2.6.2`
- PaddleSpeech: `1.5.0`
- NumPy: `1.26.4`
- GPU: RTX 3090
- Train/dev/test utterances: `6754 / 249 / 580`
- Model: official PaddleSpeech `DeepSpeech2Model`
- Architecture: 2 convolution layers, 5 bidirectional LSTM layers, hidden size 1024
- Features: 161-bin fbank with CMVN
- Decoder: CTC beam search, `beam_size: 1`, no external LM
- Best dev checkpoint: epoch 10
- Test WER: `0.362968`
- Independent WER: `0.362968231389284`
- Independent CER with spaces: `0.07469918006602065`
- Independent CER without spaces: `0.07085997112806765`

## Main Files To Read

- `REMOTE_SERVER.md`: current remote GPU access and Python environment notes.
- `conf/deepspeech2.yaml`: model/training config.
- `data/test/test.meta.json`: frozen test split audit metadata.
- `data/cross_validation_splits/raw_train_val/source.meta.json`: non-test
  train/validation source pool audit metadata.
- `scripts/data.sh`: VEPRAD data preparation stages.
- `scripts/run.sh`: recipe driver for data preparation, training, averaging, and
  testing.
- `results/2026-06-13_ds2_baseline_veprad/summary.md`: archived run summary.
- `results/2026-06-13_ds2_baseline_veprad/details/metrics.json`:
  independent metric calculation.
- `results/2026-06-13_ds2_baseline_veprad/details/training_curve.tsv`:
  parsed epoch/dev-loss curve.

## Important Notes

The train-only vocabulary choice is intentional. `scripts/data.sh` builds
`data/lang_char/vocab.txt` only from `data/manifest.train.raw`, so dev/test
transcripts do not leak into the vocabulary.

The preprocessing config has no augmentation. This avoids applying train-time
SpecAugment-style transforms to dev/test through the shared PaddleSpeech
preprocessing path.

The recipe uses relative paths from this folder. It should not depend on the old
`Projekt - DeepSpeech2` location or a hardcoded `/workspace` path.
