# PaddleSpeech DeepSpeech2 VEPRAD

Self-contained folder for the official PaddleSpeech DeepSpeech2 VEPRAD run.

This folder replaces the old toy/local DeepSpeech2 experiments. It contains the
VEPRAD split metadata, PaddleSpeech source checkout, Python pipeline scripts,
and archived completed-run outputs needed to reproduce or audit the
from-scratch PaddleSpeech DeepSpeech2 baseline.

GitHub note: the public repository intentionally excludes the official
`PaddleSpeech/` checkout, raw VEPRAD corpus, JSONL transcript manifests,
generated PaddleSpeech data, checkpoint/model files, and per-utterance decoded
transcript outputs. A local/private full bundle is needed to rerun the pipeline
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
- `conf/`: training, preprocessing, and decode configs.
- `scripts/`: direct Python pipeline and helper utilities for VEPRAD.
- `scripts/create_frozen_test_split.py`: one-time script that physically
  separated the frozen test split from the train/validation source pool.
- `scripts/create_cv_split.py`: creates one train/validation split from
  `data/cross_validation_splits/raw_train_val/source.jsonl`.
- `scripts/convert_project_manifests_to_paddlespeech.py`: converts bundled
  VEPRAD manifests into PaddleSpeech raw JSONL.
- `scripts/run_ds2_pipeline.py`: creates or reuses a CV split, prepares
  PaddleSpeech manifests/vocabulary/CMVN, trains DeepSpeech2 by directly
  calling PaddleSpeech Python entrypoints, averages the checkpoint, and
  evaluates validation/test seen/unseen subsets.
- `scripts/train_kenlm_lm.py`: trains a run-owned KenLM n-gram language model
  from a single train-only JSONL manifest.
- `scripts/recompute_asr_metrics.py`: independently recomputes WER/CER from a
  PaddleSpeech result file.
- `results/ds2_cv_20260613_001/`: current clean CV baseline with separate
  validation/test seen-speaker and unseen-speaker evaluations.
- `results/2026-06-13_ds2_baseline_veprad/`: completed baseline run summary
  and tracked supporting details from the earlier pre-CV baseline.

The checkpoint weights are not stored here. Public result folders keep summaries,
configs, metrics, and logs; local ignored details may contain decoded outputs.
The Python pipeline can regenerate checkpoints.

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

The preferred run path is the direct Python pipeline:

```bash
python scripts/run_ds2_pipeline.py \
  --run-name ds2_cv_001 \
  --split-dir data/cross_validation_splits/cv_20260613_001 \
  --device gpu \
  --gpu-id 0 \
  --avg-num 1
```

If `--split-dir` is omitted, the script creates a fresh random CV split from
`data/cross_validation_splits/raw_train_val/source.jsonl`. The script writes the
model, generated PaddleSpeech files, logs, metrics, and run summary under
`results/<run-name>/`. It calls PaddleSpeech Python functions directly.

The KenLM stage can be tested or run separately once `lmplz` and `build_binary`
are available on `PATH`:

```bash
python scripts/train_kenlm_lm.py \
  --train-manifest data/cross_validation_splits/cv_20260613_001/train.jsonl \
  --output-dir results/<run-name>/details/lm
```

For a CV `train.jsonl`, the KenLM script automatically filters the LM corpus
against the sibling `val.jsonl` and frozen `data/test/test.jsonl`. Train rows
whose normalized transcript exactly matches, or fuzzily matches at `>= 0.90`,
held-out validation/test text are excluded from `train_text.txt` before KenLM is
built. The removed rows are written to the ignored
`excluded_text_matches.jsonl` audit file, and aggregate counts are recorded in
`lm.meta.json`.

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

## Transcript Targets

VEPRAD transcripts include non-speech/event annotations such as `<uzdah>` and
`<sil>`. The local project manifests keep the original transcript in `text_raw`
and keep extracted annotations in `non_speech_events`, but the ASR target field
`text` removes these event tags before PaddleSpeech training and evaluation.

This is intentional. The current PaddleSpeech DeepSpeech2 setup uses
character-level CTC targets (`unit_type: char`). In that mode, PaddleSpeech
tokenizes text as individual characters, so an annotation like `<uzdah>` would
be learned as the character sequence `<`, `u`, `z`, `d`, `a`, `h`, `>` rather
than as one acoustic event label. Word-level or subword targets could represent
the event as a single token, and a custom tokenizer could also be written, but
those approaches would move this run away from the intended character-level
DeepSpeech2 baseline and add extra decoding/metric complexity. Therefore
non-speech annotations are excluded from the ASR targets while remaining
auditable in metadata.

## Current CV Baseline

- Date: 2026-06-13
- Run folder: `results/ds2_cv_20260613_001/`
- CV split: `data/cross_validation_splits/cv_20260613_001/`
- Training utterances: `3225`
- Validation utterances: `1037`
- Frozen test utterances: `993`
- Model: official PaddleSpeech `DeepSpeech2Model`
- Architecture: 2 convolution layers, 5 bidirectional LSTM layers, hidden size 1024
- Features: 161-bin fbank with CMVN
- Decoder: CTC beam search path, `beam_size: 1`, no external LM
- Validation WER: `0.326573`
- Validation seen-speaker WER: `0.270430`
- Validation unseen-speaker WER: `0.355705`
- Test WER: `0.384420`
- Test seen-speaker WER: `0.280320`
- Test unseen-speaker WER: `0.486651`

KenLM beam-search decoding was also evaluated for this run in the local/remote
run folder `results/ds2_cv_20260613_001_lm_beam10/`. That diagnostic used a
5-gram KenLM trained from all 3225 train transcripts for the split. It was not
the final stricter LM protocol because the LM corpus was not yet filtered
against validation/test transcript repetition.

- Decode: KenLM 5-gram, CTC beam search, `beam_size: 10`, `alpha: 2.5`,
  `beta: 0.3`
- Validation WER: `0.079664`
- Validation seen-speaker WER: `0.061952`
- Validation unseen-speaker WER: `0.088821`
- Test WER: `0.121155`
- Test seen-speaker WER: `0.065549`
- Test unseen-speaker WER: `0.175764`

## Paper-Small Approximation Run

- Date: 2026-06-13
- Run folders: `results/paper_small_001_nolm_beam1/` and
  `results/paper_small_001_kenlm5_beam10/`
- CV split: `data/cross_validation_splits/cv_paper_small_001/`
- Training utterances: `3057`
- Validation utterances: `1205`
- Frozen test utterances: `993`
- Model: official PaddleSpeech `DeepSpeech2Model`, configured to approximate
  the smaller Deep Speech 2 paper model as closely as possible without editing
  PaddleSpeech source
- Architecture: 2 convolution layers, 5 bidirectional GRU layers, hidden size
  650
- Parameters: `36.21M`
- Features: 161-bin `fbank_kaldi` with CMVN
- Best checkpoint: epoch 11, selected by validation loss `35.604879`

No-LM decoding used the CTC beam-search path with `beam_size: 1`.

| Subset | WER |
| --- | ---: |
| Validation | `0.420641` |
| Validation seen speakers | `0.344741` |
| Validation unseen speakers | `0.451422` |
| Test | `0.452892` |
| Test seen speakers | `0.350400` |
| Test unseen speakers | `0.553545` |

KenLM decoding used a split-owned 5-gram LM, `beam_size: 10`, `alpha: 2.5`,
and `beta: 0.3`. Unlike the earlier LM diagnostic, the LM corpus was filtered
against the validation split and frozen test set: 2290 of 3057 train rows were
kept, while 767 rows were excluded because their normalized transcript exactly
or fuzzily matched held-out text.

| Subset | WER |
| --- | ---: |
| Validation | `0.125783` |
| Validation seen speakers | `0.101836` |
| Validation unseen speakers | `0.135494` |
| Test | `0.154980` |
| Test seen speakers | `0.097115` |
| Test unseen speakers | `0.211807` |

The paper-small approximation is worse than the earlier CV baseline. The main
difference is model capacity: the earlier run used 5 bidirectional LSTM layers
with hidden size 1024, while the paper-small approximation uses 5 bidirectional
GRU layers with hidden size 650. The comparison is not a clean ablation because
the CV split also changed and the later KenLM run used stricter held-out-text
filtering.

## Earlier Archived Run

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
- `conf/deepspeech2_paper_small.yaml`: paper-small approximation config using
  unmodified PaddleSpeech DS2 options.
- `data/test/test.meta.json`: frozen test split audit metadata.
- `data/cross_validation_splits/raw_train_val/source.meta.json`: non-test
  train/validation source pool audit metadata.
- `scripts/run_ds2_pipeline.py`: direct split/train/eval pipeline.
- `scripts/create_cv_split.py`: cross-validation split generator.
- `results/2026-06-13_ds2_baseline_veprad/summary.md`: archived run summary.
- `results/2026-06-13_ds2_baseline_veprad/details/metrics.json`:
  independent metric calculation.
- `results/2026-06-13_ds2_baseline_veprad/details/training_curve.tsv`:
  parsed epoch/dev-loss curve.

## Important Notes

The train-only vocabulary choice is intentional. `scripts/run_ds2_pipeline.py`
builds the character vocabulary only from the training manifest, so validation
and test transcripts do not leak into the vocabulary.

The preprocessing config has no augmentation. This avoids applying train-time
SpecAugment-style transforms to dev/test through the shared PaddleSpeech
preprocessing path.

KenLM language models are treated as run-owned artifacts. Train them from the
current run's `train.jsonl` only, after filtering out train transcripts that
exactly or fuzzily match the current validation split or frozen test set. Store
the LM under `results/<run>/details/lm/`, and do not reuse it across different
CV splits. The generated text corpus, ARPA file, binary LM, match audit, and
logs are ignored by Git; the small `lm.meta.json` audit file may be tracked for
completed runs.

The first LM diagnostic showed that exact or near-exact transcript repetition is
present in VEPRAD because many weather-report sentences recur. Filtering eval
utterances after decoding changed the LM-assisted test WER only modestly
(`0.121155` unfiltered, `0.131239` after exact-match removal, `0.132189` after
`>= 0.90` fuzzy-match removal), but future KenLM models avoid this text leakage
at training time by filtering the LM corpus instead of deleting validation/test
examples.

The pipeline uses relative paths from this folder. It should not depend on the
old `Projekt - DeepSpeech2` location or a hardcoded `/workspace` path.
