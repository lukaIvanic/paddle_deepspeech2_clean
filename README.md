# PaddleSpeech DeepSpeech2 VEPRAD

Self-contained folder for the official PaddleSpeech DeepSpeech2 VEPRAD run.

This folder replaces the old toy/local DeepSpeech2 experiments. It contains the
VEPRAD audio, manifests, PaddleSpeech source checkout, recipe files, helper
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
- `data/raw/veprad/`: VEPRAD audio/text corpus.
- `data/manifests/veprad/`: canonical VEPRAD manifests and speaker-CV manifests.
- `data/paddlespeech/`: generated PaddleSpeech raw manifests. This directory is
  created by `scripts/data.sh` during stage `-1` and is not required before the
  first run.
- `conf/`: training, preprocessing, and decode configs.
- `scripts/`: adapted PaddleSpeech recipe scripts for VEPRAD plus helper
  utilities. `scripts/run.sh` is the main recipe entrypoint.
- `scripts/convert_project_manifests_to_paddlespeech.py`: converts bundled
  VEPRAD manifests into PaddleSpeech raw JSONL.
- `scripts/recompute_asr_metrics.py`: independently recomputes WER/CER from a
  PaddleSpeech result file.
- `results/paddlespeech_ds2_veprad_full_2026-06-13/`: archived completed run
  outputs, logs, configs, decoded hypotheses, metrics, and error analysis.

The checkpoint weights are not stored here. The archived results contain decoded
outputs and logs, and the bundled recipe/data can regenerate checkpoints.

## Environment Setup

Use a Linux GPU machine with CUDA-compatible PaddlePaddle support. The successful
run used Python `3.10.14`, PaddlePaddle `2.6.2`, PaddleSpeech `1.5.0`, NumPy
`1.26.4`, and an RTX 3090.

## Current Vast.ai Box

The current PaddleSpeech/DeepSpeech2 box is:

- Vast.ai instance: `40802979`
- Label: `rogj-paddlespeech-ds2`
- GPU: RTX 3090
- Status at last check: running
- SSH host: `ssh1.vast.ai`
- SSH port: `12978`
- SSH key: `~/.ssh/vast_sentiment_ed25519`

Connect with either:

```bash
ssh -p 12978 -i ~/.ssh/vast_sentiment_ed25519 root@ssh1.vast.ai
```

or, from this Mac after `~/.ssh/config` is up to date:

```bash
ssh vast_rogj_asr
```

The host/port can change if the instance is destroyed and recreated. Check the
current values with:

```bash
~/.local/share/codex-vastai/bin/vastai show instances --raw
```

Look for the instance labeled `rogj-paddlespeech-ds2`.

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

Then run the recipe:

```bash
bash scripts/run.sh --stage 0 --stop_stage 0 --gpus 0
bash scripts/run.sh --stage 1 --stop_stage 1 --gpus 0
bash scripts/run.sh --stage 2 --stop_stage 3 --gpus 0 --avg_num 1
```

See `REMOTE_REPRODUCTION.md` for the full Vast.ai sync and result-copy workflow.

## Planned Dataset Split Pipeline

The archived baseline below used the existing VEPRAD manifests copied from the
earlier project. For the next reproducible experiment series, use a stricter
two-dimensional split protocol that separates both speakers and utterances.

First, create one frozen test split and do not regenerate it during model
selection. The test split should contain all utterances from 3 randomly selected
speakers. From all remaining speakers, also add 10% of each speaker's utterances,
rounded up, so speakers with very few utterances still contribute at least one
test example. After this step, the test manifest is fixed and should not be
changed unless the whole experiment protocol is explicitly restarted.

Cross-validation runs are then sampled only from the remaining non-test data.
For each validation run, randomly select 3 validation speakers and use all of
their remaining utterances for validation. From all other available speakers,
also add 10% of each speaker's remaining utterances, rounded up, to validation.
All remaining utterances form the training split for that run. These validation
folds are intentionally random per run; the frozen manifests written for a run
are the reproducibility record rather than a hard-coded seed.

| Stage | Pool | Speaker-held-out component | Same-speaker utterance component | Output |
| --- | --- | --- | --- | --- |
| Source corpus | All normalized VEPRAD utterances | None | None | Full manifest |
| Frozen test split | Full manifest | All utterances from 3 random speakers | 10% of utterances from all remaining speakers, rounded up per speaker | Fixed test manifest |
| CV input pool | Full manifest minus frozen test | Test speakers and test utterances removed | Test utterances removed from remaining speakers | Remaining pool |
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

- `REMOTE_REPRODUCTION.md`: exact steps to run this folder on a remote GPU box.
- `conf/deepspeech2.yaml`: model/training config.
- `scripts/data.sh`: VEPRAD data preparation stages.
- `scripts/run.sh`: recipe driver for data preparation, training, averaging, and
  testing.
- `results/paddlespeech_ds2_veprad_full_2026-06-13/README.md`: archived run
  summary.
- `results/paddlespeech_ds2_veprad_full_2026-06-13/exp/deepspeech2/checkpoints/avg_1.rsl`:
  test references and hypotheses.
- `results/paddlespeech_ds2_veprad_full_2026-06-13/metrics.json`: independent
  metric calculation.
- `results/paddlespeech_ds2_veprad_full_2026-06-13/top_30_errors.md`: worst
  decoded test examples.

## Important Notes

The train-only vocabulary choice is intentional. `scripts/data.sh` builds
`data/lang_char/vocab.txt` only from `data/manifest.train.raw`, so dev/test
transcripts do not leak into the vocabulary.

The preprocessing config has no augmentation. This avoids applying train-time
SpecAugment-style transforms to dev/test through the shared PaddleSpeech
preprocessing path.

The recipe uses relative paths from this folder. It should not depend on the old
`Projekt - DeepSpeech2` location or a hardcoded `/workspace` path.
