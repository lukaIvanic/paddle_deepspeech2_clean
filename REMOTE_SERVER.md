# Remote Server Notes

This file records how to access the current GPU server and where the Python
environment lives. Code should be moved with Git because the project repository
is public:

```bash
https://github.com/lukaIvanic/paddle_deepspeech2_clean.git
```

The public repository intentionally does not contain `PaddleSpeech/`, raw VEPRAD
data, generated PaddleSpeech data, checkpoints, or decoded per-utterance result
files. Those ignored assets must already exist on the remote machine, or be
provisioned separately, before training can run end to end.

## Current Server

The current PaddleSpeech/DeepSpeech2 Vast.ai box is:

- Instance: `40802979`
- Label: `rogj-paddlespeech-ds2`
- GPU: RTX 3090
- SSH host: `ssh1.vast.ai`
- SSH port: `12978`
- SSH key: `~/.ssh/vast_sentiment_ed25519`

Connect directly:

```bash
ssh -p 12978 -i ~/.ssh/vast_sentiment_ed25519 root@ssh1.vast.ai
```

Or use the local SSH alias:

```bash
ssh vast_rogj_asr
```

The host and port can change if the instance is destroyed and recreated. Refresh
the current values with:

```bash
~/.local/share/codex-vastai/bin/vastai show instances --raw
```

Look for the instance labeled `rogj-paddlespeech-ds2`.

## Remote Project Path

The clean remote base directory for new work is:

```bash
BASE=/workspace/rogj_paddlespeech_ds2_clean
PROJECT=$BASE/paddle_deepspeech2_clean
```

Do not use `/workspace/rogj_paddlespeech_ds2` for new runs. That older
directory contains the exploratory/baseline recipe setup and old outputs, and is
kept only as a legacy reference.

First-time checkout:

```bash
mkdir -p "$BASE"
git clone https://github.com/lukaIvanic/paddle_deepspeech2_clean.git "$PROJECT"
```

Update an existing checkout:

```bash
cd "$PROJECT"
git pull --ff-only
```

The public repo ignores the official PaddleSpeech source checkout. Provision it
inside the clean project clone:

```bash
cd "$PROJECT"
git clone https://github.com/PaddlePaddle/PaddleSpeech.git PaddleSpeech
git -C PaddleSpeech checkout 6b25a400008d393f9c3af837b3c692b17f29ee1a
```

## Python Environment

The project virtual environment is expected at:

```bash
$PROJECT/.venv
```

Create it on the remote:

```bash
cd "$PROJECT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install paddlepaddle-gpu==2.6.2 paddlespeech==1.5.0 numpy==1.26.4
```

Use it:

```bash
cd "$PROJECT"
. .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"
```

The package versions used for the completed run were:

```text
paddlepaddle-gpu==2.6.2
paddlespeech==1.5.0
numpy==1.26.4
```

NumPy `1.26.4` is intentional because PaddleSpeech `1.5.0` imports APIs that
are removed in NumPy 2.x.

## Pipeline Command

Once the ignored local assets are in place, run the direct Python pipeline:

```bash
python scripts/run_ds2_pipeline.py \
  --run-name ds2_cv_001 \
  --split-dir data/cross_validation_splits/cv_20260613_001 \
  --device gpu \
  --gpu-id 0 \
  --avg-num 1
```

Outputs are written under `results/<run-name>/`. Keep model weights, generated
data, and per-utterance outputs out of Git.
