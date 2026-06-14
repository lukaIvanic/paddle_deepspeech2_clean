# HF Wav2Vec2 Clean VEPRAD Reproduction Track

This track is independent of PaddleSpeech/DeepSpeech2. It fine-tunes a
Hugging Face CTC ASR model on the audited VEPRAD train split, evaluates on all
clean validation/test subsets, and trains a run-local KenLM only from filtered
train text.

Default acoustic checkpoint:

```text
classla/wav2vec2-xls-r-parlaspeech-hr
```

That checkpoint is Croatian ASR based on XLS-R 300M. We use it without its
external LM variant, fine-tune on VEPRAD train only, and decode with either
greedy CTC or our own clean KenLM.

## Remote Setup

Install the non-Paddle dependencies in the remote Python environment:

```bash
python -m pip install \
  transformers accelerate soundfile pyctcdecode kenlm tqdm pyyaml
```

KenLM command-line tools must also be on `PATH`:

```bash
export PATH=/workspace/kenlm_tools/build/bin:$PATH
```

## Run

```bash
python experiment_tracks/hf_wav2vec2_clean_20260614/run_hf_ctc_finetune.py \
  --device cuda \
  --epochs 5 \
  --train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --eval-batch-size 8
```

The runner creates a new CV split unless the split folder already exists. It
then validates split isolation, trains/fine-tunes the acoustic model, trains a
KenLM corpus after exact/fuzzy held-out text filtering, validates the post-filter
LM corpus, and evaluates:

- `val`
- `val_seen_speakers`
- `val_unseen_speakers`
- `test`
- `test_seen_speakers`
- `test_unseen_speakers`

Outputs go under:

```text
results/hf_wav2vec2_clean_20260614/
```

Aggregate WER/CER metrics are written to `track_report.json` and per-subset
metric files. Transcript-level predictions are deliberately kept under ignored
`details/predictions/`.
