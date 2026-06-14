# HF Wav2Vec2 Clean VEPRAD Reproduction Track

This track is independent of PaddleSpeech/DeepSpeech2. It reproduces the
Hugging Face Wav2Vec2 + LM fine-tuning methodology on the audited VEPRAD split:
fine-tune on train only, evaluate on all clean validation/test subsets, and keep
the bundled language model fixed.

Default checkpoint:

```text
classla/wav2vec2-large-slavic-parlaspeech-hr-lm
```

That checkpoint is Croatian ASR based on `facebook/wav2vec2-large-slavic-
voxpopuli-v2`, fine-tuned on ParlaSpeech-HR and enhanced with a 5-gram LM based
on ParlaMint. The VEPRAD run does not train that bundled LM and does not use
VEPRAD validation/test text for training.

## Remote Setup

Install the non-Paddle dependencies in the remote Python environment:

```bash
python -m pip install \
  "transformers==4.18.0" "tokenizers==0.12.1" "numpy==1.26.4" \
  accelerate soundfile pyctcdecode kenlm tqdm pyyaml
```

KenLM command-line tools are only needed if `--train-local-veprad-kenlm` is
enabled:

```bash
export PATH=/workspace/kenlm_tools/build/bin:$PATH
```

## Run

```bash
python experiment_tracks/hf_wav2vec2_clean_20260614/run_hf_ctc_finetune.py \
  --model-name /workspace/hf_models/classla_wav2vec2_large_slavic_parlaspeech_hr_lm_local_tokenizer_fix \
  --device cuda \
  --force-run
```

The local model path above is a snapshot of the upstream CLASSLA LM checkpoint
with tokenizer metadata normalized so the documented `Wav2Vec2ProcessorWithLM`
path can load it. The acoustic weights and bundled LM files are not changed.

The runner creates a new CV split unless the split folder already exists. It
then validates split isolation, evaluates the starting checkpoint, fine-tunes
the acoustic model on VEPRAD train, and evaluates the fine-tuned checkpoint:

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

If `--train-local-veprad-kenlm` is enabled, the script additionally trains a
run-owned VEPRAD KenLM only from train transcripts after exact/fuzzy filtering
against validation and frozen test text, validates that filter, and reports it
separately from the bundled checkpoint LM.
