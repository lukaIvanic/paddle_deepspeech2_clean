# PaddleSpeech DeepSpeech2 VEPRAD Full Run

Date: 2026-06-13

This folder archives the first full VEPRAD run using the actual PaddleSpeech DeepSpeech2 implementation on a Vast.ai GPU box. This is not the local toy DeepSpeech2-shaped script.

## Environment

- Remote workdir: `/workspace/rogj_paddlespeech_ds2`
- PaddleSpeech repo commit: `6b25a40`
- Python: `3.10.14`
- PaddlePaddle: `2.6.2`
- PaddleSpeech: `1.5.0`
- NumPy: `1.26.4`
- GPU instance: Vast.ai RTX 3090, instance `40802979`, label `rogj-paddlespeech-ds2`

## Data

Local source manifests:

- Train: `Projekt - DeepSpeech2/data/manifests/veprad/train.jsonl`, 6754 utterances
- Dev: `Projekt - DeepSpeech2/data/manifests/veprad/dev.jsonl`, 249 utterances
- Test: `Projekt - DeepSpeech2/data/manifests/veprad/test.jsonl`, 580 utterances

The PaddleSpeech recipe used the existing project split. The character vocabulary was built from the training manifest only. Test audio/transcripts were used only for the final held-out decode after selecting the checkpoint by dev loss.

## Model

- Toolkit/model: official PaddleSpeech `DeepSpeech2Model`
- Encoder: convolutional subsampling plus recurrent stack
- Convolution layers: 2
- Recurrent layers: 5 bidirectional LSTM layers
- Recurrent hidden size: 1024
- Feature type: 161-bin fbank with CMVN
- Parameters: 113.96M elements
- Epochs: 15
- Batch size: 32
- Learning rate: `5.0e-4`
- LR decay: `0.93`
- Decoder: CTC beam search with `beam_size: 1`
- External language model: none

## Dev Curve

| Epoch | Dev loss |
| ---: | ---: |
| 0 | 259.172779 |
| 1 | 258.923302 |
| 2 | 259.089157 |
| 3 | 258.675516 |
| 4 | 259.080117 |
| 5 | 258.217734 |
| 6 | 92.569345 |
| 7 | 42.180667 |
| 8 | 28.267362 |
| 9 | 22.519129 |
| 10 | 21.200106 |
| 11 | 21.950419 |
| 12 | 24.272049 |
| 13 | 25.319377 |
| 14 | 27.002593 |

Best dev loss was epoch 10. The later epochs increased dev loss, so this run shows overfitting after epoch 10.

## Test Result

The final decode used the dev-selected best checkpoint via `avg_num=1`.

- PaddleSpeech WER: `0.362968`
- Independently recomputed WER: `0.362968231389284`
- Independently recomputed CER, spaces included: `0.07469918006602065`
- Independently recomputed CER, spaces removed: `0.07085997112806765`
- Exact utterance matches: 41 / 580, or `0.0706896551724138`

This is a legitimate from-scratch PaddleSpeech DeepSpeech2 baseline on VEPRAD, but it is far worse than the separate pretrained wav2vec2-style baseline used elsewhere in the project.

## Files

- `conf/deepspeech2.yaml`: training config used by the recipe
- `conf/preprocess.yaml`: feature/CMVN preprocessing config
- `conf/tuning/decode.yaml`: decode config
- `local/data.sh`: VEPRAD data preparation script for the PaddleSpeech recipe
- `run.sh`: copied PaddleSpeech recipe driver
- `exp/deepspeech2/checkpoints/avg_1.rsl`: decoded held-out test references and hypotheses
- `exp/deepspeech2/eval_avg1.log`: final evaluation log
- `metrics.json`: independent WER/CER calculation from `avg_1.rsl`
- `utterance_metrics.tsv`: per-utterance WER/CER/error counts
- `top_30_errors.md`: highest-WER test examples for manual error analysis
- `training_curve.tsv`: parsed epoch/dev-loss curve
- `environment_versions.txt`: remote software versions
- `code_snapshot/`: copied recipe scripts/configs and the relevant official PaddleSpeech DeepSpeech2 source files used by this run

The checkpoint weights were not copied into this folder.
