# 2026-06-13 DeepSpeech2 Baseline on VEPRAD

This was the first full VEPRAD training run using the official PaddleSpeech
DeepSpeech2 implementation. It was run from scratch on a Vast.ai GPU box, not
with the earlier local toy DeepSpeech2-shaped script.

## Run Metadata

- Date: 2026-06-13
- Remote workdir: `/workspace/rogj_paddlespeech_ds2/paddle_deepspeech2_clean`
- PaddleSpeech repo commit: `6b25a400008d393f9c3af837b3c692b17f29ee1a`
- Python: `3.10.14`
- PaddlePaddle: `2.6.2`
- PaddleSpeech: `1.5.0`
- NumPy: `1.26.4`
- GPU: Vast.ai RTX 3090, instance `40802979`, label `rogj-paddlespeech-ds2`

## Data Split

This run used the existing VEPRAD manifests copied from the earlier project:

- Train: 6754 utterances
- Dev: 249 utterances
- Test: 580 utterances

The character vocabulary was built from the training manifest only. Test
audio/transcripts were used only for the final held-out decode after selecting
the checkpoint by dev loss.

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

## Historical Run Path

The run was initiated through the earlier adapted PaddleSpeech shell recipe.
That active recipe has since been replaced by the direct Python pipeline in
`scripts/run_ds2_pipeline.py`.

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

Best dev loss was epoch 10. The later epochs increased dev loss, so this run
shows overfitting after epoch 10.

The full curve is in `details/training_curve.tsv`.

## Results

The final decode used the dev-selected best checkpoint via `avg_num=1`.

- PaddleSpeech WER: `0.362968`
- Independently recomputed WER: `0.362968231389284`
- Independently recomputed CER, spaces included: `0.07469918006602065`
- Independently recomputed CER, spaces removed: `0.07085997112806765`
- Exact utterance matches: 41 / 580, or `0.0706896551724138`

This is a legitimate from-scratch PaddleSpeech DeepSpeech2 baseline on VEPRAD,
but it is far worse than the separate pretrained wav2vec2-style baseline used
elsewhere in the project.

## Details Layout

Tracked supporting files:

- `details/metrics.json`: independent WER/CER calculation
- `details/training_curve.tsv`: parsed epoch/dev-loss curve
- `details/environment_versions.txt`: remote software versions
- `details/config/`: training, preprocessing, decode, and effective PaddleSpeech configs
- `details/logs/`: training, averaging, and evaluation logs
- `details/checkpoints/avg_1.avg.json`: checkpoint averaging metadata

Local ignored supporting files, if present:

- `details/decoded/avg_1.rsl`: decoded held-out references and hypotheses
- `details/analysis/utterance_metrics.tsv`: per-utterance WER/CER/error counts
- `details/analysis/top_30_errors.md`: highest-WER test examples for manual error analysis
- `details/generated/`: generated CMVN/vocabulary artifacts

Checkpoint weights are not stored in this repository. Source code is represented
by the repository commit history instead of copying scripts into the run folder.
