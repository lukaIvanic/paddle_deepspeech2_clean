# paper_small_001_nolm_beam1

DeepSpeech2 VEPRAD pipeline run.

## Split

- CV split: `cv_paper_small_001`
- Split folder: `data/cross_validation_splits/cv_paper_small_001`
- Created at UTC: `2026-06-13T16:15:26.042204+00:00`

## Data

- `train`: 3057 utterances, 19 speakers, 4.722465 h
- `val`: 1205 utterances, 22 speakers, 1.806637 h
- `val_seen_speakers`: 349 utterances, 19 speakers, 0.544708 h
- `val_unseen_speakers`: 856 utterances, 3 speakers, 1.261928 h
- `test`: 993 utterances, 25 speakers, 1.50394 h
- `test_seen_speakers`: 486 utterances, 22 speakers, 0.741424 h
- `test_unseen_speakers`: 507 utterances, 3 speakers, 0.762517 h

## Model

- Output: `results/paper_small_001_nolm_beam1/details/model`
- Averaged checkpoint prefix: `results/paper_small_001_nolm_beam1/details/model/checkpoints/avg_1`
- Avg num: `1`
- Device: `gpu:0`

## Evaluation

| Subset | Utterances | WER | CER spaces | CER no spaces | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val` | 1205 | 0.420641 | 0.094172 | 0.087317 | 0.053942 |
| `val_seen_speakers` | 349 | 0.344741 | 0.074713 | 0.067970 | 0.088825 |
| `val_unseen_speakers` | 856 | 0.451422 | 0.102040 | 0.095136 | 0.039720 |
| `test` | 993 | 0.452892 | 0.109974 | 0.103349 | 0.062437 |
| `test_seen_speakers` | 486 | 0.350400 | 0.074190 | 0.065854 | 0.104938 |
| `test_unseen_speakers` | 507 | 0.553545 | 0.144615 | 0.139546 | 0.021696 |

## Training Details

- Config: `conf/deepspeech2_paper_small.yaml`
- Requested epochs: 20
- Selected checkpoint: epoch 11 (`avg_1`) by validation loss 35.604879
- Features: `fbank_kaldi` with global CMVN, 161 feature dimensions
- Architecture: fixed PaddleSpeech 2-layer convolutional subsampling front end, 5 bidirectional GRU layers, hidden size 650, no FC layers, CTC output over 31 vocabulary entries
- Parameters: 36.21M
- Decoding in this run: CTC beam-search path with no external language model, beam size 1

## Notes

- The pipeline calls PaddleSpeech Python functions directly; it does not shell out through the old recipe scripts.
- Training and evaluation use the normalized transcript `text` field. Non-speech/event tags remain auditable in local `text_raw` and `non_speech_events` metadata, but PaddleSpeech DeepSpeech2 character mode would split tags such as `<uzdah>` into characters instead of learning them as one acoustic event token.
- Per-utterance decode files and generated PaddleSpeech manifests are local artifacts and are ignored by Git.
