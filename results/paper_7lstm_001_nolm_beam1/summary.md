# paper_7lstm_001_nolm_beam1

DeepSpeech2 VEPRAD pipeline run.

## Split

- CV split: `cv_paper_small_001`
- Split folder: `data/cross_validation_splits/cv_paper_small_001`
- Created at UTC: `2026-06-13T17:35:21.007909+00:00`

## Data

- `train`: 3057 utterances, 19 speakers, 4.722465 h
- `val`: 1205 utterances, 22 speakers, 1.806637 h
- `val_seen_speakers`: 349 utterances, 19 speakers, 0.544708 h
- `val_unseen_speakers`: 856 utterances, 3 speakers, 1.261928 h
- `test`: 993 utterances, 25 speakers, 1.50394 h
- `test_seen_speakers`: 486 utterances, 22 speakers, 0.741424 h
- `test_unseen_speakers`: 507 utterances, 3 speakers, 0.762517 h

## Model

- Output: `results/paper_7lstm_001_nolm_beam1/details/model`
- Averaged checkpoint prefix: `results/paper_7lstm_001_nolm_beam1/details/model/checkpoints/avg_1`
- Avg num: `1`
- Device: `gpu:0`

## Evaluation

| Subset | Utterances | WER | CER spaces | CER no spaces | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val` | 1205 | 0.514571 | 0.123983 | 0.114171 | 0.035685 |
| `val_seen_speakers` | 349 | 0.430092 | 0.097338 | 0.087337 | 0.063037 |
| `val_unseen_speakers` | 856 | 0.548832 | 0.134744 | 0.125002 | 0.024533 |
| `test` | 993 | 0.535434 | 0.141152 | 0.133413 | 0.031219 |
| `test_seen_speakers` | 486 | 0.440417 | 0.098550 | 0.088075 | 0.051440 |
| `test_unseen_speakers` | 507 | 0.628745 | 0.182366 | 0.177151 | 0.011834 |

## Notes

- The pipeline calls PaddleSpeech Python functions directly; it does not shell out through the old recipe scripts.
- Training and evaluation use the normalized transcript `text` field. Non-speech/event tags remain auditable in local `text_raw` and `non_speech_events` metadata, but PaddleSpeech DeepSpeech2 character mode would split tags such as `<uzdah>` into characters instead of learning them as one acoustic event token.
- Per-utterance decode files and generated PaddleSpeech manifests are local artifacts and are ignored by Git.
