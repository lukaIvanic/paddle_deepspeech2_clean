# paper_7lstm_001_kenlm5_beam20_a2p5_b0p3

DeepSpeech2 VEPRAD pipeline run.

## Split

- CV split: `cv_paper_small_001`
- Split folder: `data/cross_validation_splits/cv_paper_small_001`
- Created at UTC: `2026-06-13T17:54:50.836626+00:00`

## Data

- `train`: 3057 utterances, 19 speakers, 4.722465 h
- `val`: 1205 utterances, 22 speakers, 1.806637 h
- `val_seen_speakers`: 349 utterances, 19 speakers, 0.544708 h
- `val_unseen_speakers`: 856 utterances, 3 speakers, 1.261928 h
- `test`: 993 utterances, 25 speakers, 1.50394 h
- `test_seen_speakers`: 486 utterances, 22 speakers, 0.741424 h
- `test_unseen_speakers`: 507 utterances, 3 speakers, 0.762517 h

## Model

- Output: `results/paper_7lstm_001_kenlm5_beam20_a2p5_b0p3/details/model`
- Averaged checkpoint prefix: `results/paper_7lstm_001_kenlm5_beam20_a2p5_b0p3/details/model/checkpoints/avg_1`
- Avg num: `1`
- Device: `gpu:0`

## Evaluation

| Subset | Utterances | WER | CER spaces | CER no spaces | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val` | 1205 | 0.144268 | 0.062935 | 0.060623 | 0.370124 |
| `val_seen_speakers` | 349 | 0.104549 | 0.044323 | 0.042868 | 0.452722 |
| `val_unseen_speakers` | 856 | 0.160460 | 0.070577 | 0.067950 | 0.336449 |
| `test` | 993 | 0.174736 | 0.086631 | 0.086784 | 0.326284 |
| `test_seen_speakers` | 486 | 0.114333 | 0.047866 | 0.045409 | 0.465021 |
| `test_unseen_speakers` | 507 | 0.234055 | 0.124132 | 0.126700 | 0.193294 |

## Notes

- The pipeline calls PaddleSpeech Python functions directly; it does not shell out through the old recipe scripts.
- Training and evaluation use the normalized transcript `text` field. Non-speech/event tags remain auditable in local `text_raw` and `non_speech_events` metadata, but PaddleSpeech DeepSpeech2 character mode would split tags such as `<uzdah>` into characters instead of learning them as one acoustic event token.
- Per-utterance decode files and generated PaddleSpeech manifests are local artifacts and are ignored by Git.
