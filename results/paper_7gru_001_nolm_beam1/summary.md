# paper_7gru_001_nolm_beam1

DeepSpeech2 VEPRAD pipeline run.

## Split

- CV split: `cv_paper_small_001`
- Split folder: `data/cross_validation_splits/cv_paper_small_001`
- Created at UTC: `2026-06-13T17:08:42.434444+00:00`

## Data

- `train`: 3057 utterances, 19 speakers, 4.722465 h
- `val`: 1205 utterances, 22 speakers, 1.806637 h
- `val_seen_speakers`: 349 utterances, 19 speakers, 0.544708 h
- `val_unseen_speakers`: 856 utterances, 3 speakers, 1.261928 h
- `test`: 993 utterances, 25 speakers, 1.50394 h
- `test_seen_speakers`: 486 utterances, 22 speakers, 0.741424 h
- `test_unseen_speakers`: 507 utterances, 3 speakers, 0.762517 h

## Model

- Output: `results/paper_7gru_001_nolm_beam1/details/model`
- Averaged checkpoint prefix: `results/paper_7gru_001_nolm_beam1/details/model/checkpoints/avg_1`
- Avg num: `1`
- Device: `gpu:0`

## Evaluation

| Subset | Utterances | WER | CER spaces | CER no spaces | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val` | 1205 | 0.400108 | 0.088596 | 0.081903 | 0.072199 |
| `val_seen_speakers` | 349 | 0.328464 | 0.069313 | 0.062533 | 0.106017 |
| `val_unseen_speakers` | 856 | 0.429248 | 0.096419 | 0.089747 | 0.058411 |
| `test` | 993 | 0.428946 | 0.106670 | 0.100878 | 0.067472 |
| `test_seen_speakers` | 486 | 0.324422 | 0.071603 | 0.064509 | 0.109053 |
| `test_unseen_speakers` | 507 | 0.531593 | 0.140550 | 0.135913 | 0.027613 |

## Notes

- The pipeline calls PaddleSpeech Python functions directly; it does not shell out through the old recipe scripts.
- Training and evaluation use the normalized transcript `text` field. Non-speech/event tags remain auditable in local `text_raw` and `non_speech_events` metadata, but PaddleSpeech DeepSpeech2 character mode would split tags such as `<uzdah>` into characters instead of learning them as one acoustic event token.
- Per-utterance decode files and generated PaddleSpeech manifests are local artifacts and are ignored by Git.
