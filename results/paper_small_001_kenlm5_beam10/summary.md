# paper_small_001_kenlm5_beam10

DeepSpeech2 VEPRAD pipeline run.

## Split

- CV split: `cv_paper_small_001`
- Split folder: `data/cross_validation_splits/cv_paper_small_001`
- Created at UTC: `2026-06-13T16:34:42.597695+00:00`

## Data

- `train`: 3057 utterances, 19 speakers, 4.722465 h
- `val`: 1205 utterances, 22 speakers, 1.806637 h
- `val_seen_speakers`: 349 utterances, 19 speakers, 0.544708 h
- `val_unseen_speakers`: 856 utterances, 3 speakers, 1.261928 h
- `test`: 993 utterances, 25 speakers, 1.50394 h
- `test_seen_speakers`: 486 utterances, 22 speakers, 0.741424 h
- `test_unseen_speakers`: 507 utterances, 3 speakers, 0.762517 h

## Model

- Output: `results/paper_small_001_kenlm5_beam10/details/model`
- Averaged checkpoint prefix: `results/paper_small_001_kenlm5_beam10/details/model/checkpoints/avg_1`
- Avg num: `1`
- Device: `gpu:0`

## Evaluation

| Subset | Utterances | WER | CER spaces | CER no spaces | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val` | 1205 | 0.125783 | 0.056706 | 0.054695 | 0.408299 |
| `val_seen_speakers` | 349 | 0.101836 | 0.047519 | 0.046108 | 0.467049 |
| `val_unseen_speakers` | 856 | 0.135494 | 0.060420 | 0.058166 | 0.384346 |
| `test` | 993 | 0.154980 | 0.078183 | 0.078027 | 0.358510 |
| `test_seen_speakers` | 486 | 0.097115 | 0.040685 | 0.038738 | 0.495885 |
| `test_unseen_speakers` | 507 | 0.211807 | 0.114459 | 0.115930 | 0.226824 |

## Language Model Details

- Acoustic checkpoint: copied from `paper_small_001_nolm_beam1`, selected epoch 11 (`avg_1`) by validation loss 35.604879
- Decode config: CTC beam search with KenLM 5-gram, beam size 10, alpha 2.5, beta 0.3
- KenLM binary: `results/paper_small_001_kenlm5_beam10/details/lm/lm.klm`
- LM training manifest: `data/cross_validation_splits/cv_paper_small_001/train.jsonl`
- LM text filtering: kept 2290/3057 train rows; excluded 767 rows (507 exact, 260 fuzzy) against validation plus frozen test text
- Fuzzy filter: threshold 0.9, minimum length ratio 0.8

## Notes

- The pipeline calls PaddleSpeech Python functions directly; it does not shell out through the old recipe scripts.
- Training and evaluation use the normalized transcript `text` field. Non-speech/event tags remain auditable in local `text_raw` and `non_speech_events` metadata, but PaddleSpeech DeepSpeech2 character mode would split tags such as `<uzdah>` into characters instead of learning them as one acoustic event token.
- Per-utterance decode files and generated PaddleSpeech manifests are local artifacts and are ignored by Git.
