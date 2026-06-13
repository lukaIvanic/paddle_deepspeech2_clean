# ds2_cv_20260613_001

DeepSpeech2 VEPRAD pipeline run.

## Split

- CV split: `cv_20260613_001`
- Split folder: `data/cross_validation_splits/cv_20260613_001`
- Created at UTC: `2026-06-13T14:18:25.437113+00:00`

## Data

- `train`: 3225 utterances, 19 speakers, 4.888142 h
- `val`: 1037 utterances, 22 speakers, 1.640959 h
- `val_seen_speakers`: 367 utterances, 19 speakers, 0.557562 h
- `val_unseen_speakers`: 670 utterances, 3 speakers, 1.083397 h
- `test`: 993 utterances, 25 speakers, 1.50394 h
- `test_seen_speakers`: 486 utterances, 22 speakers, 0.741424 h
- `test_unseen_speakers`: 507 utterances, 3 speakers, 0.762517 h

## Model

- Output: `results/ds2_cv_20260613_001/details/model`
- Averaged checkpoint prefix: `results/ds2_cv_20260613_001/details/model/checkpoints/avg_1`
- Avg num: `1`
- Device: `gpu:0`

## Evaluation

| Subset | Utterances | WER | CER spaces | CER no spaces | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val` | 1037 | 0.326573 | 0.067607 | 0.062750 | 0.109932 |
| `val_seen_speakers` | 367 | 0.270430 | 0.055290 | 0.051089 | 0.166213 |
| `val_unseen_speakers` | 670 | 0.355705 | 0.073983 | 0.068792 | 0.079104 |
| `test` | 993 | 0.384420 | 0.091377 | 0.087630 | 0.099698 |
| `test_seen_speakers` | 486 | 0.280320 | 0.058488 | 0.054233 | 0.172840 |
| `test_unseen_speakers` | 507 | 0.486651 | 0.123193 | 0.119848 | 0.029586 |

## Notes

- The pipeline calls PaddleSpeech Python functions directly; it does not shell out through the old recipe scripts.
- Training and evaluation use normalized transcript text with non-speech/event tags removed. The original `text_raw` and extracted `non_speech_events` metadata are preserved in the local manifests, but PaddleSpeech DeepSpeech2 character mode would split tags such as `<uzdah>` into characters instead of learning them as one acoustic event token.
- Training completed on the remote GPU first; evaluation was resumed with `--skip-train --reuse-run-dir` after fixing eval-output directory creation in the wrapper.
- Per-utterance decode files and generated PaddleSpeech manifests are local artifacts and are ignored by Git.
