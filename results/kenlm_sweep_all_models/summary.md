# kenlm_sweep_all_models

Cross-model summary of the shared KenLM decoder-parameter sweep.

All rows use the same CV split (`cv_paper_small_001`), the same evaluation
subsets, and a filtered word-level 5-gram KenLM trained from the split's train
transcripts. Each acoustic model was decoded with the same grid:

- `beam=5,10,20,40` at `alpha=2.5`, `beta=0.3`
- `beam=20`, `beta=0.3`, `alpha=2.0` and `alpha=3.0`
- `beam=20`, `alpha=2.5`, `beta=0.0` and `beta=0.6`

The full machine-readable sweep output is stored in `metrics.json` in this
folder. It contains every candidate, every evaluated subset, and the full metric
objects (`wer`, CER variants, exact-match rate, utterance counts, and edit
counts).

## Best Validation Rows

| Model | Params | Best validation setting | Val WER | Test WER | Test seen WER | Test unseen WER |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `paper_small_001` | 36.21M | `beam=40, alpha=2.5, beta=0.3` | 0.088331 | 0.118312 | 0.075366 | 0.160338 |
| `paper_7gru_001` | 50.73M | `beam=40, alpha=2.5, beta=0.3` | 0.093750 | 0.127067 | 0.082616 | 0.170721 |
| `paper_7lstm_001` | 67.61M | `beam=40, alpha=2.5, beta=0.3` | 0.123254 | 0.150864 | 0.094850 | 0.205874 |

## Beam-20 Comparison

| Model | Best beam-20 setting | Val WER | Test WER | Test seen WER | Test unseen WER |
| --- | --- | ---: | ---: | ---: | ---: |
| `paper_small_001` | `alpha=2.0, beta=0.3` | 0.099771 | 0.126169 | 0.079142 | 0.172352 |
| `paper_7gru_001` | `alpha=2.5, beta=0.6` | 0.103023 | 0.138143 | 0.088053 | 0.187333 |
| `paper_7lstm_001` | `alpha=2.0, beta=0.3` | 0.139571 | 0.168076 | 0.107990 | 0.226787 |

## Takeaways

- `beam=40, alpha=2.5, beta=0.3` is the best validation-selected decoding
  setting for all three trained acoustic models.
- The 5-layer GRU paper-small model is best in this sweep, despite having fewer
  parameters than the 7-layer variants.
- The 7-layer GRU model is consistently better than the 7-layer LSTM model.
- Seen-speaker subsets are substantially easier than unseen-speaker subsets for
  all models, both on validation and frozen test.
- This sweep changes only decoder parameters. It does not retrain acoustic
  models and does not change the KenLM n-gram order.

## Detailed Summaries

- `results/paper_small_001_kenlm_sweep_all/summary.md`
- `results/paper_7gru_001_kenlm_sweep_all/summary.md`
- `results/paper_7lstm_001_kenlm_sweep_all/summary.md`
