# paper_7gru_001_kenlm_sweep_all

KenLM decoder-parameter sweep for the 7-layer GRU DeepSpeech2 VEPRAD run.

This sweep reused the trained acoustic checkpoint from
`paper_7gru_001_nolm_beam1` and the filtered word-level 5-gram KenLM from
`paper_7gru_001_kenlm5_beam20_a2p5_b0p3`. No acoustic model training was run
for these rows.

## Fixed Settings

- CV split: `cv_paper_small_001`
- Acoustic checkpoint: `results/paper_7gru_001_nolm_beam1/details/model/checkpoints/avg_1`
- Acoustic model: 7 bidirectional GRU layers, hidden size 650, 50.73M params
- LM: filtered word-level KenLM 5-gram trained from 2290/3057 train transcripts
- LM filtering: 767 train transcripts excluded by exact/fuzzy overlap with val/test text
- PaddleSpeech scorer: `is_character_based = 0`, `max_order = 5`, `dict_size = 1181`
- Decode method: CTC beam search with external scorer
- `cutoff_prob`: 1.0
- `cutoff_top_n`: 40
- `num_proc_bsearch`: 1
- Evaluated subsets: `val`, `val_seen_speakers`, `val_unseen_speakers`, `test`, `test_seen_speakers`, `test_unseen_speakers`

## Results

The rows vary only `beam_size`, `alpha`, and `beta`. WER is reported for each
evaluation subset. The existing `beam=20, alpha=2.5, beta=0.3` configuration is
included as the comparison row.

| Run | Beam | Alpha | Beta | Val WER | Val seen WER | Val unseen WER | Test WER | Test seen WER | Test unseen WER | Approx seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `paper_7gru_001_sweep_all_b05_a2p5_bt0p3` | 5 | 2.5 | 0.3 | 0.185814 | 0.153381 | 0.198460 | 0.234304 | 0.144238 | 0.322753 | 224 |
| `paper_7gru_001_sweep_all_b10_a2p5_bt0p3` | 10 | 2.5 | 0.3 | 0.131081 | 0.108514 | 0.140234 | 0.170471 | 0.108745 | 0.231237 | 253 |
| `paper_7gru_001_sweep_all_b20_a2p5_bt0p3` | 20 | 2.5 | 0.3 | 0.105070 | 0.087437 | 0.112221 | 0.141660 | 0.090470 | 0.191931 | 240 |
| `paper_7gru_001_sweep_all_b40_a2p5_bt0p3` | 40 | 2.5 | 0.3 | 0.093750 | 0.076586 | 0.100711 | 0.127067 | 0.082616 | 0.170721 | 229 |
| `paper_7gru_001_sweep_all_b20_a2p0_bt0p3` | 20 | 2.0 | 0.3 | 0.103986 | 0.084725 | 0.111713 | 0.140088 | 0.088808 | 0.190448 | 246 |
| `paper_7gru_001_sweep_all_b20_a3p0_bt0p3` | 20 | 3.0 | 0.3 | 0.111272 | 0.092654 | 0.118822 | 0.148769 | 0.097115 | 0.199496 | 248 |
| `paper_7gru_001_sweep_all_b20_a2p5_bt0p0` | 20 | 2.5 | 0.0 | 0.106575 | 0.088898 | 0.113744 | 0.145252 | 0.093490 | 0.196381 | 228 |
| `paper_7gru_001_sweep_all_b20_a2p5_bt0p6` | 20 | 2.5 | 0.6 | 0.103023 | 0.083681 | 0.110867 | 0.138143 | 0.088053 | 0.187333 | 240 |

## Takeaways

- The best validation WER in this sweep is `beam=40, alpha=2.5, beta=0.3`.
  It also gives the best WER on every test subset.
- Among beam-20 settings, `alpha=2.5, beta=0.6` is best on validation and test.
- Beam 5 and beam 10 are substantially worse, so the current model benefits
  from a wider CTC beam.
- Increasing alpha to 3.0 hurts compared with 2.0-2.5, suggesting that the LM
  should not be weighted more strongly for this checkpoint.
- Removing the insertion bonus (`beta=0.0`) hurts compared with `beta=0.3` and
  `beta=0.6`.
- The frozen test subsets were included here for reporting and diagnostics. For
  formal hyperparameter selection, the choice should be justified from
  validation results first; in this sweep, that still selects the beam-40 row.

## Notes

- This was not an n-gram order sweep. All rows use the same filtered word-level
  5-gram KenLM.
- Per-candidate run folders were generated on the remote GPU workspace under
  `results/paper_7gru_001_sweep_all_*`.
- Full metric JSON files and PaddleSpeech logs remain in the remote run
  folders. This summary records the report-level numbers.
