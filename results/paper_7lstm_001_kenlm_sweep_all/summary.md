# paper_7lstm_001_kenlm_sweep_all

KenLM decoder-parameter sweep for the 7-layer LSTM DeepSpeech2 VEPRAD run.

This sweep reused the trained acoustic checkpoint from `paper_7lstm_001_nolm_beam1`
and the filtered word-level 5-gram KenLM from
`paper_7lstm_001_kenlm5_beam20_a2p5_b0p3`. No acoustic model training was run
for these rows.

## Fixed Settings

- CV split: `cv_paper_small_001`
- Acoustic checkpoint: `results/paper_7lstm_001_nolm_beam1/details/model/checkpoints/avg_1`
- Acoustic model: 7 bidirectional LSTM layers, hidden size 650, 67.61M params
- LM: filtered word-level KenLM 5-gram trained from 2290/3057 train transcripts
- LM filtering: 767 train transcripts excluded by exact/fuzzy overlap with val/test text
- Decode method: CTC beam search with external scorer
- `cutoff_prob`: 1.0
- `cutoff_top_n`: 40
- `num_proc_bsearch`: 1
- Evaluated subsets: `val`, `val_seen_speakers`, `val_unseen_speakers`, `test`, `test_seen_speakers`, `test_unseen_speakers`

## Results

| Run | Beam | Alpha | Beta | Val WER | Val seen WER | Val unseen WER | Test WER | Test seen WER | Test unseen WER | Approx seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `paper_7lstm_001_sweep_all_b05_a2p5_bt0p3` | 5 | 2.5 | 0.3 | 0.274446 | 0.212020 | 0.299763 | 0.311233 | 0.212355 | 0.408336 | 254 |
| `paper_7lstm_001_sweep_all_b10_a2p5_bt0p3` | 10 | 2.5 | 0.3 | 0.188764 | 0.143155 | 0.207177 | 0.219412 | 0.146655 | 0.290715 | 226 |
| `paper_7lstm_001_sweep_all_b20_a2p5_bt0p3` | 20 | 2.5 | 0.3 | 0.144268 | 0.104549 | 0.160460 | 0.174736 | 0.114333 | 0.234055 | 232 |
| `paper_7lstm_001_sweep_all_b40_a2p5_bt0p3` | 40 | 2.5 | 0.3 | 0.123254 | 0.090150 | 0.136679 | 0.150864 | 0.094850 | 0.205874 | 260 |
| `paper_7lstm_001_sweep_all_b20_a2p0_bt0p3` | 20 | 2.0 | 0.3 | 0.139571 | 0.103506 | 0.154198 | 0.168076 | 0.107990 | 0.226787 | 233 |
| `paper_7lstm_001_sweep_all_b20_a3p0_bt0p3` | 20 | 3.0 | 0.3 | 0.153661 | 0.114357 | 0.169601 | 0.185138 | 0.126567 | 0.242510 | 226 |
| `paper_7lstm_001_sweep_all_b20_a2p5_bt0p0` | 20 | 2.5 | 0.0 | 0.147278 | 0.107262 | 0.163676 | 0.177655 | 0.118713 | 0.235538 | 254 |
| `paper_7lstm_001_sweep_all_b20_a2p5_bt0p6` | 20 | 2.5 | 0.6 | 0.140113 | 0.102254 | 0.155383 | 0.171668 | 0.111312 | 0.230940 | 240 |

## Takeaways

- The best validation WER is `beam=40, alpha=2.5, beta=0.3`.
- That same row also gives the best WER on every test subset.
- Among beam-20 settings, `alpha=2.0, beta=0.3` is best on validation and test.
- Raising alpha to 3.0 hurts, and removing beta hurts compared with the better beam-20 settings.
- This was not an n-gram order sweep. All rows use the same filtered word-level 5-gram KenLM.
