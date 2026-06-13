# paper_small_001_kenlm_sweep_all

KenLM decoder-parameter sweep for the paper-small DeepSpeech2 VEPRAD run.

This sweep reused the trained acoustic checkpoint from `paper_small_001_nolm_beam1`
and the filtered word-level 5-gram KenLM from `paper_small_001_kenlm5_beam10`.
No acoustic model training was run for these rows.

## Fixed Settings

- CV split: `cv_paper_small_001`
- Acoustic checkpoint: `results/paper_small_001_nolm_beam1/details/model/checkpoints/avg_1`
- Acoustic model: 5 bidirectional GRU layers, hidden size 650, 36.21M params
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
| `paper_small_001_sweep_all_b05_a2p5_bt0p3` | 5 | 2.5 | 0.3 | 0.177384 | 0.144825 | 0.190758 | 0.218289 | 0.139103 | 0.296055 | 224 |
| `paper_small_001_sweep_all_b10_a2p5_bt0p3` | 10 | 2.5 | 0.3 | 0.125783 | 0.101836 | 0.135494 | 0.154980 | 0.097115 | 0.211807 | 250 |
| `paper_small_001_sweep_all_b20_a2p5_bt0p3` | 20 | 2.5 | 0.3 | 0.101397 | 0.083055 | 0.108835 | 0.129312 | 0.081106 | 0.176654 | 239 |
| `paper_small_001_sweep_all_b40_a2p5_bt0p3` | 40 | 2.5 | 0.3 | 0.088331 | 0.073664 | 0.094279 | 0.118312 | 0.075366 | 0.160338 | 228 |
| `paper_small_001_sweep_all_b20_a2p0_bt0p3` | 20 | 2.0 | 0.3 | 0.099771 | 0.078673 | 0.108412 | 0.126169 | 0.079142 | 0.172352 | 245 |
| `paper_small_001_sweep_all_b20_a3p0_bt0p3` | 20 | 3.0 | 0.3 | 0.108020 | 0.088689 | 0.115860 | 0.134326 | 0.086241 | 0.181549 | 247 |
| `paper_small_001_sweep_all_b20_a2p5_bt0p0` | 20 | 2.5 | 0.0 | 0.103023 | 0.085142 | 0.110274 | 0.129761 | 0.080955 | 0.177692 | 226 |
| `paper_small_001_sweep_all_b20_a2p5_bt0p6` | 20 | 2.5 | 0.6 | 0.100614 | 0.080760 | 0.108582 | 0.126693 | 0.079595 | 0.172946 | 237 |

## Takeaways

- The best validation WER is `beam=40, alpha=2.5, beta=0.3`.
- That same row also gives the best WER on every test subset.
- Among beam-20 settings, `alpha=2.0, beta=0.3` is best on validation and test.
- Raising alpha to 3.0 hurts, and removing beta hurts compared with the better beam-20 settings.
- This was not an n-gram order sweep. All rows use the same filtered word-level 5-gram KenLM.
