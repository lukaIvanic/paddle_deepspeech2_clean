# paper_small_001_kenlm_sweep_val

Validation-only KenLM decode sweep for the paper-small DeepSpeech2 run.

The sweep reused the acoustic checkpoint from `paper_small_001_nolm_beam1`
(`avg_1`, selected epoch 11) and the filtered 5-gram KenLM from
`paper_small_001_kenlm5_beam10`. Only the full validation set was decoded, so
the frozen test set was not used for hyperparameter selection.

The full machine-readable validation sweep output is stored in `metrics.json`
in this folder.

## Fixed Settings

- Split: `cv_paper_small_001`
- Evaluation component: `val`
- Validation utterances: 1205
- Acoustic model: 5 bidirectional GRU layers, hidden size 650, 36.21M params
- LM: filtered KenLM 5-gram trained from 2290/3057 train transcripts
- Decode method: CTC beam search with external scorer
- `cutoff_prob`: 1.0
- `cutoff_top_n`: 40
- `num_proc_bsearch`: 1

## Results

The existing `paper_small_001_kenlm5_beam10` result is included as the baseline
row. The other rows were run as validation-only sweep candidates on the remote
GPU and took about 75-85 seconds each.

| Run | Beam | Alpha | Beta | Val WER | Val CER spaces | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `paper_small_001_sweep_val_b20_a2p5_bt0p3` | 20 | 2.5 | 0.3 | 0.101397 | 0.040365 | 0.447303 |
| `paper_small_001_sweep_val_b10_a1p5_bt0p3` | 10 | 1.5 | 0.3 | 0.118136 | 0.048066 | 0.415768 |
| `paper_small_001_sweep_val_b10_a2p5_bt0p6` | 10 | 2.5 | 0.6 | 0.122230 | 0.054387 | 0.412448 |
| `paper_small_001_kenlm5_beam10` | 10 | 2.5 | 0.3 | 0.125783 | 0.056706 | 0.408299 |
| `paper_small_001_sweep_val_b10_a2p5_bt0p0` | 10 | 2.5 | 0.0 | 0.129697 | 0.059098 | 0.403320 |
| `paper_small_001_sweep_val_b10_a3p5_bt0p3` | 10 | 3.5 | 0.3 | 0.151433 | 0.073028 | 0.375934 |
| `paper_small_001_sweep_val_b05_a2p5_bt0p3` | 5 | 2.5 | 0.3 | 0.177384 | 0.091945 | 0.331120 |

## Takeaways

- Increasing beam size from 10 to 20 gave the best validation result in this
  sweep.
- Lowering alpha from 2.5 to 1.5 helped at beam 10; raising alpha to 3.5 hurt.
- At beam 10 and alpha 2.5, beta 0.6 was slightly better than beta 0.3, while
  beta 0.0 was worse.
- The next focused sweep should try beam 20 with alpha around 1.5-2.5 and beta
  around 0.3-0.6, then run the selected setting once on validation and frozen
  test seen/unseen subsets.
