# hf_wav2vec2_clean_20260614

- Model: `/workspace/hf_models/classla_wav2vec2_large_slavic_parlaspeech_hr_lm_local_tokenizer_fix`
- Split: `cv_hf_wav2vec2_clean_20260614_001`
- Train utterances: `3298`
- Epochs: `8.0`
- Bundled LM: `True`

## Split

| Subset | Utterances | Hours | Speakers |
|---|---:|---:|---:|
| train | 3298 | 5.026953 | 19 |
| val | 964 | 1.502149 | 22 |
| val_seen_speakers | 375 | 0.570410 | 19 |
| val_unseen_speakers | 589 | 0.931739 | 3 |
| test | 993 | 1.503940 | 25 |
| test_seen_speakers | 486 | 0.741424 | 22 |
| test_unseen_speakers | 507 | 0.762517 | 3 |

Validation unseen speakers: `m05`, `m11`, `z07`.
Test unseen speakers: `m04`, `z06`, `z14`.

## Training

| Setting | Value |
|---|---:|
| epochs | 8 |
| train batch size | 16 |
| gradient accumulation | 4 |
| effective batch size | 64 |
| optimizer updates | 408 |
| learning rate | 0.0003 |
| warmup steps | 500 |
| fp16 | true |
| freeze feature encoder | true |
| gradient checkpointing | true |

## Notes

- The bundled LM is the checkpoint's fixed external ParlaMint LM; no VEPRAD KenLM was trained for this run.
- The split validator passed with no structural errors and no `sm04` data.
- The validator reports repeated or near-repeated transcript text across train and held-out sets. This matters for any text-only LM trained on VEPRAD, but this run did not train such an LM.
- After fine-tuning, greedy decoding was slightly better than bundled-LM decoding on this clean split.

## Metrics

| Decoder | Subset | WER | CER spaces | CER no spaces |
|---|---:|---:|---:|---:|
| pre_finetune_bundled_lm | val | 0.283855 | 0.146039 | 0.147165 |
| pre_finetune_bundled_lm | val_seen_speakers | 0.275025 | 0.145283 | 0.145929 |
| pre_finetune_bundled_lm | val_unseen_speakers | 0.288475 | 0.146363 | 0.147773 |
| pre_finetune_bundled_lm | test | 0.294470 | 0.152677 | 0.155102 |
| pre_finetune_bundled_lm | test_seen_speakers | 0.269295 | 0.137527 | 0.139644 |
| pre_finetune_bundled_lm | test_unseen_speakers | 0.319935 | 0.166998 | 0.169625 |
| pre_finetune_greedy | val | 0.302893 | 0.143126 | 0.148892 |
| pre_finetune_greedy | val_seen_speakers | 0.295360 | 0.143473 | 0.147546 |
| pre_finetune_greedy | val_unseen_speakers | 0.307922 | 0.143134 | 0.149921 |
| pre_finetune_greedy | test | 0.309287 | 0.147885 | 0.154151 |
| pre_finetune_greedy | test_seen_speakers | 0.280622 | 0.132262 | 0.138595 |
| pre_finetune_greedy | test_unseen_speakers | 0.337140 | 0.162798 | 0.168950 |
| finetuned_greedy | val | 0.030788 | 0.008415 | 0.008153 |
| finetuned_greedy | val_seen_speakers | 0.024482 | 0.007512 | 0.007133 |
| finetuned_greedy | val_unseen_speakers | 0.034359 | 0.008898 | 0.008696 |
| finetuned_greedy | test | 0.038614 | 0.008061 | 0.007529 |
| finetuned_greedy | test_seen_speakers | 0.026884 | 0.005634 | 0.004923 |
| finetuned_greedy | test_unseen_speakers | 0.049689 | 0.010298 | 0.009888 |
| finetuned_bundled_lm | val | 0.037183 | 0.010147 | 0.009567 |
| finetuned_bundled_lm | val_seen_speakers | 0.033169 | 0.009623 | 0.009136 |
| finetuned_bundled_lm | val_unseen_speakers | 0.039489 | 0.010458 | 0.009822 |
| finetuned_bundled_lm | test | 0.042356 | 0.009730 | 0.008995 |
| finetuned_bundled_lm | test_seen_speakers | 0.032775 | 0.007527 | 0.006779 |
| finetuned_bundled_lm | test_unseen_speakers | 0.051468 | 0.011616 | 0.010874 |
