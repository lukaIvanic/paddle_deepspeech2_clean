# Clean VEPRAD Reproduction Track

This track reruns the DeepSpeech2 VEPRAD experiments on a fresh clean
cross-validation split while keeping the frozen test set isolated.

## Goals

- Create or reuse one clean CV split from `data/cross_validation_splits/raw_train_val/source.jsonl`.
- Validate that train, validation, and frozen test utterance IDs do not overlap.
- Validate that quarantined `sm04` rows are not present in active splits.
- Train the selected DeepSpeech2 configurations from scratch on the clean split.
- Train KenLM only from train transcripts after filtering exact/fuzzy matches
  against validation and frozen test text.
- Report WER and CER for validation/test all, seen-speaker, and unseen-speaker
  subsets.

## Default Run

```bash
python3 experiment_tracks/clean_repro_20260614/run_clean_repro.py
```

Default outputs:

- split: `data/cross_validation_splits/cv_clean_repro_20260614_001/`
- results: `results/clean_repro_20260614/`
- track report: `results/clean_repro_20260614/track_report.json`

By default, the runner trains:

- `paper_small`: `conf/deepspeech2_paper_small.yaml`
- `paper_7gru`: `conf/deepspeech2_paper_7gru.yaml`
- `paper_7lstm`: `conf/deepspeech2_paper_7lstm.yaml`

Each model gets a no-LM beam1 run and a KenLM beam-search eval run using
`beam=40`, `alpha=2.5`, `beta=0.3`.

## Validation

Split validation can be run independently:

```bash
python3 experiment_tracks/clean_repro_20260614/validate_clean_split.py \
  --split-dir data/cross_validation_splits/cv_clean_repro_20260614_001 \
  --out results/clean_repro_20260614/split_validation.json
```

After KenLM training, the same validator can audit the LM corpus:

```bash
python3 experiment_tracks/clean_repro_20260614/validate_clean_split.py \
  --split-dir data/cross_validation_splits/cv_clean_repro_20260614_001 \
  --lm-dir results/clean_repro_20260614/paper_small_kenlm_b40_a2p5_b0p3/details/lm \
  --out results/clean_repro_20260614/paper_small_kenlm_validation.json
```

The LM audit checks the post-filter KenLM text against validation and frozen
test transcripts using the same exact/fuzzy matching policy used during LM
training.
