# Cross-Validation Splits

`raw_train_val/` contains the non-test VEPRAD source pool used to create
train/validation splits.

Future split folders should contain only JSONL manifests and metadata. They
should reference files under `raw_train_val/raw/veprad/` and must not reference
`data/test/`.

Generated split folders use this shape:

```text
<split_id>/
  train.jsonl
  train_seen_speakers.jsonl
  train_unseen_speakers.jsonl
  val.jsonl
  val_seen_speakers.jsonl
  val_unseen_speakers.jsonl
  split.meta.json
```

`val_unseen_speakers.jsonl` contains all utterances from the fully held-out
validation speakers. `val_seen_speakers.jsonl` contains the 10% same-speaker
utterance holdout. `train_seen_speakers.jsonl` contains the training rows from
speakers that are also represented in same-speaker validation.
`train_unseen_speakers.jsonl` is intentionally empty because fully held-out
validation speakers must not appear in training.

Current non-test pool:

- Utterances: 4262
- Speakers: 22
- Hours: 6.529102
- Excluded from this pool: frozen test utterances and the quarantined `sm04`
  subset

The JSONL manifests and raw audio/transcript files are intentionally ignored by
Git. Track only metadata and documentation.
