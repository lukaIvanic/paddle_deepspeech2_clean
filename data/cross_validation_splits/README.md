# Cross-Validation Splits

`raw_train_val/` contains the non-test VEPRAD source pool used to create
train/validation splits.

Future split folders should contain only JSONL manifests and metadata. They
should reference files under `raw_train_val/raw/veprad/` and must not reference
`data/test/`.

Current non-test pool:

- Utterances: 4262
- Speakers: 22
- Hours: 6.529102
- Excluded from this pool: frozen test utterances and the quarantined `sm04`
  subset

The JSONL manifests and raw audio/transcript files are intentionally ignored by
Git. Track only metadata and documentation.
