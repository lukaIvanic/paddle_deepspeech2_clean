# Frozen Test Split

This folder contains the one-time held-out VEPRAD test split.

The JSONL manifest contains transcript text and is intentionally ignored by Git.
The tracked `test.meta.json` file is the public audit record for the split.

Do not use this folder for model selection or cross-validation.

Current frozen split:

- Full held-out speakers: `m04`, `z06`, `z14`
- Same-speaker component: 10% of every other speaker, rounded up
- Utterances: 993
- Hours: 1.50394
- Random seed used for the one-time test draw: `20260613`

The local raw audio/transcript files live under `raw/veprad/` and are ignored by
Git.
