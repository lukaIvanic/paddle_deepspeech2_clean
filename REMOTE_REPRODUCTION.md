# Remote Reproduction

These commands reproduce the PaddleSpeech DeepSpeech2 VEPRAD run on a Linux GPU
machine from this self-contained folder. The previous successful run used a
Vast.ai RTX 3090 instance, but the steps are written for any compatible CUDA GPU
box.

## 1. Set Variables

The current PaddleSpeech/DeepSpeech2 Vast.ai box is instance `40802979`, labeled
`rogj-paddlespeech-ds2`. At last check it was running as:

- SSH host: `ssh1.vast.ai`
- SSH port: `12978`
- SSH key: `~/.ssh/vast_sentiment_ed25519`

Direct SSH command:

```bash
ssh -p 12978 -i ~/.ssh/vast_sentiment_ed25519 root@ssh1.vast.ai
```

There is also a local SSH alias:

```bash
ssh vast_rogj_asr
```

If the box is destroyed/recreated, refresh these values with:

```bash
~/.local/share/codex-vastai/bin/vastai show instances --raw
```

Look for the instance labeled `rogj-paddlespeech-ds2`.

For the commands below, set:

```bash
REMOTE=root@ssh1.vast.ai
PORT=12978
KEY=~/.ssh/vast_sentiment_ed25519
BASE=/workspace/rogj_paddlespeech_ds2
```

Adjust `REMOTE`, `PORT`, and `KEY` for the current Vast.ai instance.

## 2. Sync This Folder

From `/Users/lukaivanic/projects/faks/Računalna obrada govora i jezika`:

```bash
ssh -p "$PORT" -i "$KEY" "$REMOTE" "mkdir -p '$BASE'"

rsync -a --delete --info=progress2 -e "ssh -p $PORT -i $KEY" \
  "paddle_deepspeech2_clean/" \
  "$REMOTE:$BASE/paddle_deepspeech2_clean/"
```

No separate VEPRAD data sync is needed. The audio, manifests, PaddleSpeech
checkout, recipe files, and archived results are inside `paddle_deepspeech2_clean/`.

## 3. Install Python Dependencies

On the remote:

```bash
cd "$BASE/paddle_deepspeech2_clean"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install paddlepaddle-gpu==2.6.2 paddlespeech==1.5.0 numpy==1.26.4
```

NumPy `1.26.4` is intentional. PaddleSpeech `1.5.0` imports APIs that are
removed in NumPy 2.x.

The required remote Python packages for the completed run were:

```text
paddlepaddle-gpu==2.6.2
paddlespeech==1.5.0
numpy==1.26.4
```

## 4. Verify The Bundled PaddleSpeech Checkout

On the remote:

```bash
cd "$BASE/paddle_deepspeech2_clean"
git -C PaddleSpeech rev-parse HEAD
```

Expected commit:

```text
6b25a400008d393f9c3af837b3c692b17f29ee1a
```

## 5. Train And Evaluate

On the remote:

```bash
cd "$BASE/paddle_deepspeech2_clean"
. .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"

bash scripts/run.sh --stage 0 --stop_stage 0 --gpus 0
bash scripts/run.sh --stage 1 --stop_stage 1 --gpus 0
bash scripts/run.sh --stage 2 --stop_stage 3 --gpus 0 --avg_num 1
```

The completed run took about 15 minutes for training on an RTX 3090. The best
dev checkpoint was epoch 10.

Expected final PaddleSpeech line:

```text
Final error rate [wer] (580/580) = 0.362968
```

## 6. Recompute Metrics Independently

On the remote, after evaluation:

```bash
python scripts/recompute_asr_metrics.py \
  --result-file exp/deepspeech2/checkpoints/avg_1.rsl \
  --metrics-out exp/deepspeech2/checkpoints/avg_1.metrics.json \
  --utterance-out exp/deepspeech2/checkpoints/avg_1.utterance_metrics.tsv
```

Expected independent metrics:

```text
WER = 0.362968231389284
CER with spaces = 0.07469918006602065
CER without spaces = 0.07085997112806765
```

## 7. Copy New Results Back

From the local Mac:

```bash
RESULT_DIR="paddle_deepspeech2_clean/results/paddlespeech_ds2_veprad_full_$(date +%F)"
mkdir -p "$RESULT_DIR/data/lang_char"

rsync -a -e "ssh -p $PORT -i $KEY" \
  "$REMOTE:$BASE/paddle_deepspeech2_clean/exp/" \
  "$RESULT_DIR/exp/"

rsync -a -e "ssh -p $PORT -i $KEY" \
  "$REMOTE:$BASE/paddle_deepspeech2_clean/data/mean_std.json" \
  "$RESULT_DIR/data/mean_std.json"

rsync -a -e "ssh -p $PORT -i $KEY" \
  "$REMOTE:$BASE/paddle_deepspeech2_clean/data/lang_char/vocab.txt" \
  "$RESULT_DIR/data/lang_char/vocab.txt"
```

Copy additional logs/configs as needed. Do not copy checkpoint weights unless
they are specifically needed.
