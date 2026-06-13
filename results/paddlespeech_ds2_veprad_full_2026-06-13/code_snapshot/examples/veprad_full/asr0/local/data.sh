#!/bin/bash
set -e

stage=-1
stop_stage=100
unit_type=char
dict_dir=data/lang_char
veprad_root=/workspace/rogj_paddlespeech_ds2/veprad_full

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;

mkdir -p data ${dict_dir}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    cp ${veprad_root}/manifests/train.raw.jsonl data/manifest.train.raw
    cp ${veprad_root}/manifests/dev.raw.jsonl data/manifest.dev.raw
    cp ${veprad_root}/manifests/test.raw.jsonl data/manifest.test.raw
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
      --manifest_path=data/manifest.train.raw \
      --num_samples=2000 \
      --spectrum_type=fbank \
      --feat_dim=161 \
      --delta_delta=false \
      --sample_rate=16000 \
      --stride_ms=10 \
      --window_ms=25 \
      --use_dB_normalization=False \
      --num_workers=8 \
      --output_path=data/mean_std.json
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${MAIN_ROOT}/utils/build_vocab.py \
      --unit_type ${unit_type} \
      --count_threshold=0 \
      --vocab_path=${dict_dir}/vocab.txt \
      --manifest_paths data/manifest.train.raw
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for split in train dev test; do
      python3 ${MAIN_ROOT}/utils/format_data.py \
        --cmvn_path data/mean_std.json \
        --unit_type ${unit_type} \
        --vocab_path=${dict_dir}/vocab.txt \
        --manifest_paths data/manifest.${split}.raw \
        --output_path data/manifest.${split}
    done
fi

echo "VEPRAD full data preparation done."
