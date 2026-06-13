#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path.sh"
cd "${PROJECT_ROOT}"

gpus=0
stage=0
stop_stage=3
conf_path=conf/deepspeech2.yaml
ips=            #xx.xx.xx.xx,xx.xx.xx.xx
decode_conf_path=conf/tuning/decode.yaml
avg_num=1
source "${MAIN_ROOT}/utils/parse_options.sh" || exit 1;

avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}') ###ckpt = deepspeech2
echo "checkpoint name ${ckpt}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash "${SCRIPT_DIR}/data.sh" || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} "${SCRIPT_DIR}/train.sh" "${conf_path}" "${ckpt}" ${ips}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # test ckpt avg_n
    CUDA_VISIBLE_DEVICES=${gpus} "${SCRIPT_DIR}/test.sh" "${conf_path}" "${decode_conf_path}" "exp/${ckpt}/checkpoints/${avg_ckpt}" || exit -1
fi
