#!/bin/bash

if [ -z "${PROJECT_ROOT:-}" ] || [ -z "${MAIN_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "${SCRIPT_DIR}/path.sh"
fi
cd "${PROJECT_ROOT}"

if [ $# != 3 ];then
    echo "usage: ${0} config_path decode_config_path ckpt_path_prefix"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
decode_config_path=$2
ckpt_prefix=$3

# download language model

python3 -u "${BIN_DIR}/test.py" \
--ngpu ${ngpu} \
--config ${config_path} \
--decode_cfg ${decode_config_path} \
--result_file ${ckpt_prefix}.rsl \
--checkpoint_path ${ckpt_prefix}

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
