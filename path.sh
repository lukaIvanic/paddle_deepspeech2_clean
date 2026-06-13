PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

export MAIN_ROOT="${PROJECT_ROOT}/PaddleSpeech"

export PATH=${MAIN_ROOT}:${MAIN_ROOT}/utils:${PATH}
export LC_ALL=C

# Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8 
export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/


MODEL=deepspeech2
export BIN_DIR=${MAIN_ROOT}/paddlespeech/s2t/exps/${MODEL}/bin
