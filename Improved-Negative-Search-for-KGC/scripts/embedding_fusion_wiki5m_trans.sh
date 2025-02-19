#!/usr/bin/env bash

set -x
set -e

model_path="bert"
task="wiki5m_trans"
DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

test_path="$DATA_DIR/test.txt.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    test_path=$1
    shift
fi

neighbor_weight=0.05

python3 -u embedding_fusion_wiki5m_trans.py \
--task "${task}" \
--is-test \
--neighbor-weight "${neighbor_weight}" \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "${test_path}" "$@"
