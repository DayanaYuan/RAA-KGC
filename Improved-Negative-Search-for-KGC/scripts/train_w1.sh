#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u /home/horanchen/ydy/study/code/5Improved-Negative-neibor/main.py \
--model-dir "/home/horanchen/ydy/study/code/5Improved-Negative-neibor/WN18RR_v1/checkpoint/" \
--pretrained-model /home/horanchen/ydy/study/code/5Improved-Negative-neibor/bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "/home/horanchen/ydy/study/code/5Improved-Negative-neibor/WN18RR_v1/train.txt.json" \
--valid-path "/home/horanchen/ydy/study/code/5Improved-Negative-neibor/WN18RR_v1/valid.txt.json" \
--task ${TASK} \
--batch-size 32 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 10 \
--workers 4 \
--max-to-keep 3 "$@"
