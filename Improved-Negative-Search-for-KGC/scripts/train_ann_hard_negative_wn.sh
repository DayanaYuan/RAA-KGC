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

python3 -u main.py \
--model-dir "WN18RR_v1/checkpoint/wn18rr_tail_entity_2hop_neighbours_hard_negative_1pos1neg/" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "WN18RR_v1/train.forward.ann.hard.negative.tail.entity.2hop.neighbours.json,WN18RR_v1/train.backward.ann.hard.negative.tail.entity.2hop.neighbours.json" \
--valid-path "/homeWN18RR_v1/valid.forward.ann.hard.negative.top30.json,WN18RR_v1/valid.backward.ann.hard.negative.top30.json" \
--task WN18RR \
--batch-size 32 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--use-hard-negative \
--num-negatives 1 \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 1 \
--max-to-keep 3 "$@"
