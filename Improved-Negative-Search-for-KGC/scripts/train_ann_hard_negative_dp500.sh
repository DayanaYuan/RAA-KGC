#!/usr/bin/env bash

set -x
set -e

TASK="dbpedia500"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 1e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.forward.ann.hard.negative.tail.entity.2hop.neighbours.json,${DATA_DIR}/train.backward.ann.hard.negative.tail.entity.2hop.neighbours.json" \
--valid-path "${DATA_DIR}/valid.forward.ann.hard.negative.top30.json,${DATA_DIR}/valid.backward.ann.hard.negative.top30.json" \
--task ${TASK} \
--batch-size 768 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--use-hard-negative \
--num-negatives 1 \
--pre-batch 0 \
--finetune-t \
--epochs 5 \
--workers 4 \
--max-to-keep 3 "$@"
