#!/usr/bin/env bash

set -x
set -e

model_path="bert"
task="WN18RR"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    task=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

test_path="${DATA_DIR}/test.txt.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    test_path=$1
    shift
fi

neighbor_weight=0.05
rerank_n_hop=2
if [ "${task}" = "WN18RR" ]; then
# WordNet is a sparse graph, use more neighbors for re-rank
  rerank_n_hop=5
fi
if [ "${task}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
  neighbor_weight=0.0
fi

python3 -u embedding_fusion.py \
--task "WN18RR" \
--is-test \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "/home/horanchen/ydy/study/code/5Improved-Negative-Search-for-KGC/data/WN18RR_v1/train.txt.json" \
--valid-path "/home/horanchen/ydy/study/code/5Improved-Negative-Search-for-KGC/data/WN18RR_v1/test.txt.json" "$@"
