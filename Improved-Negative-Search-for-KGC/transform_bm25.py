import json
import csv
import os

from triplet import reverse_triplet, EntityExample, TripletDict, EntityDict
from tqdm import tqdm
import numpy as np


# todo: step1 -> transform training data into formats required by pyserini
path = 'data/FB15k237/entities.json'
with open(path, 'r', encoding='utf-8') as f:
    entity_exs = [EntityExample(**obj) for obj in json.load(f)]

index_collections = []
for entity in entity_exs:
    entity_name = entity.entity
    entity_name = '' if entity_name is None else ' '.join(entity_name.split('_')[:-2])
    entity_desc = entity.entity_desc
    if entity_desc.startswith(entity_name):
        entity_desc = entity_desc[len(entity_name):].strip()
    entity_content = '{}: {}'.format(entity_name.strip(), entity_desc.strip())

    index_collections.append({
        'id': entity.entity_id,
        'contents': entity_content
    })

id2entity = {ex.entity_id: ex for ex in entity_exs}
entity2idx = {ex.entity_id: i for i, ex in enumerate(entity_exs)}

with open('data/FB15k237/bm25_collections.jsonl', 'w') as f:
    for doc in index_collections:
        f.write(json.dumps(doc))
        f.write('\n')

with open('data/FB15k237/train.txt.json') as f:
    examples = json.load(f)
examples += [reverse_triplet(obj) for obj in examples]

query_collections = []
for example in examples:
    # use head as query to retrieve hard negatives using BM25
    head_entity = id2entity[example['head_id']]
    rel = example['relation']
    head_entity_name = ' '.join(head_entity.entity.split('_')[:-2])
    head_desc = head_entity.entity_desc
    if head_desc.startswith(head_entity_name):
        head_desc = head_desc[len(head_entity_name):].strip()
    query = '{}: {}'.format(head_entity_name.strip(), head_desc.strip())
    query_collections.append(query)

    # # use the correct tail as query to retrieve similar entities in terms of lexicons as hard negatives using BM25
    # tail_entity = id2entity[example['tail_id']]
    # tail_entity_name = ' '.join(tail_entity.entity.split('_')[:-2])
    # tail_desc = tail_entity.entity_desc
    # if tail_desc.startswith(tail_entity_name):
    #     tail_desc = tail_desc[len(tail_entity_name):].strip()
    # query = '{}: {}'.format(tail_entity_name.strip(), tail_desc.strip())
    # query_collections.append(query)

with open('data/FB15k237/train.query.tsv', 'w') as f:
    w = csv.writer(f, delimiter='\t')
    for i, row in enumerate(query_collections):
        data = [[i + 1, row]]
        w.writerows(data)

# for wikidata, split training data into multiple shards
# shard = 0
# SHARD_SIZE = 6000000
# for start in range(0, len(query_collections), SHARD_SIZE):
#     end = start + SHARD_SIZE
#     shard_id = start // SHARD_SIZE
#     with open('data/wiki5m_trans/train.query_shard{}.tsv'.format(shard_id), 'w') as f:
#         w = csv.writer(f, delimiter='\t')
#         for i, row in enumerate(query_collections[start:end]):
#             data = [[i + 1, row]]
#             w.writerows(data)


# todo: step 2 -> use pyserini to retrieve bm25 negatives, then use following code to transform negatives into
#  required formats
# with open('data/FB15k237/train.txt.json') as f:
#     examples = json.load(f)
# examples += [reverse_triplet(obj) for obj in examples]
#
# negative_collections = [[] for _ in range(len(examples))]
# negatives = []
# with open('data/FB15k237/train.bm25.hard.negatives.txt', 'r') as f:
#     for line in tqdm(f.readlines()):
#         line = line.strip().split(' ')
#         assert len(line) == 6, line
#         query_id, _, negative_id, _, _, _ = line
#         query_id = int(query_id) - 1
#         negative_collections[query_id].append(negative_id)
#
# print(len(negative_collections))
#
# train_triplet_dict = TripletDict(path_list=['data/FB15k237/train.txt.json'])
#
# topk = 30
# entity_dict = EntityDict(entity_dict_dir='data/FB15k237/')
# cnt = 0
# with open('data/FB15k237/train.bm25.head.query.top30.hard.negative.json', 'w') as f:
#     for i, example in enumerate(tqdm(examples)):
#         gold_neighbor_ids = train_triplet_dict.get_neighbors(example['head_id'], example['relation'])
#         negatives = negative_collections[i]
#         negatives = [negative_id for negative_id in negatives if negative_id != example['head_id'] and
#                      negative_id not in gold_neighbor_ids]
#         if len(negatives) == 0:
#             cnt += 1
#             print('no bm25 hard negatives, use random negatives instead')
#             gold_neighbour_ids = set(train_triplet_dict.get_neighbors(example['head_id'], example['relation']))
#             while len(negatives) < topk:
#                 entity_idx = np.random.choice(len(entity_dict))
#                 entity_id = entity_dict.get_entity_by_idx(entity_idx).entity_id
#                 if entity_id not in gold_neighbour_ids:
#                     negatives.append(entity_id)
#         example['negatives'] = negatives[:topk]
#         negative_collections[i] = None
#
#     f.write(json.dumps(examples, ensure_ascii=False, indent=4))
#     print(cnt)


# for wikidata
# with open('data/wiki5m_trans/train.txt.json') as f:
#     examples = json.load(f)
# examples += [reverse_triplet(obj) for obj in examples]
#
# train_triplet_dict = TripletDict(path_list=['data/wiki5m_trans/train.txt.json'])
# topk = 5
# entity_dict = EntityDict(entity_dict_dir='data/wiki5m_trans/')
# cnt = 0
#
# shard = 0
# SHARD_SIZE = 6000000
# for start in range(0, len(examples), SHARD_SIZE):
#     end = start + SHARD_SIZE
#     shard_id = start // SHARD_SIZE
#     negative_collections = [[] for _ in range(len(examples[start:end]))]
#     negatives = []
#     with open('data/wiki5m_trans/train.bm25.hard.negatives_shard{}.txt'.format(shard_id), 'r') as f:
#         for line in tqdm(f.readlines()):
#             line = line.strip().split(' ')
#             assert len(line) == 6, line
#             query_id, _, negative_id, _, _, _ = line
#             query_id = int(query_id) - 1
#             negative_collections[query_id].append(negative_id)
#     print(len(negative_collections))
#
#     output_path = 'data/wiki5m_trans/train.bm25.tail.query.top30.hard.negative_shard{}.json'.format(shard_id)
#     with open(output_path, 'w') as f:
#         for i, example in enumerate(tqdm(examples[start:end])):
#             gold_neighbor_ids = train_triplet_dict.get_neighbors(example['head_id'], example['relation'])
#             negatives = negative_collections[i]
#             negatives = [negative_id for negative_id in negatives if negative_id != example['head_id'] and
#                          negative_id not in gold_neighbor_ids]
#             if len(negatives) == 0:
#                 cnt += 1
#                 print('no bm25 hard negatives, use random negatives instead')
#                 gold_neighbour_ids = set(train_triplet_dict.get_neighbors(example['head_id'], example['relation']))
#                 while len(negatives) < topk:
#                     entity_idx = np.random.choice(len(entity_dict))
#                     entity_id = entity_dict.get_entity_by_idx(entity_idx).entity_id
#                     if entity_id not in gold_neighbour_ids:
#                         negatives.append(entity_id)
#             example['negatives'] = negatives[:topk]
#             negative_collections[i] = None
#
#         print('write {} examples to {}'.format(len(examples[start:end]), output_path))
#         f.write(json.dumps(examples[start:end], ensure_ascii=False, indent=4))
#         print(cnt)
