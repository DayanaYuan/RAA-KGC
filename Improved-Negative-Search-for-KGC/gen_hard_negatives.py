import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from typing import List

import torch
import tqdm

from config import args
from dict_hub import get_entity_dict, get_train_triplet_dict, get_link_graph, get_all_triplet_dict
from doc import load_data, Example
from logger_config import logger
from predict import BertPredictor
from triplet import EntityDict


def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


entity_dict = _setup_entity_dict()
train_triplet_dict = get_train_triplet_dict()
all_triplet_dict = get_all_triplet_dict()
get_link_graph()


@dataclass
class PredInfo:
    head_id: str
    relation: str
    tail_id: str
    negatives: List


# use head or tail entity as query to find top-k nearest neighbours in the embedding space as hard negatives
def retrieve_entity_similar_negatives(path: str, examples, entity_tensor: torch.tensor, eval_forward=True, topk=30,
                                      batch_size=128, use_head=False):
    start_time = time.time()
    all_topk_scores, all_topk_indices = [], []

    total = len(examples)
    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        query_ids = []
        for idx in range(start, end):
            cur_ex = examples[idx]
            if use_head:
                query_ids.append(cur_ex.head_id)
            else:
                query_ids.append(cur_ex.tail_id)
        query_idx = [entity_dict.entity_to_idx(e_id) for e_id in query_ids]
        query_idx = torch.LongTensor(query_idx).to(entity_tensor.device)
        query_entity_tensor = entity_tensor[query_idx]
        batch_score = torch.mm(query_entity_tensor, entity_tensor.t())

        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = train_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        all_topk_scores.extend(batch_sorted_score[:, :topk].tolist())
        all_topk_indices.extend(batch_sorted_indices[:, :topk].tolist())

    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    pred_infos = []
    for idx, ex in enumerate(examples):
        cur_topk_scores = all_topk_scores[idx]
        cur_topk_indices = all_topk_indices[idx]
        negatives = [entity_dict.get_entity_by_idx(topk_idx).entity_id
                     for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices) if topk_idx != target[idx]
                     and topk_score != -1]
        if len(negatives) == 0:
            continue

        pred_info = PredInfo(head_id=ex.head_id, relation=ex.relation,
                             tail_id=ex.tail_id, negatives=negatives)
        pred_infos.append(pred_info)

    eval_dir = 'forward' if eval_forward else 'backward'
    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(path)
    output_path = '{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename)
    logger.info('writing {} samples to {}'.format(len(pred_infos), output_path))
    with open(output_path, 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))
    dist_path = os.path.dirname(path) + '/{}.{}.ann.hard.negative.{}.entity.similarity.top{}.json'.format(
        split.split('.')[0], eval_dir, 'head' if use_head else 'tail', topk)
    shutil.move(output_path, dist_path)

    logger.info('Evaluation takes {} seconds'.format(round(time.time() - start_time, 3)))


def retriever_replaced_head_relation_negatives(path: str, examples, entity_tensor: torch.tensor, eval_forward=True,
                                               topk=30, predictor=None, batch_size=128, replaced_head=False):
    start_time = time.time()
    query_examples = []
    if replaced_head:
        for idx, cur_ex in enumerate(tqdm.tqdm(examples)):
            cur_head_id = cur_ex.head_id
            cur_head_idx = entity_dict.entity_to_idx(cur_head_id)
            cur_head_tensor = entity_tensor[cur_head_idx]
            # [num_entities]
            entity_similarity_scores = torch.mm(cur_head_tensor.unsqueeze(0), entity_tensor.t()).squeeze(0)
            entity_similarity_scores[cur_head_idx] = -1
            entity_similarity_scores, entity_sorted_indices = torch.sort(entity_similarity_scores, dim=-1,
                                                                         descending=True)
            closest_entity_idx = entity_sorted_indices[0]
            closest_entity_id = entity_dict.get_entity_by_idx(closest_entity_idx.item()).entity_id
            new_example = Example(head_id=closest_entity_id, relation=cur_ex.relation, tail_id=cur_ex.tail_id)
            query_examples.append(new_example)

        hr_tensor = predictor.predict_by_examples(query_examples, only_head_embedding=True)
    else:
        hr_tensor = predictor.predict_by_examples(examples, only_head_embedding=True)

    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]

    assert hr_tensor.size(1) == entity_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entity_tensor.size(0)
    topk_scores, topk_indices = [], []

    for start in range(0, total, batch_size):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entity_tensor.t())
        assert entity_cnt == batch_score.size(1)

        # filter known triplets
        for idx in range(batch_score.size(0)):
            cur_ex = examples[start + idx]
            gold_neighbor_ids = train_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation) \
                if 'valid' not in path else all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if replaced_head:
                cur_query_ex = query_examples[start + idx]
                gold_query_neighbor_ids = train_triplet_dict.get_neighbors(cur_query_ex.head_id, cur_query_ex.relation) \
                    if 'valid' not in path else all_triplet_dict.get_neighbors(cur_query_ex.head_id, cur_query_ex.relation)
                gold_neighbor_ids.update(gold_query_neighbor_ids)

            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            mask_indices = [entity_dict.entity_to_idx(e_id) for e_id in gold_neighbor_ids]
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        topk_scores.extend(batch_sorted_score[:, :topk].tolist())
        topk_indices.extend(batch_sorted_indices[:, :topk].tolist())

    pred_infos = []
    for idx, ex in enumerate(examples):
        cur_topk_scores = topk_scores[idx]
        cur_topk_indices = topk_indices[idx]
        negatives = [entity_dict.get_entity_by_idx(topk_idx).entity_id
                     for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices) if topk_idx != target[idx]
                     and topk_score != -1]
        if len(negatives) == 0:
            continue

        pred_info = PredInfo(head_id=ex.head_id, relation=ex.relation,
                             tail_id=ex.tail_id, negatives=negatives)
        pred_infos.append(pred_info)

    eval_dir = 'forward' if eval_forward else 'backward'
    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(path)
    output_path = '{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename)
    logger.info('writing {} samples to {}'.format(len(pred_infos), output_path))
    with open(output_path, 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))
    if replaced_head:
        dist_path = os.path.dirname(path) + '/{}.{}.ann.hard.negative.top{}.head.replace.json'.format(
            split.split('.')[0], eval_dir, topk)
    else:
        dist_path = os.path.dirname(path) + '/{}.{}.ann.hard.negative.top{}.json'.format(split.split('.')[0], eval_dir,
                                                                                         topk)
    shutil.move(output_path, dist_path)

    logger.info('Evaluation takes {} seconds'.format(round(time.time() - start_time, 3)))


def retrieve_structure_aware_negatives(path: str, examples, eval_forward=True, topk=30, n_hop=2):
    # use the correct tail entity as centroid node to find its n-hop neighbours as hard negatives
    import numpy as np
    start_time = time.time()
    pred_infos = []
    for idx, cur_ex in enumerate(tqdm.tqdm(examples)):
        gold_neighbor_ids = train_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
        query_ids = [cur_ex.tail_id]
        negatives = set()

        for e_id in query_ids:
            n_hop_neighbours = get_link_graph().get_n_hop_entity_indices(e_id, entity_dict, n_hop=n_hop,
                                                                         return_eids=True)
            for n_hop_e_id in n_hop_neighbours:
                if n_hop_e_id not in gold_neighbor_ids:
                    negatives.add(n_hop_e_id)
        negatives = [negative for negative in negatives if negative != cur_ex.tail_id and negative != cur_ex.head_id]

        if len(negatives) == 0:
            print('no n-hop hard negatives, use random negatives instead')
            while len(negatives) < topk:
                entity_idx = np.random.choice(len(entity_dict))
                entity_id = entity_dict.get_entity_by_idx(entity_idx).entity_id
                if entity_id not in gold_neighbor_ids:
                    negatives.append(entity_id)

        pred_info = PredInfo(head_id=cur_ex.head_id, relation=cur_ex.relation,
                             tail_id=cur_ex.tail_id, negatives=negatives[:topk])
        pred_infos.append(pred_info)

    eval_dir = 'forward' if eval_forward else 'backward'
    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(path)
    output_path = '{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename)
    logger.info('writing {} samples to {}'.format(len(pred_infos), output_path))
    with open(output_path, 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))
    dist_path = os.path.dirname(path) + '/{}.{}.ann.hard.negative.tail.entity.{}hop.neighbours.json'.format(
        split.split('.')[0], eval_dir, n_hop)
    shutil.move(output_path, dist_path)

    logger.info('Evaluation takes {} seconds'.format(round(time.time() - start_time, 3)))

    # use the head entity as centroid node to find its n-hop neighbours as hard negatives
    start_time = time.time()
    pred_infos = []
    for idx, cur_ex in enumerate(tqdm.tqdm(examples)):
        gold_neighbor_ids = train_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
        negatives = set()
        n_hop_neighbours = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id, entity_dict, n_hop=n_hop,
                                                                     return_eids=True)
        for n_hop_e_id in n_hop_neighbours:
            if n_hop_e_id not in gold_neighbor_ids:
                negatives.add(n_hop_e_id)

        negatives = [negative for negative in negatives if negative != cur_ex.tail_id and negative != cur_ex.head_id]
        if len(negatives) == 0:
            negatives = set()
            idx = 1
            while len(negatives) == 0 and idx <= 3:
                n_hop_neighbours = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id, entity_dict,
                                                                             n_hop=n_hop + idx, return_eids=True)
                for n_hop_e_id in n_hop_neighbours:
                    if n_hop_e_id not in gold_neighbor_ids:
                        negatives.add(n_hop_e_id)

                idx += 1
            negatives = [negative for negative in negatives if
                         negative != cur_ex.tail_id and negative != cur_ex.head_id]

        if len(negatives) == 0:
            print('no n-hop hard negatives, use random negatives instead')
            while len(negatives) < topk:
                entity_idx = np.random.choice(len(entity_dict))
                entity_id = entity_dict.get_entity_by_idx(entity_idx).entity_id
                if entity_id not in gold_neighbor_ids and entity_id != cur_ex.head_id:
                    negatives.append(entity_id)

        pred_info = PredInfo(head_id=cur_ex.head_id, relation=cur_ex.relation,
                             tail_id=cur_ex.tail_id, negatives=negatives[:topk])
        pred_infos.append(pred_info)

    eval_dir = 'forward' if eval_forward else 'backward'
    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(path)
    output_path = '{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename)
    logger.info('writing {} samples to {}'.format(len(pred_infos), output_path))
    with open(output_path, 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))
    dist_path = os.path.dirname(path) + '/{}.{}.ann.hard.negative.head.entity.{}hop.neighbours.json'.format(
        split.split('.')[0], eval_dir, n_hop)
    shutil.move(output_path, dist_path)

    logger.info('Evaluation takes {} seconds'.format(round(time.time() - start_time, 3)))


if __name__ == '__main__':
    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path, use_data_parallel=False)

    train_forward_examples = load_data(args.train_path, add_forward_triplet=True, add_backward_triplet=False)
    train_backward_examples = load_data(args.train_path, add_forward_triplet=False, add_backward_triplet=True)

    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    # structure-aware
    logger.info("generating structure-aware negatives......")
    retrieve_structure_aware_negatives(args.train_path, train_forward_examples, eval_forward=True, topk=30, n_hop=2)
    retrieve_structure_aware_negatives(args.train_path, train_backward_examples, eval_forward=False, topk=30, n_hop=2)

    # entity similar
    # use head as query
    logger.info("generating head entity-similar negatives......")
    retrieve_entity_similar_negatives(args.train_path, train_forward_examples, entity_tensor, eval_forward=True,
                                      topk=30, use_head=True)
    retrieve_entity_similar_negatives(args.train_path, train_backward_examples, entity_tensor, eval_forward=False,
                                      topk=30, use_head=True)
    # use tail as query
    logger.info("generating tail entity-similar negatives......")
    retrieve_entity_similar_negatives(args.train_path, train_forward_examples, entity_tensor, eval_forward=True,
                                      topk=30, use_head=False)
    retrieve_entity_similar_negatives(args.train_path, train_backward_examples, entity_tensor, eval_forward=False,
                                      topk=30, use_head=False)

    # head-relation neural negatives
    logger.info("generating head-relation negatives......")
    retriever_replaced_head_relation_negatives(args.train_path, train_forward_examples, entity_tensor,
                                               eval_forward=True, topk=30, predictor=predictor, replaced_head=False)
    retriever_replaced_head_relation_negatives(args.train_path, train_backward_examples, entity_tensor,
                                               eval_forward=False, topk=30, predictor=predictor, replaced_head=False)
    # replaced head-relation neural negatives
    logger.info("generating replaced head-relation negatives......")
    retriever_replaced_head_relation_negatives(args.train_path, train_forward_examples, entity_tensor,
                                               eval_forward=True, topk=30, predictor=predictor, replaced_head=True)
    retriever_replaced_head_relation_negatives(args.train_path, train_backward_examples, entity_tensor,
                                               eval_forward=False, topk=30, predictor=predictor, replaced_head=True)

    # generate hard negatives for development sets
    logger.info("generating negatives for development sets......")
    valid_forward_examples = load_data(args.valid_path, add_forward_triplet=True, add_backward_triplet=False)
    valid_backward_examples = load_data(args.valid_path, add_forward_triplet=False, add_backward_triplet=True)
    retriever_replaced_head_relation_negatives(args.valid_path, valid_forward_examples, entity_tensor,
                                               eval_forward=True, topk=30, predictor=predictor, replaced_head=False)
    retriever_replaced_head_relation_negatives(args.valid_path, valid_backward_examples, entity_tensor,
                                               eval_forward=False, topk=30, predictor=predictor, replaced_head=False)
