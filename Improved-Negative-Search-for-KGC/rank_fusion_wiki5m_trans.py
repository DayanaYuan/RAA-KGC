import json
import os
from time import time
from typing import List, Tuple

import torch
import tqdm
from dataclasses import dataclass

from config import args
from dict_hub import get_entity_dict, get_all_triplet_dict
from doc import load_data, Example
from logger_config import logger
from rerank import rerank_by_graph
from triplet import EntityDict


def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()
SHARD_SIZE = 1000000


@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


def _get_shard_path(model_dir, shard_id=0):
    return '{}/shard_{}'.format(model_dir, shard_id)


def _load_entity_embeddings(model_dir):
    shard_tensors = []
    for start in range(0, len(entity_dict), SHARD_SIZE):
        shard_id = start // SHARD_SIZE
        shard_path = _get_shard_path(model_dir=model_dir, shard_id=shard_id)
        shard_entity_tensor = torch.load(shard_path, map_location=lambda storage, loc: storage)
        logger.info('Load {} entity embeddings from {}'.format(shard_entity_tensor.size(0), shard_path))
        shard_tensors.append(shard_entity_tensor)

    entity_tensor = torch.cat(shard_tensors, dim=0)
    logger.info('{} entity embeddings in total'.format(entity_tensor.size(0)))
    assert entity_tensor.size(0) == len(entity_dict.entity_exs)
    return entity_tensor


@torch.no_grad()
def get_entity_ranks(hr_tensor: torch.tensor,
                     entities_tensor: torch.tensor,
                     target: List[int],
                     examples: List[Example],
                     k=3, batch_size=256):
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    indices = []

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())

        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        if args.task.lower() != 'dbpedia500':
            rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)

        indices.extend(batch_sorted_indices.tolist())

    entity_ranks = [[0] * entity_cnt for _ in range(total)]
    for i, entity_indices in enumerate(indices):
        for j, idx in enumerate(entity_indices):
            entity_ranks[i][idx] = j + 1

    return entity_ranks


def get_rank_fusion_scores(hr_tensor: List[torch.tensor],
                           entities_tensor: List[torch.tensor],
                           target: List[int],
                           examples: List[Example],
                           weights,
                           k=3, batch_size=256,
                           rank_fusion_k=0):
    assert hr_tensor[0].size(1) == entities_tensor[0].size(1)
    total = hr_tensor[0].size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor[0].size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(entities_tensor[0].device)
    entity_range = torch.arange(1, entity_cnt + 1, dtype=hr_tensor[0].dtype, device=hr_tensor[0].device)
    mean_rank, mrr, hit1, hit3, hit10, hit50 = 0, 0, 0, 0, 0, 0
    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        batch_gold_neighbor_ids = [all_triplet_dict.get_neighbors(examples[idx].head_id, examples[idx].relation)
                                   for idx in range(start, min(end, total))]
        batch_rank_fusion_scores = []
        for i in range(len(hr_tensor)):
            # batch_size * entity_cnt
            batch_score = torch.mm(hr_tensor[i][start:end, :].to(entities_tensor[i].device), entities_tensor[i].t())

            assert entity_cnt == batch_score.size(1)

            # re-ranking based on topological structure
            if args.task.lower() != 'dbpedia500':
                rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

            # filter known triplets
            for idx in range(batch_score.size(0)):
                mask_indices = []
                cur_ex = examples[start + idx]
                # gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
                gold_neighbor_ids = batch_gold_neighbor_ids[idx]
                if len(gold_neighbor_ids) > 10000:
                    logger.debug(
                        '{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
                for e_id in gold_neighbor_ids:
                    if e_id == cur_ex.tail_id:
                        continue
                    mask_indices.append(entity_dict.entity_to_idx(e_id))
                mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
                batch_score[idx].index_fill_(0, mask_indices, -1)

            batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)

            batch_ranks = torch.zeros_like(batch_sorted_indices, dtype=hr_tensor[0].dtype)
            entity_range = entity_range.to(batch_sorted_indices.device)
            for j in range(batch_score.size(0)):
                batch_ranks[j][batch_sorted_indices[j]] = entity_range
            batch_rank_fusion_scores.append((1 / (batch_ranks + rank_fusion_k) * weights[i]).to(entities_tensor[0].device))

        batch_rank_fusion_scores = torch.stack(batch_rank_fusion_scores, dim=0)
        batch_rank_fusion_scores = torch.sum(batch_rank_fusion_scores, dim=0)

        batch_target = target[start:end]
        # filter known triplets
        for idx in range(batch_rank_fusion_scores.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            # gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            gold_neighbor_ids = batch_gold_neighbor_ids[idx]
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_rank_fusion_scores.device)
            batch_rank_fusion_scores[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_rank_fusion_scores, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_rank_fusion_scores.size(0)
        for idx in range(batch_rank_fusion_scores.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            hit50 += 1 if cur_rank <= 50 else 0

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10, 'hit@50': hit50}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}

    return metrics


def predict_by_split():
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)

    # wiki5m_trans
    entity_tensor_1 = _load_entity_embeddings('checkpoint/wiki5m_trans_ann_hard_negative_head_replace_1pos1neg')
    entity_tensor_2 = _load_entity_embeddings('checkpoint/wiki5m_trans_ann_hard_negative_1pos1neg_rerun')
    entity_tensor_3 = _load_entity_embeddings(
        'checkpoint/wiki5m_trans_ann_hard_negative_tail_entity_similarity_1pos1neg')
    entity_tensor_4 = _load_entity_embeddings(
        'checkpoint/wiki5m_trans_tail_entity_2hop_neighbours_hard_negative_1pos1neg')
    entity_tensor_5 = _load_entity_embeddings('checkpoint/wiki5m_trans_bm25_tail_query_hard_negative_1pos1neg')

    cuda0 = torch.device('cuda:0')
    cuda1 = torch.device('cuda:1')
    cuda2 = torch.device('cuda:2')

    entity_tensor = [entity_tensor_1.cuda(cuda0), entity_tensor_2.cuda(cuda1), entity_tensor_3.cuda(cuda1),
                     entity_tensor_4.cuda(cuda2), entity_tensor_5.cuda(cuda2)]

    forward_metrics = eval_single_direction(entity_tensor=entity_tensor,
                                            eval_forward=True,
                                            batch_size=4)
    backward_metrics = eval_single_direction(entity_tensor=entity_tensor,
                                             eval_forward=False,
                                             batch_size=4)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))


def eval_single_direction(entity_tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

    eval_dir = 'forward' if eval_forward else 'backward'

    # wiki5m_trans
    # average metrics: {"mean_rank": 8162.5211, "mrr": 0.4096, "hit@1": 0.3702, "hit@3": 0.4267, "hit@10": 0.4801, "hit@50": 0.5662}
    hr_tensor_1 = torch.load(
        'checkpoint/wiki5m_trans_ann_hard_negative_head_replace_1pos1neg/{}_hr_tensor'.format(eval_dir),
        map_location=lambda storage, loc: storage)
    # average metrics: {"mean_rank": 11249.3549, "mrr": 0.4022, "hit@1": 0.3642, "hit@3": 0.4177, "hit@10": 0.4677, "hit@50": 0.552}
    hr_tensor_2 = torch.load(
        'checkpoint/wiki5m_trans_ann_hard_negative_1pos1neg_rerun/{}_hr_tensor'.format(eval_dir),
        map_location=lambda storage, loc: storage)
    # average metrics: {"mean_rank": 46236.0905, "mrr": 0.3925, "hit@1": 0.3545, "hit@3": 0.4087, "hit@10": 0.4597, "hit@50": 0.5391}
    hr_tensor_3 = torch.load(
        'checkpoint/wiki5m_trans_ann_hard_negative_tail_entity_similarity_1pos1neg/{}_hr_tensor'.format(eval_dir),
        map_location=lambda storage, loc: storage)
    # average metrics: {"mean_rank": 5106.9625, "mrr": 0.3795, "hit@1": 0.3338, "hit@3": 0.3968, "hit@10": 0.4618, "hit@50": 0.5722}
    hr_tensor_4 = torch.load(
        'checkpoint/wiki5m_trans_tail_entity_2hop_neighbours_hard_negative_1pos1neg/{}_hr_tensor'.format(eval_dir),
        map_location=lambda storage, loc: storage)
    # average metrics: {"mean_rank": 5355.7488, "mrr": 0.3669, "hit@1": 0.3231, "hit@3": 0.3835, "hit@10": 0.4453, "hit@50": 0.5509}
    hr_tensor_5 = torch.load(
        'checkpoint/wiki5m_trans_bm25_tail_query_hard_negative_1pos1neg/{}_hr_tensor'.format(eval_dir),
        map_location=lambda storage, loc: storage)
    hr_tensor = [hr_tensor_1, hr_tensor_2, hr_tensor_3, hr_tensor_4, hr_tensor_5]

    weights = [1.2, 0.9, 0.6, 0.45, 0.3]
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    # for large datasets: dbpedia500 and wiki5m_trans
    metrics = get_rank_fusion_scores(hr_tensor=hr_tensor, entities_tensor=entity_tensor, target=target,
                                     examples=examples, weights=weights, batch_size=batch_size,
                                     rank_fusion_k=0)

    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))

    return metrics


if __name__ == '__main__':
    predict_by_split()
