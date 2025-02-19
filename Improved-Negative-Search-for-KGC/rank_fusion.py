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
            batch_score = torch.mm(hr_tensor[i][start:end, :], entities_tensor[i].t())

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
            for j in range(batch_score.size(0)):
                batch_ranks[j][batch_sorted_indices[j]] = entity_range
            batch_rank_fusion_scores.append(1 / (batch_ranks + rank_fusion_k) * weights[i])

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

    # WN18RR
    entity_tensor_1 = torch.load('checkpoint/wn18rr_tail_entity_2hop_neighbours_hard_negative_1pos1neg/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor_2 = torch.load('checkpoint/wn18rr_bm25_head_query_hard_negative_1pos1neg/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor_3 = torch.load('checkpoint/wn18rr_ann_hard_negative_head_entity_similarity_1pos1neg/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor_4 = torch.load('checkpoint/wn18rr_ann_hard_negative_head_replace_1pos1neg/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor_5 = torch.load('checkpoint/wn18rr_ann_hard_negative_1pos1neg_rerun/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor = [entity_tensor_1.cuda(), entity_tensor_2.cuda(), entity_tensor_3.cuda(), entity_tensor_4.cuda(),
                     entity_tensor_5.cuda()]

    # # FB15k237
    # # entity_tensor_1 = torch.load('checkpoint/fb15k237_ann_hard_negative_nearest3_head_replace_1pos5neg/entity_tensor',
    # #                              map_location=lambda storage, loc: storage)
    # entity_tensor_1 = torch.load('checkpoint/fb15k237_ann_hard_negative_head_replace_1pos3neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor_2 = torch.load('checkpoint/fb15k237_ann_hard_negative_1pos3neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor_3 = torch.load('checkpoint/fb15k237_ann_hard_negative_tail_entity_similarity_1pos3neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor_4 = torch.load('checkpoint/fb15k237_bm25_tail_query_hard_negative_1pos3neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor_5 = torch.load('checkpoint/fb15k237_tail_entity_2hop_neighbours_hard_negative_1pos3neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor = [entity_tensor_1.cuda(), entity_tensor_2.cuda(), entity_tensor_3.cuda(), entity_tensor_4.cuda(),
    #                  entity_tensor_5.cuda()]

    # # dbpedia500
    # entity_tensor_1 = torch.load('checkpoint/dp500_ann_hard_negative_head_replace_1pos3neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor_2 = torch.load('checkpoint/dp500_ann_hard_negative_tail_entity_similarity_1pos3neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # # entity_tensor_3 = torch.load('checkpoint/dp500_ann_hard_negative_nearest3_head_replace_1pos3neg/entity_tensor',
    # #                              map_location=lambda storage, loc: storage)
    # entity_tensor_3 = torch.load('checkpoint/dp500_ann_hard_negative_1pos3neg_rerun/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor_4 = torch.load('checkpoint/dp500_tail_entity_2hop_neighbours_hard_negative_1pos1neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor_5 = torch.load('checkpoint/dp500_bm25_tail_query_hard_negative_1pos1neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    # entity_tensor = [entity_tensor_1.cuda(), entity_tensor_2.cuda(), entity_tensor_3.cuda(), entity_tensor_4.cuda(),
    #                  entity_tensor_5.cuda()]

    forward_metrics = eval_single_direction(entity_tensor=entity_tensor,
                                            eval_forward=True,
                                            batch_size=256)
    backward_metrics = eval_single_direction(entity_tensor=entity_tensor,
                                             eval_forward=False,
                                             batch_size=256)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))


def eval_single_direction(entity_tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

    eval_dir = 'forward' if eval_forward else 'backward'

    # WN18RR
    # average metrics: {"mean_rank": 235.09, "mrr": 0.6784, "hit@1": 0.6085, "hit@3": 0.7246, "hit@10": 0.8004, "hit@50": 0.8893}
    hr_tensor_1 = torch.load(
        'checkpoint/wn18rr_tail_entity_2hop_neighbours_hard_negative_1pos1neg/{}_hr_tensor'.format(eval_dir),
        map_location=lambda storage, loc: storage)
    # average metrics: {"mean_rank": 222.016, "mrr": 0.6554, "hit@1": 0.5892, "hit@3": 0.6913, "hit@10": 0.7763, "hit@50": 0.8784}
    hr_tensor_2 = torch.load('checkpoint/wn18rr_bm25_head_query_hard_negative_1pos1neg/{}_hr_tensor'.format(eval_dir),
                             map_location=lambda storage, loc: storage)
    # average metrics: {"mean_rank": 228.7018, "mrr": 0.6545, "hit@1": 0.5884, "hit@3": 0.6875, "hit@10": 0.7773, "hit@50": 0.8759}
    hr_tensor_3 = torch.load(
        'checkpoint/wn18rr_ann_hard_negative_head_entity_similarity_1pos1neg/{}_hr_tensor'.format(eval_dir),
        map_location=lambda storage, loc: storage)
    # average metrics: {"mean_rank": 253.111, "mrr": 0.6526, "hit@1": 0.588, "hit@3": 0.6884, "hit@10": 0.7748, "hit@50": 0.8747}
    hr_tensor_4 = torch.load('checkpoint/wn18rr_ann_hard_negative_head_replace_1pos1neg/{}_hr_tensor'.format(eval_dir),
                             map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 249.9823, "mrr": 0.6455, "hit@1": 0.5767, "hit@3": 0.6828, "hit@10": 0.7668, "hit@50": 0.8716}
    # hr_tensor_5 = torch.load(
    #     'checkpoint/wn18rr_ann_hard_negative_nearest3_head_replace_1pos1neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # average metrics: {"mean_rank": 171.4975, "mrr": 0.6434, "hit@1": 0.5707, "hit@3": 0.6862, "hit@10": 0.7755, "hit@50": 0.88}
    hr_tensor_5 = torch.load('checkpoint/wn18rr_ann_hard_negative_1pos1neg_rerun/{}_hr_tensor'.format(eval_dir),
                             map_location=lambda storage, loc: storage)

    hr_tensor = [hr_tensor_1.cuda(), hr_tensor_2.cuda(), hr_tensor_3.cuda(), hr_tensor_4.cuda(), hr_tensor_5.cuda()]

    # Averaged metrics: {'mean_rank': 181.6871, 'mrr': 0.6889, 'hit@1': 0.6187, 'hit@3': 0.7291, 'hit@10': 0.8171, 'hit@50': 0.8981}
    weights = [1.0, 0.3, 0.3, 0.3, 0.1]
    # Averaged metrics: {'mean_rank': 165.1514, 'mrr': 0.6865, 'hit@1': 0.6166, 'hit@3': 0.7267, 'hit@10': 0.816, 'hit@50': 0.9002}
    # weights = [1, 1, 1, 1, 1]

    # # FB15k237
    # # # average metrics: {"mean_rank": 137.5796, "mrr": 0.3611, "hit@1": 0.2745, "hit@3": 0.3913, "hit@10": 0.5338, "hit@50": 0.7184}
    # # hr_tensor_1 = torch.load(
    # #     'checkpoint/fb15k237_ann_hard_negative_nearest3_head_replace_1pos5neg/{}_hr_tensor'.format(eval_dir),
    # #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 132.3612, "mrr": 0.3651, "hit@1": 0.2757, "hit@3": 0.3991, "hit@10": 0.5418, "hit@50": 0.7253}
    # hr_tensor_1 = torch.load(
    #     'checkpoint/fb15k237_ann_hard_negative_head_replace_1pos3neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 132.293, "mrr": 0.3617, "hit@1": 0.2744, "hit@3": 0.3916, "hit@10": 0.5371, "hit@50": 0.7221}
    # hr_tensor_2 = torch.load(
    #     'checkpoint/fb15k237_ann_hard_negative_1pos3neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 138.8972, "mrr": 0.3538, "hit@1": 0.2686, "hit@3": 0.3838, "hit@10": 0.5229, "hit@50": 0.7104}
    # hr_tensor_3 = torch.load(
    #     'checkpoint/fb15k237_ann_hard_negative_tail_entity_similarity_1pos3neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 139.2998, "mrr": 0.3456, "hit@1": 0.2585, "hit@3": 0.3747, "hit@10": 0.522, "hit@50": 0.7147}
    # hr_tensor_4 = torch.load(
    #     'checkpoint/fb15k237_bm25_tail_query_hard_negative_1pos3neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 149.5862, "mrr": 0.3307, "hit@1": 0.2437, "hit@3": 0.357, "hit@10": 0.5066, "hit@50": 0.7082}
    # hr_tensor_5 = torch.load(
    #     'checkpoint/fb15k237_tail_entity_2hop_neighbours_hard_negative_1pos3neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    #
    # hr_tensor = [hr_tensor_1.cuda(), hr_tensor_2.cuda(), hr_tensor_3.cuda(), hr_tensor_4.cuda(), hr_tensor_5.cuda()]

    # # dbpedia500
    # # average metrics: {"mean_rank": 2186.8164, "mrr": 0.2708, "hit@1": 0.2112, "hit@3": 0.2933, "hit@10": 0.3813, "hit@50": 0.516}
    # hr_tensor_1 = torch.load(
    #     'checkpoint/dp500_ann_hard_negative_head_replace_1pos3neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 1976.9334, "mrr": 0.2683, "hit@1": 0.2068, "hit@3": 0.2898, "hit@10": 0.3829, "hit@50": 0.5321}
    # hr_tensor_2 = torch.load(
    #     'checkpoint/dp500_ann_hard_negative_tail_entity_similarity_1pos3neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # # # average metrics: {"mean_rank": 2374.2607, "mrr": 0.2626, "hit@1": 0.2066, "hit@3": 0.284, "hit@10": 0.3653, "hit@50": 0.4925}
    # # hr_tensor_3 = torch.load(
    # #     'checkpoint/dp500_ann_hard_negative_nearest3_head_replace_1pos3neg/{}_hr_tensor'.format(eval_dir),
    # #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 2318.486, "mrr": 0.2633, "hit@1": 0.2054, "hit@3": 0.2858, "hit@10": 0.37, "hit@50": 0.4992}
    # hr_tensor_3 = torch.load(
    #     'checkpoint/dp500_ann_hard_negative_1pos3neg_rerun/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 1680.042, "mrr": 0.2563, "hit@1": 0.1879, "hit@3": 0.2751, "hit@10": 0.3944, "hit@50": 0.5608}
    # hr_tensor_4 = torch.load(
    #     'checkpoint/dp500_tail_entity_2hop_neighbours_hard_negative_1pos1neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    # # average metrics: {"mean_rank": 1831.8533, "mrr": 0.2485, "hit@1": 0.184, "hit@3": 0.2666, "hit@10": 0.3757, "hit@50": 0.542}
    # hr_tensor_5 = torch.load(
    #     'checkpoint/dp500_bm25_tail_query_hard_negative_1pos1neg/{}_hr_tensor'.format(eval_dir),
    #     map_location=lambda storage, loc: storage)
    #
    # hr_tensor = [hr_tensor_1.cuda(), hr_tensor_2.cuda(), hr_tensor_3.cuda(), hr_tensor_4.cuda(), hr_tensor_5.cuda()]

    # weights = [1.0, 0.3, 0.3, 0.3, 0.1]
    # weights = [1, 1, 1, 1, 1]
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    # for small datasets: WN18RR and FB15k237
    total_entity_ranks = []
    for i in range(len(hr_tensor)):
        entity_ranks = get_entity_ranks(hr_tensor=hr_tensor[i], entities_tensor=entity_tensor[i],
                                        target=target, examples=examples, batch_size=batch_size)
        total_entity_ranks.append(entity_ranks)

    rank_fusion_k = 0
    rank_fusion_scores = []
    for i in tqdm.tqdm(range(len(total_entity_ranks[0]))):
        hr_rank_fusion_scores = []
        for k in range(len(entity_dict)):
            rank_fusion_score = 0
            for j in range(len(hr_tensor)):
                rank_fusion_score += 1 / (rank_fusion_k + total_entity_ranks[j][i][k]) * weights[j]
            hr_rank_fusion_scores.append(rank_fusion_score)
        rank_fusion_scores.append(hr_rank_fusion_scores)
    rank_fusion_scores = torch.tensor(rank_fusion_scores).cuda()
    logger.info(rank_fusion_scores.size())
    logger.info(len(entity_dict))
    total = len(rank_fusion_scores)

    target = torch.LongTensor(target).unsqueeze(-1).cuda()
    mean_rank, mrr, hit1, hit3, hit10, hit50 = 0, 0, 0, 0, 0, 0
    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        batch_target = target[start:end]
        batch_score = rank_fusion_scores[start:end]

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
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
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

    # # for large datasets: dbpedia500 and wiki5m_trans
    # metrics = get_rank_fusion_scores(hr_tensor=hr_tensor, entities_tensor=entity_tensor, target=target,
    #                                  examples=examples, weights=weights, batch_size=batch_size,
    #                                  rank_fusion_k=0)

    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))

    return metrics


if __name__ == '__main__':
    predict_by_split()
