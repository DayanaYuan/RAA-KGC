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


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List[Example],
                    k=3, batch_size=256) -> Tuple:
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10, hit50 = 0, 0, 0, 0, 0, 0

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
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10, 'hit@50': hit50}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks


def predict_by_split():
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)

    # WN18RR
    entity_tensor_1 = torch.load('/home/horanchen/ydy/study/code/5Improved-Negative-Search-for-KGC/data/WN18RR_v1/checkpoint/wn18rr_tail_entity_2hop_neighbours_hard_negative_1pos1neg/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor_2 = torch.load('/home/horanchen/ydy/study/code/5Improved-Negative-Search-for-KGC/data/WN18RR_v1/checkpoint/wn18rr_bm25_head_query_hard_negative_1pos1neg/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor_3 = torch.load('/home/horanchen/ydy/study/code/5Improved-Negative-Search-for-KGC/data/WN18RR_v1/checkpoint/wn18rr_ann_hard_negative_head_entity_similarity_1pos1neg/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor_4 = torch.load('/home/horanchen/ydy/study/code/5Improved-Negative-Search-for-KGC/data/WN18RR_v1/checkpoint/wn18rr_ann_hard_negative_head_replace_1pos1neg/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    # entity_tensor_5 = torch.load('checkpoint/wn18rr_ann_hard_negative_nearest3_head_replace_1pos1neg/entity_tensor',
    #                              map_location=lambda storage, loc: storage)
    entity_tensor_5 = torch.load('/home/horanchen/ydy/study/code/5Improved-Negative-Search-for-KGC/data/WN18RR_v1/checkpoint/wn18rr_ann_hard_negative_1pos1neg_rerun/entity_tensor',
                                 map_location=lambda storage, loc: storage)
    entity_tensor = torch.cat([entity_tensor_1, entity_tensor_2, entity_tensor_3, entity_tensor_4, entity_tensor_5],
                              dim=-1)
    entity_tensor = entity_tensor.cuda()

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
    # entity_tensor = torch.cat([entity_tensor_1, entity_tensor_2, entity_tensor_3, entity_tensor_4, entity_tensor_5],
    #                           dim=-1)
    # entity_tensor = entity_tensor.cuda()

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
    #
    # entity_tensor = torch.cat([entity_tensor_1, entity_tensor_2, entity_tensor_3, entity_tensor_4, entity_tensor_5],
    #                           dim=-1)
    # entity_tensor = entity_tensor.cuda()

    forward_metrics = eval_single_direction(entity_tensor=entity_tensor,
                                            eval_forward=True,
                                            batch_size=256)
    backward_metrics = eval_single_direction(entity_tensor=entity_tensor,
                                             eval_forward=False,
                                             batch_size=256)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))


def eval_single_direction(entity_tensor: torch.tensor,
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
    # average metrics: {"mean_rank": 171.4975, "mrr": 0.6434, "hit@1": 0.5707, "hit@3": 0.6862, "hit@10": 0.7755, "hit@50": 0.88}
    hr_tensor_5 = torch.load('checkpoint/wn18rr_ann_hard_negative_1pos1neg_rerun/{}_hr_tensor'.format(eval_dir),
                             map_location=lambda storage, loc: storage)

    # Averaged metrics: {'mean_rank': 216.752, 'mrr': 0.6919, 'hit@1': 0.6265, 'hit@3': 0.7278, 'hit@10': 0.8111, 'hit@50': 0.8966}
    weights = [1.2, 0.3, 0.3, 0.3, 0.1]
    # Averaged metrics: {'mean_rank': 207.9573, 'mrr': 0.6892, 'hit@1': 0.6248, 'hit@3': 0.7247, 'hit@10': 0.8081, 'hit@50': 0.8972}
    weights = [1, 1, 1, 1, 1]
    hr_tensor = torch.cat([weights[0] * hr_tensor_1, weights[1] * hr_tensor_2, weights[2] * hr_tensor_3,
                           weights[3] * hr_tensor_4, weights[4] * hr_tensor_5], dim=-1)
    hr_tensor = hr_tensor.to(entity_tensor.device)

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
    # weights = [0.9, 0.6, 0.2, 0.05, 0.01]
    # hr_tensor = torch.cat([weights[0] * hr_tensor_1, weights[1] * hr_tensor_2, weights[2] * hr_tensor_3,
    #                        weights[3] * hr_tensor_4, weights[4] * hr_tensor_5], dim=-1)
    # hr_tensor = hr_tensor.to(entity_tensor.device)

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
    # weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    # hr_tensor = torch.cat([weights[0] * hr_tensor_1, weights[1] * hr_tensor_2, weights[2] * hr_tensor_3,
    #                        weights[3] * hr_tensor_4, weights[4] * hr_tensor_5], dim=-1)
    # hr_tensor = hr_tensor.to(entity_tensor.device)

    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                target=target, examples=examples,
                                                                batch_size=batch_size)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics


if __name__ == '__main__':
    predict_by_split()
