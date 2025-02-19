import torch
from typing import List
from config import args
from triplet import EntityDict
from dict_hub import get_link_graph
from doc import Example
# from doc_norela import Example
import numpy as np


# 通过这些步骤，代码可以根据链接图中的 n 跳实体索引，为 batch_score 张量中对应的实体位置添加权重值，以便进一步加权影响模型的输出。
def rerank_by_graph(related_batch_score: torch.tensor,
                    batch_score: torch.tensor,
                    examples: List[Example],
                    entity_dict: EntityDict):

    if args.task == 'wiki5m_ind':
        assert args.neighbor_weight < 1e-6, 'Inductive setting can not use re-rank strategy'

    if args.neighbor_weight < 1e-6:
        return
    related_batch_sorted_score, related_batch_sorted_indices = torch.sort(related_batch_score, dim=-1, descending=True)
    # 下述代码片段中的目的是根据获取的链接图中的 n 跳实体索引，将权重值添加到 batch_score 张量的相应位置。让我们逐行分析代码：
    for idx in range(batch_score.size(0)): # 对于 batch_score 张量的第一维（通常是 batch 维）进行循环迭代，batch_score.size(0) 表示批次的大小。
        cur_ex = examples[idx]             # 从 examples 列表中获取当前索引 idx 对应的示例，examples 包含当前批次的示例。
        n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
                                                                  entity_dict=entity_dict,
                                                                  n_hop=args.rerank_n_hop)
        # 使用 get_link_graph() 函数从链接图中获取以当前示例的头实体为中心的 n 跳实体索引。这些索引表示链接图中与当前示例的头实体具有 n 跳关系的其他实体。

        # 创建一个张量 delta，其中包含了与 n_hop_indices 长度相同的权重值，这些权重值都是通过 args.neighbor_weight 参数指定的。这些权重将被添加到 batch_score 张量中相应位置的值。
        delta = torch.tensor([args.neighbor_weight for _ in n_hop_indices]).to(batch_score.device)
        # 将 n_hop_indices 转换为 PyTorch 的张量类型，并确保在与 batch_score 张量相同的设备上。
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)
        # 将 delta 中的值添加到 batch_score 张量的指定位置，这些位置由 n_hop_indices 张量中的索引指定。此处使用的是 PyTorch 中的 index_add_ 函数，它会原地修改 batch_score 张量。
        batch_score[idx].index_add_(0, n_hop_indices, delta)

        # related_n_hop_indices = related_batch_sorted_indices[idx][:len(delta)]
        # batch_score[idx].index_add_(0, related_n_hop_indices, delta)

        related_n_hop_indices = related_batch_sorted_indices[idx][:len(delta)]
        redelta = torch.tensor([0.0 for _ in related_n_hop_indices]).to(batch_score.device)
        batch_score[idx].index_add_(0, related_n_hop_indices, redelta)

        # The test set of FB15k237 removes triples that are connected in train set,
        # so any two entities that are connected in train set will not appear in test,
        # however, this is not a trick that could generalize.
        # by default, we do not use this piece of code .

        # if args.task == 'FB15k237':
        #     n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
        #                                                               entity_dict=entity_dict,
        #                                                               n_hop=1)
        #     n_hop_indices.remove(entity_dict.entity_to_idx(cur_ex.head_id))
        #     delta = torch.tensor([-0.5 for _ in n_hop_indices]).to(batch_score.device)
        #     n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)
        #
        #     batch_score[idx].index_add_(0, n_hop_indices, delta)
