import json
import os
from collections import OrderedDict
from typing import List

import torch
import torch.utils.data
import tqdm

from config import args
from dict_hub import build_tokenizer
from doc import collate, Example, Dataset, collate_test
from logger_config import logger
from models import build_model
from utils import AttrDict, move_to_cuda


class BertPredictor:

    def __init__(self, model=None, train_args=None):
        self.model = model
        self.train_args = train_args if train_args is not None else AttrDict()
        self.use_cuda = False

    def load(self, ckt_path, use_data_parallel=False):
        assert os.path.exists(ckt_path), ckt_path
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        build_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'hr_queue' in k or 'hr_queue_ptr' in k or 'tail_queue' in k or 'tail_queue_ptr' in k:
                continue
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        #     , strict=True
        self.model.load_state_dict(new_state_dict, strict=True)

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            # self.model.cuda()
            self.use_cuda = True
        logger.info('Load model from {} successfully'.format(ckt_path))
        self.model.eval()

    def _setup_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        self.train_args.momentum = False
        logger.info('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        args.use_link_graph = self.train_args.use_link_graph
        args.is_test = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example], only_head_embedding=False):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task, teset_set=False),
            num_workers=args.workers,
            batch_size=max(args.batch_size, 512 * torch.cuda.device_count()),
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list = [], []
        for batch_dict in tqdm.tqdm(data_loader):
            batch_dict['only_head_embedding'] = only_head_embedding
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            # if self.use_cuda and ('A100' not in torch.cuda.get_device_name(0) or torch.cuda.device_count() == 1 or
            #                       args.distributed):
            #     batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'].detach().cpu())
            if not only_head_embedding:
                tail_tensor_list.append(outputs['tail_vector'])

        if only_head_embedding:
            return torch.cat(hr_tensor_list, dim=0)
        else:
            return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task, teset_set=False),
            num_workers=args.workers,
            batch_size=args.batch_size,
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            if self.use_cuda and ('A100' not in torch.cuda.get_device_name(0) or torch.cuda.device_count() == 1 or
                                  args.distributed):
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)
