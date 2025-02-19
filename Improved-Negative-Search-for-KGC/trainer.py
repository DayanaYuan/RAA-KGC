import gc
import json
import os
import random
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.utils.data
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from dict_hub import build_tokenizer
from doc import Dataset, collate
from logger_config import logger
from metric import accuracy
from models import build_model, ModelOutput
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj


class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        self.train_dataset = Dataset(path=args.train_path, task=args.task, is_train=True,
                                     num_negatives=args.num_negatives, teset_set=False)
        valid_dataset = Dataset(path=args.valid_path, task=args.task, is_train=False, num_negatives=10, teset_set=False) \
            if args.valid_path else None
        num_training_steps = args.epochs * len(self.train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        if args.task.lower() not in ['dbpedia500', 'wiki5m_trans' 'wiki5m_ind']:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True)

        self.valid_loader = None
        eval_batch_size = 384 if args.task.lower() in ['dbpedia500', 'wiki5m_trans' 'wiki5m_ind'] and \
                                 args.use_hard_negative else 512
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=eval_batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metric_dict = self.eval_epoch(epoch)
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1'])
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        new_state_dict = OrderedDict()
        for k, v in self.model.state_dict().items():
            if k.startswith('module.hr_bert_k.') or k.startswith('module.tail_bert_k.'):
                continue
            new_state_dict[k] = v
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': new_state_dict,
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_step(self, batch_dict, top1, top3, inv_t, losses, hr_losses, related_losses):
        if torch.cuda.is_available():
            batch_dict = move_to_cuda(batch_dict)
        batch_size = len(batch_dict['batch_data'])

        # compute output
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                output_dict = self.model(**batch_dict)
        else:
            output_dict = self.model(**batch_dict)
        outputs = get_model_obj(self.model).compute_logits(output_dict=output_dict, batch_dict=batch_dict)
        outputs = ModelOutput(**outputs)
        related_logits, related_labels = outputs.related_logits, outputs.related_labels
        assert related_logits.size(0) == self.args.batch_size
        # head + relation -> tail
        related_loss = self.criterion(related_logits, related_labels)
        # tail -> head + relation
        related_loss = related_loss + self.criterion(related_logits[:, :self.args.batch_size].t(), related_labels)

        logits, labels = outputs.logits, outputs.labels
        assert logits.size(0) == batch_size
        # head + relation -> tail
        hr_loss = self.criterion(logits, labels)
        # tail -> head + relation
        hr_loss += self.criterion(logits[:, :batch_size].t(), labels)

        loss = 0.2 * related_loss + 0.8 * hr_loss

        acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
        top1.update(acc1.item(), batch_size)
        top3.update(acc3.item(), batch_size)

        inv_t.update(outputs.inv_t, 1)
        losses.update(loss.item(), self.args.batch_size)
        related_losses.update(related_loss.item(), self.args.batch_size)
        hr_losses.update(hr_loss.item(), self.args.batch_size)

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        if self.args.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
        self.scheduler.step()

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        related_losses = AverageMeter('Loss', ':.4')
        hr_losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader) if self.args.task.lower() not in ['wiki5m_ind', 'wiki5m_trans', 'dbpedia500']
            else len(self.train_dataset.examples) // self.args.batch_size,
            [losses, inv_t, top1, top3, hr_losses, related_losses],
            prefix="Epoch: [{}]".format(epoch))

        self.model.train()

        # split large datasets into chunks to avoid OOM
        if self.args.task.lower() in ['dbpedia500', 'wiki5m_trans' 'wiki5m_ind']:
            global_step = 0
            self.train_loader = []
            shard_size = 1000 * self.args.batch_size
            random.shuffle(self.train_dataset.examples)
            for start in range(0, len(self.train_dataset), shard_size):
                end = start + shard_size
                train_loader = torch.utils.data.DataLoader(
                    Dataset(path='', examples=self.train_dataset.examples[start:end], task=self.args.task,
                            is_train=True, num_negatives=self.args.num_negatives),
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    num_workers=self.args.workers,
                    pin_memory=True,
                    drop_last=True)

                for batch_dict in train_loader:
                    self.train_step(batch_dict, top1, top3, inv_t, losses)

                    if global_step % self.args.print_freq == 0:
                        progress.display(global_step)
                    if (global_step + 1) % self.args.eval_every_n_step == 0:
                        self._run_eval(epoch=epoch, step=global_step + 1)
                        self.model.train()
                    global_step += 1
        else:
            for i, batch_dict in enumerate(self.train_loader):
                self.train_step(batch_dict, top1, top3, inv_t, losses, hr_losses, related_losses)

                if i % self.args.print_freq == 0:
                    progress.display(i)
                if (i + 1) % self.args.eval_every_n_step == 0:
                    self._run_eval(epoch=epoch, step=i + 1)
                    self.model.train()

        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model).cuda()
        # elif torch.cuda.is_available():
        #     self.model.cuda()
        # else:
        #     logger.info('No gpu will be used')

    def load(self, model, ckt_path, use_data_parallel=False):
        assert os.path.exists(ckt_path), ckt_path
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel predictor')
            model = torch.nn.DataParallel(model).cuda()
        elif torch.cuda.is_available():
            model.cuda()
        logger.info('Load model from {} successfully'.format(ckt_path))
        return model

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
