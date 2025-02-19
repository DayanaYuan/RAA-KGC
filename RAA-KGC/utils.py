import os
import glob
import torch
import shutil

import numpy as np
import torch.nn as nn

from logger_config import logger


class AttrDict:
    pass


def save_checkpoint(state: dict, is_best: bool, filename: str):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.mdl')
    shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.mdl')


def delete_old_ckt(path_pattern: str, keep=5):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        logger.info('Delete old checkpoint1 {}'.format(f))
        os.system('rm -f {}'.format(f))


def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    assert isinstance(model, torch.nn.Module), 'Argument must be nn.Module'

    num_parameters = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_parameters += np.prod(list(p.size()))
            logger.info('{}: {}'.format(name, np.prod(list(p.size()))))

    logger.info('Number of parameters: {}M'.format(num_parameters // 10**6))
    return num_parameters


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model

def move_to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device, non_blocking=True)
    elif isinstance(tensor, dict):
        return {key: move_to_device(value, device) for key, value in tensor.items()}
    elif isinstance(tensor, list):
        return [move_to_device(x, device) for x in tensor]
    elif isinstance(tensor, tuple):
        return tuple(move_to_device(x, device) for x in tensor)
    else:
        return tensor

def split_and_move_to_cuda(tensor_list, device_ids):
    num_gpus = len(device_ids)
    split_batch_dict = [[] for _ in device_ids]

    for i, tensor in enumerate(tensor_list):
        device_id = device_ids[i % num_gpus]
        split_batch_dict[device_id].append(move_to_device(tensor, device_id))

    return split_batch_dict


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            # return maybe_tensor.cuda(non_blocking=True)
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
            # return split_and_move_to_cuda(maybe_tensor, list(range(torch.cuda.device_count())))
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
