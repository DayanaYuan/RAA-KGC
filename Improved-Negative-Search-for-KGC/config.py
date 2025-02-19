import glob
import os
import random
import torch
import argparse
import warnings

import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='SimKGC arguments')
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--task', default='wn18rr', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default="WN18RR_v1/train.txt.json" , type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default="WN18RR_v1/test.txt.json" , type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--model-dir', default='WN18RR_v1/checkpoint50/wn18rr_tail_entity_2hop_neighbours_hard_negative_1pos1neg/', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=3, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--pooling', default='mean', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--dropout', default=0.1, type=float, metavar='N',
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true',
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=5000, type=int,
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.02, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature as a trainable parameter or not')
parser.add_argument('--max-num-tokens', default=50, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', action='store_true',
                    help='use head entity as negative')
parser.add_argument('--output-dim', default=128, type=int)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=2333, type=int,
                    help='seed for initializing training. ')

# only used for evaluation
parser.add_argument('--is-test', action='store_true',
                    help='is in test mode or not')
parser.add_argument('--rerank-n-hop', default=5, type=int,
                    help='use n-hops node for re-ranking entities, only used during evaluation')
parser.add_argument('--neighbor-weight', default=0.05, type=float,
                    help='weight for re-ranking entities')
parser.add_argument('--eval-model-path', default='WN18RR_v1/checkpoint50/wn18rr_tail_entity_2hop_neighbours_hard_negative_1pos1neg/model_best.mdl', type=str, metavar='N',
                    help='path to model, only used for evaluation')

parser.add_argument('--use-hard-negative', action='store_true',
                    help="use hard negatives or not")
parser.add_argument('--num-negatives', default=1, type=int,
                    help="num of negative examples used for training")


args = parser.parse_args()

if args.task.lower() in ['wiki5m_trans', 'wiki5m_ind'] and args.use_hard_negative:
    directory = os.path.dirname(args.train_path.split(',')[0])
    if 'bm25' in args.train_path:
        path_pattern = '{}/train.bm25.tail.query.top30.hard.negative_shard*.json'.format(directory)
        path_list = glob.glob(path_pattern)
    elif 'train.forward.ann.hard.negative.tail.entity.similarity.top10.shard0.json' in args.train_path:
        path_pattern = '{}/train.*.ann.hard.negative.tail.entity.similarity.top10.shard*.json'.format(directory)
        path_list = glob.glob(path_pattern)
    elif 'train.forward.ann.hard.negative.top10.head.replace.shard0.json' in args.train_path:
        path_pattern = '{}/train.*.ann.hard.negative.top10.head.replace.shard*.json'.format(directory)
        path_list = glob.glob(path_pattern)
    elif 'train.forward.ann.hard.negative.tail.entity.2hop.neighbours.shard0.json' in args.train_path:
        path_pattern = '{}/train.*.ann.hard.negative.tail.entity.2hop.neighbours.shard*.json'.format(directory)
        path_list = glob.glob(path_pattern)
    else:
        forward_path_pattern = '{}/train.forward.ann.hard.*.json'.format(directory)
        backward_path_pattern = '{}/train.backward.ann.hard.*.json'.format(directory)
        path_list = glob.glob(forward_path_pattern) + glob.glob(backward_path_pattern)
    args.train_path = ','.join(path_list)

assert not args.train_path or all([os.path.exists(path) for path in args.train_path.split(',')])
assert args.pooling in ['cls', 'mean', 'max']
assert args.task.lower() in ['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans', 'dbpedia50', 'dbpedia500']
assert args.lr_scheduler in ['linear', 'cosine']

if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)
else:
    # assert os.path.exists(args.eval_model_path), 'One of args.model_dir and args.eval_model_path should be valid path'
    args.model_dir = os.path.dirname(args.eval_model_path)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn('GPU is not available, set use_amp=False and print_freq=1')
