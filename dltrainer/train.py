from __future__ import print_function
from imp import reload
from PIL import Image

import os
import sys
import argparse
import io

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import pdb
import numpy as np
from datetime import datetime
import calendar
import time
import cProfile

from dataset import ImageNetDataset
from model import Model

from tensorboardX import SummaryWriter

from logger import Logger
from profiler import Profiler

USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Large scale ImageNet training!')
parser.add_argument('--expid', type=str, default="", required=False)
parser.add_argument('--model', type=str, default="resnet50", required=False)
parser.add_argument('--epochs', type=int, default=1, metavar='N')
parser.add_argument('--iter', type=int, default=300, metavar='N')

parser.add_argument('--batch-size', type=int, default=32, metavar='N')
parser.add_argument('--use-remote', action='store_true', default=False)

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--devices', type=int, default=1, metavar='N')
parser.add_argument('--world-size', type=int, default=1, metavar='N')
parser.add_argument('--rank', type=int, default=0, metavar='N')

parser.add_argument('--no-tensorboard', action='store_true', default=False)
parser.add_argument('--no-profiler', action='store_true', default=False)
parser.add_argument('--profile-freq', type=float, default=1, metavar='N')
parser.add_argument('--profile-networkio', action='store_true', default=False)

args = parser.parse_args()

# Global variables
tb_logger = None
profiler = None 
dataset = None
flat_profiler = None
log_dir = ""

net = Model(args.model)
if USE_CUDA:
    net.cuda()

    devices = []
    for i in range(args.devices):
        devices.append(i)

    if len(devices)>1:
        net = torch.nn.DataParallel(net, device_ids=devices)
        cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[args.epochs*.25, args.epochs*.5,args.epochs*.75], gamma=0.1)

# Training
def train(iter_count, inputs, targets):
    # Step count for decaying learning rate
    scheduler.step()

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    epoch_start_time = time.time()

    optimizer.zero_grad()

    outputs = net(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    process_sync_grads(net)
    optimizer.step()

    curr_mini_loss = loss.data[0]

    epoch_end_time = time.time()
    epoch_time_taken = epoch_end_time - epoch_start_time

    if iter_count%args.log_interval==0:
        msg = '[iter: %d, ] Train Loss: %.3f' % (iter_count, curr_mini_loss)
        print(iter_count, msg)

        if not args.no_tensorboard:
            tb_logger.add_scalar('train/loss', curr_mini_loss, iter_count)
            tb_logger.add_scalar('train/epoch_time_taken', epoch_time_taken, iter_count)

def test(iter_count, inputs, targets):
    net.eval()

    epoch_start_time = time.time()

    outputs = net(inputs)
    loss = criterion(outputs, targets)
    curr_mini_loss = loss.data[0]

    epoch_end_time = time.time()
    epoch_time_taken = epoch_end_time - epoch_start_time

    if iter_count%args.log_interval==0:
        msg = '[iter: %d, ] Test Loss: %.3f' % (iter_count, curr_mini_loss)
        print(iter_count, msg)

        if not args.no_tensorboard:
            tb_logger.add_scalar('test/loss', curr_mini_loss, iter_count)
            tb_logger.add_scalar('test/epoch_time_taken', epoch_time_taken, iter_count)

def main():
    global tb_logger, profiler, dataset, flat_profiler, log_dir

    if not args.no_tensorboard:
        log_dir = os.path.join('log', args.expid, 
            args.model, 
            "batch_size_"+str(args.batch_size/args.world_size),
            "use_remote"+str(args.use_remote),
            "profile_freq"+str(args.profile_freq),
            datetime.now().isoformat())

        tb_logger = SummaryWriter(log_dir=log_dir)


    logger = Logger()

    if not args.no_profiler:
        profiler = Profiler(logger, tb_logger, freq=args.profile_freq)
        profiler.log(log_network=args.profile_networkio)

    tb_logger.add_text('params/model', str(args.model), 1)
    tb_logger.add_text('params/is-gpu-cached', str(args.gpu_cached), 1)
    tb_logger.add_text('params/is-cpu-cached', str(args.cpu_cached), 1)
    tb_logger.add_text('params/batch_size', str(args.batch_size/args.world_size), 1)

    iter_count = 0

    flat_profiler = cProfile.Profile()
    flat_profiler.enable()

    world_size = dist.get_world_size()
    sync_batch = args.batch_size / float(world_size)

    if args.use_remote:
        dataset = ImageNetDataset(shard_spec="http://storage.googleapis.com/lpr-imagenet/imagenet_train-@000003.tgz", 
                                zcom_port=args.zcom_port,
                                mini_batch_size=sync_batch,
                                num_epochs=args.epochs,
                                log_dir=log_dir,
                                num_workers=args.num_workers,
                                prefetch_workers=args.prefetch_workers,
                                prefetch_size=args.prefetch_size)    
    else:
        dataset = ImageNetDataset(shard_spec="testdata/imagenet_train-@000003.tgz", 
                                zcom_port=args.zcom_port,
                                mini_batch_size=sync_batch,
                                num_epochs=args.epochs,
                                log_dir=log_dir,
                                num_workers=args.num_workers,
                                prefetch_workers=args.prefetch_workers,
                                prefetch_size=args.prefetch_size)  
    while True:
        if iter_count > args.iter:
            flat_profiler.disable()
            flat_profiler.dump_stats(log_dir+"/main_profile.stats")
            return

        iter_count = iter_count + 1

        inputs, targets, time_to_create_batch =  dataset.getNextBatch()
        inputs, targets = Variable(inputs), Variable(targets)
        inputs, targets = inputs.cuda(), targets.cuda()

        tb_logger.add_scalar('train/time_to_create_batch', time_to_create_batch, iter_count)
        train(iter_count, inputs, targets)
        # test(iter_count, inputs, targets)


def init_processes(rank, w_size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '10.142.0.4' # REPLACE
    os.environ['MASTER_PORT'] = '6988' # REPLACE
    dist.init_process_group(backend, rank=rank, world_size=w_size)
    fn(rank, size)


def process_sync_grads(network):
    world_size = dist.get_world_size()
    for param in network.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= world_size

if __name__ == '__main__':
    try: 
        init_processes(args.rank, args.world_size, main)
        print('\nDone!')
        tb_logger.close() if tb_logger!=None else None
        profiler.stop() if profiler!=None else None
        sys.exit(0)
    except KeyboardInterrupt as e:
        tb_logger.close() if tb_logger!=None else None
        profiler.stop() if profiler!=None else None
        print('\nCancelled by user. Bye!')