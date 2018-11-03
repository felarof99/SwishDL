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
from model import *

from tensorboardX import SummaryWriter

from logger import Logger
from profiler import Profiler

USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Large scale ImageNet training!')
parser.add_argument('--expid', type=str, default="", required=False)
parser.add_argument('--model', type=str, default="resnet50", required=False)
parser.add_argument('--shard-spec', type=str, default="http://storage.googleapis.com/lpr-demo/cifar10-train-000000.tgz", required=False)

parser.add_argument('--epochs', type=int, default=1, metavar='N')
parser.add_argument('--iter', type=int, default=300, metavar='N')
parser.add_argument('--log-interval', type=int, default=1, metavar='N')

parser.add_argument('--devices', type=int, default=1, metavar='N')
parser.add_argument('--batch-size', type=int, default=32, metavar='N')

parser.add_argument('--use-remote', action='store_true', default=False)

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--no-tensorboard', action='store_true', default=False)
parser.add_argument('--no-profiler', action='store_true', default=False)
parser.add_argument('--profile-freq', type=float, default=1, metavar='N')
parser.add_argument('--profile-networkio', action='store_true', default=False)
parser.add_argument('--sync', action='store_true', default=False)

parser.add_argument('--world-size', type=int, default=1, metavar='N')
parser.add_argument('--rank', type=int, default=0, metavar='N')

args = parser.parse_args()

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Global variables
tb_logger = None
profiler = None
dataset = None
log_dir = ""

# net = Model(args.model)
net = ResNet50()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
    epoch_start_time = time.time()

    # Step count for decaying learning rate
    scheduler.step()

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    process_sync_grads(net)
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    curr_mini_loss = train_loss

    epoch_end_time = time.time()
    epoch_time_taken = epoch_end_time - epoch_start_time

    if iter_count%args.log_interval==0:
        msg = '[iter: %d, ] Train Loss: %.3f' % (iter_count, curr_mini_loss)
        print(iter_count, msg)

        if not args.no_tensorboard:
            tb_logger.add_scalar('train/loss', curr_mini_loss, iter_count)
            tb_logger.add_scalar('train/epoch_time_taken', epoch_time_taken, iter_count)

            if args.sync:
                try:
                    cmd = "gsutil rsync -r log gs://cloud-infra-logs/"
                    # cmd = "gsutil cp train.py gs://large-scale-dl/train.py"

                    print("Running gsutil cmd", cmd)

                    assert os.system(cmd) == 0
                except Exception as e:
                    print("ERROR: gsutil failed!", e)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            curr_mini_loss = test_loss/(batch_idx+1)
            curr_mini_accuracy = 100.*correct/total

            if iter_count%args.log_interval==0:
                msg = '[iter: %d, ] Test Loss: %.3f %.3f' % (iter_count, curr_mini_loss, curr_accuracy)
                print(iter_count, msg)

                if not args.no_tensorboard:
                    tb_logger.add_scalar('test/loss', curr_mini_loss, iter_count)
                    tb_logger.add_scalar('test/accuracy', curr_mini_accuracy, iter_count)
                    tb_logger.add_scalar('test/epoch_time_taken', epoch_time_taken, iter_count)

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

def main(rank, w_size):
    global tb_logger, profiler, dataset, log_dir

    if not args.no_tensorboard:
        log_dir = os.path.join('log', 
            args.expid,
            args.model,
            datetime.now().isoformat())

        tb_logger = SummaryWriter(log_dir=log_dir)

    logger = Logger()

    if not args.no_profiler:
        profiler = Profiler(logger, tb_logger, freq=args.profile_freq)
        profiler.log(log_network=args.profile_networkio)

    tb_logger.add_text('params/model', str(args.model), 1)
    tb_logger.add_text('params/batch_size', str(args.batch_size/args.world_size), 1)

    world_size = dist.get_world_size()
    sync_batch = args.batch_size / world_size

    dataset = ImageNetDataset(shard_spec=args.shard_spec,
                            mini_batch_size=sync_batch,
                            num_epochs=args.epochs)

    for iter_count in range(args.iter):
        inputs, targets, time_to_create_batch =  dataset.getNextBatch()
        tb_logger.add_scalar('train/time_to_create_batch', time_to_create_batch, iter_count)

        inputs, targets = Variable(inputs), Variable(targets)
        if USE_CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()

        train(iter_count, inputs, targets)
        test(iter_count, inputs, targets)


def init_processes(rank, w_size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'master-service'
    os.environ['MASTER_PORT'] = '6988'
    dist.init_process_group(backend, rank=rank, world_size=w_size)
    fn(rank, w_size)


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
