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
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--log-interval', type=int, default=1, metavar='N')

parser.add_argument('--devices', type=int, default=1, metavar='N')
parser.add_argument('--batch-size', type=int, default=32, metavar='N')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--no-tensorboard', action='store_true', default=False)
parser.add_argument('--no-profiler', action='store_true', default=False)
parser.add_argument('--profile-freq', type=float, default=1, metavar='N')
parser.add_argument('--profile-networkio', action='store_true', default=False)

parser.add_argument('--world-size', type=int, default=1, metavar='N')
parser.add_argument('--rank', type=int, default=0, metavar='N')

parser.add_argument('--split-by', type=int, default=1, metavar='N')
parser.add_argument('--split-to-use', type=int, default=0, metavar='N')


args = parser.parse_args()

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Global variables
tb_logger = None
profiler = None
dataset = None
log_dir = ""
trainloader = None
testloader = None
iter_count = 0

net = ResNet50()

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

def train(epoch):
    print('\nEpoch: %d' % epoch)
    global iter_count
    
    epoch_start_time = time.time()
    scheduler.step()

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        iter_count += 1

        inputs, targets = Variable(inputs), Variable(targets)
        if USE_CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if args.world_size>1:
            process_sync_grads(net)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if iter_count%args.log_interval==0:
            msg = '[epoch: %d iter: %d] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, iter_count, train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            print(msg)

            if not args.no_tensorboard:
                tb_logger.add_scalar('train/loss', train_loss/(batch_idx+1), iter_count)
                tb_logger.add_scalar('train/accuracy', 100.*correct/total, iter_count)

    epoch_end_time = time.time()
    epoch_time_taken = epoch_end_time - epoch_start_time

    if not args.no_tensorboard:
        tb_logger.add_scalar('train/epoch_time_taken', epoch_time_taken, iter_count)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            num_batches += 1

            if USE_CUDA:
                inputs, targets = inputs.cuda(), targets.cuda()
                
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        msg = '[epoch: %d, ] Test Loss: %.3f %.3f' % (epoch, test_loss/num_batches, 100.*correct/total)
        print(msg)

        if not args.no_tensorboard:
            tb_logger.add_scalar('test/loss', test_loss/num_batches, epoch)
            tb_logger.add_scalar('test/accuracy', 100.*correct/total, epoch)


def main(rank, world_size):
    global tb_logger, profiler, dataset, log_dir, trainloader, testloader

    if not args.no_tensorboard:
        log_dir = os.path.join('log', 
            args.expid,
            datetime.now().isoformat())

        tb_logger = SummaryWriter(log_dir=log_dir)

    logger = Logger()

    if not args.no_profiler:
        profiler = Profiler(logger, tb_logger, freq=args.profile_freq)
        profiler.log(log_network=args.profile_networkio)

    tb_logger.add_text('params/batch_size', str(args.batch_size/world_size), 1)

    sync_batch = args.batch_size / world_size

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainset.train_data = np.split(trainset.train_data, args.split_by)[args.split_to_use]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=sync_batch, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)


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
        if args.world_size>1:
            init_processes(args.rank, args.world_size, main)
        else:
            main(0, 1)
        print('\nDone!')
        tb_logger.close() if tb_logger!=None else None
        profiler.stop() if profiler!=None else None
        sys.exit(0)
    except KeyboardInterrupt as e:
        tb_logger.close() if tb_logger!=None else None
        profiler.stop() if profiler!=None else None
        print('\nCancelled by user. Bye!')
