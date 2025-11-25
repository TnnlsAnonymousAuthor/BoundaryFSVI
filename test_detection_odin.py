###################################################################################################
# Measure the detection performance: reference code is https://github.com/ShiyuLiang/odin-pytorch #
###################################################################################################
# Writer: Kimin Lee
from __future__ import print_function
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn

import data_loader
import calculate_log as callog
import models

from torch.autograd import Variable

import bayesianize.bnn as bnn
from helpers import seed_everything, compute_acc, compute_ece, selective_acc, visualize_dataloader

# Training settings
parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
parser.add_argument('--batch-size', type=int, default=50, help='batch size')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')

parser.add_argument('--dataset', default='cifar10', help='target dataset: svhn | cifar10')
parser.add_argument('--ctx_dataset', default='gan', help='none | svhn_augmentation | cifar10_augmentation | eurosat | cifar100_augmentation | gan')
parser.add_argument('--cost_type', type=str, default='kl_cost', help='gp_cost | kl_cost')
parser.add_argument('--out_dataset', default='iNaturalist', help='out-of-dist dataset: imagenet | lsun | dtd | sun | oxford_flowers | iNaturalist | eurosat | places')

locals()
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
seed_everything(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

args.outf = f"{args.outf}/{args.cost_type}/{args.dataset}_(ctx){args.ctx_dataset}"
args.pre_trained_net = f"{args.outf}/model_epoch_20.pth"

# 创建当前OOD数据集的输出目录
args.outf = f"{args.outf}/(ood_ODIN){args.out_dataset}"
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

print('Load model')
model = models.vgg13()
bnn.bayesianize_(model, inference='ffg', init_sd=1e-4, prior_weight_sd=1e6, prior_bias_sd=1e6)
model.load_state_dict(torch.load(args.pre_trained_net))
model.cuda()
print(model)

print('load target data: ',args.dataset)
_, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)

print('load non target data: ',args.out_dataset)
nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size, args.imageSize, args.dataroot)

visualize_dataloader(test_loader, title=f"Target Data: {args.dataset}")
visualize_dataloader(nt_test_loader, title=f"Non-Target Data: {args.out_dataset}")

def testData(model, criterion, CUDA_DEVICE, testloader10, testloader, dataName, noiseMagnitude1, temper):
    """
    使用ODIN方法评估模型在ID（in-distribution）和OOD（out-of-distribution）数据上的置信度输出

    :param criterion: nn.CrossEntropyLoss()
    :param testloader10: dataloader for ID dataset (e.g., CIFAR-10)
    :param testloader: dataloader for OOD dataset (e.g., imagenet)
    :param dataName: name of ID dataset
    :param noiseMagnitude1: epsilon for pertubation in Eq. (S15)
    :param temper: temperature scaling factor in Eq. (S14)
    """
    f1 = open(f'{args.outf}/confidence_Base_In.txt', 'w')   # ID数据的输出文件
    f2 = open(f'{args.outf}/confidence_Base_Out.txt', 'w')  # OOD数据的输出文件

    print("Processing in-distribution images")

    # ----------------------- In-Distribution Data Processing ------------------------
    for j, data in enumerate(testloader10):
        images, _ = data                       # images shape: [B, 3, 32, 32]
        images = images.cuda(CUDA_DEVICE)
        images.requires_grad = True

        outputs = model(images)               # Feed-forward output logits，shape: [B, num_classes]
        outputs_softmax = torch.softmax(outputs.data / temper, dim=1)  # Softmax scores after temperature scaling in Eq. (S14), shape: [B, num_classes]

        labels = torch.argmax(outputs_softmax, dim=1).detach()         # Max score index for each sample (confidence), shape: [B]

        # Gradients computed by backpropagation (based on temperature-scaled logits and predicted labels)
        loss = criterion(outputs / temper, labels)
        loss.backward()

        gradient = torch.ge(images.grad.data, 0).float()  # shape [B, 3, 32, 32]
        gradient = (gradient - 0.5) * 2  # {-1, +1}

        # Channel normalization
        gradient[:, 0] = gradient[:, 0] / (63.0 / 255.0)
        gradient[:, 1] = gradient[:, 1] / (62.1 / 255.0)
        gradient[:, 2] = gradient[:, 2] / (66.7 / 255.0)

        # Adding perturbation and re-forwarding
        tempInputs = images.data - noiseMagnitude1 * gradient
        outputs = model(tempInputs)
        outputs = outputs / temper
        outputs_softmax = torch.softmax(outputs.data, dim=1)

        max_confidences = torch.max(outputs_softmax, dim=1)[0]  # shape: [B]
        for conf in max_confidences:
            f1.write(f"{conf.item()}\n")

    # ----------------------- OOD Data Processing ------------------------
    print("Processing out-of-distribution images")

    for j, data in enumerate(testloader):
        images, _ = data
        images = images.cuda(CUDA_DEVICE)
        images.requires_grad = True

        outputs = model(images)
        outputs_softmax = torch.softmax(outputs.data / temper, dim=1)
        labels = torch.argmax(outputs_softmax, dim=1).detach()

        loss = criterion(outputs / temper, labels)
        loss.backward()

        gradient = torch.ge(images.grad.data, 0).float()
        gradient = (gradient - 0.5) * 2
        gradient[:, 0] = gradient[:, 0] / (63.0 / 255.0)
        gradient[:, 1] = gradient[:, 1] / (62.1 / 255.0)
        gradient[:, 2] = gradient[:, 2] / (66.7 / 255.0)

        tempInputs = images.data - noiseMagnitude1 * gradient
        outputs = model(tempInputs)
        outputs = outputs / temper
        outputs_softmax = torch.softmax(outputs.data, dim=1)

        max_confidences = torch.max(outputs_softmax, dim=1)[0]
        for conf in max_confidences:
            f2.write(f"{conf.item()}\n")

criterion = nn.CrossEntropyLoss()
testData(model, criterion, 0, test_loader, nt_test_loader, args.out_dataset, 0.0001, 10)
print('calculate metrics')
callog.metric(args.outf)
