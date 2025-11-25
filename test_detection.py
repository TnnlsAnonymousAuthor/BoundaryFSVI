###################################################################################################
# Measure the detection performance: reference code is https://github.com/ShiyuLiang/odin-pytorch #
###################################################################################################
# Writer: Kimin Lee
from __future__ import print_function
import argparse
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import data_loader
import calculate_log as callog
import models

from torch.autograd import Variable

import bayesianize.bnn as bnn
from helpers import seed_everything, compute_acc, compute_ece, selective_acc, visualize_dataloader

# Training settings
parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')

parser.add_argument('--dataset', default='cifar10', help='target dataset: svhn | cifar10')
parser.add_argument('--cost_type', type=str, default='kl_cost', help='md_cost | kl_cost')
parser.add_argument('--ctx_dataset', default='gan', help='none | svhn_augmentation | cifar10_augmentation | eurosat | cifar100_augmentation | gan')
parser.add_argument('--out_dataset', default='imagenet', help='out-of-dist dataset: imagenet | lsun | dtd | sun | oxford_flowers | iNaturalist')

locals()
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
seed_everything(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

args.outf = f"{args.outf}/{args.cost_type}/{args.dataset}_(ctx){args.ctx_dataset}"
args.pre_trained_net = f"{args.outf}/model_epoch_20.pth"

args.outf = f"{args.outf}/(ood){args.out_dataset}"
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

print('Load model')
model = models.vgg13()
bnn.bayesianize_(model, inference='ffg', init_sd=1e-4, prior_weight_sd=1e6, prior_bias_sd=1e6)
pre_trained_state_dict = torch.load(args.pre_trained_net, map_location='cpu')
model.load_state_dict(torch.load(args.pre_trained_net))
print(model)

print('load target data: ',args.dataset)
_, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)

print('load non target data: ',args.out_dataset)
nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size, args.imageSize, args.dataroot)

visualize_dataloader(test_loader, title=f"Target Data: {args.dataset}")
visualize_dataloader(nt_test_loader, title=f"Non-Target Data: {args.out_dataset}")

if args.cuda:
    model.cuda()

def generate_target():
    model.eval()
    correct = 0
    total = 0
    f1 = open('%s/confidence_Base_In.txt'%args.outf, 'w')

    all_labels, all_output = [], []

    for data, target in test_loader:
        total += data.size(0)
        #vutils.save_image(data, '%s/target_samples.png'%args.outf, normalize=True)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)  # target=[bts]
        batch_output = model(data)  # [bts, num_classes]
        all_labels.append(target.data)
        all_output.append(batch_output.data)

        # compute the accuracy
        pred = batch_output.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()
        for i in range(data.size(0)):
            # confidence score: max_y p(y|x)
            output = batch_output[i].view(1,-1)
            soft_out = F.softmax(output)
            soft_out = torch.max(soft_out.data)
            f1.write("{}\n".format(soft_out))

    all_labels = torch.cat(all_labels, dim=0)
    all_output = torch.cat(all_output, dim=0)
    all_probs = F.softmax(all_output)
    all_labels, all_output, all_probs = all_labels.cpu().numpy(), all_output.cpu().numpy(), all_probs.cpu().numpy()

    print(f'Accuracy: {compute_acc(all_probs, all_labels):.2f}, '
          f'AUC: {roc_auc_score(all_labels, all_probs, multi_class="ovr") * 100:.2f}, '
          f'AUPRC: {average_precision_score(all_labels, all_probs, average="macro") * 100:.2f}, '
          f'NLL: {F.nll_loss(F.log_softmax(torch.tensor(all_output), dim=-1), torch.tensor(all_labels)).item():.2f}, '
          f'Sel. ACC: {selective_acc(all_probs, all_labels)[0]:.2f}, '
          f'ECE: {compute_ece(all_probs, all_labels, args) * 100:.2f}, ')

def generate_non_target():
    model.eval()
    total = 0
    f2 = open('%s/confidence_Base_Out.txt'%args.outf, 'w')

    for data, target in nt_test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        batch_output = model(data)
        for i in range(data.size(0)):
            # confidence score: max_y p(y|x)
            output = batch_output[i].view(1,-1)
            soft_out = F.softmax(output)
            soft_out = torch.max(soft_out.data)
            f2.write("{}\n".format(soft_out))

print('generate log from in-distribution data')
generate_target()
print('generate log  from out-of-distribution data')
generate_non_target()
print('calculate metrics')
callog.metric(args.outf)
