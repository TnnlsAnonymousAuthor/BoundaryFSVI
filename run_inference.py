##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Kimin Lee

from __future__ import print_function
import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import models

from torch.autograd import Variable

import bayesianize.bnn as bnn
from helpers import seed_everything, visualize_dataloader, mahalanobis_distance_cost_function
from itertools import cycle

# Training settings
parser = argparse.ArgumentParser(description='Training code - cross entropy')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--dataset', default='svhn', help='cifar10 | svhn')
parser.add_argument('--ctx_dataset', default='none', help='none | svhn_augmentation | cifar10_augmentation | cifar100_augmentation')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--cost_type', type=str, default='kl_cost', help='md_cost | kl_cost')
parser.add_argument('--beta', type=float, default=1, help='For mahalanobis distance cost function: β = 0.00001 for cifar10 and 0.0005 for svhn. For KL divergence cost function: β = 1 for both datasets.')

args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
seed_everything(args.seed)

args.outf = f"{args.outf}/{args.cost_type}/{args.dataset}_(ctx){args.ctx_dataset}"
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)
visualize_dataloader(train_loader, title=f"Target Data: {args.dataset}")

if args.ctx_dataset != 'none':
    print('load context data: ',args.ctx_dataset)
    ctx_train_loader, _ = data_loader.getContextDataSet(args.ctx_dataset, args.batch_size, args.imageSize, args.dataroot)
    ctx_iter = cycle(ctx_train_loader)  # 确保 ctx_train_loader 可以无限迭代
    visualize_dataloader(ctx_train_loader, title=f"Context Data: {args.ctx_dataset}")

print('Load model')
model = models.vgg13()
bnn.bayesianize_(model, inference='ffg', init_sd=1e-4, prior_weight_sd=1e6, prior_bias_sd=1e6)
print("Parameter size: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))

if args.cuda:
    model.cuda()

print('Setup optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        # Forward for target data
        output = F.log_softmax(model(data))
        loss = F.nll_loss(output, target)
        KL_loss_cls = sum(m.kl_divergence() for m in model.modules() if hasattr(m, "kl_divergence"))

        cost_loss = torch.Tensor([0.0]).cuda()
        if args.ctx_dataset != 'none':
            # get a context batch from ctx_train_loader
            ctx_data, _ = next(ctx_iter)
            ctx_data = ctx_data.cuda()

            if args.cost_type == "md_cost":
                ctx_output, fake_feature = model(ctx_data, return_feature=True)
                cost_loss = mahalanobis_distance_cost_function(ctx_output, fake_feature, beta=args.beta)
            elif args.cost_type == "kl_cost":
                uniform_dist = torch.Tensor(ctx_data.size(0), args.num_classes).fill_((1. / args.num_classes)).cuda()
                ctx_output = model(ctx_data, return_feature=False)
                ctx_output = F.log_softmax(ctx_output)
                cost_loss += args.beta * F.kl_div(ctx_output, uniform_dist) * args.num_classes

        total_loss = loss + 0.1 * KL_loss_cls + cost_loss
        total_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.6f}\tKL Loss: {KL_loss_cls.data.item():.6f}\tCost Loss: {cost_loss.data.item():.6f}"
            )


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for data, target in test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data))
        test_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))

for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    train(epoch)
    end_time = time.time()
    print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds, Estimated time remaining: {(args.epochs - epoch) * (end_time - start_time) / 60:.2f} (mins)")
    if epoch in decreasing_lr:
        optimizer.param_groups[0]['lr'] *= args.droprate
    test(epoch)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))
