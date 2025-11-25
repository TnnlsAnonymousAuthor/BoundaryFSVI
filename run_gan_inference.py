##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Kimin Lee
from __future__ import print_function
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import torchvision.utils as vutils
import models
import os

from torch.autograd import Variable

import bayesianize.bnn as bnn
from helpers import seed_everything, mahalanobis_distance_cost_function

# Training settings
parser = argparse.ArgumentParser(description='Training code - joint confidence')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | svhn')
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

args.outf = f"{args.outf}/{args.cost_type}/{args.dataset}_(ctx)gan"
if not os.path.exists(args.outf):
    os.makedirs(args.outf)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)

print('Load model')
model = models.vgg13()
bnn.bayesianize_(model, inference='ffg', init_sd=1e-4, prior_weight_sd=1e6, prior_bias_sd=1e6)
print("Parameter size: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))

print('load GAN')
nz = 100
netG = models.Generator(1, nz, 64, 3) # ngpu, nz, ngf, nc
netD = models.Discriminator(1, 3, 64) # ngpu, nc, ndf
# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)

if args.cuda:
    model.cuda()
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    fixed_noise = fixed_noise.cuda()
fixed_noise = Variable(fixed_noise)

print('Setup optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        start_batch_time = time.time()  # The start time for the current batch

        gan_target = torch.FloatTensor(target.size()).fill_(0)
        uniform_dist = torch.Tensor(data.size(0), args.num_classes).fill_((1./args.num_classes))

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            gan_target, uniform_dist = gan_target.cuda(), uniform_dist.cuda()

        data, target, uniform_dist = Variable(data), Variable(target), Variable(uniform_dist)

        t0 = time.time()
        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterion(output.squeeze(), targetv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output.squeeze(), targetv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterion(output.squeeze(), targetv)
        D_G_z2 = output.data.mean()

        # minimize the true distribution
        if args.cost_type == "md_cost":
            fake_output, fake_feature = model(fake, return_feature=True)
            costG = mahalanobis_distance_cost_function(fake_output, fake_feature, beta=args.beta)
        elif args.cost_type == "kl_cost":
            fake_output = model(fake, return_feature=False)
            fake_output = F.log_softmax(fake_output)
            costG = args.beta * F.kl_div(fake_output, uniform_dist) * args.num_classes
        generator_loss = errG +  costG
        generator_loss.backward()
        optimizerG.step()
        gan_time = time.time() - t0  # boundaryGAN training time

        t1 = time.time()
        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        output = F.log_softmax(model(data))
        loss = F.nll_loss(output.squeeze(), target)
        KL_loss_cls = sum(m.kl_divergence() for m in model.modules() if hasattr(m, "kl_divergence"))

        # KL divergence
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        if args.cost_type == "md_cost":
            fake_output, fake_feature = model(fake, return_feature=True)
            cost_loss = mahalanobis_distance_cost_function(fake_output, fake_feature, beta=args.beta)
        elif args.cost_type == "kl_cost":
            fake_output = model(fake, return_feature=False)
            fake_output = F.log_softmax(fake_output)
            cost_loss = args.beta * F.kl_div(fake_output, uniform_dist)*args.num_classes
        total_loss = loss + 0.1 * KL_loss_cls +  cost_loss
        total_loss.backward()
        optimizer.step()

        clf_time = time.time() - t1  # BNN classifier training time
        total_batch_time = time.time() - start_batch_time

        if batch_idx % args.log_interval == 0:
            gan_ratio = 100. * gan_time / total_batch_time
            clf_ratio = 100. * clf_time / total_batch_time

            print(
                f"Classification Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.data.item():.6f}, KL Classifier Loss: {KL_loss_cls.data.item():.6f}, Cost Loss: {cost_loss.data.item():.6f} | "
                f"Time: Total {total_batch_time:.3f}s | GAN {gan_time:.3f}s ({gan_ratio:.1f}%) | Classifier {clf_time:.3f}s ({clf_ratio:.1f}%)"
            )

            fake = netG(fixed_noise)
            vutils.save_image(fake.data, '%s/gan_samples_epoch_%03d.png'%(args.outf, epoch), normalize=True)

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
    test(epoch)
    end_time = time.time()
    print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds, Estimated time remaining: {(args.epochs - epoch) * (end_time - start_time) / 60:.2f} (mins)")

    if epoch in decreasing_lr:
        optimizerG.param_groups[0]['lr'] *= args.droprate
        optimizerD.param_groups[0]['lr'] *= args.droprate
        optimizer.param_groups[0]['lr'] *= args.droprate
    if epoch % 10 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))
