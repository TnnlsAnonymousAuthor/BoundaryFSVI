import argparse
import os
import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import data_loader
import models
import bayesianize.bnn as bnn
from helpers import seed_everything, visualize_dataloader

# Training settings
parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
parser.add_argument('--batch-size', type=int, default=100, help='batch size')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--dataroot', default='../data', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes (default: 10)')

parser.add_argument('--dataset', default='cifar10CatHorse', help='target dataset: cifar10CatHorse')
parser.add_argument('--cost_type', type=str, default='kl_cost', help='md_cost | kl_cost')
parser.add_argument('--ctx_dataset', default='gan', help='none | cifar10_augmentation | eurosat | cifar100_augmentation | gan')
parser.add_argument('--out_dataset', default='imagenet', help='out-of-dist dataset: imagenet1k-lion | imagenet')

# UMAP可调参数
parser.add_argument('--umap-n-neighbors', type=int, default=30, help='UMAP n_neighbors')
parser.add_argument('--umap-min-dist', type=float, default=0.6, help='UMAP min_dist')
parser.add_argument('--umap-components', type=int, default=2, help='UMAP output dims (default: 2)')
args = parser.parse_args()
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
seed_everything(args.seed)

args.outf = f"{args.outf}/{args.cost_type}/{args.dataset}_(ctx){args.ctx_dataset}/(ood){args.out_dataset}"
if not os.path.exists(args.outf):
    os.makedirs(args.outf)
args.pre_trained_net = f"output/{args.cost_type}/{args.dataset}_(ctx){args.ctx_dataset}/model_epoch_40.pth"

model = models.vgg13(num_classes=2)
bnn.bayesianize_(model, inference='ffg', init_sd=1e-4, prior_weight_sd=1e6, prior_bias_sd=1e6)
if os.path.exists(args.pre_trained_net):
    state = torch.load(args.pre_trained_net, map_location='cpu')
    model.load_state_dict(state)
if args.cuda:
    model.cuda()
model.eval()

_, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot, num_workers=0, persistent_workers=False)
nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size, args.imageSize, args.dataroot, num_workers=0, persistent_workers=False)

visualize_dataloader(nt_test_loader)

in_features_list, in_labels_list = [], []
ood_features_list = []
ctx_features_list = []
ood_images_list = []

# Limit the maximum number of samples to visualize
max_ood_samples = 1000
max_ctx_samples = 1000
# Current count of OOD samples collected
ood_count = 0
ctx_count = 0

# 如果是GAN上下文，需要准备生成器路径与模型
gan_netG_path = f"output/{args.cost_type}/{args.dataset}_(ctx)gan/netG_epoch_40.pth"

with torch.no_grad():
    # Extract in-distribution features
    for data, target in test_loader:
        data = data.cuda()
        logits, feats = model(data, return_feature=True)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        in_features_list.append(feats.cpu())
        in_labels_list.append(target.cpu())

    # Extract out-of-distribution features
    for data, _ in nt_test_loader:
        data = data.cuda()
        logits, feats = model(data, return_feature=True)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        ood_features_list.append(feats.cpu())
        ood_images_list.append(data.cpu())

        ood_count += data.size(0)
        if ood_count >= max_ood_samples:
            break

    # Extract context features
    if args.ctx_dataset != 'none':
        # Using boundaryGAN to generate context samples in our method
        if args.ctx_dataset == 'gan':
            nz = 100
            netG = models.Generator(1, nz, 64, 3)
            state = torch.load(gan_netG_path, map_location='cpu')
            netG.load_state_dict(state)
            netG.cuda()
            netG.eval()

            # Generate 1000 context samples and corresponding features
            remaining = max_ctx_samples
            batch_g = min(args.batch_size, 256)
            while remaining > 0:
                bs = min(batch_g, remaining)
                noise = torch.randn(bs, nz, 1, 1)
                noise = noise.cuda()
                fake = netG(noise)
                logits, feats = model(fake, return_feature=True)
                if feats.dim() > 2:
                    feats = feats.view(feats.size(0), -1)
                ctx_features_list.append(feats.cpu())
                ctx_count += bs
                remaining -= bs
        else:
            # Using 1000 context samples in the manually specified context dataset
            ctx_loader, _ = data_loader.getContextDataSet(args.ctx_dataset, args.batch_size, args.imageSize, args.dataroot, num_workers=0, persistent_workers=False)
            for data, _ in ctx_loader:
                data = data.cuda()
                logits, feats = model(data, return_feature=True)
                if feats.dim() > 2:
                    feats = feats.view(feats.size(0), -1)
                remaining = max_ctx_samples - ctx_count
                if remaining <= 0:
                    break
                if feats.size(0) > remaining:
                    feats = feats[:remaining]
                ctx_features_list.append(feats.cpu())
                ctx_count += feats.size(0)
                if ctx_count >= max_ctx_samples:
                    break

# 拼接与标签
in_features = torch.cat(in_features_list, dim=0)          # [N_in, D]
in_labels = torch.cat(in_labels_list, dim=0).numpy()      # [N_in]
ood_features = torch.cat(ood_features_list, dim=0)        # [N_ood, D]
ctx_features = torch.cat(ctx_features_list, dim=0) if len(ctx_features_list) > 0 else None  # [N_ctx, D] or None

# Incorporate In-distribution, OOD, and Context features into X_all
if ctx_features is not None:
    X_in = in_features.numpy()
    X_ctx = ctx_features.numpy()
    X_ood = ood_features.numpy()
    X_all = np.concatenate([X_in, X_ctx, X_ood], axis=0)
else:
    X_in = in_features.numpy()
    X_ood = ood_features.numpy()
    X_all = np.concatenate([X_in, X_ood], axis=0)

# PCA is first applied to reduce the feature dimension to 50
X_all = StandardScaler().fit_transform(X_all)
X_all = PCA(n_components=50, random_state=args.seed).fit_transform(X_all)
# UMAP is then applied to reduce the feature dimension to 2
reducer = umap.UMAP(
    n_neighbors=args.umap_n_neighbors,
    min_dist=args.umap_min_dist,
    n_components=args.umap_components,
    random_state=args.seed,
    metric='euclidean'
)
all_2d = reducer.fit_transform(X_all)

# Convert 2D features back to in, ctx, ood sets
n_in = X_in.shape[0]
if ctx_features is not None:
    n_ctx = X_ctx.shape[0]
    in_2d = all_2d[:n_in]
    ctx_2d = all_2d[n_in:n_in + n_ctx]
    ood_2d = all_2d[n_in + n_ctx:]
else:
    in_2d = all_2d[:n_in]
    ctx_2d = None
    ood_2d = all_2d[n_in:]

# The closest (Euclidean distance) OOD points to the In-distribution centroid in 2D space
closest_ood_indices = [821, 772] if args.out_dataset == 'imagenet1k-lion' else [272, 394]
# This can be verified using the following code:
##############################################################################################
#diff = ood_2d[:, None, :] - ctx_2d[None, :, :]                                               #
#dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))   # [N_ood, N_ctx]                          #
#min_dist_to_ctx = dist_matrix.min(axis=1)  # The closest distance to ctx for each OOD sample #
#topk = min(10, ood_2d.shape[0])                                                              #
#closest_ood_indices = np.argsort(min_dist_to_ctx)[:topk]  # [topk]                           #
###############################################################################################

# Output the features as a dictionary
save_path = f"{args.outf}/features.pkl"
data_dict = {
        "in_2d": in_2d,                 # [N_in, D]
        "ctx_2d": ctx_2d,               # [N_ctx, D] 或 None
        "ood_2d": ood_2d,               # [N_ood, D]
        "in_labels": in_labels,       # [N_in]
        "closest_ood_indices": closest_ood_indices
    }
with open(save_path, "wb") as f:
    pickle.dump(data_dict, f)

