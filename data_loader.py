# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
import random

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy.random as nr
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        """
        :param mean: 高斯噪声的均值
        :param std: 高斯噪声的标准差
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        :param tensor: 输入图像张量，形状为 (C, H, W)，数值范围为 [0,1]
        :return: 添加噪声后的图像
        """
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)  # 限制范围在[0,1]之间，避免数值溢出

def getSVHN(batch_size, img_size=32, data_root='/tmp/public_dataset/pytorch', train=True, val=True, custom_tfm=None, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        # SVHN 的标签是数字 1 到 10，且“0” 被标为 10；
        # 为了让标签从 0~9 对应正常的类别索引，执行以下处理：
        # 所有标签减 1；
        # 原来的 10 就变成了 9；
        # 原来的 1~9 变成 0~8；
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()] + (custom_tfm if custom_tfm else [])),
                target_transform=target_transform,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=transforms.Compose([
                    transforms.Scale(img_size),
                    transforms.ToTensor(),
                ]),
                target_transform=target_transform
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getCIFAR10(batch_size, img_size=32, data_root='/tmp/public_dataset/pytorch', train=True, val=True, custom_tfm=None, custom_tfm_test=None, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()] + (custom_tfm if custom_tfm else []))),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([transforms.Scale(img_size), transforms.ToTensor(),] + (custom_tfm_test if custom_tfm_test else []))),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getTargetDataSet(data_type, batch_size, imageSize, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == '2d':
        train_loader, test_loader = get_two_gaussian(batch_size=batch_size, context_type="sparse-margin")

    return train_loader, test_loader

def getNonTargetDataSet(data_type, batch_size, imageSize, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'imagenet':
        testsetout = datasets.ImageFolder(dataroot+"/imagenet-data", transform=transforms.Compose([transforms.Scale(imageSize),transforms.ToTensor()]))
        train_loader, test_loader = None, torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'lsun':
        testsetout = datasets.ImageFolder(dataroot+"/lsun-data", transform=transforms.Compose([transforms.Scale(imageSize),transforms.ToTensor()]))
        train_loader, test_loader = None, torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'dtd':
        testsetout = datasets.ImageFolder(dataroot + "/dtd-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader, test_loader = None, torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'eurosat':
        testsetout = datasets.ImageFolder(dataroot + "/eurosat-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader, test_loader = None, torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'oxford_flowers':
        testsetout = datasets.ImageFolder(dataroot + "/oxford_flowers-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader, test_loader = None, torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'iNaturalist':
        testsetout = datasets.ImageFolder(dataroot + "/iNaturalist-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader, test_loader = None, torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'sun':
        testsetout = datasets.ImageFolder(dataroot + "/SUN-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader, test_loader = None, torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'places':
        testsetout = datasets.ImageFolder(dataroot + "/Places-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader, test_loader = None, torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

    return test_loader

def getContextDataSet(data_type, batch_size, imageSize, dataroot):
    if data_type == 'svhn_augmentation':
        custom_tfm = [transforms.RandomRotation(60), transforms.RandomHorizontalFlip(), transforms.GaussianBlur(kernel_size=(3, 3), sigma=10), transforms.RandomSolarize(threshold=0.5), transforms.ColorJitter(brightness=0.5, contrast=0.5)]
        # custom_tfm = None
        train_loader, test_loader = getSVHN(batch_size=batch_size, img_size=imageSize, data_root=dataroot, custom_tfm=custom_tfm, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'cifar10_augmentation':
        custom_tfm = [transforms.RandomRotation(60), transforms.RandomHorizontalFlip(), transforms.GaussianBlur(kernel_size=(3, 3), sigma=10), transforms.RandomSolarize(threshold=0.5), transforms.ColorJitter(brightness=0.5, contrast=0.5)]
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, img_size=imageSize, data_root=dataroot, custom_tfm=custom_tfm, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'cifar100':
        train_data = datasets.CIFAR100(root=dataroot + "/cifar100-data", train=True, download=True, transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
    elif data_type == 'cifar100_augmentation':
        custom_tfm = [transforms.Scale(imageSize), transforms.ToTensor(), transforms.RandomRotation(60), transforms.RandomHorizontalFlip(), transforms.GaussianBlur(kernel_size=(3, 3), sigma=10), transforms.RandomSolarize(threshold=0.5), transforms.ColorJitter(brightness=0.5, contrast=0.5)]
        train_data = datasets.CIFAR100(root=dataroot + "/cifar100-data", train=True, download=True, transform=transforms.Compose(custom_tfm))
        train_loader, test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True), None
    elif data_type == 'caltech10':
        train_data = datasets.ImageFolder(dataroot+"/caltech10-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)),transforms.ToTensor()]))
        train_loader, test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True), None
    elif data_type == 'caltech101':
        train_data = datasets.ImageFolder(dataroot + "/caltech101-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader, test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True), None
    elif data_type == 'eurosat':
        train_data = datasets.ImageFolder(dataroot + "/eurosat-data", transform=transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()]))
        train_loader, test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True), None
    return train_loader, test_loader

def get_two_gaussian(batch_size=24, context_type="compact-margin"):
    """
    Generate the training and context data loaders for the 2D illustrative examples used in the paper (see Fig. 1).
    :param context_type: The relationship between context data and training data (none | cover | right-top-far | left-bottom-far | sparse-margin | compact-margin)
    """
    random.seed(0)
    np.random.seed(0)
    num_samples_per_class = 100
    num_samples = num_samples_per_class * 2

    # Class 0: left-bottom Gaussian distribution, mean at (-4, -4) and covariance as diagonal matrix
    mean0 = np.array([-2, -2])
    cov0 = np.array([[0.5, 0.0], [0.0, 0.5]])
    data0 = np.random.multivariate_normal(mean0, cov0, num_samples_per_class)
    label0 = np.zeros((num_samples_per_class,))

    # Class 1: right-top Gaussian distribution, mean at (2, 2) and covariance as diagonal matrix
    mean1 = np.array([2, 2])
    cov1 = np.array([[0.5, 0.0], [0.0, 0.5]])
    data1 = np.random.multivariate_normal(mean1, cov1, num_samples_per_class)
    label1 = np.ones((num_samples_per_class,))

    #  Incorporate both classes
    train_data = np.vstack((data0, data1))  # [num_samples, 2]
    train_labels = np.concatenate((label0, label1))  # [num_samples,]

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # *** Generate context data based on the specified context type ***
    context_samples = []
    max_trials = num_samples * 50  #  Max rejection sampling trials for avoiding infinite loops
    trials = 0

    # Determine the sampling range and distance to the training data (threshold) of the context samples
    if context_type == "cover":
        context_range = [-3, 3]
        selected_threshold = 0
    elif context_type == "right-top-far":
        context_range = [15, 25]
        selected_threshold = 0
    elif context_type == "left-bottom-far":
        context_range = [-25, -15]
        selected_threshold = 0
    elif context_type == "sparse-margin":
        context_range = [-30, 30]
        selected_threshold = 0.5
    elif context_type == "compact-margin":
        radius = 9
        selected_threshold = 0.5
        context_samples = []
        trials = 0
        max_trials = 10000

        # Rejection sampling from two circles centered at (-2, -2) and (2, 2) to construct context samples around the training distribution
        centers = [torch.tensor([-2.0, -2.0]), torch.tensor([2.0, 2.0])]
        while len(context_samples) < 150 and trials < max_trials:
            trials += 1
            # 从极坐标中均匀采样
            angle = torch.rand(1) * 2 * np.pi  # 均匀角度
            r = radius * torch.sqrt(torch.rand(1))  # 面积均匀分布采样

            # 在两个圆中随机选择一个中心
            center = centers[np.random.randint(0, 2)]
            x = center[0] + r * torch.cos(angle)
            y = center[1] + r * torch.sin(angle)
            point = torch.stack([x, y], dim=1).squeeze()

            # 与训练集所有点距离进行比较
            dists = torch.norm(train_data - point, dim=1)  # [num_train]
            if torch.all(dists > selected_threshold):
                context_samples.append(point)

        # 构建 OOD loader
        ood_data = torch.stack(context_samples, dim=0)  # [num_samples, 2]
        ood_labels = torch.full((ood_data.size(0),), -1, dtype=torch.long)  # 设为 OOD 标签

        ood_dataset = TensorDataset(ood_data, ood_labels)
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, ood_loader

    elif context_type == "none":
        context_loader = train_loader
        return train_loader, context_loader

    # Rejection sampling for the context samples until the desired number of samples is reached
    while len(context_samples) < num_samples and trials < max_trials:
        trials += 1
        point = torch.FloatTensor(2).uniform_(*context_range)
        dists = torch.norm(train_data - point, dim=1)
        if torch.all(dists > selected_threshold):
            context_samples.append(point)
    if len(context_samples) < num_samples:
        print(f"Warning: Only generated {len(context_samples)} OOD samples out of requested {num_samples}.")

    context_dataset = TensorDataset(torch.stack(context_samples), torch.zeros(len(context_samples), dtype=torch.long))
    context_loader = DataLoader(context_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, context_loader

def rejection_sample_ood(train_loader, num_samples=1000, bounds=[-10, 10], threshold=1.0):
    """
    从 [-10, 10]^2 中 rejection 采样与训练数据不重合的 OOD 样本

    :param train_loader: 用于获取原始训练分布
    :param num_samples: 需要的 OOD 样本数量
    :param bounds: 采样边界
    :param threshold: 判定是否靠近训练分布的欧氏距离阈值
    :return: Tensor [num_samples, 2] 的 OOD 样本
    """
    train_data = []
    for x, _ in train_loader:
        train_data.append(x)
    train_data = torch.cat(train_data, dim=0)  # [N, 2]

    ood_samples = []
    max_trials = num_samples * 50  # 最大尝试次数防止死循环
    trials = 0

    while len(ood_samples) < num_samples and trials < max_trials:
        trials += 1
        point = torch.FloatTensor(2).uniform_(*bounds)
        dists = torch.norm(train_data - point, dim=1)
        if torch.all(dists > threshold):
            ood_samples.append(point)

    if len(ood_samples) < num_samples:
        print(f"Warning: Only generated {len(ood_samples)} OOD samples out of requested {num_samples}.")
    return torch.stack(ood_samples)  # [num_samples, 2]