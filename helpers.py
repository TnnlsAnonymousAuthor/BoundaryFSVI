import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import random
import torch

eps = 1e-6

def gaussian_process_cost_function(preds_f, preds_feature, tau_f=0.01):
    """
    :param preds_f: 模型预测输出(未经过激活) [bts, K]
    :param preds_feature:  特征提取器出书 [bts, ndim]
    :param tau_f: 代价函数权重参数
    :return:
    """
    K = preds_f.size(1)  # K, 类别数
    y = torch.zeros_like(preds_f)  # [bts, K], 上下文集标签（希望所有位置的预测输出接近于0）
    cov = preds_feature @ preds_feature.T  # [bts, bts]
    cov += torch.ones_like(cov)  #
    cov += torch.eye(cov.size(0)).to(cov.device) * 0.1  # 添加一个小的噪声项以确保正定性

    # 通过将所有样本的第k维preds_f[:, k]视为“一个bts维的随机变量”，总共有K个这样的独立随机变量，可以用batch Gaussian来加速计算
    preds_f = preds_f.T # [K, bts]
    y = y.T # [K, bts]
    # 在第0个维度复制K个协方差矩阵
    cov = cov.unsqueeze(0).expand(K, -1, -1)  # [K, bts, bts]
    gp = torch.distributions.MultivariateNormal(loc=y, covariance_matrix=cov)  # [K, bts], 由y与特征相似矩阵cov定义的预测函数上的GP
    cost = - tau_f * gp.log_prob(preds_f)  # [K, bts], pred_f越接近于y，log_prob越大，代价cost越小

    # # /*** 以上计算等价于以下代码 ***/
    # for k in range(K):
    #     gp = torch.distributions.MultivariateNormal(loc=preds_f[:, k], covariance_matrix=cov)  # [bts, K], 由y与特征相似矩阵cov定义的预测函数上的GP
    #     cost += - tau_f * gp.log_prob(y[:, k])  # [bts, K], pred_f越接近于y，log_prob越大，代价cost越小
    return cost.sum()  # [bts, K], 代价函数

def mahalanobis_distance_cost_function(preds_f, preds_feature, beta=0.01, noise=0.1):
    """
    :param preds_f: Model's logits output (without activation) [bts, K]
    :param preds_feature:  Feature extractor output [bts, ndim]
    :param beta: Cost function weight scalar parameter
    """
    K = preds_f.size(1)  # K, number of classes
    y = torch.zeros_like(preds_f)  # [bts, K] context set labels (we want all positions of the predicted output to be close to 0)
    cov = preds_feature @ preds_feature.T  # [bts, bts]
    cov += torch.ones_like(cov)  #
    cov += torch.eye(cov.size(0)).to(cov.device) * noise  # Add a small noise term to ensure positive definiteness

    # Through treating each sample's k-th dimension preds_f[:, k] as a "bts-dimensional random variable",
    # we have K such independent random variables, which can be accelerated using batch Gaussian
    preds_f = preds_f.T # [K, bts]
    y = y.T # [K, bts]

    # Copy the covariance matrix K times along the first dimension
    cov = cov.unsqueeze(0).expand(K, -1, -1)  # [K, bts, bts]

    # Gaussian Process (GP) defined by y and covariance matrix cov can be used to compute the Mahalanobis distance
    mahalanobis_distance = torch.distributions.MultivariateNormal(loc=y, covariance_matrix=cov)  # [K, bts]
    # The closer preds_f is to y, the larger the log_prob, and the smaller the cost
    cost = - beta * mahalanobis_distance.log_prob(preds_f)  # [K, bts]

    # *** The following code is equivalent to the above calculations ***
    # for k in range(K):
    #     # Gaussian Process (GP) defined by y and covariance matrix cov can be used to compute the Mahalanobis distance
    #     mahalanobis_distance = torch.distributions.MultivariateNormal(loc=preds_f[:, k], covariance_matrix=cov)  # [bts, K]
    #     # The closer preds_f is to y, the larger the log_prob, and the smaller the cost
    #     cost += - beta * mahalanobis_distance.log_prob(y[:, k])  # [bts, K]
    return cost.sum()  # [bts, K], 代价函数

def seed_everything(seed=42):
    """
    设置全局随机种子来提高可复现性。

    参数:
    - seed (int): 随机种子的值
    """
    random.seed(seed)  # Python内置的随机库
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希随机性
    np.random.seed(seed)  # Numpy库的随机种子
    torch.manual_seed(seed)  # PyTorch的随机种子
    torch.cuda.manual_seed(seed)  # GPU操作的种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True  # 确保CUDA的确定性

def visualize_dataloader(dataloader, title=None):
    """
    从 DataLoader 中随机可视化 64 张图像，支持灰度和 RGB 图像

    :param dataloader: torch.utils.data.DataLoader, 任意一个已加载数据的 DataLoader 对象
    """
    # 随机获取一个 batch
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # 如果 batch 数量不足 64，重复抽样
    if images.size(0) < 64:
        indices = torch.randint(0, images.size(0), (64,))
        images = images[indices]
    else:
        images = images[:64]

    # 创建 8x8 网格图像
    grid_img = vutils.make_grid(images, nrow=8, padding=2, normalize=True)

    # 将 tensor 转为 numpy 格式
    np_img = grid_img.numpy()

    # 单通道图像（灰度）
    if np_img.shape[0] == 1:
        plt.imshow(np_img[0], cmap='gray')
    else:  # 多通道图像（RGB）
        plt.imshow(np.transpose(np_img, (1, 2, 0)))

    # plt.title(title if title else "Random Sample from DataLoader")
    plt.axis("off")
    plt.show()

def visualize_single_image(dataloader, i, title=None):
    """
    可视化数据集中第 i 张图片，支持灰度和 RGB 图像

    :param dataloader: torch.utils.data.DataLoader, 已加载数据的 DataLoader 对象
    :param i: int, 数据集中图片的索引（从0开始）
    :param title: str or None, 图片标题，默认无标题
    """
    # 将所有数据加载到内存（注意：数据集较大时会耗费较多内存）
    images = []
    for batch in dataloader:
        imgs, _ = batch
        images.append(imgs)
    images = torch.cat(images, dim=0)  # 拼接所有batch的图片，shape=[总样本数, C, H, W]

    # 判断索引是否越界
    if i < 0 or i >= images.size(0):
        raise IndexError(f"Index {i} out of range. Dataset size: {images.size(0)}")

    # 取出第i张图片
    img = images[i]

    # 归一化到 [0,1] 方便展示（假设原图是[0,1]或[-1,1]区间）
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min + 1e-5)

    # 转成 numpy 格式
    np_img = img.numpy()

    # 判断通道数，单通道灰度图或多通道RGB图
    if np_img.shape[0] == 1:
        plt.imshow(np_img[0], cmap='gray')
    else:
        plt.imshow(np.transpose(np_img, (1, 2, 0)))

    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

def compute_acc(pred_prob, true_labels):
    """
    :param pred_prob: 预测概率（经过Softmax） [N, K]
    :param true_labels: 真实标签 [N]
    :return: 准确率
    """
    predicted_label = np.argmax(pred_prob, axis=1)
    accuracy = np.mean(predicted_label == true_labels)
    return accuracy * 100

def compute_ece(pred_prob, true_labels, args, M=10):
    """
    Reference to : https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/
    :param pred_prob: predictive probability（after softmax） [N, K]
    :param true_labels: labels [N]
    :param M: Number of bins for calibration
    :return: expected calibration error (ECE)
    """
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(pred_prob, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(pred_prob, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = 0.0
    plt.figure(figsize=(5, 4))

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower &amp; upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

            # Plot horizontal line: accuracy (blue)
            plt.hlines(y=accuracy_in_bin,
                       xmin=bin_lower,
                       xmax=bin_upper,
                       color='C0',
                       linewidth=4,
                       label='Accuracy' if bin_lower == 0 else "")

            # Plot horizontal line: confidence (green)
            plt.hlines(y=avg_confidence_in_bin,
                       xmin=bin_lower,
                       xmax=bin_upper,
                       color='C2',
                       linewidth=4,
                       label='Confidence' if bin_lower == 0 else "")

            # Plot the gap between accuracy and confidence (pink)
            lower_y = min(accuracy_in_bin, avg_confidence_in_bin)
            upper_y = max(accuracy_in_bin, avg_confidence_in_bin)
            plt.fill_between([bin_lower, bin_upper],
                             y1=lower_y, y2=upper_y,
                             color='gray', alpha=0.2,
                             label='Gap' if bin_lower == 0 else "")

    # 理想校准对角线
    plt.plot([0, 100], [0, 100], '--', color='gray')

    # plt.xlabel("Confidence", fontsize=16)
    # plt.ylabel("Accuracy", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/figsInPaper/figS3/{args.dataset}_(ctx){args.ctx_dataset}_ece.png", bbox_inches='tight', dpi=300)
    plt.show()
    return ece

def compute_entropy(preds_prob):
    """
    :param preds_prob: 预测概率（经过Softmax） [N, K]
    :return:
    """
    entropy = - np.sum(preds_prob * np.log(preds_prob + eps), axis=1)
    return entropy

def selective_acc(p, Y):
    """
    计算 Selective Prediction Accuracy 和其对应的 AUC 曲线（根据预测熵阈值）

    :param p: ndarray, shape (B, d)，每个样本的分类概率分布
    :param Y: ndarray, shape (B,)，每个样本的真实标签
    :return:
        auc_sel_id: float，预测选择曲线的面积（AUC）
        thresholded_accuracies: list[float]，不同阈值下的准确率
    """
    # 创建101个阈值：从100到1的百分位，再加一个0.1
    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p.argmax(axis=-1)  # shape: (B,)
    accuracies_test = (predictions_test == Y).astype(np.float32)  # shape: (B,)
    scores_id = compute_entropy(p)  # shape: (B,)，预测的不确定性（熵）

    thresholded_accuracies = []

    for threshold in thresholds:
        perc = np.percentile(scores_id, threshold)  # 计算scores_id的百分位
        mask = (scores_id <= perc)  # 保留“更确定”的样本

        if np.sum(mask) == 0:
            mean_accuracy = 0.0
        else:
            masked_accuracies = accuracies_test * mask.astype(np.float32)
            mean_accuracy = masked_accuracies.sum() / mask.sum()

        thresholded_accuracies.append(mean_accuracy)

    thresholded_accuracies = np.array(thresholded_accuracies)

    # 计算 AUC（选择曲线下面积）
    auc_sel = 0.0
    for i in range(len(thresholds) - 1):
        if i == 0:
            x = 100 - thresholds[i + 1]
        else:
            x = thresholds[i] - thresholds[i + 1]
        auc_sel += (x * thresholded_accuracies[i] + x * thresholded_accuracies[i + 1]) / 2.0

    return auc_sel, thresholded_accuracies

def plot_train_loader_2d(train_loader, num_bg_points=5000):
    """
    可视化 train_loader 中的二维高斯数据分布，模仿双高斯环状图示。

    :param train_loader: PyTorch 的 DataLoader，包含二维数据和标签
    :param num_bg_points: 背景灰色点的数量，用于构造密度背景
    """
    all_data = []
    all_labels = []

    # 收集所有数据
    for batch_data, batch_labels in train_loader:
        all_data.append(batch_data)
        all_labels.append(batch_labels)
    data = torch.cat(all_data, dim=0).numpy()  # [N, 2]
    labels = torch.cat(all_labels, dim=0).numpy()  # [N]

    # 绘制背景灰色散点，用于体现密度
    if data.shape[0] > num_bg_points:
        idx = np.random.choice(data.shape[0], num_bg_points, replace=False)
    else:
        idx = np.arange(data.shape[0])
    bg_data = data[idx]

    plt.scatter(bg_data[:, 0], bg_data[:, 1], s=1, color='gray', alpha=0.3)  # 灰色背景点

    # 绘制红色星形采样点，区分两个类别
    class0 = data[labels == 0]
    class1 = data[labels == 1]
    plt.scatter(class0[:, 0], class0[:, 1], marker='o', color='blue', edgecolors='black', s=40, alpha=0.2)
    plt.scatter(class1[:, 0], class1[:, 1], marker='o', color='red', edgecolors='black', s=40, alpha=0.2)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.show()

def plot_reject_ood(ood_samples):
    ood_samples = ood_samples.detach().cpu().numpy()

    plt.scatter(ood_samples[:, 0], ood_samples[:, 1], marker='x', alpha=0.5, s=10, c='green')

def plot_decision_bound(model):
    """
    可视化模型的决策边界
    :param model: PyTorch 模型
    """
    # 生成网格数据
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid_data = np.c_[xx1.ravel(), xx2.ravel()]

    # 转换为 tensor 并进行预测
    grid_tensor = torch.tensor(grid_data, dtype=torch.float32).to('cuda')
    with torch.no_grad():
        pred_probs = torch.nn.functional.softmax(model(grid_tensor), dim=1)[:, 1].cpu().numpy()

    # 绘制决策边界
    plt.contourf(xx1, xx2, pred_probs.reshape(xx1.shape), alpha=0.5, levels=np.linspace(0, 1, 20), cmap='coolwarm')

    # 绘制颜色条
    plt.colorbar(label='Predicted Probability')