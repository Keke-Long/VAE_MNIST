import os
import numpy as np
import torch
from model import *
from data_loader import get_data_loaders
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

number = 7
num_cluster = 35
images_to_plot = 500
scale_factor = 0.5  # 缩小比例


train_loader, test_loader = get_data_loaders(train_batch_size=512, test_batch_size=512)

model = VAE_CNN().to(device)
model.load_state_dict(torch.load('vae_model_state.pth'))

latent_vectors = []  # 用来存储标签为1的样本的潜在向量
indices_for_1 = []  # 用来存储标签为1的样本的全局索引

current_index = 0  # 初始化当前的全局索引
for data, labels in test_loader:
    data = data.to(device)
    labels_cpu = labels.to('cpu').numpy()
    with torch.no_grad():
        mu, _ = model.encoder(data)
        z = mu.to('cpu').numpy()  # 获取潜在向量
    is_label_1 = labels_cpu == number
    latent_vectors.extend(z[is_label_1])  # 只保存标签为1的潜在向量

    # 计算并保存标签为1的样本的全局索引
    indices_for_this_batch = np.where(is_label_1)[0] + current_index
    indices_for_1.extend(indices_for_this_batch)

    current_index += len(labels)  # 更新当前的全局索引以反映这个批次的大小

latent_vectors = np.array(latent_vectors)


# kmeans聚类
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(latent_vectors)
cluster_labels = kmeans.labels_

# 聚类可视化
for cluster_id in range(num_cluster):
    selected_cluster_indices = np.where(cluster_labels == cluster_id)[0]

    # 从标签为1的样本索引中选择对应聚类的索引
    selected_original_indices = np.array(indices_for_1)[selected_cluster_indices]

    # 如果选中的数据点超过500个，就只取前500个
    if len(selected_original_indices) > images_to_plot:
        selected_original_indices = selected_original_indices[:images_to_plot]

    # 可视化
    fig, axes = plt.subplots(nrows=20, ncols=25, figsize=(25 * scale_factor, 20 * scale_factor))
    axes = axes.flatten()
    for ax_idx, data_idx in enumerate(selected_original_indices):
        if ax_idx >= images_to_plot:
            break
        image, _ = test_loader.dataset[data_idx]
        image = image.squeeze()  # 假设图像是单通道的
        ax = axes[ax_idx]
        ax.imshow(image.numpy(), cmap='gray')
        ax.axis('off')
    for ax in axes[len(selected_original_indices):]: # 隐藏多余的子图
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'./results latent space cluster/KMeans z_dim=10/num{number}/cluster_{num_cluster}_{cluster_id}.png')
    plt.close(fig)



# # 应用DBSCAN进行聚类
# dbscan = DBSCAN(eps=0.6, min_samples=5)  # eps和min_samples根据你的数据进行调整
# cluster_labels = dbscan.fit_predict(latent_vectors)
#
# # 获取唯一的聚类标签（排除噪声点，噪声点标签为-1）
# unique_clusters = set(cluster_labels)
# if -1 in unique_clusters:
#     unique_clusters.remove(-1)  # 如果不想可视化噪声点，去除标签为-1的聚类
#
# # 聚类可视化
# for cluster_id in unique_clusters:
#     selected_cluster_indices = np.where(cluster_labels == cluster_id)[0]
#
#     # 从标签为1的样本索引中选择对应聚类的索引
#     selected_original_indices = np.array(indices_for_1)[selected_cluster_indices]
#
#     # 如果选中的数据点超过500个，就只取前500个
#     images_to_plot = min(len(selected_original_indices), 500)
#
#     # 可视化
#     nrows, ncols = 20, 25
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25 * scale_factor, 20 * scale_factor))
#     axes = axes.flatten()
#     for ax_idx, data_idx in enumerate(selected_original_indices[:images_to_plot]):
#         image, _ = test_loader.dataset[data_idx]
#         image = image.squeeze()
#         ax = axes[ax_idx]
#         ax.imshow(image.numpy(), cmap='gray')
#         ax.axis('off')
#     for ax in axes[images_to_plot:]: # 隐藏多余的子图
#         ax.axis('off')
#     plt.tight_layout()
#     folder_path = f'./results latent space cluster/DBSCAN z_dim=10/num{number}'
#     os.makedirs(folder_path, exist_ok=True)
#     plt.savefig(f'{folder_path}/cluster_{cluster_id}.png')
#     plt.close(fig)