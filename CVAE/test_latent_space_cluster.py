import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

number = 1
num_cluster = 25
images_to_plot = 500
scale_factor = 0.5  # 缩小比例


# 加载模型
model = CVAE_CNN().to(device)
model.load_state_dict(torch.load('cvae_model.pth'))
model.eval()  # 设置为评估模式


# 遍历测试数据集，提取标签为1的样本的隐向量
latent_vectors = []  # 用于存储隐向量
indices_for_1 = []  # 用于存储标签为1的样本索引
for data, labels in test_loader:
    data, labels = data.to(device), labels.to(device)
    is_label_1 = (labels == number)
    if is_label_1.any():
        data, labels = data[is_label_1], labels[is_label_1]
        with torch.no_grad():
            mu, _ = model.encoder(data.view(-1, 28 * 28), labels)
            latent_vectors.extend(mu.cpu().numpy())
latent_vectors = np.array(latent_vectors)


# KMeans聚类
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(latent_vectors)
cluster_labels = kmeans.labels_

# 聚类可视化
# for cluster_id in range(num_cluster):
#     selected_indices = np.where(cluster_labels == cluster_id)[0]
#     if len(selected_indices) == 0:
#         continue  # 如果聚类中没有样本，则跳过
#
#     # 可视化
#     fig, axes = plt.subplots(nrows=20, ncols=25, figsize=(25 * scale_factor, 20 * scale_factor))
#     axes = axes.flatten()
#     for i, idx in enumerate(selected_indices[:500]):  # 最多可视化500个样本
#         z = torch.from_numpy(latent_vectors[idx]).unsqueeze(0).to(device)
#         with torch.no_grad():
#             recon = model.decoder(z, torch.tensor([[number]]).float().to(device)).cpu()
#         image = recon.view(28, 28).numpy()
#         ax = axes[i]
#         ax.imshow(image, cmap='gray')
#         ax.axis('off')
#     for ax in axes[len(selected_indices):]:
#         ax.axis('off')
#     plt.tight_layout()
#     folder_path = f'./results latent space cluster/KMeans z_dim=15/num{number}'
#     os.makedirs(folder_path, exist_ok=True)
#     plt.savefig(f'{folder_path}/cluster_{num_cluster}_{cluster_id}.png')
#     plt.close(fig)


# t-SNE进行降维后画图 2维
tsne = TSNE(n_components=2, random_state=0)
latent_vectors_2d = tsne.fit_transform(latent_vectors)
colors = plt.cm.rainbow(np.linspace(0, 1, num_cluster))
plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.83, top=0.9, bottom=0.1)
for cluster_id in range(num_cluster):
    indices = np.where(cluster_labels == cluster_id)
    plt.scatter(latent_vectors_2d[indices, 0], latent_vectors_2d[indices, 1],
                color=colors[cluster_id], label=f'Cluster {cluster_id}', alpha=0.7, edgecolor='none', s=15)
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.legend(markerscale=2., bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f'./results latent space cluster/KMeans z_dim=15/num{number}/t-SNE Visualization of Latent Space.png', dpi=300)



# t-SNE进行降维后画图 3维
# tsne = TSNE(n_components=3, random_state=0)
# latent_vectors_3d = tsne.fit_transform(latent_vectors)
# colors = plt.cm.rainbow(np.linspace(0, 1, num_cluster))
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# for cluster_id in range(num_cluster):
#     indices = np.where(cluster_labels == cluster_id)
#     ax.scatter(latent_vectors_3d[indices, 0], latent_vectors_3d[indices, 1], latent_vectors_3d[indices, 2], color=colors[cluster_id], label=f'Cluster {cluster_id}', alpha=0.5)
# ax.set_xlabel('t-SNE Feature 1')
# ax.set_ylabel('t-SNE Feature 2')
# ax.set_zlabel('t-SNE Feature 3')
# ax.set_title('3D t-SNE Visualization of Latent Space')
# plt.legend(markerscale=2., loc='upper left', frameon=False)
# plt.show()