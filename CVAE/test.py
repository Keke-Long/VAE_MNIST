from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import *
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model = CVAE_CNN().to(device)
# model.load_state_dict(torch.load('cvae_model.pth'))
# model.eval()  # 将模型设置为评估模式
#
# with torch.no_grad():
#     # 获取一个原始样本和对应标签
#     original_sample, original_label = next(iter(test_loader))
#     original_sample = original_sample[0].unsqueeze(0).to(device)
#     original_label = original_label[0].unsqueeze(0).to(device)
#
#     # 使用CVAE编码器从原始图像中提取特征
#     mu, logvar = model.encoder(original_sample.view(-1, 28*28), original_label)
#     z = model.reparameterize(mu, logvar)
#
#     # 创建一个大图
#     fig, axes = plt.subplots(1, 11, figsize=(11, 1))
#
#     # 显示原始样本
#     original_image = original_sample[0].cpu().numpy().squeeze()
#     axes[0].imshow(original_image, cmap='gray')
#     axes[0].set_title('Original')
#     axes[0].axis('off')
#
#     # 使用提取的特征和不同的标签生成新图像
#     for i in range(10):
#         new_label = torch.tensor([i], dtype=torch.float32).to(device)
#         generated_sample = model.decoder(z, new_label).cpu()
#         #generated_image = generated_sample[:, :-10] # 去掉每一行末尾的 one-hot 向量并转换为图像
#         axes[i + 1].imshow(generated_sample.view(28, 28).squeeze().numpy(), cmap='gray')
#         axes[i + 1].set_title(f'{i}')
#         axes[i + 1].axis('off')
#
#     plt.tight_layout()
#     plt.savefig('CVAE_CNN result.jpg')




model = CVAE_CNN().to(device)
model.load_state_dict(torch.load('cvae_model.pth'))
model.eval()  # 设置为评估模式

model.eval()
images_list = []
labels_list = []
mus = []
ys = []
with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        mu, _ = model.encoder(data.view(-1, 28*28), labels)

        images_list.append(data.cpu())
        labels_list.append(labels.cpu())

        mus.append(mu)
        ys.append(labels)
mus = torch.cat(mus, dim=0).cpu().numpy()
ys = torch.cat(ys, dim=0).cpu().numpy()

images_list = torch.cat(images_list, dim=0).cpu().numpy()
labels_list = torch.cat(labels_list, dim=0).cpu().numpy()

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
mus_tsne = tsne.fit_transform(mus)

# 创建一个图来展示t-SNE降维后的图像
fig, ax = plt.subplots(figsize=(30, 30), dpi=350)  # 调整画布的大小
for i in range(1, len(mus_tsne), 15):

    x, y = mus_tsne[i, :]  # t-SNE的结果作为坐标
    img_data = images_list[i].reshape(28, 28)
    img_data_inverted = 1.0 - img_data # 反转图像颜色：黑底变为白底

    # 创建RGBA图像
    img_rgba = np.zeros((28, 28, 4), dtype=np.uint8)
    img_rgba[..., :3] = img_data_inverted[:, :, None] * 255  # 设置RGB为反转后的值
    # 设置透明度：纯白(1.0)或接近白色的像素变为透明，其他保持不变
    threshold = 0.75  # 定义阈值，高于此亮度的像素将被视为透明
    img_rgba[..., 3] = np.where(img_data_inverted > threshold, 0, 255)

    # 使用PIL创建图像
    img = Image.fromarray(img_rgba, 'RGBA')
    img = OffsetImage(img, zoom=0.2)  # zoom控制图像大小
    # 创建注释框放置图像
    ab = AnnotationBbox(img, (x, y), frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)

ax.set_xlim(min(mus_tsne[:, 0]), max(mus_tsne[:, 0]))
ax.set_ylim(min(mus_tsne[:, 1]), max(mus_tsne[:, 1]))
ax.axis('off')
plt.savefig('CVAE_CNN t_SNE all number image.png', dpi=350)