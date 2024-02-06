import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from model import *
from data_loader import get_data_loaders
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def test_encoding_decoding(model, x):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        z_mu, z_logvar = model.encoder(x)# 编码
        x_decoded = model.decoder(model.sampling(z_mu, z_logvar))# 解码
        x_decoded = x_decoded.view(1, 28, 28)# 还原图像的形状
        return x_decoded


def generate_intermediate_image(model, x_L, x_R, lamda):
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        # 编码
        z_L_mu, z_L_logvar = model.encoder(x_L.view(-1, 1, 28, 28))
        z_R_mu, z_R_logvar = model.encoder(x_R.view(-1, 1, 28, 28))

        # 从均值和对数方差获得潜在表示
        z_L = model.sampling(z_L_mu, z_L_logvar)
        z_R = model.sampling(z_R_mu, z_R_logvar)

        # 加权求和以生成中间潜在表示
        z_intermediate = (1 - lamda) * z_L + lamda * z_R

        # 解码
        x_intermediate = model.decoder(z_intermediate)

        return x_intermediate.view(1, 28, 28)


x_L = torch.load('x_L_all_images.pt')
x_R = torch.load('x_R_all_images.pt')
x_L = [image.to(device) for image in x_L]
x_R = [image.to(device) for image in x_R]

model = VAE_CNN().to(device)
model.load_state_dict(torch.load('vae_model_state.pth'))


# 画图 style插值结果
# fig, axs = plt.subplots(10, 11, figsize=(11, 10))  # 10行, 11列
# for i in range(10):
#     for j, lamda in enumerate([k / 10.0 for k in range(11)]):
#         img = generate_intermediate_image(model, x_L[i], x_R[i], lamda)
#         img = img.cpu().numpy().squeeze()  # 将图像转换为NumPy数组
#         axs[i, j].imshow(img, cmap='gray')
#         axs[i, j].axis('off')
# plt.tight_layout()
# plt.savefig('VAE result.jpg')




def plot_latent_space(model, data_loader, n_components=2):
    model.eval()
    mus = []
    ys = []
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            mu, _ = model.encoder(data.view(-1, 1, 28, 28))
            mus.append(mu)
            ys.append(labels)
    mus = torch.cat(mus, dim=0).cpu().numpy()
    ys = torch.cat(ys, dim=0).cpu().numpy()

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=n_components, random_state=0)
    mus_tsne = tsne.fit_transform(mus)

    return mus_tsne, ys

# t—SNE隐空间可视化
train_loader, test_loader = get_data_loaders(train_batch_size=512, test_batch_size=512)
mus_tsne, ys = plot_latent_space(model, test_loader)
plt.figure(figsize=(10, 8))
plt.scatter(mus_tsne[:, 0], mus_tsne[:, 1], c=ys, cmap='viridis', s=2, alpha=0.6)
plt.colorbar()
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig('2D plot of latent space using t-SNE.jpg')






# 隐空间可视化（带图像）
train_loader, test_loader = get_data_loaders(train_batch_size=512, test_batch_size=512)

model.eval()
images = []
labels = []
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        mu, _ = model.encoder(data.view(-1, 1, 28, 28))
        images.append(data.cpu())
        labels.append(label.cpu())
images = torch.cat(images, dim=0).cpu().numpy()
labels = torch.cat(labels, dim=0).cpu().numpy()

# 使用t-SNE进行降维
mus = images.reshape(images.shape[0], -1)  # 将图像展平
tsne = TSNE(n_components=2, random_state=0)
mus_tsne = tsne.fit_transform(mus)


# 创建一个图来展示t-SNE降维后的图像
fig, ax = plt.subplots(figsize=(30, 30), dpi=350)  # 调整画布的大小

for i in range(len(mus_tsne)):
    # 选择特定数字显示
    if labels[i] != 8:
        continue

    x, y = mus_tsne[i, :]  # t-SNE的结果作为坐标
    img_data = images[i].reshape(28, 28)
    img_data_inverted = 1.0 - img_data # 反转图像颜色：黑底变为白底

    # 创建RGBA图像
    img_rgba = np.zeros((28, 28, 4), dtype=np.uint8)
    img_rgba[..., :3] = img_data_inverted[:, :, None] * 255  # 设置RGB为反转后的值
    # 设置透明度：纯白(1.0)或接近白色的像素变为透明，其他保持不变
    threshold = 0.75  # 定义阈值，高于此亮度的像素将被视为透明
    img_rgba[..., 3] = np.where(img_data_inverted > threshold, 0, 255)

    # 使用PIL创建图像
    img = Image.fromarray(img_rgba, 'RGBA')
    img = OffsetImage(img, zoom=0.25)  # zoom控制图像大小
    # 创建注释框放置图像
    ab = AnnotationBbox(img, (x, y), frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)

ax.set_xlim(min(mus_tsne[:, 0]), max(mus_tsne[:, 0]))
ax.set_ylim(min(mus_tsne[:, 1]), max(mus_tsne[:, 1]))
ax.axis('off')
# plt.savefig('label7 new.png')
plt.show()

