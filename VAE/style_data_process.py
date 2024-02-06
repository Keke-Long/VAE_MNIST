import matplotlib.pyplot as plt
import torchvision
import numpy as np
from data_loader import get_data_loaders
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)

def imshow(img):
    plt.imshow(img, cmap='gray')
    plt.show()

# # 遍历数据集并可视化每个数字的几个样本
# for i in range(10):
#     # 找到属于类别 i 的所有图像索引
#     indices = torch.where(dataset.targets == i)[0]
#     # 从每个类别中随机选择 5 个样本进行可视化
#     sample_indices = torch.randperm(len(indices))[:5]
#     for idx in sample_indices:
#         image, label = dataset[indices[idx]]
#         plt.figure()  # 创建一个新的图形
#         plt.imshow(image.squeeze(), cmap='gray')  # 显示图像
#         plt.savefig(f'/media/ubuntu/U_KOKO/mnist/Label {label}_{idx}.jpg')
#         plt.close()  # 关闭图形，避免在屏幕上显示


def save_tensor_image(label, idx):
    indices = torch.where(dataset.targets == label)[0] # 找到属于给定标签的所有图像索引
    if idx >= len(indices):
        print(f"索引{idx}超出了标签{label}的图像范围。")
        return
    image_tensor, _ = dataset[indices[idx]] # 获取特定索引的图像张量
    torch.save(image_tensor, f'./x_R_pictures/{label}_{idx}.pt')

# L
# save_tensor_image(0, 1826)
# save_tensor_image(1, 4833)
# save_tensor_image(2, 732)
# save_tensor_image(3, 3098)
# save_tensor_image(4, 5656)
# save_tensor_image(5, 4400)
# save_tensor_image(6, 3837)
# save_tensor_image(7, 750)
# save_tensor_image(8, 4981)
# save_tensor_image(9, 4634)

# R
save_tensor_image(0, 3849)
save_tensor_image(1, 2215)
save_tensor_image(2, 2077)
save_tensor_image(3, 2635)
save_tensor_image(4, 872)
save_tensor_image(5, 5052)
save_tensor_image(6, 3922)
save_tensor_image(7, 2677)
save_tensor_image(8, 4475)
save_tensor_image(9, 1350)


# 图像张量合并到一个单独的大文件x_L_all_images.pt
# folder_path = './x_L_pictures/'
# images = [None] * 10  # 假设我们知道有10个图像，标签从0到9
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.pt'):
#         label = int(file_name.split('_')[0])
#         image_tensor = torch.load(os.path.join(folder_path, file_name))
#         images[label] = image_tensor
# torch.save(images, 'x_L_all_images.pt')