import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import csv
import matplotlib.pyplot as plt
from model import *
from data_loader import get_data_loaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []

def imshow(img):
    img = img.cpu()  # img是一个PyTorch张量，形状为[1, 28, 28] # 去除批次维度
    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    plt.show()


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # 计算平均训练损失并将其添加到列表中
    average_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(average_train_loss)
    print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


train_loader, test_loader = get_data_loaders(train_batch_size=512, test_batch_size=512)
train_dataset_size = len(train_loader.dataset)
test_dataset_size = len(test_loader.dataset)
print(f"Training dataset size: {train_dataset_size}")
print(f"Testing dataset size: {test_dataset_size}")

model = VAE_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.05)
for epoch in range(1, 200):
    train(epoch)
    scheduler.step()
torch.save(model.state_dict(), 'vae_model_state.pth') # 保存模型参数


# 在训练结束后将 train_losses 保存到 CSV 文件
with open('train_losses vae.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss'])
    for epoch, loss in enumerate(train_losses, 1):
        writer.writerow([epoch, loss])
