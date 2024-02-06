from __future__ import print_function
import torch.utils.data
import matplotlib.pyplot as plt
import csv
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []

def train(epoch):
    # Sets the module in training mode.
    model.train()
    train_loss = 0

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)  # [64, 1, 28, 28]

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label)
        # 训练样本展平，在每个样本后面连接标签的one-hot向量
        flat_data = data.view(-1, data.shape[2] * data.shape[3])
        y_condition = model.to_categorical(label)
        y_condition = y_condition.to(device)
        con = torch.cat((flat_data, y_condition), 1)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # 计算平均训练损失并将其添加到列表中
    average_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(average_train_loss)
    print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    # Sets the module in evaluation mode
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            recon_batch, mu, logvar = model(data, label)

            flat_data = data.view(-1, data.shape[2] * data.shape[3])

            y_condition = model.to_categorical(label)
            y_condition = y_condition.to(device)
            con = torch.cat((flat_data, y_condition), 1)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.4f}'.format(test_loss))


model = CVAE_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
for epoch in range(1, 150):
    train(epoch)
    if epoch % 20 == 0:
        test(epoch)
    #scheduler.step()
torch.save(model.state_dict(), 'cvae_model.pth')


# 训练结束后将 train_losses 保存到 CSV 文件
with open('./train_loss/train_losses cvae_cnn z_dim=10.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss'])
    for epoch, loss in enumerate(train_losses, 1):
        writer.writerow([epoch, loss])
