import torch
from torchvision import datasets, transforms
import os

def get_data_loaders(train_batch_size, test_batch_size, data_dir='../MNIST_data/'):
    # 检查数据目录是否存在，如果不存在，则创建
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader
