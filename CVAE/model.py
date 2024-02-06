import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelBinarizer


class CVAE(nn.Module):
    def __init__(self, input_dim=794, hidden_dim=512, z_dim=20, output_dim=784):
        super(CVAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        # Decoder layers
        self.fc3 = nn.Linear(z_dim + 10, hidden_dim)  # Added 10 for one-hot labels
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        self.lb = LabelBinarizer()

    # 将标签进行one-hot编码
    def to_categorical(self, y: torch.FloatTensor):
        y = y.cpu()
        y_n = y.numpy()
        self.lb.fit(list(range(0, 10)))
        y_one_hot = self.lb.transform(y_n)
        floatTensor = torch.FloatTensor(y_one_hot)
        return floatTensor

    def encoder(self, x, y):
        y_c = self.to_categorical(y)
        y_c = y_c.to(x.device)
        # 输入样本和标签y的one-hot向量连接
        con = torch.cat((x, y_c), 1)
        h1 = F.relu(self.fc1(con))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, z, y):
        y_c = self.to_categorical(y)
        y_c = y_c.to(z.device)
        cat = torch.cat((z, y_c), 1) # 解码器的输入：将z和y的one-hot向量连接
        h3 = F.relu(self.fc3(cat))
        return F.sigmoid(self.fc4(h3)).view(-1, 1, 28, 28)

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar



class CVAE_CNN(nn.Module):
    def __init__(self, h_dim1=256, h_dim2=256, z_dim=5, dropout_rate=0):
        super(CVAE_CNN, self).__init__()

        # Encoder layers
        self.encoder_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7 + 10, h_dim1)  # Added 10 for one-hot labels
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.dropout = nn.Dropout(dropout_rate)
        # Decoder layers
        self.fc4 = nn.Linear(z_dim+10, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, 64 * 7 * 7)
        self.decoder_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        self.lb = LabelBinarizer()

    def to_categorical(self, y: torch.FloatTensor): # 将标签进行one-hot编码
        y = y.cpu()
        y_n = y.numpy()
        self.lb.fit(list(range(0, 10)))
        y_one_hot = self.lb.transform(y_n)
        floatTensor = torch.FloatTensor(y_one_hot)
        return floatTensor.to(y.device)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encoder(self, x, y):
        y_c = self.to_categorical(y)
        y_c = y_c.to(x.device)
        x = x.view(-1, 1, 28, 28)  # 将输入重新调整为图像的形状
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x = x.view(-1, 64 * 7 * 7)  # 将卷积结果展平
        con = torch.cat((x, y_c), 1) # 输入样本和标签y的one-hot向量连接
        h = F.relu(self.fc1(con))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def decoder(self, z, y):
        y_c = self.to_categorical(y)
        y_c = y_c.to(z.device)
        cat = torch.cat((z, y_c), 1) # 解码器的输入：将z和y的one-hot向量连接
        h = F.relu(self.fc4(cat))
        h = self.dropout(h)
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        h = h.view(-1, 64, 7, 7)
        h = torch.sigmoid(self.decoder_conv1(h))
        return torch.sigmoid(self.decoder_conv2(h))

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar


def loss_function(recon_x, x, mu, log_var):
    # print("Shape of recon_x:", recon_x.shape)  # 打印重建图像的形状
    # print("Shape of x:", x.shape)  # 打印原始图像的形状
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../MNIST_data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=512, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../MNIST_data', train=False, transform=transforms.ToTensor()),
    batch_size=512, shuffle=True, **kwargs)