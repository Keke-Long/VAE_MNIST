import csv
import matplotlib.pyplot as plt

# 读取 CSV 文件并提取损失值
def read_losses(filename):
    losses = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            losses.append(float(row[1]))
    return losses

# 读取数据
vae_losses = read_losses('train_losses vae.csv')
vae_cnn_losses = read_losses('train_losses vae_cnn.csv')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(vae_losses, label='VAE Losses')
plt.plot(vae_cnn_losses, label='VAE_CNN Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.savefig('comparison_loss_plot.png')
