import os
import csv
import matplotlib.pyplot as plt

def read_and_plot_losses(directory):
    plt.figure(figsize=(10, 6))

    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 确保只处理CSV文件
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            losses = []
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # 跳过标题行
                for row in reader:
                    losses.append(float(row[1]))

            # 绘制损失曲线，使用文件名（不包含.csv）作为标签
            plt.plot(losses, label=filename[:-4])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

# 调用函数，传入包含损失CSV文件的目录
read_and_plot_losses('./train_loss/')