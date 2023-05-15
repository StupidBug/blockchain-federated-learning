import torch.nn as nn


# 继承 nn.Module 类
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleCNN, self).__init__()

        # 定义两个卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数为3，输出通道数为6，卷积核大小为5
        self.pool = nn.MaxPool2d(2, 2)  # 池化窗口大小为2，步长大小为2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道数为6，输出通道数为16，卷积核大小为5

        # 定义三个全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # 输入特征数为input_dim，输出特征数为hidden_dims[0]
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # 输入特征数为hidden_dims[0]，输出特征数为hidden_dims[1]
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)  # 输入特征数为hidden_dims[1]，输出特征数为output_dim

    def forward(self, x):
        # 卷积层1和池化层1
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # 卷积层2和池化层2
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # 扁平化，变成一维向量
        x = x.view(x.size(0), -1)
        # 全连接层1和激活函数
        x = nn.functional.relu(self.fc1(x))
        # 全连接层2和激活函数
        x = nn.functional.relu(self.fc2(x))
        # 输出层，不需要激活函数
        x = self.fc3(x)
        return x


class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelIndicator:
    def __init__(self, accuracy, f1_score, loss):
        self.accuracy = accuracy
        self.f1_score = f1_score
        self.loss = loss


class ModelBuilder:
    @staticmethod
    def build_pathmnist_cnn():
        return SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=9)

    @staticmethod
    def build_cifar10_cnn():
        return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
