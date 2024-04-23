import torch
import torch.nn as nn
import torch.nn.functional as F

# 分类CNN网络, 用于CIFAR-10数据集


class MyNet3(nn.Module):
    # input: 3x32x32
    def __init__(self, num_classes=10, dropout_rate=0) -> None:
        super(MyNet3, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5,  num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 裁剪了卷积核和线性层尺寸
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 调整了Linear层的尺寸
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net = MyNet1()


class Alexnet(nn.Module):
    # input: 3x32x32
    def __init__(self, num_classes=10, dropout_rate=0):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=2),  # 32x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 32x7x7

            nn.Conv2d(32, 96, kernel_size=5, padding=2),  # 96x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),  # 192x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 128x7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x3x3
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128*3*3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg_block(num_convs, in_channels, out_channels):
    # 一个 vgg 的 block，传入三个参数，第一个是模型层数，第二个是输入的通道数，第三个是输出的通道数，
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     padding=1), nn.ReLU(True)]  # 定义第一层

    for i in range(num_convs-1):
        net.append(nn.Conv2d(out_channels, out_channels,
                   kernel_size=3, padding=1))
        net.append(nn.ReLU(True))

    net.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*net)


def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vgg_stack(
            (1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
