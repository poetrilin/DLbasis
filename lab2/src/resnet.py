import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ShortcutProjection(nn.Module):
    """Projection shortcut in the bottleneck block"""

    def __init__(self, in_channels, out_channels, stride):
        super(ShortcutProjection, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, model_path=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        assert (model_path is not None)
        pretrained_dict = torch.load(model_path)
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    if num_classes != 1000:
        inchann = model.fc.in_features
        model.fc = nn.Linear(in_features=inchann,
                             out_features=num_classes, bias=True)
    return model


def resnet34(pretrained=False, model_path=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert (model_path is not None)
        pretrained_dict = torch.load(model_path)
        model.load_state_dict(pretrained_dict)
    if num_classes != 1000:
        inchann = model.fc.in_features
        model.fc = nn.Linear(in_features=inchann,
                             out_features=num_classes, bias=True)
    return model


def resnet50(pretrained=False, model_path=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert (model_path is not None)
        pretrained_dict = torch.load(model_path)
        model.load_state_dict(pretrained_dict)
    if num_classes != 1000:
        inchann = model.fc.in_features
        model.fc = nn.Linear(in_features=inchann,
                             out_features=num_classes, bias=True)
    return model


def resnet101(pretrained=False, model_path=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        assert (model_path is not None)
        pretrained_dict = torch.load(model_path)
        model.load_state_dict(pretrained_dict)
    if num_classes != 1000:
        inchann = model.fc.in_features
        model.fc = nn.Linear(in_features=inchann,
                             out_features=num_classes, bias=True)
    return model


def resnet152(pretrained=False, model_path=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        assert (model_path is not None)
        pretrained_dict = torch.load(model_path)
        model.load_state_dict(pretrained_dict)
    if num_classes != 1000:
        inchann = model.fc.in_features
        model.fc = nn.Linear(in_features=inchann,
                             out_features=num_classes, bias=True)
    return model


_model_factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}


def create_model(arch, pretrained=False, model_path=None, num_classes=1000):
    get_model = _model_factory[arch]
    model = get_model(pretrained=pretrained,
                      model_path=model_path, num_classes=num_classes)
    return model


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters()
              if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' %
          ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        # print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


if __name__ == '__main__':
    model_path = 'models/resnet18.pth'

    resnet = create_model('res18')
    print(resnet.fc.weight)
    resnet = create_model('res18', pretrained=True, model_path=model_path)
    print(resnet.fc.weight)
    print(resnet.fc.weight.shape)
    resnet = create_model('res18', pretrained=True,
                          model_path=model_path, num_classes=2)
    print(resnet.fc.weight)
    print(resnet.fc.weight.shape)

    model_info(resnet)
    # resnet = resnet18(pretrained=True, model_path=model_path, num_classes = 1000)
    # 固定除了全连接层所有层的权重，反向传播时将不会计算梯度
    for param in resnet.parameters():
        param.requires_grad = False
    for param in resnet.fc.parameters():
        param.requires_grad = True

    model_info(resnet)
    inchann = resnet.fc.in_features
    # 重新定义全连接层 预训练模型在ImageNet上为1000类，因此需要根据自己数据集的类别进行改动
    # 本例子单独训练该全连接层部分（也可以固定底层几层，其余部分参数都进入训练）
    resnet.fc = nn.Linear(in_features=inchann, out_features=2, bias=True)
    # print(resnet)
