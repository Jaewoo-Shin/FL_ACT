import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import cuda
import numpy as np
import torch

'''
<Reference>

ResNet for Cifar-100 is from:
[1] Clova AI Research, GitHub repository, https://github.com/clovaai/overhaul-distillation
'''

__all__=['resnet8']


def simple_tanh(input):
    device = torch.device(input.device)
    size = input.shape
    ones = torch.ones(size).to(device)
    minusones = torch.ones(size).to(device) * -1
    output = torch.minimum(ones, torch.maximum(input, minusones))
    return output
class Itself(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class STanh(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return simple_tanh(input)




def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes,activation, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'simple_tanh':
            self.activation = STanh()
        elif activation == 'none':
            self.activation = Itself()
        # self.activation = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.activation(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, activation, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'sigmoid':
#             self.activation = nn.Sigmoid()
#         elif activation == 'leaky_relu':
#             self.activation = nn.LeakyReLU()
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'simple_tanh':
#             self.activation = STanh()
#         elif activation == 'none':
#             self.activation = Itself()

        self.activation = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.activation(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, width, depth, num_classes, activation, bottleneck=False):
        super(ResNet, self).__init__()
        self.inplanes = 16 * width
        if bottleneck == True:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock
        
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, activation, 16* width, n)
        self.layer2 = self._make_layer(block, activation, 32* width, n, stride=2)
        self.layer3 = self._make_layer(block, activation, 64* width, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * width * block.expansion, num_classes)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'simple_tanh':
            self.activation = STanh()
        elif activation == 'none':
            self.activation = Itself()


    def _make_layer(self, block, activation, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, activation, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, activation, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.activation(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            print('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def get_channel_num(self):

        return [16, 32, 64]

    def extract_feature(self, x, preReLU=True):

        x = self.conv1(x)
        x = self.bn1(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        x = self.activation(feat3)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        if not preReLU:
            feat1 = self.activation(feat1)
            feat2 = self.activation(feat2)
            feat3 = self.activation(feat3)

#         return [feat1, feat2, feat3], out
        return out


def resnet8(num_classes, activation):
    model = ResNet(width=1, depth=8, num_classes=num_classes, activation=activation, bottleneck=False)
    
    return model