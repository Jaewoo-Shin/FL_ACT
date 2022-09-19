import torch
import torch.nn as nn
import torch.nn.functional as F

def simple_tanh(input):
    device = torch.device(input.device)
    size = input.shape
    ones = torch.ones(size).to(device)
    minusones = torch.ones(size).to(device) * -1
    output = torch.minimum(ones, torch.maximum(input, minusones))
    return output

def simple_tanhv2(input):
    device = torch.device(input.device)
    size = input.shape
    ones = torch.ones(size).to(device)
    minusones = torch.ones(size).to(device) * -1
    output = torch.minimim(ones, torch.maximum(input * 0.5, minusones))
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

class STanhv2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return simple_tanhv2(input)

class simpleNet3(nn.Module):
    def __init__(self, num_classes = 100, num_filter = 16, activation = 'relu'):
        super(simpleNet3, self).__init__()
        num_filter = int(num_filter)
        self.conv1 = nn.Conv2d(3, num_filter, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(num_filter*2*16*16 , num_classes)
        self.ceriation = nn.CrossEntropyLoss()
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
        elif activation == 'simple_tanhv2':
            self.activation = STanhv2()
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = torch.flatten(x,1)

        out = self.fc(x)
        return out


class simpleNet4_sc(nn.Module):
    def __init__(self, num_classes = 100, num_filter = 16, activation = 'relu'):
        super(simpleNet4_sc, self).__init__()
        print(num_filter)
        num_filter = int(num_filter)
        self.conv1 = nn.Conv2d(3, num_filter, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_filter*2 ,num_filter*2 , kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(num_filter*2*16*16 , num_classes)
        self.ceriation = nn.CrossEntropyLoss()
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
        elif activation == 'simple_tanhv2':
            self.activation = STanhv2()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)+ x
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x) + x
        x = self.activation(x)
        x = self.maxpool(x)
        x = torch.flatten(x,1)

        out = self.fc(x)
        return out

class simpleNet4_bn(nn.Module):
    def __init__(self, num_classes = 100, num_filter = 16, activation = 'relu'):
        super(simpleNet4_bn, self).__init__()
        num_filter = int(num_filter)
        self.conv1 = nn.Conv2d(3, num_filter, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filter*2)
        self.conv4 = nn.Conv2d(num_filter*2 ,num_filter*2 , kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filter*2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(num_filter*2*16*16 , num_classes)
        self.ceriation = nn.CrossEntropyLoss()
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
        elif activation == 'simple_tanhv2':
            self.activation = STanhv2()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = torch.flatten(x,1)

        out = self.fc(x)
        return out

class simpleNet4(nn.Module):
    def __init__(self, num_classes = 100, num_filter = 16, activation = 'relu'):
        super(simpleNet4, self).__init__()
        print(num_filter)
        num_filter = int(num_filter)
        self.conv1 = nn.Conv2d(3, num_filter, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_filter*2 ,num_filter*2 , kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(num_filter*2*16*16 , num_classes)
        self.ceriation = nn.CrossEntropyLoss()
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
        elif activation == 'simple_tanhv2':
            self.activation = STanhv2()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = torch.flatten(x,1)

        out = self.fc(x)
        return out

class simpleNet5(nn.Module):
    def __init__(self, num_classes = 100, num_filter = 16, activation = 'relu'):
        super(simpleNet5, self).__init__()
        num_filter = int(num_filter)
        self.conv1 = nn.Conv2d(3, num_filter, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_filter*2 ,num_filter*2 , kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_filter*2 ,num_filter*4 , kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(num_filter*4*16*16 , num_classes)
        self.ceriation = nn.CrossEntropyLoss()
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
        elif activation == 'simple_tanhv2':
            self.activation = STanhv2()
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = torch.flatten(x,1)

        out = self.fc(x)
        return out

class simpleNet6(nn.Module):
    def __init__(self, num_classes = 100, num_filter = 16, activation = 'relu'):
        super(simpleNet6, self).__init__()
        num_filter = int(num_filter)
        self.conv1 = nn.Conv2d(3, num_filter, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_filter*2 ,num_filter*2 , kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_filter*2 ,num_filter*4 , kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(num_filter*4 ,num_filter*4 , kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(num_filter*4*16*16 , num_classes)
        self.ceriation = nn.CrossEntropyLoss()
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
        elif activation == 'simple_tanhv2':
            self.activation = STanhv2()
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = torch.flatten(x,1)

        out = self.fc(x)
        return out

class simpleNet7(nn.Module):
    def __init__(self, num_classes = 100, num_filter = 16, activation = 'relu'):
        super(simpleNet7, self).__init__()
        num_filter = int(num_filter)
        self.conv1 = nn.Conv2d(3, num_filter, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_filter*2 ,num_filter*2 , kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_filter*2 ,num_filter*4 , kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(num_filter*4 ,num_filter*4 , kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(num_filter*4 ,num_filter*8 , kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(num_filter*8*16*16 , num_classes)
        self.ceriation = nn.CrossEntropyLoss()
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
        elif activation == 'simple_tanhv2':
            self.activation = STanhv2()
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.conv7(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = torch.flatten(x,1)

        out = self.fc(x)
        return out

