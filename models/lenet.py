import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

__all__=['LeNet', 'LeNetContainer']


############## LeNet for MNIST ###################
'''
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 5*5*50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
'''

############## LeNet for CIFAR ###################
class LeNet(nn.Module):
    def __init__(self, num_classes=10, num_filter = 16):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(6, num_filter, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(num_filter*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    def forward(self, x, return_feature=False):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        if return_feature:
            x_ = copy.deepcopy(x)
        x = self.fc(x)
        if return_feature:
            return x, x_
        else:
            return x

    
class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = self.fc1(x)
        x = self.fc2(x)
        return x