import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class FCNet(nn.Module):

    def __init__(self, activation_function_name):
        super(FCNet, self).__init__()
        if activation_function_name == "relu":
            self.activation_function = torch.relu
        if activation_function_name == "sigmoid":
            self.activation_function = torch.sigmoid
        # doing: initialize the layers for the fully-connected neural network (please do not change layer names!)
        self.linear1 = nn.Linear(3 * 32 * 32, 500)
        self.linear2 = nn.Linear(500, 100)
        self.linear3 = nn.Linear(100, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        # doing: complete the forward pass (use self.activation_function)
        x = x.view(x.size(0), -1)
        x = self.activation_function(self.linear1(x))
        x = self.activation_function(self.linear2(x))
        x = self.linear3(x)
        return x

class ConvNet(nn.Module):

    def __init__(self, activation_function_name):
        super(ConvNet, self).__init__()
        if activation_function_name == "relu":
            self.activation_function = torch.relu
        if activation_function_name == "sigmoid":
            self.activation_function = torch.sigmoid
        # doing: initialize the layers for the convolutional neural network (please do not change layer names!)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        # doing: complete the forward pass (use self.activation_function)
        x = self.activation_function(self.conv1(x))
        x = self.maxpool2d(x)
        x = self.activation_function(self.conv2(x))
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x