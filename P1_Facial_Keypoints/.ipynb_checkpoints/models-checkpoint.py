## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
                
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # try this paper's architecture: https://arxiv.org/pdf/1710.00977.pdf
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, stride=1, padding=2)
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.drop1 = nn.Dropout()
        
        self.fc1 = nn.Linear(in_features = 200704, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000, out_features = 136)
        
                
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.reshape(x.size(0), -1)
        
        x = self.drop1(x)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
