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
                
        # Implemented the architecture described here: https://arxiv.org/pdf/1710.00977.pdf

        self.conv1 = nn.Conv2d(1, 32, kernel_size = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 1)
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.drop = nn.Dropout()

        self.fc1 = nn.Linear(in_features = 6400, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000, out_features = 550)
        self.fc3 = nn.Linear(in_features = 550, out_features = 136)
            
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.conv1(x); # 2. conv 1 [ 32, 93, 93 ]
        x = F.elu(x);  # 3. activation 1
        x = self.pool(x); # 4. max pool
        x = self.drop(x); # 5. dropout 1
        
        # print(x.shape) # [16, 32, 46, 46]
        
        x = self.conv2(x); # 6. conv 2 [ 64, 44, 44 ]
        x = F.elu(x); # 7. activateion 2
        x = self.pool(x); # 8. max pool 2
        x = self.drop(x); # 9. dropout 2
        
        # print(x.shape) # [16, 64, 22, 22]
        
        x = self.conv3(x); # 10. conv 3 [ 128, 21, 21 ]
        x = F.elu(x); # 11. activateion 3
        x = self.pool(x); # 12. max pool 3
        x = self.drop(x); # 13. dropout 3
        
        # print(x.shape) # [16, 128, 10, 10]
        
        x = self.conv4(x); # 14. conv 4 [ 256, 10, 10 ]
        x = F.elu(x); # 15. activateion 4
        x = self.pool(x); # 16. max pool 4
        x = self.drop(x); # 17. dropout 4 [256, 5, 5]
        
        # print(x.shape) # [16, 256, 5, 5]
        
        x = x.view(x.size(0), -1); # 18. flatten [6400]
        
        x = self.fc1(x); # 19. dense linear [1000]
        x = F.elu(x);  # 20. activation 5
        x = self.drop(x); # 21. dropout 5
        
        # print(x.shape) # [16, 1000]
        
        x = self.fc2(x); # 22. dense linear [1000]
        x = F.elu(x);  # 23. activation 6
        x = self.drop(x); # 24. dropout 6
        
        # print(x.shape) # [16, 550]
        
        x = self.fc3(x); # 25. dense linear [550]
        
        # print(x.shape) # [16, 136]
        
        return x
