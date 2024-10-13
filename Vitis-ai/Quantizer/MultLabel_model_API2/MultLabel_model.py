import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution , input_shape=(3,480,256)
        self.layer1 = self.conv_module(3, 16, 3, 1) #output_shape=(16,239,127)
        self.layer2 = self.conv_module(16, 16, 4, 1) #output_shape=(16,118,62)
        self.layer3 = self.conv_twice(16, 32, 3, 1) #output_shape=(32,58,30)
        self.layer4 = self.conv_twice(32, 64, 3, 1) #output_shape=(64,28,14)
        self.layer5 = self.conv_twice(64, 8, 5, 1) #output_shape=(8,12,5)
        # # Fully connected 1
        self.fc1 = nn.Linear(8*12*5, 4) 
        
    def forward(self, x):
        # Convolutions
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # Resize
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out
    
    def conv_module(self, in_channels, out_channels, kernel_size, stride, padding=0):
        "By giving the convolutional parameters, the function will return a convolutional module"
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    def conv_twice(self, in_channels, out_channels, kernel_size, stride, padding=0):
        "By giving the convolutional parameters, the function will return a double convolutional module"
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),    # expand the channels
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )