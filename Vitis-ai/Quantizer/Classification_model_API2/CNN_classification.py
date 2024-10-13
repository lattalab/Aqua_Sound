import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,128,480)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=25, kernel_size=3, stride=1, padding=0) #output_shape=(25,126,478)
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(25,63,239)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=25, out_channels=32, kernel_size=6, stride=1, padding=0) #output_shape=(32,58,234)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,29,117)
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0) #output_shape=(32,26,114)
        self.relu3 = nn.ReLU() # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,13,57)
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0) #output_shape=(32,26,114)
        self.relu4 = nn.ReLU() # activation
        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,6,28)
        # Fully connected 1 ,#input_shape=(32*13*57)
        self.fc1 = nn.Linear(32 * 6 * 28, 1024) 
        # Fully connected 2
        self.fc2 = nn.Linear(1024, 32)
        # Fully connected 3
        self.fc3 = nn.Linear(32, 4)
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2
        out = self.maxpool2(out)
        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)
        # Max pool 3
        out = self.maxpool3(out)
        # Convolution 4
        out = self.cnn4(out)
        out = self.relu4(out)
        # Max pool 4
        out = self.maxpool4(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out