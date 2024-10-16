import torch.nn as nn

# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution , input_shape=(3,480,256)
        self.layer1 = self.conv_module(3, 8, 5, 1) #output_shape=(8,238,126)
        self.layer2 = self.conv_module(8, 14, 3, 1) #output_shape=(14,118,62)
        self.layer3 = self.conv_twice(14, 44, 3, 1) #output_shape=(44,58,30)
        self.layer4 = self.conv_twice(44, 20, 3, 1) #output_shape=(20,28,14)
        self.layer5 = self.conv_twice(20, 8, 5, 1) #output_shape=(8,12,5)
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
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2)
        )
    def conv_twice(self, in_channels, out_channels, kernel_size, stride, padding=0):
        "By giving the convolutional parameters, the function will return a double convolutional module"
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),    # expand the channels
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2)
        )