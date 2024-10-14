# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CNN_Model::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=25, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Conv2d[cnn1]/input.2
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/ReLU[relu1]/25
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/MaxPool2d[maxpool1]/input.3
        self.module_4 = py_nndct.nn.Conv2d(in_channels=25, out_channels=32, kernel_size=[6, 6], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Conv2d[cnn2]/input.4
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/ReLU[relu2]/42
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/MaxPool2d[maxpool2]/input.5
        self.module_7 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[4, 4], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Conv2d[cnn3]/input.6
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/ReLU[relu3]/59
        self.module_9 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/MaxPool2d[maxpool3]/input.7
        self.module_10 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Conv2d[cnn4]/input.8
        self.module_11 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/ReLU[relu4]/76
        self.module_12 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/MaxPool2d[maxpool4]/82
        self.module_13 = py_nndct.nn.Module('shape') #CNN_Model::CNN_Model/84
        self.module_14 = py_nndct.nn.Module('reshape') #CNN_Model::CNN_Model/input.9
        self.module_15 = py_nndct.nn.Linear(in_features=5376, out_features=1024, bias=True) #CNN_Model::CNN_Model/Linear[fc1]/input.10
        self.module_16 = py_nndct.nn.Linear(in_features=1024, out_features=32, bias=True) #CNN_Model::CNN_Model/Linear[fc2]/input
        self.module_17 = py_nndct.nn.Linear(in_features=32, out_features=4, bias=True) #CNN_Model::CNN_Model/Linear[fc3]/99

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_5 = self.module_5(self.output_module_4)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_9 = self.module_9(self.output_module_8)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_13 = self.module_13(input=self.output_module_12, dim=0)
        self.output_module_14 = self.module_14(input=self.output_module_12, size=[self.output_module_13,-1])
        self.output_module_15 = self.module_15(self.output_module_14)
        self.output_module_16 = self.module_16(self.output_module_15)
        self.output_module_17 = self.module_17(self.output_module_16)
        return self.output_module_17
