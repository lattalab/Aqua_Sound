# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CNN_Model::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[layer1]/Conv2d[0]/input.2
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/Sequential[layer1]/ReLU[1]/44
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[layer1]/MaxPool2d[2]/input.3
        self.module_4 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[4, 4], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[layer2]/Conv2d[0]/input.4
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/Sequential[layer2]/ReLU[1]/61
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[layer2]/MaxPool2d[2]/input.5
        self.module_7 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[layer3]/Conv2d[0]/input.6
        self.module_8 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[layer3]/Conv2d[1]/input.7
        self.module_10 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/Sequential[layer3]/ReLU[3]/93
        self.module_11 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[layer3]/MaxPool2d[4]/input.9
        self.module_12 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[layer4]/Conv2d[0]/input.10
        self.module_13 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[layer4]/Conv2d[1]/input.11
        self.module_15 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/Sequential[layer4]/ReLU[3]/125
        self.module_16 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[layer4]/MaxPool2d[4]/input.13
        self.module_17 = py_nndct.nn.Conv2d(in_channels=64, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[layer5]/Conv2d[0]/input.14
        self.module_18 = py_nndct.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[layer5]/Conv2d[1]/input.15
        self.module_20 = py_nndct.nn.ReLU(inplace=False) #CNN_Model::CNN_Model/Sequential[layer5]/ReLU[3]/157
        self.module_21 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[layer5]/MaxPool2d[4]/163
        self.module_22 = py_nndct.nn.Module('shape') #CNN_Model::CNN_Model/165
        self.module_23 = py_nndct.nn.Module('reshape') #CNN_Model::CNN_Model/input
        self.module_24 = py_nndct.nn.Linear(in_features=480, out_features=4, bias=True) #CNN_Model::CNN_Model/Linear[fc1]/172

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
        self.output_module_10 = self.module_10(self.output_module_8)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_13 = self.module_13(self.output_module_12)
        self.output_module_15 = self.module_15(self.output_module_13)
        self.output_module_16 = self.module_16(self.output_module_15)
        self.output_module_17 = self.module_17(self.output_module_16)
        self.output_module_18 = self.module_18(self.output_module_17)
        self.output_module_20 = self.module_20(self.output_module_18)
        self.output_module_21 = self.module_21(self.output_module_20)
        self.output_module_22 = self.module_22(input=self.output_module_21, dim=0)
        self.output_module_23 = self.module_23(input=self.output_module_21, size=[self.output_module_22,-1])
        self.output_module_24 = self.module_24(self.output_module_23)
        return self.output_module_24
