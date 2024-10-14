import os
import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
import sys
import argparse
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from MultLabel_model import CNN_Model
from custom_dataset import MultiLabelImageDataset, dict_for_label
from torch.utils.data import DataLoader

LR = 0.02
#batch_size_train = 20
#batch_size_valid = 20

NUM_EPOCHS = 25

IMAGE_SIZE = (480, 256)
transform = transforms.Compose([
        transforms.Resize( IMAGE_SIZE ), 
        transforms.ToTensor()
    ])

# 設定dataset
def generate_dataset(image_folder_path,  batch_size = 32, train_or_valid = True):
    # 使用自定義的多標籤資料集
    dataset = MultiLabelImageDataset(image_folder=image_folder_path, label_mapping=dict_for_label, transform=transform)

    # 使用 DataLoader 加載資料
    if train_or_valid:
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader
        
def Accuracy(y_pred, y_true):
    count = 0
    for i in range(y_true.shape[0]):  # 幾個樣本
        p = sum(np.logical_and(y_pred[i], y_true[i]))
        q = sum(np.logical_or(y_pred[i], y_true[i]))
        count += p / q
    return count / y_true.shape[0]

def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0
    # 新增這邊記錄真實跟預測值
    pred_list = []
    label_list = []

    for data in val_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        
        predicted_prob = torch.sigmoid(outputs)
        
        # 將每個類別的預測概率與閾值 0.5 比較，大於 0.5 的表示預測該類別存在 (標記為 1)
        predicted = (predicted_prob > 0.5).int()
        # accumlating loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # 計算每個樣本的預測是否與真實標籤完全相同
        correct += (predicted == labels).all(dim=1).sum().item()
        total += labels.size(0)

        pred_list.extend(predicted.cpu().numpy())
        label_list.extend(labels.cpu().numpy())
    
    avg_loss = total_loss/ len(val_loader)
    acc = Accuracy(np.array(pred_list), np.array(label_list))
    return acc, avg_loss


def quantize(model, quantized_model_name, quant_mode, batchsize, quantize_output_dir, finetune, deploy, train_dataset_path, val_dataset_path, device):
    # Construct paths
    quant_model_path = os.path.join(quantize_output_dir, quantized_model_name)

    # Override batch size if in test mode
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batchsize != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batchsize = 1
    
    
    # Generate a random input tensor for quantization
    rand_in = torch.randn([batchsize, 3, 480, 256])

    # Create quantizer
    quantizer = torch_quantizer(quant_mode, model, rand_in, output_dir=quantize_output_dir)
    quantized_model = quantizer.quant_model

    
    # Load Images from a Folder
    image_folder_path = val_dataset_path
    print(f"Loading validating data from {image_folder_path}.")
    valid_dataloader = generate_dataset(image_folder_path, batch_size = batchsize, train_or_valid = False)
    
    # to get loss value after evaluation
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    # fast finetune model or load finetuned parameter before test
    if finetune == True:
        if quant_mode == 'calib':
            print("Start to finetune the quantized model")
            
            # Load Images from a Folder
            image_folder_path = train_dataset_path
            print(f"Loading training data from {image_folder_path}.")
            train_dataloader = generate_dataset(image_folder_path, batch_size = batchsize, train_or_valid = True)
            
            quantizer.fast_finetune(evaluate, (quantized_model, train_dataloader, loss_fn, device))
            quantized_model = quantizer.quant_model # Update quantized_model to use the fine-tuned parameters
            
        elif quant_mode == 'test':
            quantizer.load_ft_param()
            quantized_model = quantizer.quant_model # Update quantized_model to use the fine-tuned parameters
    		
    		
    # start to evaluate the model
    avg_accuracy = 0
    avg_loss = 0
    
    avg_accuracy, avg_loss = evaluate(quantized_model, valid_dataloader, loss_fn, device)
    
	# evaluation result
    print(f"avg_accuracy : {avg_accuracy}, avg_loss : {avg_loss}")
	
	# handle quantization result, export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
        # Save the quantized model
        torch.save(quantized_model.state_dict(), quant_model_path)
    if deploy:
        quantizer.export_xmodel(deploy_check=True, output_dir=quantize_output_dir)
	
    

    return

def main():
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}.")
	
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  type=str, default='./MyModel_v3.pt',    help='Path to float model')
    ap.add_argument('--quantized_model_name',  type=str, default='quantized_model_v3.pt',    help='Quantized model name')
    ap.add_argument('--quant_mode', type=str, default='calib', choices=['calib', 'test'], help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('--batchsize',  type=int, default=32, help='Testing batch size - must be an integer. Default is 16')
    ap.add_argument('--quantize',  type=bool, default=True, help='Whether to do quantization, default is True')
    ap.add_argument('--quantize_output_dir',  type=str, default='./quantized/', help='Directory to save the quantized model')
    ap.add_argument('--finetune',  type=bool, default=True, help='Whether to do the finetune befroe quantization')
    ap.add_argument('--deploy',  type=bool, default=True, help='Whether to export xmodel for deployment')
    ap.add_argument('--train_dataset_path',  type=str, default='./dataset_v3/train', help='Give the train dataset path')
    # use test data as validation here
    ap.add_argument('--val_dataset_path',  type=str, default='./dataset_v3/test')
    args = ap.parse_args()

    print('\n----------------------------------------')
    print('PyTorch version:', torch.__version__)
    print('Python version:', sys.version)
    print('----------------------------------------')
    print('Command line options:')
    print('--model:', args.model)
    print('--quantized_model_name:', args.quantized_model_name)
    print('--quant_mode:', args.quant_mode)
    print('--batchsize:', args.batchsize)
    print('--quantize:', args.quantize)
    print('--quantize_output_dir:', args.quantize_output_dir)
    print('--finetune:', args.finetune)
    print('--deploy:', args.deploy)
    print('--train_dataset_path:', args.train_dataset_path)
    print('--val_dataset_path:', args.val_dataset_path)
    
    
    print('----------------------------------------')

    # Load the model and weights
    model = CNN_Model()
    model.to(device)
    print("Loading model")
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    if args.quantize: # Check if quantization is requested
        print("Quantizing model")
        quantize(model, 
                 args.quantized_model_name, 
                 args.quant_mode, 
                 args.batchsize, 
                 args.quantize_output_dir, 
                 args.finetune, 
                 args.deploy, 
                 args.train_dataset_path, 
                 args.val_dataset_path, 
                 device)
                 
        print('Quantization finished. Results saved in:', args.quantize_output_dir)

if __name__ == "__main__":
    main()
