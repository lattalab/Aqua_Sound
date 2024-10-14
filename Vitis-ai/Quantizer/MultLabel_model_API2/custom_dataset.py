import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as dset

dict_for_label = {
                    'boat':[1,0,0,0], 'dolphin':[0,1,0,0], 'fish':[0,0,1,0], 'whale':[0,0,0,1],
                    'boat+fish':[1,0,1,0], 'boat+whale':[1,0,0,1],
                    'dolphin+boat':[1,1,0,0], 'dolphin+whale':[0,1,0,1], 'dolphin+fish':[0,1,1,0],
                    'fish+whale':[0,0,1,1]
                  }
# 自定義資料集，基於 ImageFolder 並支持多標籤
class MultiLabelImageDataset(Dataset):
    def __init__(self, image_folder, label_mapping, transform=None):
        # 使用 ImageFolder 來獲取圖片和單標籤
        self.image_folder = dset.ImageFolder(image_folder, transform=transform)
        
        # 依照圖片標籤名稱來對應多標籤
        self.label_mapping = label_mapping  # 字典對應表

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        # 獲取圖片及其對應的單一標籤（由 ImageFolder 生成）
        img, class_idx = self.image_folder[idx]
        
        # 使用 ImageFolder 的 class_to_idx 來獲取標籤名稱
        label_name = self.image_folder.classes[class_idx]

        # 根據 label_name 查詢對應的 one-hot 編碼
        one_hot_label = torch.tensor(self.label_mapping[label_name], dtype=torch.float32)
        
        return img, one_hot_label