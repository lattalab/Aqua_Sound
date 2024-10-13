import cv2
import numpy as np
import os

# 設定檔案路徑
filepath = './boat'
files = os.listdir(filepath)

for file in files:
    print("image filename:", file)
    img = cv2.imread(os.path.join(filepath, file)) # 讀取圖片
    # 裁切區域的 x 與 y 座標（左上角）
    # 以及裁切區域的寬度與高度
    # 調出這些參數來裁切圖片
    x = 75
    y = 35
    w = 445
    h = 395
    crop_img = img[y:y+h, x:x+w]        # 取出陣列的範圍
    cv2.imwrite(os.path.join(filepath, file), crop_img) # 儲存圖片