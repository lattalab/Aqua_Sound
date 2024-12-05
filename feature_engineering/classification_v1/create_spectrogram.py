import warnings
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
import time
import sys
from math import ceil
from random import choice

# Suppress specific UserWarning from numba
warnings.filterwarnings('ignore')

# modify some parameters
# Audio params
# 這個參數表示音訊的採樣率，即每秒採集的樣本數。
SAMPLE_RATE = 22050 
# duration in second 
# 這個參數表示音訊的持續時間，單位為秒。在這個例子中，音訊的持續時間為 5 秒。
DURATION = 5.0
# 這個參數計算了音訊數據的總長度，以樣本數表示。
# 由於採樣率為 16000 Hz，持續時間為 5 秒，因此音訊數據的總長度為 16000 * 5 = 80000 個樣本。
AUDIO_LEN = int(SAMPLE_RATE * DURATION)

# Spectrogram params
N_MELS = 128     # freq axis
N_FFT = 2048
HOP_LEN = 512    # non-overlap region, which means 1/4 portion overlapping
SPEC_WIDTH = AUDIO_LEN // HOP_LEN + 1  # time axis
FMAX = SAMPLE_RATE//2   # max frequency
SPEC_SHAPE = [N_MELS, SPEC_WIDTH]  # expected output spectrogram shape

# Create a directory to store the spectrogram images
destination_folder = "boat"
source_folder = "./聲音資料/船/M1/20201016_120407.wav"

def process_audio_file(filepath):
    start_time = time.time()    # 計時
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)  # 讀取音檔

    # Generate the mel spectrogram
    spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    spec = librosa.power_to_db(spec)

    # Plot the mel spectrogram
    librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=HOP_LEN, x_axis='time', y_axis='mel', cmap='viridis')
    filename = os.path.split(filepath)[-1]
    plt.title(f"Spectrogram_" + filename, fontsize=17)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Save the spectrogram image with a meaningful filename
    save_filepath = os.path.join(destination_folder, filename.replace(".wav", ".png"))
    plt.savefig(save_filepath)
    # Close the figure to free up resources
    plt.close()
    
    end_time = time.time()
    print(f"Processed {os.path.basename(filepath)} in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":

    os.makedirs(destination_folder, exist_ok=True)  # 存放spectrogram的資料夾
    files = os.listdir(source_folder)
    for file in files:
        if not file.endswith(".wav"):
            continue
        print("image filename:", file)
        filepath = os.path.join(source_folder, file)    # 從這邊讀取音檔
        process_audio_file(filepath)

        print(f"Spectrogram images saved to {destination_folder}")
