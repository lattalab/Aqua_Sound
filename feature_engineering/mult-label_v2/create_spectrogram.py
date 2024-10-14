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
import noisereduce as nr

# Suppress specific UserWarning from numba
warnings.filterwarnings('ignore')

# modify some parameters
# Audio params
# 這個參數表示音訊的採樣率，即每秒採集的樣本數。
SAMPLE_RATE = 44100 

# Spectrogram params
N_MELS = 128     # freq axis
N_FFT = 2048
HOP_LEN = 512    # non-overlap region, which means 1/4 portion overlapping
FMAX = SAMPLE_RATE//2   # max frequency

# Create a directory to store the spectrogram images
source_folder = "../../MultLabel_model/合併聲音/魚+鯨魚"
destination_folder = "../../聲音資料/Merge_without_denoise/fish+whale"

def process_audio_file(filepath):
    start_time = time.time()    # 計時
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)  # 讀取音檔
    #audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)    # Denoise the audio

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
