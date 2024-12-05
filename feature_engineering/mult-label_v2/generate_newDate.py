import numpy as np
import librosa
import time
import matplotlib.pyplot as plt
import os
import sys
import noisereduce as nr
import math

SAMPLE_RATE = 44100  # (samples/sec)
N_MELS = 128  # freq axis, number of filters
N_FFT = 2048  # frame size
HOP_LEN = 512  # non-overlap region, which means 1/4 portion overlapping
FMAX = SAMPLE_RATE // 2  # max frequency, based on the rule, it should be half of SAMPLE_RATE
noise_factor = 0.01
SNR = 20
speed_factor = 1.5
shift_amount = SAMPLE_RATE // 2

# use librosa to load audio file
def load_audio(file_path, sr=SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

# add noise by calculating the signal-to-noise ratio
def get_white_noise(signal,SNR) :
    #RMS value of signal
    RMS_s=math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise=np.random.normal(0, STD_n, signal.shape[0])
    return signal + noise

# add noise to audio file
def noise_addition(data, noise_factor):
    noise = np.random.randn(len(data))  # random assign value by normal distribution
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

# pitch modification
def pitch_modified(data, sampling_rate=SAMPLE_RATE, n_steps=8):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)

# speed change
def speed_change(data, speed_factor=1.0):
    return librosa.effects.time_stretch(data, rate=speed_factor)

# time shift
def time_shift(data, shift_amount):
    return np.roll(data, shift_amount)

# dictionary for data augmentation
wayForDataAug = {
    "noise": get_white_noise,
    "pitch": pitch_modified,
    "speed": speed_change,
    "shift": time_shift
}

def process_audio_file(filepath, dataAug):
    start_time = time.time()    # 計時
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)  # 讀取音檔
    # Apply noise reduction
    if str(dataAug) != 'noise':
        audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.9, n_fft=N_FFT, hop_length=HOP_LEN)

    way = wayForDataAug[dataAug]
    if way == get_white_noise:
        audio = way(audio, SNR)
    elif way == pitch_modified:
        audio = way(audio, SAMPLE_RATE)
    elif way == speed_change:
        audio = way(audio, speed_factor)
    else:
        audio = way(audio, shift_amount)

    # Generate the mel spectrogram
    spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    spec = librosa.power_to_db(spec)

    # Plot the mel spectrogram
    librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=HOP_LEN, x_axis='time', y_axis='mel', cmap='viridis')
    filename = os.path.split(filepath)[-1]
    plt.title(f"Spectrogram_" + str(dataAug) + "_" + filename, fontsize=17)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Save the spectrogram image with a meaningful filename
    save_filepath = os.path.join(destination_folder, str(dataAug) + "_" + filename.replace(".wav", ".png"))
    plt.savefig(save_filepath)
    # Close the figure to free up resources
    plt.close()
    
    end_time = time.time()
    print(f"Processed {os.path.basename(filepath)} in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_newData.py <source_folder> <destination_folder> <dataAugmentation>")
        sys.exit(1)

    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]
    dataAug = sys.argv[3]

    os.makedirs(destination_folder, exist_ok=True)  # 存放spectrogram的資料夾
    files = os.listdir(source_folder)
    for file in files:
        if not file.endswith(".wav"):
            continue
        print("image filename:", file)
        filepath = os.path.join(source_folder, file)    # 從這邊讀取音檔
        process_audio_file(filepath, dataAug)

        print(f"Spectrogram images saved to {destination_folder}")