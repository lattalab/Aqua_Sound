import librosa
from observe_audio_function import denoise, FMAX, N_MELS, HOP_LEN, N_FFT
import matplotlib.pyplot as plt
import os

# deal with the error: main thread is not in main loop
import matplotlib
matplotlib.use('Agg')

SAMPLE_RATE = 44100
destination_folder = "./class/boat"
source_folder = "./聲音資料/船/台灣攤"

# 讀取音檔
def load_audio(file_path, SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return y, sr


# Load the audio file
images = os.listdir(source_folder)
for filename in images:
    print("audio filename:", filename)
    if not filename.endswith(".wav"):
        continue
    filepath = os.path.join(source_folder, filename)
    audio, sr = load_audio(filepath, SAMPLE_RATE)

    # Denoise the audio
    audio = denoise(audio)

    # Get the mel spectrogram
    spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    spec = librosa.power_to_db(spec)

    # Plot the mel spectrogram
    librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=HOP_LEN, x_axis='time', y_axis='mel', cmap='viridis')
    plt.title(f"Spectrogram_" + filename, fontsize=17)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Save the spectrogram image with a meaningful filename
    save_filepath = os.path.join(destination_folder,  "denoise_" + (filename.replace(".wav", ".png")))
    print("save_filename:", os.path.split(save_filepath)[-1])
    plt.savefig(save_filepath)
    # Close the figure to free up resources
    plt.close()