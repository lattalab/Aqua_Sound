from pydub import AudioSegment
import random
import os
import sys

def merge_audio(file1, file2, output_file):
    # Load the audio files
    audio1 = AudioSegment.from_file(file1)
    audio2 = AudioSegment.from_file(file2)

    # Combine the audio files
    # repeat file2 until it's the same length as file1
    combined_audio = audio1.overlay(audio2, loop=True) # overlay

    # merge the audio filepath
    output_path = output_file + os.path.split(file1)[1][:-4] + "_" + os.path.split(file2)[1][:-4] + ".wav"

    # Export the combined audio file
    combined_audio.export(output_path, format="wav")
    print(f"Combined {audio1} and {audio2} into {output_path}")

if len(sys.argv) < 4:
    print("Usage: python create_new_audio.py <audio_folder_path1> <audio_folder_path2> <output_filepath>")
    sys.exit(1)

# get the list of audio files and shuffle them
audios_1 = os.listdir(sys.argv[1])
random.shuffle(audios_1)
audios_2 = os.listdir(sys.argv[2])
random.shuffle(audios_2)

# merge the audio files
for audio1, audio2 in zip(audios_1, audios_2):
    audio1 = os.path.join(sys.argv[1], audio1)
    audio2 = os.path.join(sys.argv[2], audio2)
    merge_audio(audio1, audio2, sys.argv[3])