#!/bin/bash

# Make sure your program exist the folder that you specified in the input

echo "It's going to plot the audios in one folder into a bunch of mel_spec.img(s)"
if [ "$#" -lt 2 ]; then
    echo "Please follow this format: sh plot_spectrogram.sh <AUDIO_FOLDER_name> <TXT_FOLDER_name> <IMG_FOLDER_name>"
    exit 1
fi

AUDIO_FOLDER=$1
TXT_FOLDER=$2
IMG_FOLDER=$3

# Iterate over each file in the audio folder
for audio_file in "$AUDIO_FOLDER"/*; do
	echo "$audio_file"
    # Run the mel-spectrogram calculation for each file
    ./cal_mel_spec "$audio_file" "$TXT_FOLDER"
    echo "Done! Generated txt files in $TXT_FOLDER from $audio_file"
    
    # Iterate over each .txt file in the txt folder
    for txt_file in "$TXT_FOLDER"/*.txt; do
        # Get the base name of the audio file (without the directory and extension)
        base_name=$(basename "$audio_file" .wav)
        # Get the base name of the txt file (without the directory and extension)
        base_name2=$(basename "$txt_file" .txt)
        # Correct string concatenation for the image name
        image_name="${base_name}_image_${base_name2}"
        echo $image_name
        # Run the plot script for each txt file and save the image
        ./plot_mel_spec "$txt_file" "$IMG_FOLDER/$image_name.png"
        rm $txt_file
    done
    echo "Done! Generated images in $IMG_FOLDER from $TXT_FOLDER"
done

