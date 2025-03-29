import json
import logging
import os
import shutil
import torch
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file, get_all_files

logger = logging.getLogger(__name__)
SAMPLERATE = 16000

print(torch.__version__)
def prepare_mini_librispeech(
    data_folder, save_json_train, save_json_valid, save_json_test
):
    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return
    print("----------------preparing data--------------------")
    # If the dataset doesn't exist yet, download it
    train_folder = os.path.join(data_folder, "train-clean-5", "LibriSpeech","train-clean-5")
    valid_folder = os.path.join(data_folder,  "dev-clean-2", "LibriSpeech","dev-clean")
    test_folder = os.path.join(data_folder, "test-clean","LibriSpeech","test-clean")
    if not check_folders(train_folder, valid_folder, test_folder):
        print("not right folder")

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".flac"]

    # List of flac audio files
    wav_list_train = get_all_files(train_folder, match_and=extension)
    wav_list_valid = get_all_files(valid_folder, match_and=extension)
    wav_list_test = get_all_files(test_folder, match_and=extension)

    # List of transcription file
    extension = [".trans.txt"]
    trans_list = get_all_files(data_folder, match_and=extension)
    trans_dict = get_transcription(trans_list)


    # Create the json files
    create_json(wav_list_train, trans_dict, save_json_train)
    create_json(wav_list_valid, trans_dict, save_json_valid)
    create_json(wav_list_test, trans_dict, save_json_test)
    print("------------finish preparing data--------------------")

def get_transcription(trans_list):

    # Processing all the transcription files in the list
    trans_dict = {}
    for trans_file in trans_list:
        # Reading the text file
        with open(trans_file) as f:
            for line in f:
                uttid = line.split(" ")[0]
                text = line.rstrip().split(" ")[1:]
                text = " ".join(text)
                trans_dict[uttid] = text

    logger.info("Transcription files read!")
    return trans_dict


def create_json(wav_list, trans_dict, json_file):
    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:

        # Reading the signal (to retrieve duration in seconds)
        #print("wav file",wav_file)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-5:])

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "words": trans_dict[uttid],
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):

    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


    
# data_folder = r'C:\Users\s6071\Desktop\RL_final_project\speechbrain\templates\speech_recognition\data'
# prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')