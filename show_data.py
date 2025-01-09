import argparse
import os.path
from time import time

import numpy as np
import torch
import cv2
import h5py
from tqdm import tqdm
from data_generator.data_creation.events_representations import Histogram


def get_arguments():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        '--path',
        required=True,
        type=str,
        help='Path of the dataset folder to show!'
    )
    return arg_parser.parse_args()

def read_a_sequence(sequence_folder_path: str):
    if not (os.path.isfile(os.path.join(sequence_folder_path, "disparity.h5")) and
            os.path.isfile(os.path.join(sequence_folder_path, "event_left.h5")) and
            os.path.isfile(os.path.join(sequence_folder_path, "event_right.h5"))):
        raise Exception(f"The dataset folder path [{sequence_folder_path}] does not contain one of the following files:"
                        f" disparity.h5, event_left.h5, event_right.h5!")
    start = time()
    with h5py.File(os.path.join(sequence_folder_path, "disparity.h5"), "r") as h5file:
        disparity_dataset = h5file["dataset"][:]
    with h5py.File(os.path.join(sequence_folder_path, "event_left.h5"), "r") as h5file:
        left_dataset = h5file["dataset"][:]
    with h5py.File(os.path.join(sequence_folder_path, "event_right.h5"), "r") as h5file:
        right_dataset = h5file["dataset"][:]
    reading_time = time() - start
    print(f"Reading Time : {reading_time}")
    print(f"Disparity {disparity_dataset.shape}")
    print(f"Left {left_dataset.shape}")
    print(f"Right {right_dataset.shape}")
    print("Showing images...")
    histo = Histogram(height=left_dataset.shape[1], width=left_dataset.shape[2], normalize=False)
    for i in tqdm(range(disparity_dataset.shape[0])):
        a = disparity_dataset[i, :]/disparity_dataset[i, :].max()
        a = np.stack((a, a, a), axis=-1)
        b = histo.to_rgb_mono(torch.from_numpy(left_dataset[i]))
        c = histo.to_rgb_mono(torch.from_numpy(right_dataset[i]))
        all = np.concatenate((a, b, c), axis=1)
        cv2.imshow("Disparity", all)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("Finished to show images!")

if __name__ == '__main__':
    my_args = get_arguments()
    if not (os.path.isdir(my_args.path)):
        raise Exception(f"The dataset folder path [{my_args.path}] does not exist!")
    sequences = {int(a_dir):a_dir for a_dir in os.listdir(my_args.path)}
    sequences_min = min(list(sequences.keys()))
    sequences_max = max(list(sequences.keys()))
    while True:
        selected_sequence = input(f"Please select a sequence in [{sequences_min}; {sequences_max}]: ")
        if selected_sequence.lower() == "all":
            for sequence in sequences:
                read_a_sequence(os.path.join(my_args.path, sequences[sequence]))
        try:
            selected_sequence = int(selected_sequence)
        except ValueError:
            continue
        if not selected_sequence in sequences.keys():
            continue
        sequence_folder = os.path.join(my_args.path, sequences[selected_sequence])
        read_a_sequence(sequence_folder)
