import argparse
import os.path

import h5py


def get_arguments():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        '--path',
        required=True,
        type=str,
        help='Path of the dataset folder to show!'
    )
    return arg_parser.parse_args()

if __name__ == '__main__':
    my_args = get_arguments()
    if not (os.path.isdir(my_args.path)):
        raise Exception(f"The dataset folder path [{my_args.path}] does not exist!")
    if not (os.path.isfile(os.path.join(my_args.path, "disparity.h5")) and
            os.path.isfile(os.path.join(my_args.path, "event_left.h5")) and
            os.path.isfile(os.path.join(my_args.path, "event_right.h5"))):
        raise Exception(f"The dataset folder path [{my_args.path}] does not contain one of the following files:"
                        f" disparity.h5, event_left.h5, event_right.h5!")
    disparity_dataset = h5py.File(os.path.join(my_args.path, "disparity.h5"), "r")
    left_dataset = h5py.File(os.path.join(my_args.path, "event_left.h5"), "r")
    right_dataset = h5py.File(os.path.join(my_args.path, "event_right.h5"), "r")
