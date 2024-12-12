import numpy as np
import cv2
import os
import torch
import h5py
import hdf5plugin

from ..utils import lidar_to_histogram_features
from .events_representations import Histogram

STARTING_FRAME = 10000
TOTAL_NUM_OF_EVENTS = 0
TOTAL_NUM_OF_EVENTS_STEP = 0

class Callbacks:

    @staticmethod
    def set_starting_frame(starting_frame):
        global STARTING_FRAME
        STARTING_FRAME = starting_frame

    @staticmethod
    def average_num_of_events():
        return TOTAL_NUM_OF_EVENTS / TOTAL_NUM_OF_EVENTS_STEP

    # LIDAR callback
    @staticmethod
    def lidar_callback(data, disable_all_sensors, where_to_save):
        if not disable_all_sensors:
            lidar_data_raw = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
            lidar_data_raw = np.reshape(lidar_data_raw, (int(lidar_data_raw.shape[0] / 4), 4))

            # MY LIDAR
            lidar_data = lidar_to_histogram_features(lidar_data_raw[:, :3])[0]
            lidar_data = np.rot90(lidar_data)
            saved_frame = (data.frame - STARTING_FRAME)
            cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.png"), lidar_data)
    # CAMERAS callback
    @staticmethod
    def rgb_callback(data, disable_all_sensors, where_to_save):
        if not disable_all_sensors:
            rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME)
            cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.jpg"), rgb)

    # DEPTH callback
    @staticmethod
    def depth_callback(data, disable_all_sensors, where_to_save):
        if not disable_all_sensors:
            import carla
            data.convert(carla.ColorConverter.LogarithmicDepth)
            depth = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            depth = depth[:, :, 0]
            saved_frame = data.frame - STARTING_FRAME
            cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.png"), depth)

    @staticmethod
    def event_callback(data, disable_all_sensors, where_to_save):
        if not disable_all_sensors:
            x = np.array(data.to_array_x())
            y = np.array(data.to_array_y())
            t = np.array(data.to_array_t())
            pol = np.array(data.to_array_pol())
            histo = Histogram(height=data.height, width=data.width, normalize=False)
            representation = histo.convert(torch.from_numpy(x),
                                           torch.from_numpy(y),
                                           torch.from_numpy(pol),
                                           torch.from_numpy(t))
            global TOTAL_NUM_OF_EVENTS, TOTAL_NUM_OF_EVENTS_STEP
            TOTAL_NUM_OF_EVENTS += x.shape[0]
            TOTAL_NUM_OF_EVENTS_STEP += 1
            saved_frame = (data.frame - STARTING_FRAME)
            cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.png"), histo.to_rgb_mono(representation))
