import math

import numpy as np
import cv2
import os
import torch
import h5py
import hdf5plugin
import matplotlib.pyplot as plt

from ..utils import lidar_to_histogram_features
from .events_representations import Histogram
from .disparity_visualization import disp_to_rgb

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
    def lidar_callback(data, disable_all_sensors, h5_dataset):
        if not disable_all_sensors:
            lidar_data_raw = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
            lidar_data_raw = np.reshape(lidar_data_raw, (int(lidar_data_raw.shape[0] / 4), 4))

            # MY LIDAR
            lidar_data = lidar_to_histogram_features(lidar_data_raw[:, :3])[0]
            lidar_data = np.rot90(lidar_data)
            saved_frame = (data.frame - STARTING_FRAME)
            # cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.png"), lidar_data)
    # CAMERAS callback
    @staticmethod
    def rgb_callback(data, disable_all_sensors, h5_dataset):
        if not disable_all_sensors:
            bgr = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME)
            # cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.jpg"), bgr)

    # DEPTH callback
    @staticmethod
    def depth_callback(data, disable_all_sensors, h5_dataset):
        if not disable_all_sensors:
            import carla
            raw_depth = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            b = raw_depth[:, :, 0] / 256
            g = raw_depth[:, :, 1] / 256
            r = raw_depth[:, :, 2] / 256
            depth = (r + g * 256 + b * 256 * 256) / (256 * 256 * 256 - 1)
            m_depth = 1000 * depth * 256

            saved_frame = data.frame - STARTING_FRAME

            focal_length = data.width / (2 * math.tan(data.fov * math.pi / 180 / 2))
            disparity = 0.6 * focal_length / m_depth
            disparity[m_depth == m_depth.max()] = 0
            h5_dataset[saved_frame] = disparity
            # cv2.imwrite(os.path.join("/home/enrico/Downloads", f"{saved_frame}.png"), disparity*3)

    @staticmethod
    def event_callback(data, disable_all_sensors, h5_dataset):
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
            h5_dataset[saved_frame] = representation.numpy()
            # cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.png"), histo.to_rgb_mono(representation))
