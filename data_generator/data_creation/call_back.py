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

class Callbacks:
    """
    # LIDAR callback
    @staticmethod
    def lidar_callback(data, disable_all_sensors, data_list):
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
    def rgb_callback(data, disable_all_sensors, data_list):
        if not disable_all_sensors:
            bgr = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME)
            # cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.jpg"), bgr)
    """

    # DEPTH callback
    @staticmethod
    def depth_callback(data, where_to_save):
        raw_depth = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        b = raw_depth[:, :, 0] / 256
        g = raw_depth[:, :, 1] / 256
        r = raw_depth[:, :, 2] / 256
        depth = (r + g * 256 + b * 256 * 256) / (256 * 256 * 256 - 1)
        m_depth = 1000 * depth * 256

        focal_length = data.width / (2 * math.tan(data.fov * math.pi / 180 / 2))
        disparity = 0.6 * focal_length / m_depth
        disparity[m_depth == m_depth.max()] = 0
        cv2.imwrite(os.path.join(where_to_save, f"{data.frame:05d}.png"), disparity)

    @staticmethod
    def event_callback(data, data_list, starting_times):
        x = np.array(data.to_array_x())
        y = np.array(data.to_array_y())
        t = np.array(data.to_array_t())
        p = np.array(data.to_array_pol())

        data_list["x"][int(data.frame)] = x
        data_list["y"][int(data.frame)] = y
        data_list["t"][int(data.frame)] = t
        data_list["p"][int(data.frame)] = p

        if len(starting_times) == 0:
            starting_times.append(t.min())
