import numpy as np
import cv2
import os
import torch
import h5py
import hdf5plugin

from ..utils import lidar_to_histogram_features
from .events_representations import Histogram

class Callbacks:
    # LIDAR callback
    @staticmethod
    def lidar_callback(data, disable_all_sensors, starting_frame, amount_of_carla_frame_after_we_save,
                       tick_obtained_from_sensor, where_to_save, friendly_name):
        if not disable_all_sensors:
            if (data.frame - starting_frame) % amount_of_carla_frame_after_we_save == 0:
                lidar_data_raw = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
                lidar_data_raw = np.reshape(lidar_data_raw, (int(lidar_data_raw.shape[0] / 4), 4))

                # MY LIDAR
                lidar_data = lidar_to_histogram_features(lidar_data_raw[:, :3])[0]
                lidar_data = np.rot90(lidar_data)
                saved_frame = (data.frame - starting_frame)
                cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.png"), lidar_data)
            tick_obtained_from_sensor[friendly_name] += 1

    # CAMERAS callback
    @staticmethod
    def rgb_callback(data, disable_all_sensors, starting_frame, amount_of_carla_frame_after_we_save,
                       tick_obtained_from_sensor, where_to_save, friendly_name):
        if not disable_all_sensors:
            if (data.frame - starting_frame) % amount_of_carla_frame_after_we_save == 0:
                rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
                saved_frame = (data.frame - starting_frame)
                cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.jpg"), rgb)
            tick_obtained_from_sensor[friendly_name] += 1

    # DEPTH callback
    @staticmethod
    def depth_callback(data, disable_all_sensors, starting_frame, amount_of_carla_frame_after_we_save,
                       tick_obtained_from_sensor, where_to_save, friendly_name):

        if not disable_all_sensors:
            if (data.frame - starting_frame) % amount_of_carla_frame_after_we_save == 0:
                import carla
                data.convert(carla.ColorConverter.LogarithmicDepth)
                depth = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
                depth = depth[:, :, 0]
                saved_frame = (data.frame - starting_frame)
                cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.png"), depth)
            tick_obtained_from_sensor[friendly_name] += 1

    @staticmethod
    def event_callback(data, disable_all_sensors, starting_frame, amount_of_carla_frame_after_we_save,
                       tick_obtained_from_sensor, where_to_save, friendly_name, events):
        if not disable_all_sensors:
            x = np.array(data.to_array_x())
            y = np.array(data.to_array_y())
            t = np.array(data.to_array_t())
            pol = np.array(data.to_array_pol())
            # events.add(x, y, t, pol)
            if (data.frame - starting_frame) % amount_of_carla_frame_after_we_save == 0:
                saved_frame = (data.frame - starting_frame)
                histo = Histogram(height=data.height, width=data.width, normalize=True)
                representation = histo.convert(torch.from_numpy(x),
                                               torch.from_numpy(y),
                                               torch.from_numpy(pol),
                                               torch.from_numpy(t))
                print(f"Non zero pixels: {torch.nonzero(representation).shape[0]}")
                cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}*.png"), histo.to_rgb_mono(representation))
                dvs_img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
                # Blue is positive, red is negative
                dvs_img[y, x, pol + 1] = 255
                cv2.imwrite(os.path.join(where_to_save, f"{saved_frame}.png"), dvs_img)
                events.reset()
                # hf = h5py.File(f"{saved_frame}.h5", "w")
                # hf.create_dataset("x", data=x)
                # hf.create_dataset("y", data=y)
                # hf.create_dataset("t", data=t)
                # hf.create_dataset("pol", data=pol)
                # hf.close()
            tick_obtained_from_sensor[friendly_name] += 1
