import sys
import os
import signal
import time
from abc import ABC, abstractmethod
import json

import h5py
import numpy
import numpy as np
from numpy import dtype
from tabulate import tabulate
from tqdm import tqdm

from ..utils import  color_info_string
from .weather import get_a_random_weather
from .call_back import Callbacks
from .events import Events
from ..utils import NutException, color_error_string

def take_data(carla_egg_path, rpc_port, ego_vehicle_found_event, finished_taking_data_event,
              where_to_save, sensors_json, tm_ready_to_warm_up, tm_ready_to_take_data, dt_ready_to_warm_up,
              dt_ready_to_take_data, warm_up_frame, frames_to_take, starting_frame_num):
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass

    disable_all_sensors = False # TODO: CHANGED!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tick_obtained_from_sensor = {}

    # (1) Connect the client and set up bp library
    carla_tick = sensors_json["carla_tick"]
    client = carla.Client('localhost', rpc_port)
    client.set_timeout(60.0)
    world = client.get_world()

    # (2) Search the CAR
    hero = None
    while hero is None:
        print("Waiting for the ego vehicle...")
        possible_vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in possible_vehicles:
            if vehicle.attributes['role_name'] == 'hero':
                print("Ego vehicle found")
                hero = vehicle
                break
        time.sleep(1)
    ego_vehicle_found_event.set()


    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = carla_tick
    # In this case, the simulator will take x steps to recreate one second of
    # the simulated world.
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    # fixed_delta_seconds <= max_substep_delta_time * max_substeps
    # To have an optimal physical simulation,
    # the substep delta time should at least be below 0.01666 and ideally below 0.01.
    world.apply_settings(settings)
    bp_lib = world.get_blueprint_library()

    # (3) Let's set random weather
    a_random_weather, weather_dict = get_a_random_weather()
    print("WEATHER:")
    a_table_head = ["Weather Parameter", "Value"]
    a_table = []
    for key in weather_dict:
        a_table.append([key, weather_dict[key]])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))
    world.set_weather(a_random_weather)

    # (4) Let's add all the sensor in the sensor.json file!
    print(f"Simulation at {1/carla_tick:.1f} frames per second")

    # (5) Let's define some sensor Class
    class MyCarlaSensors(ABC):
        def __init__(self, sensor_cfg):
            self.sensor_cfg = sensor_cfg
            self.friendly_name = sensor_cfg["friendly_name"]
            self.callback_function_name = sensor_cfg["callback"]
            self.blue_print_name = sensor_cfg["blue_print_name"]
            self.amount_of_frame_after_we_save = sensor_cfg["attributes"]["sensor_tick"] / carla_tick
            print(f"{self.friendly_name} we save after {self.amount_of_frame_after_we_save:.2f} frames! [{1 / sensor_cfg['attributes']['sensor_tick']:.2f} fps]")

            blue_print = bp_lib.find(self.blue_print_name)
            attributes = sensor_cfg["attributes"]
            for attribute_name in attributes:
                blue_print.set_attribute(attribute_name, str(attributes[attribute_name]))
            location = sensor_cfg["location"]
            transformation = carla.Transform(
                carla.Location(x=location["x"], y=location["y"], z=location["z"]),
                carla.Rotation(pitch=location["pitch"], roll=location["roll"], yaw=location["yaw"]))
            self.actor = world.spawn_actor(blue_print, transformation, attach_to=hero)
            self.actor.listen(lambda data: self.callback(data))
            tick_obtained_from_sensor[sensor_cfg["friendly_name"]] = 0
            self.start_frame = None


            """
            if self.blue_print_name == "sensor.camera.dvs":
                self.events = Events()
            h5_file_name = sensor_cfg["h5_file_name"]
            if h5_file_name[-3:] != ".h5":
                h5_file_name += ".h5"
            self.h5_file_path = os.path.join(where_to_save, h5_file_name)
            self.data = []
            """

        @abstractmethod
        def callback(self, data):
            pass

        def shutdown(self):
            self.actor.stop()
            self.actor.destroy()

    class PngSensor(MyCarlaSensors):
        def __init__(self, sensor_cfg):
            super().__init__(sensor_cfg)
            self.timestamp_dict = {}

            self.disparity_folder_path = os.path.join(where_to_save, "disparity")
            os.mkdir(self.disparity_folder_path)
            self.disparity_raw_folder_path = os.path.join(where_to_save, "disparity_raw")
            os.mkdir(self.disparity_raw_folder_path)

            self.timestamps_to_save = []


        def callback(self, data):
            getattr(Callbacks, self.callback_function_name)(    data,
                                                                disable_all_sensors,
                                                                self.timestamp_dict,
                                                                self.disparity_raw_folder_path
                                                            )

        def check_data(self):
            print("Checking data and renaming file names...")
            # Let's rename the frame name so they start from 0
            all_frames_file_name = {int(file_name[:-4]): file_name
                                    for file_name in os.listdir(self.disparity_raw_folder_path) if file_name[-4:] == ".png"}
            """
            if len(all_frames_file_name) < frames_to_take:
                # We need to take data again!
                raise NutException(color_error_string(f"Some problem while getting al data. Something was lost!"
                                                      f" [expecting Depth to have"
                                                      f" {sensors_json['number_of_frames_to_take']}"
                                                      f" but it has {len(all_frames_file_name)}]"))
            """
            for i in range(self.start_frame, self.start_frame+frames_to_take):
                try:
                    file_name = all_frames_file_name[i]
                except KeyError:
                    error_str = f"Depth frame {i} is missing in {self.disparity_raw_folder_path}\n"
                    for ii in range(self.start_frame, self.start_frame+frames_to_take):
                        if abs(i-ii) < 10:
                            error_str += "\t\t"
                        if ii in all_frames_file_name.keys():
                            error_str += f"{ii} : {all_frames_file_name[ii]}\n"
                        else:
                            error_str += f"{ii} : MISSING\n"
                    print(color_error_string(error_str))
                    raise NutException(color_error_string("Missing a frame!"))
                os.rename(os.path.join(self.disparity_raw_folder_path, file_name),
                          os.path.join(self.disparity_folder_path, f"{i - self.start_frame:04d}.png"))
                self.timestamps_to_save.append(self.timestamp_dict[i])
            return self.timestamps_to_save[0]

        def finalize(self, starting_time):
            print("Saving Data Timestamps...")
            for i in range(len(self.timestamps_to_save)):
                self.timestamps_to_save[i] -= int(starting_time)

            with open(os.path.join(where_to_save, "timestamps.json"), "w", encoding="utf-8") as json_timestamps_file:
                json.dump(self.timestamps_to_save, json_timestamps_file, indent=4)

    class EventSensor(MyCarlaSensors):
        def __init__(self, sensor_cfg, left_right: str):
            super().__init__(sensor_cfg)
            self.events = Events()
            assert left_right in ["left", "right"]
            self.h5_file_path = os.path.join(where_to_save, f"{left_right}.h5")
            self.data = {
                "x": {}, "y": {}, "t": {}, "p": {}
            }
            self.starting_times = []
            # example_t = np.array([0, 500, 2100, 5000, 5000, 5200, 7100, 7200, 7200, 8100, 8500, 9300])
            # self.create_ms_to_index(example_t, 10)
            self.data_to_save = None

        def callback(self, data):
            getattr(Callbacks, self.callback_function_name)(    data,
                                                                disable_all_sensors,
                                                                self.data,
                                                                self.starting_times
                                                            )

        @staticmethod
        def create_ms_to_index(t, total_num_of_ms):

            ms_to_idx = np.zeros(total_num_of_ms, dtype=np.int64)
            last_idx = 0
            for i in range(total_num_of_ms):
                ms = i * 1000000
                while True:
                    if t[last_idx] >= ms:
                        ms_to_idx[i] = last_idx
                        break
                    else:
                        last_idx += 1

            print("ms_to_idx: ")
            print(ms_to_idx[:30])
            print(ms_to_idx[-30:])

            return ms_to_idx

        def check_data(self):
            print(f"Saving h5 file {self.friendly_name}...")
            self.data_to_save = {
                array_name: [] for array_name in self.data
            }
            for i in range(self.start_frame, self.start_frame + frames_to_take):
                for array_name in self.data_to_save :
                    try:
                        self.data_to_save[array_name].append(self.data[array_name][i])
                        if array_name == "t":
                            print(self.data[array_name][i].size)
                    except KeyError:
                        error_str = f"Event frame {i} is missing\n"
                        error_str += f"{self.data[array_name].keys()}"
                        print(color_error_string(error_str))
                        raise NutException(color_error_string("Missing a frame!"))
            for array_name in self.data_to_save:
                self.data_to_save[array_name] = np.concatenate(self.data_to_save[array_name])
            print(f"NUM OF EVENTS: {self.data_to_save['t'].size}")
            return self.data_to_save["t"][0]

        def finalize(self, starting_time):
            # We subtract the starting time:
            self.data_to_save["t"] -= starting_time
            total_num_of_ms = int(frames_to_take / (1/carla_tick) * 1000)
            ms_to_idx = self.create_ms_to_index(self.data_to_save["t"], total_num_of_ms)

            start = time.time()

            with h5py.File(self.h5_file_path, "w") as f:
                for array_name in self.data_to_save:
                    f.create_dataset(array_name,
                                     data=self.data_to_save[array_name],
                                     compression="gzip",
                                     )
                f.create_dataset("ms_to_idx",
                                 data=ms_to_idx,
                                 )
            time_needed = time.time() - start
            print(f"Saved h5 file {self.friendly_name} in {time_needed}")


    sensors = []
    for sensor in sensors_json["sensors"]:
        if sensor["blue_print_name"] == "sensor.camera.depth":
            sensors.append(PngSensor(sensor))
        elif sensor["blue_print_name"] == "sensor.camera.dvs":
            if "Left" in sensor["friendly_name"]:
                sensors.append(EventSensor(sensor, left_right="left"))
            else:
                sensors.append(EventSensor(sensor, left_right="right"))

    def ctrl_c(_, __):
        for a_sensor in sensors:
            a_sensor.shutdown()
        exit()


    signal.signal(signal.SIGINT, ctrl_c)

    # Warm Up there
    while True:
        if tm_ready_to_warm_up.is_set():
            break
    possible_start_frame = None
    dt_ready_to_warm_up.set()
    with tqdm(range(warm_up_frame), desc=color_info_string("Warming Up...")) as pbar:
        while True:
            try:
                possible_start_frame = world.wait_for_tick(seconds=1).frame
            except RuntimeError:
                if tm_ready_to_take_data.is_set():
                    break
            pbar.update(1)

    official_start_frame = possible_start_frame + 10
    for sensor in sensors:
        sensor.start_frame = official_start_frame
    print(f"START FRAME = {official_start_frame}")

    dt_ready_to_take_data.set()
    with tqdm(range(frames_to_take*2), desc=color_info_string("Take Data...")) as pbar:
        while True:
            try:
                world.wait_for_tick(seconds=5)
            except RuntimeError:
                break
            pbar.update(1)
    time.sleep(20)
    disable_all_sensors = True
    starting_times = []
    for sensor in sensors:
        starting_times.append(sensor.check_data())
    print(starting_times)
    real_starting_time = min(starting_times)
    for sensor in sensors:
        sensor.finalize(real_starting_time)
    finished_taking_data_event.set()
    # print(f"Average number of events: {Callbacks.average_num_of_events():.2f}")

