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
              dt_ready_to_take_data, dt_want_to_stop_taking_data, warm_up_frames, frames_to_take):
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass

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
            self.friendly_name = sensor_cfg["friendly_name"]
            self.callback_function_name = sensor_cfg["callback"]

            amount_of_frame_after_we_save = sensor_cfg["attributes"]["sensor_tick"] / carla_tick
            print(f"{self.friendly_name} we save after {amount_of_frame_after_we_save:.2f} frames!"
                  f" [{1 / sensor_cfg['attributes']['sensor_tick']:.2f} fps]")

            blue_print = bp_lib.find(sensor_cfg["blue_print_name"])
            attributes = sensor_cfg["attributes"]
            for attribute_name in attributes:
                blue_print.set_attribute(attribute_name, str(attributes[attribute_name]))
            location = sensor_cfg["location"]
            transformation = carla.Transform(
                carla.Location(x=location["x"], y=location["y"], z=location["z"]),
                carla.Rotation(pitch=location["pitch"], roll=location["roll"], yaw=location["yaw"]))

            self.actor = world.spawn_actor(blue_print, transformation, attach_to=hero)
            self.actor.listen(lambda data: self.callback(data))
            self.start_frame = None
            self.last_frame = None
            self.consecutive_frames = None

        @abstractmethod
        def callback(self, data):
            if self.last_frame is None:
                self.last_frame = data.frame
                self.consecutive_frames = 1
            elif data.frame - self.last_frame == 1:
                self.last_frame = data.frame
                self.consecutive_frames += 1
            else:
                self.last_frame = data.frame
                self.consecutive_frames = 1

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
            getattr(Callbacks, self.callback_function_name)(
                data,
                self.disparity_raw_folder_path
            )
            # Let's save the timestamp in nanoseconds
            self.timestamp_dict[int(data.frame)] = int(data.timestamp * 10 ** 9)
            super().callback(data)

        def check_data(self):
            print(f"[{self.friendly_name}] Checking that I have enough consecutive frames!")
            if self.consecutive_frames < frames_to_take:
                raise NutException(f"The sensor {self.friendly_name} has received {self.consecutive_frames} consecutive"
                                   f" frames but we were asking {frames_to_take}!")
            # Let's wait 5 seconds that all the files get correctly saved!
            for _ in tqdm(range(5), desc=color_info_string(f"[{self.friendly_name}] "
                                                           f"Wait 5 s that all the file get saved...")):
                time.sleep(1)
            # Let's get all the files names in the directory
            all_frames_file_name = {int(file_name[:-4]): file_name
                                    for file_name in os.listdir(self.disparity_raw_folder_path)
                                    if file_name[-4:] == ".png"}
            for i in range(self.start_frame, self.start_frame+frames_to_take):
                # Let's check that the file is really there
                try:
                    file_name = all_frames_file_name[i]
                except KeyError:
                    error_str = f"Depth frame {i} is missing in {self.disparity_raw_folder_path}\n"
                    for ii in range(max(i-10, self.start_frame),
                                    min(i+10, self.start_frame+frames_to_take)):
                        if ii in all_frames_file_name.keys():
                            error_str += f"{ii} : {all_frames_file_name[ii]}\n"
                        else:
                            error_str += f"{ii} : MISSING\n"
                    raise NutException(color_error_string(error_str))
                # Now we are sure that the file is there so we can move in the final official folder with a proper
                # normalized name
                os.rename(os.path.join(self.disparity_raw_folder_path, file_name),
                          os.path.join(self.disparity_folder_path, f"{i - self.start_frame:04d}.png"))
                # We save also the timestamp of the frame
                self.timestamps_to_save.append(self.timestamp_dict[i])
            return self.timestamps_to_save[0]

        def finalize(self, starting_time):
            # There we normalized the timestamps subtracting the starting time
            print(f"[{self.friendly_name}]  Saving Data Timestamps...")
            for i in range(len(self.timestamps_to_save)):
                self.timestamps_to_save[i] -= int(starting_time)
            # Finally we save the timestamps file
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
            # A test to see if create_ms_to_index is correct!
            # example_t = np.array([0, 500, 2100, 5000, 5000, 5200, 7100, 7200, 7200, 8100, 8500, 9300])
            # self.create_ms_to_index(example_t, 10)
            self.data_to_save = None

        def callback(self, data):
            getattr(Callbacks, self.callback_function_name)(    data,
                                                                self.data,
                                                                self.starting_times
                                                            )
            super().callback(data)

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
            print(f"[{self.friendly_name}] Checking Data...")
            self.data_to_save = {
                array_name: [] for array_name in self.data
            }
            # We check that all the frames were there
            for i in range(self.start_frame, self.start_frame + frames_to_take):
                for array_name in self.data_to_save :
                    try:
                        self.data_to_save[array_name].append(self.data[array_name][i])
                    except KeyError:
                        error_str = f"[{self.friendly_name}] Frame {i} is missing\n"
                        raise NutException(color_error_string(error_str))
            # We concatenate all the data
            for array_name in self.data_to_save:
                self.data_to_save[array_name] = np.concatenate(self.data_to_save[array_name])

            print(f"[{self.friendly_name}] I have got {self.data_to_save['t'].size} events in {frames_to_take} frames."
                  f" [{self.data_to_save['t'].size/frames_to_take:.1f} events per frame]")
            return self.data_to_save["t"][0]

        def finalize(self, starting_time):
            # There we normalized the timestamps subtracting the starting time
            self.data_to_save["t"] -= starting_time
            # We calculate the total number of ms
            total_num_of_ms = int(frames_to_take / (1/carla_tick) * 1000)
            # We compute the ms_to_index vector
            ms_to_idx = self.create_ms_to_index(self.data_to_save["t"], total_num_of_ms)

            # Finally we save the h5 file
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
            print(f"[{self.friendly_name}]  Saved h5 file in {time_needed:.2f} s!")


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

    # We check that we are ready to warm up
    while True:
        if tm_ready_to_warm_up.is_set():
            break
    dt_ready_to_warm_up.set()
    # We wait that carla warms up
    with tqdm(range(warm_up_frames), desc=color_info_string("Warming Up...")) as pbar:
        while True:
            try:
                world.wait_for_tick(seconds=1).frame
            except RuntimeError:
                if tm_ready_to_take_data.is_set():
                    break
            pbar.update(1)

    # We say that we are ready to take data
    dt_ready_to_take_data.set()
    for _ in tqdm(range(frames_to_take+50), desc=color_info_string("Take Data...")):
        world.wait_for_tick()

    finish_frame = world.wait_for_tick().frame
    official_start_frame = finish_frame - 25 - frames_to_take
    dt_want_to_stop_taking_data.set()

    # Let's wait that all callbacks has been executed
    with tqdm(range(frames_to_take+25), desc=color_info_string("Waiting that all callbacks complete...")) as pbar:
        while True:
            sensors_consecutive_frames = [sensor.consecutive_frames for sensor in sensors
                                          if sensor.consecutive_frames is not None]
            all_sensors_enough_consecutive_frames = True
            for cf in sensors_consecutive_frames:
                if cf < frames_to_take + 25:
                    all_sensors_enough_consecutive_frames = False

            pbar.n = min(sensors_consecutive_frames)

            if all_sensors_enough_consecutive_frames:
                break


    # We communicate the starting frame to all the sensors
    for sensor in sensors:
        sensor.start_frame = official_start_frame

    # Now we check the data, and we get from sensor their first real data time
    starting_times = []
    for sensor in sensors:
        starting_times.append(sensor.check_data())

    # We get the minimum starting time, and we put that as the official starting time
    official_starting_time = min(starting_times)
    for sensor in sensors:
        sensor.finalize(official_starting_time)
    finished_taking_data_event.set()

