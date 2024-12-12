import sys
import os
import signal
import time

import numpy
from tabulate import tabulate
from tqdm import tqdm

from ..config import CARLA_FPS, AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
from ..utils import  color_info_string
from .weather import get_a_random_weather
from .call_back import Callbacks
from .events import Events


def take_data(carla_egg_path, rpc_port, ego_vehicle_found_event, finished_taking_data_event,
              how_many_seconds, where_to_save, sensors_json):
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass

    disable_all_sensors = True
    tick_obtained_from_sensor = {}

    # (1) Connect the client and set up bp library
    frames_hz = CARLA_FPS
    client = carla.Client('localhost', rpc_port)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1 / frames_hz
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


    # (3) Let's set random weather
    a_random_weather, weather_dict = get_a_random_weather()
    print("WEATHER:")
    a_table_head = ["Weather Parameter", "Value"]
    a_table = []
    for key in weather_dict:
        a_table.append([key, weather_dict[key]])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))
    # world.set_weather(a_random_weather) TODO:UNCOMMENT FOR RANDOM WEATHER

    # (4) Let's add all the sensor in the sensor.json file!
    print(f"Simulation at {frames_hz} frames per second")
    class MyCarlaSensors:
        def __init__(self, sensor_cfg):
            self.friendly_name = sensor_cfg["friendly_name"]
            self.callback_function_name = sensor_cfg["callback"]
            self.blue_print_name = sensor_cfg["blue_print_name"]
            self.amount_of_frame_after_we_save = frames_hz * sensor_cfg["attributes"]["sensor_tick"]
            print(f"{self.friendly_name} we save after {self.amount_of_frame_after_we_save:.2f} frames! [{1 / sensor_cfg['attributes']['sensor_tick']:.2f} fps]")
            if self.blue_print_name == "sensor.camera.dvs":
                self.events = Events()
            blue_print = bp_lib.find(self.blue_print_name)
            attributes = sensor_cfg["attributes"]
            for attribute_name in attributes:
                blue_print.set_attribute(attribute_name, str(attributes[attribute_name]))
            location = sensor_cfg["location"]
            transformation = carla.Transform(
                carla.Location(x=location["x"], y=location["y"], z=location["z"]),
                carla.Rotation(pitch=location["pitch"], roll=location["roll"], yaw=location["yaw"]))
            self.actor = world.spawn_actor(blue_print, transformation, attach_to=hero)
            self.where_to_save_sensor = os.path.join(where_to_save, sensor_cfg["folder_name"])
            os.mkdir(self.where_to_save_sensor)
            self.actor.listen(lambda data: self.callback(data))
            tick_obtained_from_sensor[sensor_cfg["friendly_name"]] = 0


        def callback(self, data):
            getattr(Callbacks, self.callback_function_name)(    data,
                                                                disable_all_sensors,
                                                                self.where_to_save_sensor
                                                            )

        def shutdown(self):
            self.actor.stop()
            self.actor.destroy()

    sensors = []
    for sensor in sensors_json:
        sensors.append(MyCarlaSensors(sensor))

    def ctrl_c(_, __):
        for a_sensor in sensors:
            a_sensor.shutdown()
        exit()


    signal.signal(signal.SIGINT, ctrl_c)

    # Let's Run Some Carla's Step to let everything be set up
    disable_all_sensors = True
    for _ in tqdm(range(10), desc=color_info_string("Warming Up...")):
        world_snapshot = world.wait_for_tick()
        Callbacks.set_starting_frame(world_snapshot.frame)

    time.sleep(3)
    Callbacks.set_starting_frame(world.wait_for_tick().frame)
    disable_all_sensors = False
    for _ in tqdm(range(how_many_seconds * frames_hz),
                  desc=color_info_string("Taking data...")):
            world.wait_for_tick()

    print(f"Average number of events: {Callbacks.average_num_of_events():.2f}")

    finished_taking_data_event.set()
