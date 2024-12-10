import sys
import os
import math
import signal
import time
import numpy as np
import cv2
from tabulate import tabulate
from tqdm import tqdm
import json

from ..config import CARLA_FPS, AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE, IMAGE_W, IMAGE_H
from ..utils import lidar_to_histogram_features, color_info_string
from .weather import get_a_random_weather


def take_data_backbone(carla_egg_path, rpc_port, ego_vehicle_found_event, finished_taking_data_event,
                       you_can_tick_event, how_many_frames, where_to_save):
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass

    STARTING_FRAME = None
    PATHS = {}
    ALREADY_OBTAINED_DATA_FROM_SENSOR = []

    # Setup witch cameras is present
    cameras_indexes = [0]
    ALREADY_OBTAINED_DATA_FROM_SENSOR = {f"rgb_{i}": False for i in cameras_indexes}
    ALREADY_OBTAINED_DATA_FROM_SENSOR = dict({f"depth_{i}": False for i in cameras_indexes},
                                               **ALREADY_OBTAINED_DATA_FROM_SENSOR)
    ALREADY_OBTAINED_DATA_FROM_SENSOR = dict({"lidar": False}, **ALREADY_OBTAINED_DATA_FROM_SENSOR)

    # Connect the client and set up bp library
    client = carla.Client('localhost', rpc_port)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1. / CARLA_FPS
    # In this case, the simulator will take 20 steps (1/0.05) to recreate one second of
    # the simulated world.
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    # fixed_delta_seconds <= max_substep_delta_time * max_substeps
    # In order to have an optimal physical simulation,
    # the substep delta time should at least be below 0.01666 and ideally below 0.01.
    world.apply_settings(settings)
    bp_lib = world.get_blueprint_library()

    # Search the CAR
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


    # I will set a random weather
    a_random_weather, weather_dict = get_a_random_weather()
    print("WEATHER:")
    a_table_head = ["Weather Parameter", "Value"]
    a_table = []
    for key in weather_dict:
        a_table.append([key, weather_dict[key]])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))
    world.set_weather(a_random_weather)

    # LIDAR callback
    def lidar_callback(data):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            lidar_data_raw = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
            lidar_data_raw = np.reshape(lidar_data_raw, (int(lidar_data_raw.shape[0] / 4), 4))

            # MY LIDAR
            lidar_data = lidar_to_histogram_features(lidar_data_raw[:, :3])[0]
            lidar_data = np.rot90(lidar_data)
            saved_frame = (data.frame - STARTING_FRAME) // AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["lidar"], f"{saved_frame}.png"), lidar_data)
            ALREADY_OBTAINED_DATA_FROM_SENSOR["lidar"] = True

    # CAMERAS callback
    def rgb_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME) // AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"rgb_{number}"], f"{saved_frame}.jpg"), rgb)
            ALREADY_OBTAINED_DATA_FROM_SENSOR[f"rgb_{number}"] = True

    # DEPTH callback
    def depth_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            data.convert(carla.ColorConverter.LogarithmicDepth)
            depth = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            depth = depth[:, :, 0]
            saved_frame = (data.frame - STARTING_FRAME) // AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"depth_{number}"], f"{saved_frame}.png"), depth)
            ALREADY_OBTAINED_DATA_FROM_SENSOR[f"depth_{number}"] = True


    # LIDAR
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('noise_stddev', '0.0')
    lidar_bp.set_attribute('upper_fov', '0.0')
    lidar_bp.set_attribute('lower_fov', '-25.0')
    lidar_bp.set_attribute('channels', '32.0')
    lidar_bp.set_attribute('rotation_frequency', '20.0')
    lidar_bp.set_attribute('points_per_second', '600000')
    lidar_init_trans = carla.Transform(
        carla.Location(x=0, y=0, z=2.5),
        carla.Rotation(pitch=0, roll=0, yaw=0)
    )

    # RGB CAMERAS
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("image_size_x", f"{IMAGE_W}")
    camera_bp.set_attribute("image_size_y", f"{IMAGE_H}")

    # DEPTH CAMERAS
    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("fov", "90")
    depth_bp.set_attribute("image_size_x", f"{IMAGE_W}")
    depth_bp.set_attribute("image_size_y", f"{IMAGE_H}")

    transformations = []

    # Obvious CAMERAS
    transformations.append(carla.Transform(carla.Location(x=1.0, y=+0.0, z=2.0),
                                           carla.Rotation(pitch=0.0, roll=0, yaw=0)))
    transformations.append(carla.Transform(carla.Location(x=0.0, y=-1.0, z=2.0),
                                           carla.Rotation(pitch=-5.0, roll=0, yaw=90)))
    transformations.append(carla.Transform(carla.Location(x=-1.0, y=+0.0, z=2.0),
                                           carla.Rotation(pitch=0.0, roll=0, yaw=180)))
    transformations.append(carla.Transform(carla.Location(x=+0.0, y=1.0, z=2.0),
                                           carla.Rotation(pitch=-5.0, roll=0, yaw=270)))

    sensors = {}
    sensors["lidar"] = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=hero)
    for i in cameras_indexes:
        sensors[f"rgb_{i}"] = world.spawn_actor(camera_bp, transformations[i], attach_to=hero)
        sensors[f"depth_{i}"] = world.spawn_actor(depth_bp, transformations[i], attach_to=hero)

    # Connect Sensor and Callbacks
    sensors["lidar"].listen(lambda data: lidar_callback(data))

    for i in cameras_indexes:
        sensors[f"rgb_{i}"].listen(lambda image, j=i: rgb_callback(image, j))
        sensors[f"depth_{i}"].listen(lambda depth, j=i: depth_callback(depth, j))

    rgb_folders_name = [f"rgb_{i}" for i in cameras_indexes]
    depth_folders_name = [f"depth_{i}" for i in cameras_indexes]

    PATHS["lidar"] = os.path.join(where_to_save, "bev_lidar")
    for i in cameras_indexes:
        PATHS[f"rgb_{i}"] = os.path.join(where_to_save, rgb_folders_name[i])
        PATHS[f"depth_{i}"] = os.path.join(where_to_save, depth_folders_name[i])

    for key_path in PATHS:
        os.mkdir(PATHS[key_path])

    def cntrl_c(_, __):
        sensors["lidar"].stop()
        sensors["lidar"].destroy()
        for i in cameras_indexes:
            sensors[f"rgb_{i}"].stop()
            sensors[f"rgb_{i}"].destroy()
            sensors[f"depth_{i}"].stop()
            sensors[f"depth_{i}"].destroy()
        exit()


    signal.signal(signal.SIGINT, cntrl_c)

    # Let's Run Some Carla's Step to let everything be set up
    DISABLE_ALL_SENSORS = True
    for _ in tqdm(range(10), desc=color_info_string("Warming Up...")):
        you_can_tick_event.set()
        world_snapshot = world.wait_for_tick()
        STARTING_FRAME = world_snapshot.frame
        time.sleep(0.3)

    time.sleep(3)
    STARTING_FRAME += 1
    DISABLE_ALL_SENSORS = False

    you_can_tick_event.set()
    for _ in tqdm(range(how_many_frames * AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE),
                  desc=color_info_string("Taking data...")):
        world_snapshot = world.wait_for_tick()
        if (world_snapshot.frame - STARTING_FRAME) % AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            while True:
                if sum(ALREADY_OBTAINED_DATA_FROM_SENSOR.values()) == len(ALREADY_OBTAINED_DATA_FROM_SENSOR):
                    break
            # print("Obtained all the sensors data! B")
        you_can_tick_event.set()
        for key in ALREADY_OBTAINED_DATA_FROM_SENSOR:
            ALREADY_OBTAINED_DATA_FROM_SENSOR[key] = False

    finished_taking_data_event.set()
