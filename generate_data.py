import argparse
import json
import multiprocessing
import os
import pathlib
import time
import signal
import psutil
from ctypes import c_int
from tabulate import tabulate
import shutil

from data_generator.data_creation import take_data
from data_generator import utils
from data_generator import config
from datetime import datetime
from data_generator.carla_interface import add_carla_to_python_path, \
    launch_carla_server_and_wait_till_its_up, \
    set_up_world_and_wait_till_its_set_up, \
    set_up_traffic_manager_and_wait_till_its_up


def get_arguments():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        '--carla_ip',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)'
    )
    arg_parser.add_argument(
        '--rpc_port',
        default=2000,
        help='Carla RPC port (default: 2000)',
        type=int
    )
    arg_parser.add_argument(
        '--tm_port',
        default=8000,
        help='Traffic Manager port (default: 8000)',
        type=int
    )
    arg_parser.add_argument(
        '--town',
        default=10,
        help='Witch town to select (default: 10)',
        type=int
    )
    arg_parser.add_argument(
        '--end_of_egg_file',
        help='How the egg file should end to be valid! (default: py3.7-linux-x86_64.egg)',
        default="py3.7-linux-x86_64.egg",
        type=str
    )
    arg_parser.add_argument(
        '--carla_path',
        help='Path to the Carla Installation!',
        required=True,
        type=str
    )
    arg_parser.add_argument(
        '--num_of_vehicle',
        help='Number of Vehicle to spawn! (default: 30)',
        required=False,
        default=30,
        type=int
    )
    arg_parser.add_argument(
        '--num_of_walkers',
        help='Number of Walkers to spawn! (default: 30)',
        required=False,
        default=30,
        type=int
    )
    arg_parser.add_argument(
        '--num_of_seconds',
        help='Number of Seconds to take! (default: 10)',
        required=False,
        default=10,
        type=int
    )
    arg_parser.add_argument(
        '--dataset_path',
        help=f'Where to save the data! (default: {os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets")})',
        required=False,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets"),
        type=str
    )
    arg_parser.add_argument(
        '--show_carla_window',
        help='If ypu want to see the carla window!',
        action='store_true'
    )
    args = arg_parser.parse_args()
    if args.town not in config.TOWN_DICT:
        error = f"Invalid Town Index! [{args.town}]\n" + \
                "Possible Town Index:\n"
        for key in config.TOWN_DICT:
            error += f"{key} -> {config.TOWN_DICT[key]}\n"
        raise Exception(utils.color_error_string(error))
    return args


pids_to_be_killed = []
def kill_all():
    global pids_to_be_killed
    for a_pid in pids_to_be_killed:
        try:
            os.kill(a_pid, signal.SIGKILL)
        except:
            print(utils.color_error_string(f"Not able to kill {a_pid}! :-("))
            pass
    pids_to_be_killed = []


def run_all(args, where_to_save, carla_ue4_path, carla_log_path, sensors_json):
    # (1) LAUNCH CARLA SERVER
    print("Launching Carla Server...")
    carla_server_pid = multiprocessing.Value(c_int)
    carla_was_correctly_started_up = launch_carla_server_and_wait_till_its_up(
        rpc_port=args.rpc_port,
        carla_server_pid=carla_server_pid,
        carla_ue4_path=carla_ue4_path,
        logs_path=carla_log_path,
        how_many_seconds_to_wait=100,
        show_carla_window=args.show_carla_window,
    )

    pids_to_be_killed.append(carla_server_pid.value)

    if not carla_was_correctly_started_up:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while starting!"))

    print(utils.color_info_string("(1/3)\tCarla Server is UP!"))

    # (3) SET UP THE WORLD
    world_was_correctly_set_up = set_up_world_and_wait_till_its_set_up(
        carla_ip=args.carla_ip,
        rpc_port=args.rpc_port,
        town_number=args.town,
        carla_server_pid=carla_server_pid
    )

    if not world_was_correctly_set_up:
        raise utils.NutException(utils.color_error_string(f"Failed to set up world!"))

    print(utils.color_info_string("(2/3)\tWorld was correctly set up!"))

    # (4) SET UP TRAFFIC MANAGER
    traffic_manager_pid = multiprocessing.Value(c_int)
    carla_is_ok, \
    traffic_manager_is_ok, \
    traffic_manager_is_up, \
    set_up_traffic_manager_process = set_up_traffic_manager_and_wait_till_its_up(
        carla_ip=args.carla_ip,
        rpc_port=args.rpc_port,
        tm_port=args.tm_port,
        number_of_vehicles=args.num_of_vehicle,
        number_of_walkers=args.num_of_walkers,
        carla_server_pid=carla_server_pid,
        traffic_manager_pid=traffic_manager_pid,
        logs_path=traffic_manager_log_path
    )

    pids_to_be_killed.append(traffic_manager_pid.value)

    if not carla_is_ok:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while setting up Traffic Manager!"))
    if not traffic_manager_is_ok:
        raise utils.NutException(utils.color_error_string(f"Traffic Manager Crashed!"))

    print(utils.color_info_string("(3/3)\tTraffic Manager Set Up properly!"))

    # (5) LAUNCH DATA CREATION PROCESS
    data_creation_pid = multiprocessing.Value(c_int)
    ego_vehicle_found_event = multiprocessing.Event()
    finished_taking_data_event = multiprocessing.Event()
    data_creation_process = multiprocessing.Process(target=take_data.take_data,
                                                    args=(egg_file_path, args.rpc_port,ego_vehicle_found_event,
                                                          finished_taking_data_event,
                                                          args.num_of_seconds, where_to_save, sensors_json))
    data_creation_process.start()
    data_creation_pid.value = data_creation_process.pid
    pids_to_be_killed.append(data_creation_pid.value)

    print(utils.get_a_title(f"STARTING TO TAKE DATA [{args.num_of_seconds} s]", color="green"))
    start_time = time.time()
    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            raise utils.NutException(utils.color_error_string(f"Carla crashed!"))
        if not set_up_traffic_manager_process.is_alive():
            raise utils.NutException(utils.color_error_string(f"Traffic Manager crashed!"))
        if not data_creation_process.is_alive():
            raise utils.NutException(utils.color_error_string(f"Data Creation crashed!"))
        if not ego_vehicle_found_event.is_set() and time.time() - start_time > 10:
            raise utils.NutException(
                utils.color_error_string(f"Data Creation is not able to find out the Ego Vehicle!"))
        if finished_taking_data_event.is_set():
            break

    print(utils.get_a_title(f"FINISHED TO TAKE DATA [{args.num_of_seconds} s]", color="green"))

    # (6) CLEANING EVERYTHING
    kill_all()
    return True


if __name__ == "__main__":
    my_args = get_arguments()
    egg_file_path, my_carla_ue4_path = add_carla_to_python_path(my_args.carla_path, my_args.end_of_egg_file)
    try:
        import carla
    except:
        raise Exception(utils.color_error_string(f"Not able to import Carla from [{egg_file_path}]"))

    print(utils.get_a_title("STARTING THE PROCESS", color="blue"))
    print(utils.color_info_success(f"Find out a valid carla in {egg_file_path}!"))

    # (0) SET UP LOGS AND DATASET FOLDER
    repo_path = pathlib.Path(__file__).parent.resolve()
    # TODO: Put this log in the data folder!
    my_carla_log_path = os.path.join(repo_path, "logs", f"carla_server_logs.log")
    traffic_manager_log_path = os.path.join(repo_path, "logs", f"traffic_manager_logs.log")
    datasets_folder_path = my_args.dataset_path

    if not os.path.isdir(datasets_folder_path):
        try:
            os.mkdir(datasets_folder_path)
        except:
            Exception(utils.color_error_string(f"Unable to create [{datasets_folder_path}] dir!"))

    # (1) LET'S MAKE A TABLE TO SUMMARIZE ALL THE ARGS VALUES
    a_table_head = ["Argument", "Value"]
    a_table = []
    for arg in vars(my_args):
        a_table.append([arg, getattr(my_args, arg)])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    # (2) LET'S TRY TO GET DATA
    my_where_to_save = None
    with open(os.path.join(repo_path, "sensors.json"), "r") as file:
        sensors_json = json.load(file)
    for i in range(config.MAX_NUM_OF_ATTEMPTS):
        # (2.1) FOR EACH ATTEMPT, CREATE A FOLDER IN THE DATASETS ONE
        now = datetime.now()
        current_time = now.strftime("%Y_%m_%d__%H_%M_%S")
        my_where_to_save = os.path.join(datasets_folder_path, f"{current_time}_{config.TOWN_DICT[my_args.town]}_{i}")
        if i > 0:
            # It's not the first attempt to we need to remove the previous failed data
            shutil.rmtree(my_where_to_save, ignore_errors=True)
        os.mkdir(my_where_to_save)
        try:
            print(utils.get_a_title(f"ATTEMPT [{i + 1}/{config.MAX_NUM_OF_ATTEMPTS}]", color="blue"))
            # (2.2) LET'S RUN ALL FOR EACH ATTEMPT
            if run_all(my_args, my_where_to_save, my_carla_ue4_path, my_carla_log_path, sensors_json):
                break
        except utils.NutException as e:
            print(e.message)
            my_args.rpc_port += 1
            kill_all()
        except KeyboardInterrupt:
            kill_all()
            print(utils.get_a_title("Bye Bye!", color="yellow"))
            exit()
