import os
import sys
import subprocess
import time
from datetime import datetime
import multiprocessing
import  multiprocessing.sharedctypes as shared_ctype
import signal
import psutil

from data_generator.data_creation.generate_traffic import generate_traffic
from .utils import color_error_string, color_info_string
from .config import TOWN_DICT

def add_carla_to_python_path(carla_path:str, end_of_egg_file:str):
    """
    This function checks the integrity of the Carla Path
    """
    # (1) Check that the Carla's Path really exists
    if not os.path.isdir(carla_path):
        raise Exception(color_error_string(f"The given Carla Path doesn't exist! [{carla_path}]"))
    # (2) Check that the egg file is really present and that it works: being able to import carla!
    carla_pythonapi_dist_path = os.path.join(carla_path, "PythonAPI/carla/dist")
    if not os.path.isdir(carla_pythonapi_dist_path):
        raise Exception(color_error_string(f"The given Carla doesn't contains a PythonAPI! [{carla_pythonapi_dist_path}]"))
    egg_files = [file for file in os.listdir(carla_pythonapi_dist_path) if file[-len(end_of_egg_file):] == end_of_egg_file]
    if len(egg_files) == 0:
        raise Exception(color_error_string(f"The given Carla doesn't contains a \"*{end_of_egg_file}\" file in"
                                           f" \"{carla_pythonapi_dist_path}\""))
    if len(egg_files) > 1:
        raise Exception(color_error_string(f"The given Carla contains to many \"*{end_of_egg_file}\" files "
                                           f"in \"{carla_pythonapi_dist_path}\"\nSet a more restrict search with"
                                           f" the \"--end_of_egg_file\" arguments!"))
    egg_file_path = os.path.join(carla_pythonapi_dist_path, egg_files[0])
    # Now that we have a unique egg file we add it to the python path!
    sys.path.append(egg_file_path)
    # (3) Check that the CarlaUE4 executable is present
    carla_ue4_folder = os.path.join(carla_path, "CarlaUE4/Binaries/Linux/")
    if not os.path.isdir(carla_ue4_folder):
        raise Exception(color_error_string(f"The folder in witch I was expecting \"CarlaUE4-Linux-Shipping\""
                                           f" doesn't exists! [{carla_ue4_folder}]"))
    if "CarlaUE4-Linux-Shipping" not in os.listdir(carla_ue4_folder):
        raise Exception(color_error_string(f"I cannot find \"CarlaUE4-Linux-Shipping\" executable in"
                                           f" \"{carla_ue4_folder}\"!"))
    carla_ue4_path = os.path.join(carla_ue4_folder, "CarlaUE4-Linux-Shipping")
    return egg_file_path, carla_ue4_path

def launch_carla_server_and_wait_till_its_up(rpc_port:int, carla_server_pid:shared_ctype,
                                             carla_ue4_path:str, logs_path:str,
                                             how_many_seconds_to_wait:int, show_carla_window:bool=False):
    def start_up_carla_server():
        with open(logs_path, 'r+') as logs_file:
            command_as_list = ["/usr/bin/stdbuf",
                               "-o0",
                               carla_ue4_path,
                               "-nosound",
                               f"-carla-rpc-port={rpc_port}"]
            if not show_carla_window:
                command_as_list.append("-RenderOffScreen")
            carla_process = subprocess.Popen(
                command_as_list,
                stdout=logs_file,
                stderr=logs_file,
                universal_newlines=True
            )
        carla_server_pid.value = carla_process.pid
        # We will wait Carla to start up!
        while True:
            with open(logs_path, 'r') as logs_file:
                time.sleep(0.2)
                lines = len(logs_file.readlines())
                if lines >= 2:
                    break
            return_code = carla_process.poll()
            if return_code is not None:
                # The Carla process died before starting up!
                exit()

        print("Waiting Carla to Start...", end="", flush=True)
        try:
            import carla
        except:
            print(color_error_string("Was not possible to import carla from launch_carla_server_and_wait_till_its_up!"))
            exit()
        start_time = datetime.now()
        while True:
            try:
                client = carla.Client('localhost', rpc_port)
                client.set_timeout(1.0)
                _ = client.get_world()
                break
            except RuntimeError:
                pass
            print("*", end="", flush=True)
            if (datetime.now() - start_time).total_seconds() > how_many_seconds_to_wait:
                break
        print()
    # END start_up_carla_server

    # FIRST OF ALL KILL ALL CARLA SERVER RUNNING
    for proc in psutil.process_iter():
        if "CarlaUE4-Linux-Shipping" in proc.name():
            print(color_info_string(f"Another carla server was running, I will kill it! [{proc.name()}]"))
            os.kill(proc.pid, signal.SIGKILL)
    if not os.path.isdir(os.path.dirname(logs_path)):
        try:
            os.mkdir(os.path.dirname(logs_path))
        except:
            Exception(color_error_string(f"Unable to find out log dir! [{os.path.dirname(logs_path)}]"))
    if os.path.isfile(logs_path):
        os.remove(logs_path)
    with open(logs_path, 'w') as _:
        pass
    
    check_carla_process = multiprocessing.Process(target=start_up_carla_server)
    check_carla_process.start()
    # Let's wait till Carla Server is Up!
    while True:
        if not check_carla_process.is_alive():
            check_carla_process.join()
            if not psutil.pid_exists(carla_server_pid.value):
                return False
            else:
                return True

def set_up_world_and_wait_till_its_set_up(carla_ip:str, rpc_port:int, town_number:int,
                                          carla_server_pid:shared_ctype):
    def set_up_world(world_set_up):
        try:
            import carla
        except:
            raise Exception(color_error_string(f"Not able to import Carla!"))
        client = carla.Client(carla_ip, rpc_port)
        client.set_timeout(1000.0)
        client.load_world(TOWN_DICT[town_number])
        world_set_up.set()
    # END set_up_world

    my_world_set_up = multiprocessing.Event()
    set_up_world_process = multiprocessing.Process(target=set_up_world, args=(my_world_set_up, ))
    set_up_world_process.start()

    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            set_up_world_process.kill()
            return False
        if not set_up_world_process.is_alive():
            set_up_world_process.join()
            if my_world_set_up.is_set():
                return True
            else:
                os.kill(carla_server_pid.value, signal.SIGKILL)
                return False

def set_up_traffic_manager_and_wait_till_its_up(carla_ip:str, rpc_port:int, tm_port:int,
                                                number_of_vehicles:int, number_of_walkers:int,
                                                carla_server_pid:shared_ctype, traffic_manager_pid:shared_ctype,
                                                logs_path:str, hero:bool=True):
    traffic_manager_is_up = multiprocessing.Event()
    set_up_traffic_manager_process = multiprocessing.Process(target=generate_traffic,
                                                             args=(carla_ip, rpc_port, tm_port, number_of_vehicles,
                                                                   number_of_walkers, traffic_manager_is_up,
                                                                   logs_path, hero))
    set_up_traffic_manager_process.start()

    traffic_manager_pid.value = set_up_traffic_manager_process.pid

    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            set_up_traffic_manager_process.kill()
            return False, True, traffic_manager_is_up, set_up_traffic_manager_process # Means Carla Crashed!
        if not set_up_traffic_manager_process.is_alive():
            set_up_traffic_manager_process.join()
            os.kill(carla_server_pid.value, signal.SIGKILL)
            return True, False, traffic_manager_is_up, set_up_traffic_manager_process # Means Traffic Manager Crashed!
        if traffic_manager_is_up.is_set():
            return True, True, traffic_manager_is_up, set_up_traffic_manager_process # Means everything good!
