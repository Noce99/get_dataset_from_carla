import os
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    carla_events_folder_path = "/media/enrico/Enrico_Datasets/carla_events/"
    folders = os.listdir(carla_events_folder_path)
    folders = [el for el in folders if el.startswith("2025")]

    unified_folder_path = os.path.join(carla_events_folder_path, "events_10ms_all_cities")

    if not os.path.exists(unified_folder_path):
        os.mkdir(unified_folder_path)

    data_folder_counter = 0
    for folder in tqdm(folders):
        current_folder_path = os.path.join(carla_events_folder_path, folder)
        all_data_folders = os.listdir(current_folder_path)
        for data_folder in all_data_folders:
            data_folder_path = os.path.join(current_folder_path, data_folder)
            shutil.copytree(data_folder_path, os.path.join(unified_folder_path, f"{data_folder_counter:03d}"))
            data_folder_counter += 1

