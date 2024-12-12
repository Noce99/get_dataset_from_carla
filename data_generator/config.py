import math
import datetime

# GENERAL
# DATASET_PATH = "/home/enrico/Downloads/nut_dataset_test" # "/leonardo_work/IscrC_SSNeRF/nut_dataset"
# VALIDATION_DATASET_PATH = "/home/enrico/Downloads/nut_dataset_old/nut_dataset_test"

# TAKE DATA
MAX_NUM_OF_ATTEMPTS = 10  # maximum number of attempts to start up all the carla's chain!
CARLA_FPS = 200
IMAGE_W = 1024
IMAGE_H = 256
BEV_IMAGE_W = 256
BEV_IMAGE_H = 256
BEV_ALTITUDE = 1000
BEV_BOTTOM_ALTITUDE = 17
BEV_SQUARE_SIDE_IN_M = 40
# MAXIMUM_DISTANCE_FROM_VEHICLE_IN_BEV = BEV_SQUARE_SIDE_IN_M/math.sqrt(2)
BEV_FOV_IN_DEGREES = (2 * math.atan(BEV_SQUARE_SIDE_IN_M / (2 * BEV_ALTITUDE)))/math.pi * 180
BEV_BOTTOM_FOV_IN_DEGREES = (2 * math.atan(BEV_SQUARE_SIDE_IN_M / (2 * BEV_BOTTOM_ALTITUDE)))/math.pi * 180
AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE = 5  # minimum is 3
NUM_OF_WAYPOINTS = 10
DISTANCE_BETWEEN_WAYPOINTS = 1  # m
DISTANCE_BETWEEN_TARGETPOINTS = 30  # m
MINIMUM_DISTANCE_FOR_NEXT_TARGETPOINT = 15  # m
MINIMUM_LIDAR_HEIGHT = 0.3  # m
MAXIMUM_LIDAR_HEIGHT = 3  # m
HOW_MANY_CARLA_FRAME_FOR_CALCULATING_SPEEDS = 3
TOWN_DICT = {1:  "Town01", 2:  "Town02", 3:  "Town03", 4:  "Town04",
             5:  "Town05",  # 6:  "Town06",  # BUGGY IN DATA CREATION
             7:  "Town07", 10: "Town10HD", }  # 12: "Town12", 13: "Town13", 15: "Town15"}  # BUGGY IN DATA CREATION
MINIMUM_SPEED_DIFFERENCE_TO_PUT_ACCELERATION_TO_0 = 0.05
NUM_OF_BBS_PER_FRAME = 100

# DATA LOADER
JOB_TMP_DIR = None
JOB_TMP_DIR_NAME = "nut_tmp"
"""
DATASET_FOLDER_STRUCT = [("rgb_A_0", ".jpg"),           ("rgb_A_1", ".jpg"),        ("rgb_A_2", ".jpg"),        ("rgb_A_3", ".jpg"),
                         ("rgb_B_0", ".jpg"),           ("rgb_B_1", ".jpg"),        ("rgb_B_2", ".jpg"),        ("rgb_B_3", ".jpg"),
                         ("depth_0", ".png"),           ("depth_1", ".png"),        ("depth_2", ".png"),        ("depth_3", ".png"),
                         ("optical_flow_0", ".png"),    ("optical_flow_1", ".png"), ("optical_flow_2", ".png"), ("optical_flow_3", ".png"),
                         ("semantic_0", ".png"),        ("semantic_1", ".png"),     ("semantic_2", ".png"),     ("semantic_3", ".png"),
                         ("bev_semantic", ".png"),      ("bev_lidar", ".png")]
"""
# CACHE_SIZE_LIMIT =  0 # int(0.1 * 1e9) # Bytes (GB without the 1e9)

# TRAIN
BATCH_SIZE = 10
LEARNING_RATE = 7.5e-5
LEARNING_RATE_DECAY = 0.1
START_EPOCH = 0
PRETRAINED_WEIGHT_PATH = None
PRETRAINED_OPTIMIZER_PATH = None
PRETRAINED_SCHEDULER_PATH = None
PRETRAINED_SCALER_PATH = None
REDUCE_LR_FIRST_TIME = 30
REDUCE_LR_SECOND_TIME = 40
NUM_OF_TENSORBOARD_LOGS_PER_EPOCH = 10
FLOW_LOSS_MULTIPLIER = 10000
TRAIN_SPEED = True
TRAIN_ACCELERATION = True

# TMP
START_TIME = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

# VISUALIZATION
NUTFUSER_SEMANTIC_COLOR = { 0 : (255, 255, 255),
                            1 : (0,     0,   0),
                            2 : (0,   255, 255),
                            3 : (255, 255,   0),
                            4 : (127,   0, 255),
                            5 : (255,   0, 255),
                            6 : (0,     0, 255),
                            7 : (0,   255,   0) }

# DRIVING SYSTEM
LIDAR_RECTANGLE_WIDTH = 20
LIDAR_RECTANGLE_HEIGHT = 120
MINIMUM_AMOUNT_OF_OBSTACLES_IN_FRONT_WHILE_MOVING = 3
MINIMUM_AMOUNT_OF_OBSTACLES_IN_FRONT_WHILE_STOP = 15
NUMBER_OF_FRAMES_AFTER_WE_SAY_WE_ARE_STUCK = 10
FRAMES_NEEDED_TO_UNSTACK = 10
SPEED_GAIN_TO_KEEP_SPEED = 0.1

# EVALUATION
DISTANCE_BETWEEN_TARGETPOINTS_IN_EVALUATION_ROUTES = DISTANCE_BETWEEN_TARGETPOINTS / 4
