# coding: utf-8

INPUT_DIR = 'babbling_original_files_yml/'
OUTPUT_DIR = 'babbling/'

# INPUT_REWARD_FILE = 'dataset_color.yml' # #"movable_object_is_pushed " is equiv to recorded_button1_is_pressed.txt right now in 3D simulated learning representations
# INPUT_JOINT_POSITION_FILE = 'controller_feedback.yml'
INPUT_DATA_FILE = 'sync_dataset.yml'  # the effector (robot wrist joint) data
INPUT_DATA_FILE_TARGET = 'target_info.yml'  # the target object being moved
SUBFOLDER_CONTAINING_RECORDS_PATTERN_INPUT = 'iteration_X/'

# To be generated within each record_X file
SUBFOLDER_CONTAINING_RECORDS_PATTERN_OUTPUT= 'record_X/'
EFFECTOR_CLOSE_ENOUGH_THRESHOLD = 0.2
OUTPUT_FILE = "state_pushing_object.txt" #''  FILENAME_FOR_STATE =  -- equiv to recorded_button1_is_pressed.txt right now in 3D simulated learning representations  # FILENAME_FOR_STATE
OUTPUT_FILE_REWARD = "reward_pushing_object.txt" #"state_pushing_object.txt" #FILENAME_FOR_REWARD
SUB_DIR_IMAGE = 'baxter_pushing_objects/'
FRAME_START_INDEX = 1  # we start from 1 so that later LUA sorting keeps the order preserved

"""
We agreed on a standard data format in YAML for the data produced by the wave 1-2-3

Two files:
First one: this file is an information about the whole iteration
target_info.yml
reward:(1|0) // if the target was a moveable object
target_position:
   - x
   - y
   - z

second one: information per frame in a iteration, concerning the robot's effector (wrist joint orientation and position in our case)
sync_dataset.yml
frame_ID:
    timestamp:
        sec: float
        nsec: float
    position: !!seq
        - x
        - y
        - z
    orientation: !!seq
        - x
        - y
        - z
        - w
    reward: (0|1) // if the end effector is close from the target
    depth: !!binary " .. "
    rgb: !!binary  " .. "

This is a PAR data. You have for each time step the position and orientation of the robots end-effector, a corresponding image (color and depth) and a reward.

For the wave 1-2 the reward is equal to 1 if the robot within this timestep is interacting with an object.

The data format may have other data for the wave 3.
"""
