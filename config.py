# coding: utf-8

INPUT_DIR = './baxter_babbling/'
OUTPUT_DIR = './baxter_babbling_converted/'

### TODO: remove the next 3 files when unified new format
REWARD_FILE = 'recorded_button1_is_pressed.txt' #"movable_object_is_pushed.txt" is equiv to recorded_button1_is_pressed.txt right now in 3D simulated learning representations
INPUT_REWARD_FILE = 'dataset_color.yml'
INPUT_JOINT_POSITION_FILE = 'controller_feedback.yml'

### NEW DATA FORMATS:
INPUT_DATA_FILE = 'sync_data.yml'
INPUT_DATA_FILE_TARGET = 'target_info.yml'

# To be generated within each record_X file
SUBFOLDER_CONTAINING_IMAGES = 'recorded_cameras_head_camera_2_image_compressed/' # in our old data, TODO: change to better name?
EFFECTOR_CLOSE_ENOUGH_THRESHOLD = 0.5


"""

We agreed on a standard data format in YAML for the data produced by the wave 1-2-3

Two files:
First one : this file is an information about the whole iteration
target_info.yml
reward:(1|0) // if the target was a moveable object
target_position:
   - x
   - y
   - z

second one: information frame per frame in a iteration.
sync_data.yml
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
    reward: (0|1) // if the endeffector is close from the target
    depth: !!binary " .. "
    rgb: !!binary  " .. "

This is a PAR data. You have for each time step the position and orientation of the robots end-effector, a corresponding image (color and depth) and a reward.

For the wave 1-2 the reward is equal to 1 if the robot within this timestep is interacting with an object.

The data format may have other data for the wave 3.
"""
