# coding: utf-8

INPUT_DIR = './baxter_babbling/'
OUTPUT_DIR = './baxter_babbling_converted/'

###
REWARD_FILE = 'recorded_button1_is_pressed.txt' #"movable_object_is_pushed.txt" is equiv to recorded_button1_is_pressed.txt right now in 3D simulated learning representations
INPUT_REWARD_FILE = 'dataset_color.yml'
INPUT_JOINT_POSITION_FILE = 'controller_feedback.yml'
# To be generated within each record_X file
SUBFOLDER_CONTAINING_IMAGES = 'recorded_cameras_head_camera_2_image_compressed/' # in our old data, TODO: change to better name?
