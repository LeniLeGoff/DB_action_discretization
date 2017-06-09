# coding: utf-8
import yaml
import pandas as pd
import numpy as np
import os, sys
import cv2
import math

from config import INPUT_DIR, OUTPUT_DIR, INPUT_DATA_FILE, INPUT_DATA_FILE_TARGET, SUBFOLDER_CONTAINING_RECORDS_PATTERN_INPUT
from config import SUBFOLDER_CONTAINING_RECORDS_PATTERN_OUTPUT, EFFECTOR_CLOSE_ENOUGH_THRESHOLD
from config import OUTPUT_FILE, OUTPUT_FILE_REWARD, SUB_DIR_IMAGE, FRAME_START_INDEX

BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'
PUSHING_BUTTON_AUGMENTED = 'pushingButton3DAugmented'

LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"
DATA_FOLDER = MOBILE_ROBOT
#DATA_FOLDER = BABBLING

PLOT_CONFIGURATIONS = [2, 3]


"""
This program uses baxter joint positions in cartesian space and translates
to discrete action space processable by representation_learning_3D program. For the wrist joint we need
the 3 axis rotations given by:
right_w0 right_w1 right_w2
Because each image in the recorded data produces many frame_ID values with joint values for that frame,
we keep only one set of joint values
The output files of the program look as follows:

OUTPUT_FILE (to be FILENAME_FOR_STATE in https://github.com/Mathieu-Seurin/baxter_representation_learning_3D)
         #time         x         y         z
0  289000578.0  0.816698  0.249241 -0.179920
1  487000576.0  0.757203  0.480915  0.400777
2  110000581.0  0.816698  0.249241 -0.179920
3  206000574.0  0.695541  0.432679  0.428267
4  787000579.0  0.816698  0.249241 -0.179920

second OUTPUT FILE (to be FILENAME_FOR_STATE_DELTAS)
         #time        dx        dy        dz
0  289000578.0  0.816698  0.249241 -0.179920
1  487000576.0 -0.059495  0.231674  0.580697
2  110000581.0  0.876193  0.017567 -0.760617
3  206000574.0 -0.180652  0.415112  1.188885
4  787000579.0  0.997350 -0.165870 -1.368805

OUTPUT_FILE_REWARD
         #time  value
0  289000578.0    0.0
1  487000576.0    0.0
2  110000581.0    0.0
3  206000574.0    0.0
4  787000579.0    0.0

"""

def get_folders(directory):
    folders = os.walk(directory)
    iteration_folders = []
    for f in folders:
        if 'record' in f[0]:
            iteration_folders.append(f[0])
    return iteration_folders

def read_yaml(filename):
    with open(filename, 'r') as stream:
        try:            #print(yaml.load(stream))
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def real_file_to_simulated_file(record_id, input_f=INPUT_DATA_FILE, input_f_target = INPUT_DATA_FILE_TARGET, output_f=OUTPUT_FILE, output_f_reward=OUTPUT_FILE_REWARD):  #input_f_reward=INPUT_REWARD_FILE,
    """
    Adds secs to nanosecs for a unique timestamp, creates label =1 if an object being pushed is moving,
    and 0 otherwise (including if an object being pushed is not moving)
    Uses https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters  to convert from joint to cartesian space
    """
    content = read_yaml(INPUT_DIR+SUBFOLDER_CONTAINING_RECORDS_PATTERN_INPUT.replace('X', str(record_id))+input_f)  # sync_dataset.yml for babbling
    content_effector = read_yaml(INPUT_DIR+SUBFOLDER_CONTAINING_RECORDS_PATTERN_INPUT.replace('X', str(record_id))+input_f_target)  # target_info.yml for babbling
    object_was_pushed_reward = content_effector['reward']
    #print 'object_was_pushed in data iteration? reward in INPUT_DATA_FILE_TARGET (important to be able to apply Causality prior or not) = ', object_was_pushed_reward

    # add new format to new_content
    # time, dx, dy, dz      recorded_robot_limb_left_endpoint_action.txt
    df = pd.DataFrame(columns=('#time', 'x', 'y', 'z'))
    df_deltas = pd.DataFrame(columns=('#time', 'dx', 'dy', 'dz'))
    df_reward = pd.DataFrame(columns=('#time', 'value'))
    init_time = 0.0
    timestamps= []
    frame_id= FRAME_START_INDEX # we start from 1 so that later LUA sorting keeps the order preserved
    prev_x, prev_y, prev_z = 0,0,0
    for key in content.keys():         # for each image per action:
        x,y,z = content[key]['position']  #print x,y,z  # also ['orientation'] available
        new_time = content[key]['timestamp']['sec'] * math.pow(10, 9) + content[key]['timestamp']['nsec']
        if frame_id == FRAME_START_INDEX: # first frame per iteration or data sequence
            dx, dy, dz = x, y, z
            prev_x, prev_y, prev_z = x, y, z
            prev_time = new_time
            dtime = new_time # Not used now
        else:
            dtime = new_time - prev_time
            dx, dy, dz = x-prev_x, y-prev_y, z-prev_z

        df.loc[frame_id] = [new_time, x, y, z]
        df_deltas.loc[frame_id] = [new_time, dx, dy, dz] # dtime if needed
        if object_was_pushed_reward == 1 and content[key]['reward'] > EFFECTOR_CLOSE_ENOUGH_THRESHOLD:
            # reward 1   # we can play without loosing info on the proximity precision
            df_reward.loc[frame_id] = [new_time, 1]
        else: # reward 0
            df_reward.loc[frame_id] = [new_time, 0]
        prev_x, prev_y, prev_z = x, y, z
        prev_time = new_time
        timestamps.append(new_time)
        str_buffer = content[key]['rgb']
        # The id of the frame is its timestamp in nanosecs, because they are not ordered in the yml file
        img_in_binary2rgb_file(str_buffer, record_id, str(int(new_time))) #frame_id) 
        frame_id += 1

    output_path = OUTPUT_DIR+ SUBFOLDER_CONTAINING_RECORDS_PATTERN_OUTPUT.replace('X', str(record_id))
    
    # Sorting frames, as they are not written in the original yml files in timely consecutive real order
    df.sort_values(by='#time', inplace=True )
    df_deltas.sort_values(by='#time', inplace=True )
    df_reward.sort_values(by='#time', inplace=True )

    df.to_csv(output_path +output_f, header=True, index=False, sep='\t')
    output_f_deltas = output_f.replace('.txt', '_deltas.txt')
    df_deltas.to_csv(output_path +output_f_deltas, header=True, index=False, sep='\t')
    df_reward.to_csv(output_path +output_f_reward, header=True, index=False, sep='\t')
    #  consistency sanity check on the nr of timestamps, frames and rewards
    if len(df) != len(df_deltas) or len(df) != len(df_reward):
        print('Output data is inconsistent, length of actions and rewards should all be equal')
        sys.exit(-1)

def plot_states_and_rewards(all_data_df, all_rewards_df):

    test_file = 'saveImagesAndRepr.txt'

def get_positive_reward_ratio(df):
    return df[df.value == 1.0].shape[0] / float(df[df.value != 1.0].shape[0])


def plot_3D(x =[1,2,3,4,5,6,7,8,9,10], y =[5,6,2,3,13,4,1,2,4,8], z =[2,3,3,3,5,7,9,11,9,10], axes_labels = ['U','V','W'], title='Learned representations-rewards distribution\n', dataset=''):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')  # 'r' : red

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    ax.set_title(title+dataset)
    

################

####   MAIN program
################


data_iteration_folders = get_folders(OUTPUT_DIR)
if len(data_iteration_folders)==0:
    print OUTPUT_DIR," not found: No output data folder was created yet, run first real_data2simulated_data.py using the input directory ", INPUT_DIR
    sys.exit(-1)

records = pd.DataFrame(columns=('#time', 'x', 'y', 'z'))
records_rewards = pd.DataFrame(columns=('#time', 'value'))

positive_reward_ratio = 0

print "READING DATA FOLDER STRUCTURE DIRECTORIES: ",len(data_iteration_folders)
folder_pattern_index = 1
for folder in data_iteration_folders:
    record_folder = OUTPUT_DIR+ SUBFOLDER_CONTAINING_RECORDS_PATTERN_OUTPUT.replace('X', str(folder_pattern_index))
    # reading each record_X folder
    if os.path.exists(record_folder):
        output_path = OUTPUT_DIR+ SUBFOLDER_CONTAINING_RECORDS_PATTERN_OUTPUT.replace('X', str(folder_pattern_index))
        data_file = output_path +OUTPUT_FILE
        reward_file = output_path +OUTPUT_FILE_REWARD

        # adding each record's position data to a general overall dataframe
        data_df = pd.read_csv(data_file, sep='\t') #    df_deltas = pd.DataFrame(columns=('#time', 'dx', 'dy', 'dz'))
        rewards_df = pd.read_csv(reward_file, sep='\t')
        print data_df.head()
        print rewards_df.head()
        records = records.append(data_df)#, ignore_index= True)
        records_rewards = records_rewards.append(rewards_df)#, ignore_index=True)
        folder_pattern_index += 1

# all_data_df = pd.concat(records)
# all_rewards_df = pd.concat(records_rewards)
# Sorting frames, as they are not written in the original yml files in timely consecutive real order
records.sort_values(by='#time', inplace=True )
records_rewards.sort_values(by='#time', inplace=True )

print records.head()
print records_rewards.head()
print "Final data contains ", len(records), ' datapoints and ', len(records), ' rewards (', round(get_positive_reward_ratio(records_rewards), 3),'% positive rewards)'

# Plot all actions (position) data
print "PLOTTING ALL ACTUAL FILES WITHIN THE DIRECTORIES... Nr of records: ",len(data_iteration_folders), " in ",DATA_FOLDER
plot_states_and_rewards(records, records_rewards)
