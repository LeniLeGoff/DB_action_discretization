
# coding: utf-8

import yaml
import yaml
import pandas as pd
import numpy as np
import os
import cv2
#from cv2 import cv

INPUT_DIR = './baxter_babbling/'
OUTPUT_DIR = './baxter_babbling_converted/'
SUBFOLDER_CONTAINING_IMAGES = 'recorded_cameras_head_camera_2_image_compressed/' # in our old data, TODO: change to better name?

"""
This program uses baxter joint positions in cartesian space and translates
to discrete action space processable by representation_learning_3D program. For the wrist joint we need
the 3 axis rotations given by:
right_w0 right_w1 right_w2
Because each image in the recorded data produces many frame_ID values with joint values for that frame,
we keep only one set of joint values
"""

def img_in_binary2rgb_file(buf, record_id, frame_id, output_folder):
    #image = cv2.imdecode(buf)
    # CV2
    output_path=OUTPUT_DIR+"record_"+str(record_id)+'/'+SUBFOLDER_CONTAINING_IMAGES+'frame'+str(frame_id)+".jpg"
    np_array = np.fromstring(buf, np.uint8)
    img_np = cv2.imdecode(np_array, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    cv2.imwrite(output_path, img_np)#const String& filename, InputArray img, const vector<int>& params=vector<int>() )

def get_folders(directory):
    folders = os.walk(directory)
    #folders = [x[0] if os.path.isdir(x[0]) for x in os.walk(directory)]
    iteration_folders = []
    for f in folders:
        if 'iteration' in f[0]:
            iteration_folders.append(f[0])
    return iteration_folders

def read_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            #print(yaml.load(stream))
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def real_file_to_simulated_file(record_id, input_f='controller_feedback.yml', input_f_reward='dataset_color.yml', output_f='recorded_robot_limb_left_endpoint_action.txt', output_f_reward='recorded_button1_is_pressed.txt'):
    """
    Adds secs to nanosecs for a unique timestamp, creates label =1 if an object being pushed is moving,
    and 0 otherwise (including if an object being pushed is not moving)
    Uses https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters  to convert from joint to cartesian space
    recorded_robot_limb_left_endpoint_action.txt  #time, dx, dy, dz
    recorded_button1_is_pressed.txt  #time, value
    """
    content = read_yaml(input_f)
    # add new format to new_content
    # time x,y,z
    # time, dx, dy, dz      recorded_robot_limb_left_endpoint_action.txt
    df = pd.DataFrame(columns=('#time', 'x', 'y', 'z'))
    df_deltas = pd.DataFrame(columns=('#time', 'dx', 'dy', 'dz'))
    df_reward = pd.DataFrame(columns=('#time', 'value'))
    init_time = 0.0
    timestamps= []
    frame_id=0
    prev_x, prev_y, prev_z = 0,0,0
    for key in content.keys():         # for each image_per_action:
        x,y,z = -1,-1,-1 #TODO: content[key]['position']:   #print x,y,z  # also ['orientation'] available
        new_time = content[key]['timestamp']['sec']+ content[key]['timestamp']['nsec']
        if frame_id ==0: # first frame per iteration or data sequence
            dx, dy, dz = x, y, z
            prev_time = new_time
            dtime = new_time # Not used now
        else:
            dtime = new_time - prev_time
            dx, dy, dz = x-prev_x, y-prev_y, z-prev_z
        df.loc[frame_id] = [new_time, x, y, z]
        df_deltas.loc[frame_id] = [new_time, dx, dy, dz] # dtime if needed
        df_reward.loc[frame_id] = [new_time, 1] # TODO: [new_time, content[key]['reward']]
        prev_x, prev_y, prev_z = dx,dy,dz
        prev_time = new_time           #timestamps.append(new_time)
        #TODO: real img
        str_buffer = get_sample_img_in_binary() #str_buffer = content[key]['rgb']
        read_binary_image(str_buffer, record_id, frame_id)
        frame_id += 1

    print "Writing to files: \n",df.head(),'\n',df_deltas.head()
    print df_reward.head()
    output_path = OUTPUT_DIR+'record_'+str(record_id)+'/'
    df.to_csv(output_path +output_f, header=True, index=False, sep='\t')
    output_f_deltas = output_f.replace('.txt', '_deltas.txt')
    df_deltas.to_csv(output_path +output_f_deltas, header=True, index=False, sep='\t')
    df_reward.to_csv(output_path +output_f_reward, header=True, index=False, sep='\t')

def read_binary_image(string_buffer, record_id, frame_id):
    img_in_binary2rgb_file(string_buffer, record_id, frame_id, OUTPUT_DIR)

### test methods
def read_binary_image_example():
    content = read_yaml('./rgb.yml')
    for frame in content.keys():
        buf =content[frame]['rgb']
        img_in_binary2rgb_file(buf, 0, 86, OUTPUT_DIR)

def get_sample_img_in_binary():
    content = read_yaml('./rgb.yml')
    for frame in content.keys():
        buf =content[frame]['rgb']
        return buf
read_binary_image_example()
####

# real_data ="controller_feedback.yml"
# simulated_data = "simulated_data_pushing_objects.txt" #"recorded_robot_limb_left_endpoint_state.xml"
# real_data_rewards = "dataset_normal.yml"
# simulated_data_rewards = "movable_object_is_pushed.txt" # equiv to recorded_button1_is_pressed.txt
#real_file_to_simulated_file(real_data, simulated_data, real_data_rewards, simulated_data_rewards )

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

data_iteration_folders = get_folders(INPUT_DIR)

print "CREATING DATA FOLDER STRUCTURE DIRECTORIES: ",len(data_iteration_folders)
i = 0
for folder in data_iteration_folders:
    record_folder = OUTPUT_DIR+'record_'+str(i)+'/'
    if not os.path.exists(record_folder):
        os.makedirs(record_folder)
    actual_images_path = record_folder+SUBFOLDER_CONTAINING_IMAGES
    if not os.path.exists(actual_images_path):
        os.makedirs(actual_images_path)
    i +=1

# CREATING ACTUAL FILES WITHIN THE DIRECTORIES
record_id = 0
print "CREATING ACTUAL FILES WITHIN THE DIRECTORIES: ",len(data_iteration_folders)
for record_id in range(len(data_iteration_folders)):
    real_file_to_simulated_file(record_id)
