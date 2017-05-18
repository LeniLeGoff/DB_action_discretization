
# coding: utf-8

import yaml
import pandas as pd
import numpy as np
import os
import cv2
from config import INPUT_DIR, OUTPUT_DIR, REWARD_FILE, INPUT_REWARD_FILE
from config import INPUT_JOINT_POSITION_FILE, SUBFOLDER_CONTAINING_IMAGES


####    NOT WORKING #### NOT NEEDED WHEN CONTINUOUS PRIORS ARE APPLIED  TODO: add
# util methods to const.py

def discretize_data(record_id):
    # TODO
    print "Discretizing record ",record_id

def subsample_data(record_id, factor):
    # TODO
    print "Subsampling data ",record_id

def kmeans_clustering_data(data_x, data_y):
    # TODO
    clustered_df = pd.DataFrame(columns=['time', 'x', 'y', 'z', 'reward'])
    print "Dataframe clusterized: ",clustered_df.head()
    clusterized_df.to_csv()

data_iteration_folders = get_folders(OUTPUT_DIR)
for record_id in range(len(data_iteration_folders)):
    real_file_to_simulated_file(record_id)
    for folder in data_iteration_folders:
        record_folder = OUTPUT_DIR+'record_'+str(i)+'/'

print "CREATING CLUSTERIZED DATA FOLDER STRUCTURE DIRECTORIES: ",len(data_iteration_folders)
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
print "CREATING ACTUAL FILES WITHIN THE DIRECTORIES: ",len(data_iteration_folders)
for record_id in range(len(data_iteration_folders)):
    real_file_to_simulated_file(record_id)
