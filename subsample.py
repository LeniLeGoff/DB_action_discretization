
# coding: utf-8

import yaml
import pandas as pd
import numpy as np
import os
import cv2
#import CONFIG


####

def discretize_data(record_id):
    print "Discretizing record ",record_id

def subsample_data(record_id, factor):
    print "Subsampling data ",record_id

def kmeans_clustering_data(data_x, data_y):
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
