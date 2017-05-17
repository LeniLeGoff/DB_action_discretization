# DB_action_discretization
Preprocessing the database produced by babbling (wave 1,2,3). In particular, discretization of the controller data.


# DATA CONVERSION:

1) Place baxter_babbling in the main directory

2) Run real_data2simulated_discrete_data.py

3) baxter_babbling_converted folder will be created with the data ready to be processed by the current repository of discrete actions in
https://github.com/Mathieu-Seurin/baxter_representation_learning_3D

This folder contains a record_X file per original iteration_X folder in the input baxter_babbling directory.


# TODO:

Discretize actions by performing clustering (k-means or other?)

Subsampling of actions by performing a 1/r ratio frame selection (making sure the movement among two consecutive frames shows an actual movement of Baxter arm)
