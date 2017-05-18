# DB_action_discretization
Preprocessing the database produced by babbling (wave 1,2,3). In particular, discretization of the controller data.

# DATA FORMAT:

Dataset WG :

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


# DATA CONVERSION:

1) Place baxter_babbling in the main directory

2) Run real_data2simulated_discrete_data.py

3) baxter_babbling_converted folder will be created with the data ready to be processed by the current repository of discrete actions in
https://github.com/Mathieu-Seurin/baxter_representation_learning_3D

This folder contains a record_X file per original iteration_X folder in the input baxter_babbling directory.


# (Not needed for now) TODO:

Discretize actions by performing clustering (k-means or other?)

Subsampling of actions by performing a 1/r ratio frame selection (making sure the movement among two consecutive frames shows an actual movement of Baxter arm)
