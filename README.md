# DB_action_discretization
Preprocessing the database produced by babbling (wave 1,2,3). In particular, discretization of the controller data.

# DATA FORMAT:

Dataset WG :

We agreed on a standard data format in YAML for the data produced by the wave 1-2-3

Two files:

1. This file is an information about the whole iteration

target_info.yml

reward:(1|0) // if the target was a moveable object

target_position:  (the position of the object being interacted with)

   - x
   - y
   - z

2. Information frame per frame in a iteration (concerning the position of the effector, i.e., wrist in our case):

sync_data.yml

frame_ID:  the ID is always: ID = nsec + sec, See example(*)

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



(*) e.g.: ID is computed as sec + nsec => 875000688 = 875000000 + 688
frame_875000688:
   timestamp:
     sec: 688
     nsec: 875000000

     
Note: the frames are not written in sequential order, and therefore, this program sorts them by using the timestamp = sec.nsec). Moreover, the ID = nsec + sec.

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



# In order to generate the original babbling data:

```
Dependencies :
pcl-1.7.2 (last release) which must be compiled with c++11 :
wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.7.2.tar.gz

add around the line 22 of CMakeLists.txt of pcl :
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

Open a terminal in the pcl source directory:
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr ..       (this will overwrite the current PCL installation)
make -j4                                   (or -j8 if you have 8 cores processors)
sudo make install
opencv 2 (last release) and opencv_nonfree :
opencv2 is available in apt with the default repositories of ubuntu 14.04 or 12.04.
For opencv nonfree packages :
sudo add-apt-repository --yes ppa:xqms/opencv-nonfree
sudo apt-get update
sudo apt-get install libopencv-nonfree-dev
boost
Available with apt-get install.
eigen3
Available with apt-get install.
cafer_core
To install it follow the tutorial at this link : https://github.com/robotsthatdream/cafer
libb64
sudo apt-get install libb64-dev
baxter sdk
To install the baxter sdk follow this tutoriel : http://sdk.rethinkrobotics.com/wiki/Workstation_Setup#Step_4:_Install_Baxter_SDK_Dependencies



---
C++11 flag:


if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


