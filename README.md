# Tag Pose Estimation
Author: Valentin Hartmann, Kiran Doshi. Computational Robotics Lab, ETH Zurich, 2026.

## Installation
To use the pose estimation code, first clone the repository
```
git clone https://github.com/kirandoshi/tag_pose_estimation.git
```

Then cd into the repository and create a python virtual environment and activate it
```
cd tag_pose_estimation/
python3 -m venv venv
source venv/bin/activate
```
Install the required dependencies using pip
```
pip install -r requirements.txt
```

Additionally, the apriltag library needs to be cloned and built to create the
python bindings. Ensure that you have cmake installed. At the top level of this
repository, run
```
bash install_apriltag.sh
```
This will clone the apriltag repository, build the library and create the python
bindings. Ensure that this is done at the top level so the resulting import
paths work correctly. If import errors occur when trying to use the apriltag
detector, check this step again.

Finally, install this package in editable mode
```
pip install -e .
```

## Getting started with pose estimation

### Instrinsic camera calibration
Before running the camera calibration, connect the camera to the computer. 
Then you need to create a Charuco board for the calibration.
You can find a default intrinsic camera calibration Charuco board in 
`configs/calibration_boards/default_intrinsic_charuco_board/`.
The pdf file can be printed and used for the calibration, the json file is used
by the calibration script to load the board parameters and the png file is for
convenience to quickly view the board.

Alternatively, can use the script `scripts/generate_charuco_board.py` to 
generate a custom Charuco board which saves the board files (`.pdf`, `.json`, 
`.png`) under the `configs/calibration_boards` folder.
See the script help for more information on how to use it and the parameters to set.

The intrinsic calibration of a camera can be performed using the script
```
python scripts/camera_intrinsic_calibration.py --cam_name <camera_name_or_id> 
--charuco_board_path [path to charuco board json file] --f [focus value to set]
```
where the charuco_board_path is optional. If not provided, a default board will be used.
The camera_name_or_id can be the serial number of the camera or the device id string
which you can determine on ubuntu using the command 
```
v4l2-ctl --list-devices
```
(You might need to install v4l-utils package `sudo apt install v4l-utils`).

### Extrinsic camera calibration

To estimate the pose of tags in a common world frame, the camera extrinsics
need to be determined in that frame.
To do this either a script for a single camera extrinsic calibration can be used
or a script which works for exactly two cameras only. For more than two cameras,
the extrinsics can either be determined pairwise or the single camera extrinsic
calibration script can be used for each camera individually, but then the 
calibration board needs to be visible in each camera frame.
Only the script for single and dual camera extrinsic calibration is currently
available. Chaining multiple cameras has to be done manually, as it will depend
on the exact camera setup and which cameras have overlapping fields of view.
The dual camera extrinsic calibration script provides the relative pose between
the two cameras which can be used to chain multiple cameras together.

Single camera extrinsic calibration can be performed using the script
```
python scripts/calibrate_extrinsics_single_camera.py --camera_config_path 
[path to camera config file] --board_config_path [path to board config file]
```
where the board config file defines the geometry of the Charuco calibration board
used for the extrinsic calibration (see Intrinsic camera calibration section for
more information on Charuco board generation). The camera config file contains
the information about camera model and further required parameters such as the
intrinsic calibration parameters (thus intrinsic calibration needs to be performed
beforehand). An example camera config file can be found in
`configs/camera_config/default_webcam_config.json`.
Only a single image is needed (and used) to compute the extrinsics, so make sure
that the calibration board is fully visible in the camera frame when taking the
snapshot. The resulting extrinsic calibration file will be saved in the 
`configs/extrinsic_calibration` folder.

Dual camera extrinsic calibration can be performed using the script
```
python scripts/calibrate_extrinsics_dual_camera.py --camera_1_config_path 
[path to first camera config file] --camera_2_config_path [path to second camera 
config file] --board_config_path [path to board config file]
```
Arguments are similar to the single camera extrinsic calibration script, but
the config files for both cameras need to be provided. The resulting extrinsic
calibration files will be saved in the `configs/extrinsic_calibration` folder. 
The script generates the poses of both cameras in the board frame as well as 
the relative pose between the two cameras. To calibrate these two cameras, multiple
frames with both cameras seeing the calibration board need to be taken while 
running the calibration script. Follow the instructions provided by the script in 
the terminal. Images should be taken from different viewpoints to get a good 
calibration result. The quality of the resulting calibration can be seen in the 
printed reprojection error.

The relative pose between the two cameras is saved in a separate file
`[camera_2_name]_to_[camera_1_name]_extrinsics.json`. This homogeneous transform
represents the pose of camera 1 in the frame of camera 2 (i.e. T_camera2_camera1).
The convention is carried over from the OpenCV function which returns the 
extrinsic calibration. Use this transform to chain multiple cameras together if
needed.

### Calibration of the robot base position

The robot base position relative to the world frame (as defined by a calibration
board) needs to be determined to be able to determine the pose of an object
relative to the robot. This can be done using the script
```
python3 scripts/calibrate_robot_base_pose.py 
--robot_type [robot type, e.g. aloha]
--camera_config_path [path to camera config file]
--ee_board_path [path to end-effector board definition file]
--ee_board_tag_type [tag type of the end-effector board, e.g. charuco, aruco, apriltag]
--world_board_path [path to world board definition file]
```

where the camera config file contains the intrinsic and extrinsic calibration 
information of the camera used for the calibration, the end-effector board
definition file contains the geometry of the tags mounted on the robot's
end-effector and the world board definition file contains the geometry of the
tags mounted in the world frame (e.g. on a table). 
The calibration will save a homogeneous transform representing the pose of the robot
base in the world frame in an .npy and .txt file in the `configs/extrinsic_calibration/`
folder. The name of the file will be `[robot_type]_base_pose_(timestamp).npy` and
`.txt`.

To ensure that the robot base and the cameras are in the same world frame, you 
have to ensure that the world board used during robot base calibration is not 
moved compared to the position during the camera extrinsic calibration. 

To ensure that your robot can be calibrated correctly, you need to implement
a robot interface class in `tag_pose_estimation/robot_interfaces/` which 
implements the abstract methods defined in the class `RobotInterface` in
`tag_pose_estimation/robot_interfaces/robot_interface.py`. This interface is 
implemented for the ALOHA robot in 
`tag_pose_estimation/robot_interfaces/aloha_robot_interface.py`. Likely you will need to have another process running for the robot itself which can take the commands and move the robot, though this depends on the implementation on the robot interface. For the ALOHA robot this is the case, the robot needs to be running in a different terminal (see ALOHA documentation.)

_(A potential future alteration, which is not currently implemented, is to determine
the robot base position in the frame of the camera directly, and if the camera
was calibrated extrinsically to the world frame, the robot base position in that 
world frame can be computed by chaining the transforms)_

### Object board creation
To be able to estimate the pose of an object, we need to define the geometry of
the arrangement of tags on the object.
The tags themselves can be either Aruco or AprilTags and need to be printed and
then attached to the object. 
Websites such as this [Aruco Marker Generator](https://chev.me/arucogen/) or
this [AprilTag Generator](https://chaitanyantr.github.io/apriltag.html) can be
used to generate and download the markers for printing. Using a svg editor they
can be arranged as desired in a GUI and then printed.
To simply create such a board definition file, we provide a script
```
python scripts/build_object_board.py 
--marker_size [marker size in meters]
--tag_type [aruco or apriltag]
--tag_family [tag family, e.g. DICT_4X4_50 for aruco, tag36h11 for apriltag]
--camera_config_path [path to camera config file]
--reference_marker [id of reference marker, optional]
--name [name of the board, optional] 
--max_id [maximum marker id to consider, optional]
```
This will open a window showing the camera feed and the detected markers.
By pressing the space bar, a snapshot will be taken and the individually resulting
poses of the detected markers will be visualised. If the detection looks good,
the current frame can be accepted to be used to compute the board pressing 'y or
rejected by pressing 'n'. 
Multiple snapshots need to be taken to connect all markers to each other. For 
better accuracy, it is recommended that multiple markers are visible in each
snapshot. For 3D objects, the markers on different planes need to be visible in
the same snapshot to be able to connect them to each other.
After finishing the data collection, the poses of the markers will be optimized
and the final marker corner positions will be computed in the board frame. 
The board frame is either the center of all markers or the center of the
reference marker if provided.
(An additional offset can be provided to shift the origin to a desired location,
see script help for more information).
The resulting board definition file will be saved in the `configs/object_boards` 
folder.

The generated board definition file can be inspected visually using the
script
```
python scripts/visualise_board.py --filepath [path to board definition file]
```
If the board doesn't look as expected, especially if tags which should be aligned
are shifted strongly relative to each other, it is recommended to re-create the 
board as this inaccuracy will directly affect the pose estimation accuracy.
The accuracy of the camera calibration also should be checked in this case.

### Running pose estimation

The pose estimation process can be run with
```
python3 pose_estimation/run_pose_estimation.py --config=[path to pose estimation config file]
```

where the config has to contain the "camera_configs": a list of paths which point
to the camera configuration files, as well as the "object_board_definitions": a
list of paths which point to the object board definition files and "port" an int
which specifies the ZMQ port to use for publishing the estimated poses.
This script just ensures that the pose estimation is being published via ZMQ, to
obtain the estimated poses, see below.
Additional optional arguments are
```
--tag_type [apriltag or aruco]
--tag_family [tag family, e.g. tag36h11 for apriltag, DICT_4X4_50 for aruco]
--frequency [frequency to run the pose estimation at]
--publish_image [whether to publish the images additionally with the pose estimation]
--use_transform [whether to use an additional transform from the config]
```
these arguments can either be provided from the command line or specified in the
config file. The values provided from the command line will override the values
in the config file. The config file should be saved in the 
`configs/pose_estimation_configs` folder, an example config file can be found
there as well.

### Using the pose estimate in your pipeline
With your pose estimation running in its own terminal you are now close to having
the pose estimate of your objects be useful in your own pipeline. All we need to
do now is use a subscriber which gets the pose and filters it. This subscriber is
implemented in `tag_pose_estimation/board_pose_listener.`
To integrate it into your own script you need to do
```python
from tag_pose_estimation.board_pose_listener import BoardPoseListener

address = (f"tcp://localhost:{PORT}") 
pose_listener = BoardPoseListener(
    box_pose_socket_address=address,
    update_rate=UPDATE_RATE
)
pose_listener.start()
pose = pose_listener.get_pose(OBJ_ID)
```
where the returned pose is a 1x7 numpy array [x, y, z, qw, qx, qy, qz] with the quaternion in scalar first format