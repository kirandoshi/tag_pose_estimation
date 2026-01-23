# Tag Pose Estimation
Author: Valentin Hartmann, Kiran Doshi. Computational Robotics Lab, ETH Zurich, 2026.

## ToDos and Open Questions Repo
- Keep support for aruco and apriltag or only apriltag?
- How to make compatible with different robots? Add a robot wrapper class?
- Add in readme how to generate the python_apriltag bindings from the C code
- Merge apriltag and aruco tag files which do the same, e.g. pose_estimator
- Clean up old files, clean out commented code and unused code in all files
- Update all docstrings and comments
- Add type hints to all functions
- Add unit tests
- Separate scripts which run code from library code, put scripts in a 'scripts' 
  folder outside of the main package folder

## Installation
To use the pose estimation code, first clone the repository
```
git clone https://github.com/yourusername/tag-pose-estimation.git
```

Then create a python virtual environment and activate it
```
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
./install_apriltag.sh
```
This will clone the apriltag repository, build the library and create the python
bindings. 

Finally, install the tag-pose-estimation package in editable mode
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

### Calibration of the robot base position
*Section to be updated*

The robot base position relative to the 0-position can be calibrated using
```
python3 pose_estimation/robot_base_calibration.py --name [name] --robot_config_path [path to robot config] -s [camera serial number]
```

For this script to work, the impedance controller needs to be started before running the calibration script using

```
./impedance_controller [path to robot config]
```

Since the robot moves during the calibration, the other robot should be moved out of the way.

The calibration tool that is used needs to mounted on the correct side.
The robot configuration will be saved in the 'calibration' folder.

### Running pose estimation
*Section to be updated*

The pose estimation process can be run with
```
python3 pose_estimation/run_pose_estimation.py --config=[path to pose estimation config file]
```

where the config has to contain the "camera_configs": a list of paths which point
to the camera configuration files, as well as the "object_board_definitions": a
list of paths which point to the object board definition files and "port" an int
which specifies the ZMQ port to use for publishing the estimated poses.
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