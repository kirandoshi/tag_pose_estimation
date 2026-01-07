# Tag Pose Estimation
Author: Valentin Hartmann, Kiran Doshi. CRL, ETH Zurich, 2025.

Initial Readme copied from current pose estimation repo - needs to be updated.

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

## Getting started with pose estimation

#### Instrinsic camera calibration
Before running the camera calibration, connect the camera to the computer. 
Then you need to create a Charuco board for the calibration.
You can find a default intrinsic camera calibration Charuco board in 
`configs/calibration_boards/default_intrinsic_charuco_board`.
The pdf file can be printed and used for the calibration, the json file is used
by the calibration script to load the board parameters and the png file is for
convenience to quickly view the board.

Alternatively, can use the script `scripts/generate_charuco_board.py` to 
generate a custom Charuco board which saves the board files (`.pdf`, `.json`, 
`.png`) under the `configs/calibration_boards` folder.
See the script help for more information on how to use it and the parameters to set.

The intrinsic calibration of a camera can be performed using the script
```
python scripts/camera_intrinsic_calibration.py --serial_number 242322072500 --charuco_board_path ./configs/boards/charuco_board_5x7_100.json
```
where the charuco_board_path is optional. If not provided, a default board will be used.
A camera can be calibrated using
```
python3 pose_estimation/camera_calibration.py --serial_number 242322072500
```

### Calibration of the robot base position
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
The pose estimation process can be run with
```
python3 pose_estimation/pose_estimator.py --config=[path to env config] --detection_type=ransac_with_refinement
```

where the config has to contain the paths to the robot configurations and the camera configuration path(s) - Multiple cameras are supported.

### Visualizing the pose estimation
A simple script to visualize the pose estimation can be run using
```
python3 pose_estimation/visualize_pose_estimation.py  --mode=threaded_tracking --config [path to env config]
```

### Example of an environment config file
```
{
  "port": 5557,
  "cameras": [
    {
      "type": "D405",
      "serial_number": "238722073187",
      "calibration_file": "./calibration/238722073187_20250605_105751_homogenous_transform.npy"
    }
  ],
  "robots": ["./calibration/20250605_110533_base_pose_robot_right.npy"],
  "boards": [
    "./configs/boards/cube_1.json",
    "./configs/boards/cube_2.json",
    "./configs/boards/cube_3.json"
  ]
}

```