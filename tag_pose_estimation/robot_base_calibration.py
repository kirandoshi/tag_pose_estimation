import numpy as np

import cv2
import cv2.aruco as aruco

import copy

import json

import sys
import time
import datetime

import argparse
from scipy.spatial.transform import Rotation as R

from tag_pose_estimation.camera_wrappers import (
    RealSenseCamera,
    WebcamCamera)
from tag_pose_estimation.transform_utils import (
    pose_to_homogeneous,
)
from tag_pose_estimation.utils import load_boards, get_project_root
from tag_pose_estimation.robot_interface.robot_interface import RobotInterface
from tag_pose_estimation.robot_interface.aloha_robot_interface import (
    AlohaRobotInterface
)

def average_pose_estimates(transforms):
    # Extract translations and rotations
    translations = np.array([T[:3, 3] for T in transforms])
    rotations = np.array(
        [R.from_matrix(T[:3, :3]).as_quat() for T in transforms]
    )  # (x, y, z, w)

    # Average translation
    avg_translation = np.mean(translations, axis=0)

    # Average quaternion (normalize sum of quaternions)
    avg_quat = np.mean(rotations, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)

    # Convert back to rotation matrix
    avg_rotation = R.from_quat(avg_quat).as_matrix()

    # Assemble averaged transform
    avg_transform = np.eye(4)
    avg_transform[:3, :3] = avg_rotation
    avg_transform[:3, 3] = avg_translation

    return avg_transform

def perturb_quaternion(
        q, 
        angle_std_deg=5.0, 
        rng=None):
    """
    Apply a small random rotational perturbation to a quaternion.

    Parameters:
    - q: array-like of shape (4,) — original quaternion (x, y, z, w)
    - angle_std_deg: standard deviation of perturbation angle in degrees
    - rng: optional, numpy random generator for reproducibility

    Returns:
    - perturbed quaternion as a numpy array (x, y, z, w)
    """
    # Generate small random rotation vector (axis-angle), with angle ~ N(0, angle_std_deg)
    if rng is None:
        rng = np.random.default_rng(seed=0)
        print("Creating default rng inside perturb_quaternion with seed 0")
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)  # normalize to get random direction
    angle_rad = rng.normal(0, np.deg2rad(angle_std_deg))
    delta_rotvec = axis * angle_rad

    # Convert original quaternion to scipy Rotation
    r_orig = R.from_quat(q)  # scipy expects [x, y, z, w]
    r_delta = R.from_rotvec(delta_rotvec)

    # Apply perturbation
    r_perturbed = r_delta * r_orig
    return r_perturbed.as_quat()  # returns [x, y, z, w]


def main(
    name,
    calibration_config_path,
):
    # Start robot communication 
    robot = RobotInterface()

    # Load robot base calibration config
    # Open config file
    try:
        with open(calibration_config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {calibration_config_path} not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Config file {calibration_config_path} is not a valid JSON.")
    
    # Get boards from config
    if "boards" not in config or len(config["boards"]) == 0:
        raise ValueError("No boards specified in the config file.")
    
    boards_abs_paths = config["boards"]
    board_target_types = config.get("target_types", None)
    assert board_target_types is not None, (
        "target_types must be specified in the config file."
    )
    assert len(board_target_types) == len(boards_abs_paths), (
        "target_types and boards must have the same length."
    )
    board_types = config.get("board_types", None)
    assert board_types is not None, (
        "board_types must be specified in the config file."
    )
    assert len(board_types) == len(boards_abs_paths), (
        "board_types and boards must have the same length."
    )

    # Load the main board and ee_board based on their target types
    ee_board_path = None
    ee_board_type = None
    main_board_path = None
    main_board_type = None
    for i, target_type in enumerate(board_target_types):
        if target_type == "ee_board":
            if ee_board_path is not None:
                raise ValueError("Multiple ee_board entries found in config file.")
            ee_board_path = boards_abs_paths[i]
            ee_board_type = board_types[i]
        elif target_type == "main_board":
            if main_board_path is not None:
                raise ValueError("Multiple main_board entries found in config file.")
            main_board_path = boards_abs_paths[i]
            main_board_type = board_types[i]
        else:
            raise ValueError(f"Unknown target type {target_type} in config file.")
        
    # Load main board
    if main_board_path is not None:
        main_board = load_boards([main_board_path], [main_board_type])[0]
    else:
        raise ValueError("No main board specified in the config file.")

    assert main_board_type == "charuco", "Main board must be a charuco board."

    # Get the board dictionary from the loaded board
    main_board_dictionary = main_board.getDictionary()

    # Board that is used to calibrate the end-effector pose and is mounted to
    # the end-effector
    if ee_board_path is not None:
        ee_board = load_boards([ee_board_path], [ee_board_type])[0]
    else:
        raise ValueError("No ee_board specified in the config file.")

    # Get the board dictionary from the loaded board
    ee_board_dictionary = ee_board.getDictionary()

    # Initialize Camera
    # If neither is specified, abort.
    camera_dict_list = config["cameras"]

    if len(camera_dict_list) != 1:
        print("Error: This script only supports calibrating one camera at a time.", file=sys.stderr)
        sys.exit(1)
    
    camera_dict = camera_dict_list[0]

    serial_number = None
    if "serial_number" in camera_dict:
        serial_number = camera_dict["serial_number"]

    if camera_dict["type"] == "D405":
        camera = RealSenseCamera(
            serial_number=serial_number
        )
    elif camera_dict["type"] == "Webcam":
        camera = WebcamCamera(
            camera_id=serial_number,
            camera_matrix_path=camera_dict["camera_matrix"],
            camera_dist_path=camera_dict["dist_coeff"],
            focus_path=camera_dict.get("focus_setting")
        )
    else:
        raise ValueError(f"Unknown camera type {camera_dict['type']}")

    # Get camera matrix
    camera_matrix = camera.camera_matrix
    # Get distortion coefficients
    dist_coeffs = camera.dist_coeffs

    # move robot around a bit
    # take pictures and save robot ee-pose along with it
    # compute basepose from it, and export
    robot_start_state = None

    while True:
        # update real state
        robot_start_state = robot.get_arm_current_ee_pose()
        print("received robot state")

        if robot_start_state is not None:
            break
    robot_ee_start_pose = robot_start_state

    # fill list with poses that we are going to do
    desired_ee_poses = [robot_ee_start_pose]

    offset = 0.075
    dirs = [
        (offset / 2, offset / 2),
        (offset / 2, -offset / 2),
        (-offset / 2, -offset / 2),
        (-offset / 2, offset / 2),
    ]

    # Initialise a numpy rng
    rng = np.random.default_rng(seed=12)

    # square in xy
    for i in range(4):
        pose = copy.deepcopy(robot_ee_start_pose)
        pose[0] += dirs[i][0]
        pose[1] += dirs[i][1]

        perturbed_quat = perturb_quaternion(pose[3:], angle_std_deg=1.5, rng=rng)
        pose[3:] = perturbed_quat

        desired_ee_poses.append(pose)

    # square in z
    for i in range(4):
        pose = copy.deepcopy(robot_ee_start_pose)
        pose[0] += dirs[i][0]
        pose[2] += dirs[i][1]

        perturbed_quat = perturb_quaternion(pose[3:], angle_std_deg=2.5, rng=rng)
        pose[3:] = perturbed_quat

        desired_ee_poses.append(pose)

    robot_base_poses = []
    actual_ee_poses = []

    curr_pose_idx = 0
    while True:
        if curr_pose_idx >= len(desired_ee_poses):
            break

        robot_state = None
        attempts_to_reach_pose = 0
        while True:
            commanded_pose = desired_ee_poses[curr_pose_idx]

            robot.move_arm_to_target_pose(commanded_pose)

            robot_state = robot.get_arm_current_ee_pose()

            if robot_state is None:
                continue

            # Convert robot_state to inline pose using get_inline_pose_from_T
            robot_state_inline_pose = robot_state['pose_inline']
            position_error = np.linalg.norm(
                np.array(robot_state_inline_pose[:3])
                - np.array(desired_ee_poses[curr_pose_idx][:3])
            )
            q1_inv = R.from_quat(robot_state_inline_pose[3:]).inv()
            q_rel = q1_inv * R.from_quat(
                desired_ee_poses[curr_pose_idx][3:]
            )  # relative rotation from q1 to q2

            # Get angle difference in radians:
            angle_diff = q_rel.magnitude()

            if position_error < 9e-2 and angle_diff < 9e-2:
                curr_pose_idx += 1

                time.sleep(0.5)
                break
            
            if attempts_to_reach_pose > 10:
                print("Failed to reach pose, moving to next one")
                curr_pose_idx += 1
                break

            time.sleep(0.01)
            attempts_to_reach_pose += 1

        # take picture, estimate board pose, and ee-pose
        max_img_taking_attempts = 10
        for _ in range(max_img_taking_attempts):
            frame = camera.get_frame(flush_buffer=True)
            # print(f"id(frame): {id(frame)}")
            if frame is None:
                print("Camera frame is None, retrying...")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            calib_corners, calib_ids, _ = aruco.detectMarkers(gray, main_board_dictionary)

            if calib_ids is not None:
                aruco.drawDetectedMarkers(frame, calib_corners, calib_ids)

                # world frame calibration
                retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    calib_corners, calib_ids, gray, main_board
                )
                if retval > 0:
                    cv2.drawChessboardCorners(frame, (4, 6), charuco_corners, True)

                if retval > 4:  # Need at least 4 corners
                    rvec = np.zeros((3, 1), dtype=np.float32)
                    tvec = np.zeros((3, 1), dtype=np.float32)

                    success = aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids,
                        board=main_board,
                        cameraMatrix=camera_matrix,
                        distCoeffs=dist_coeffs,
                        rvec=rvec,
                        tvec=tvec,
                    )

                    if success:
                        cv2.drawFrameAxes(
                            frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1
                        )

                        R_x_180 = np.array(
                            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                        )

                        # Rotation matrix for 90 degrees around the z-axis
                        R_z_90 = np.array(
                            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                        )

                        # Combined transformation (first rotate around x, then around z)
                        T_board_to_world = R_z_90 @ R_x_180

                        orig_rvec = rvec
                        orig_tvec = tvec

                        hom_board_in_camera_frame = pose_to_homogeneous(
                            orig_rvec, orig_tvec
                        )
                        hom_cam_pose_in_world_frame = T_board_to_world @ np.linalg.inv(
                            hom_board_in_camera_frame
                        )
                    else:
                        print("Unable to estimate the charuco board pose")
                        continue
                else:
                    print("Unable to find the charuco-corners of the world-frame-board")
                    continue

                # ee calibration
                # detect ee board
                print("Estimating ee-pose")

                # Detect ee board, which might use a different dictionary
                ee_corners, ee_ids, _ = aruco.detectMarkers(
                    gray, ee_board_dictionary
                )
                if ee_ids is not None:
                    aruco.drawDetectedMarkers(frame, ee_corners, ee_ids)
                else:
                    print("No ee board markers detected")
                    continue
                
                if ee_board_type == "charuco":
                    retval, ee_charuco_corners, ee_charuco_ids = aruco.interpolateCornersCharuco(
                        ee_corners, ee_ids, gray, ee_board, 
                        cameraMatrix=camera.camera_matrix, 
                        distCoeffs=camera.dist_coeffs
                    )
                    rvec = np.zeros((3, 1), dtype=np.float32)
                    tvec = np.zeros((3, 1), dtype=np.float32)

                    # Gives back the pose at the top right corner of the board
                    retval = aruco.estimatePoseCharucoBoard(
                        ee_charuco_corners,
                        ee_charuco_ids,
                        ee_board,
                        camera.camera_matrix,
                        camera.dist_coeffs,
                        rvec,
                        tvec,
                    )

                    # Transform from top left corner to board center
                    if retval:
                        # Vector from top left corner to board center in
                        # frame of the board as computed by
                        # estimatePoseCharucoBoard
                        # Get the board number of squares in x and y direction
                        # and the square length
                        num_squares_x = ee_board.getChessboardSize()[0]
                        num_squares_y = ee_board.getChessboardSize()[1]
                        square_length = ee_board.getSquareLength()

                        # Vector from top right corner to board center in
                        # frame of the board as computed by 
                        # estimatePoseCharucoBoard
                        vec_topright_to_center_boardframe = np.array([
                            (num_squares_x * square_length) / 2,
                            (num_squares_y * square_length) / 2,
                            0,
                        ]).reshape((3, 1))
                        R_board_to_camera, _ = cv2.Rodrigues(rvec)
                        t_board_to_camera = tvec
                        vec_topright_to_center_cameraframe = (
                            R_board_to_camera @ vec_topright_to_center_boardframe
                            + t_board_to_camera
                        )
                        tvec = vec_topright_to_center_cameraframe
                        # Orientation of board is 180 degree rotated around 
                        # x-axis
                        R_x_180 = np.array(
                            [[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]]
                        )
                        R_board_to_camera = R_board_to_camera @ R_x_180
                        rvec, _ = cv2.Rodrigues(R_board_to_camera)
                    else:
                        print("Unable to estimate the ee charuco board pose")
                        continue

                elif ee_board_type == "aruco":
                    # Estimate the pose of the ee board
                    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                        ee_corners,
                        ee_ids,
                        ee_board,
                        camera_matrix,
                        dist_coeffs,
                        None,
                        None,
                    )

                if retval:
                    cv2.drawFrameAxes(
                        frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1
                    )

                    ee_adapter_pose_world_frame = (
                        hom_cam_pose_in_world_frame @ pose_to_homogeneous(rvec, tvec)
                    )

                    # Construct pose of EE-frame in the frame of the calibration
                    # adapter
                    # CA = Calibration adapter frame
                    # EE = End-effector frame

                    # Rotation between EE-frame and calibration adapter is fixed
                    # in constants and is known from the design of the adapter
                    # and the definition of the EE-frame depending on the robot
                    T_CA_EE = robot.get_CA_frame_to_EE_frame_transform()

                    # Rename adapter pose in world frame for clarity
                    T_W_CA = ee_adapter_pose_world_frame

                    # T_W_EE is the pose of the end-effector in the world frame
                    T_W_EE = (
                        T_W_CA @ T_CA_EE
                    )

                    # Now compute the robot base pose in the world frame
                    # Use the matrix form of the current EE-pose
                    # Use the following relation:
                    # T_W_RB = T_W_EE @ T_EE_RB
                    # T_EE_RB = inv(T_RB_EE)
                    # where:
                    # T_W_RB: Robot base pose in world frame
                    # T_W_EE: End-effector pose in world frame
                    # T_RB_EE: End-effector pose in robot base frame, which is
                    # the current robot state read from the robot (robot_state)
                    robot_state_matrix = robot_state['pose_matrix']
                    robot_base_pose = T_W_EE @ np.linalg.inv(
                        robot_state_matrix
                    )

                    cv2.imshow("Camera image", frame)

                    use_this_estimate = False
                    print("Press 'y' if this is acceptable, 'n' otherwise.")
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("y"):
                            use_this_estimate = True
                            break

                        if key == ord("n"):
                            break

                    if use_this_estimate:
                        robot_base_poses.append(robot_base_pose)
                        actual_ee_poses.append(robot_state_matrix)

                    # destroy window
                    cv2.destroyAllWindows()

                    break
                else:
                    print("Unable to locate end effector.")

            else:
                print("Was not able to identify corners.")

    # Shut down robot
    robot.shutdown()

    print("Computed base pose:")
    if len(robot_base_poses) > 1:
        print(f"Using {len(robot_base_poses)} observations to compute average.")
        # average robot_base_poses:
        estimated_base_pose = average_pose_estimates(robot_base_poses)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        root = get_project_root()
        timed_output_filepath = root / (
            "pose_estimation_scripts/"
            + f"calibration/{timestamp}_base_pose_robot_{name}.npy")

        np.save(timed_output_filepath, estimated_base_pose)

        # Print the base pose as text file as a homogeneous transformation matrix
        text_output_filepath = root / (
            "pose_estimation_scripts/"
            + f"calibration/{timestamp}_base_pose_robot_{name}.txt")

        with open(text_output_filepath, "w") as f:
            f.write(f"# Robot base pose for robot {name}\n")
            f.write(f"# Estimated on {datetime.datetime.now().isoformat()}\n")
            f.write(f"# Using {len(robot_base_poses)} observations\n")
            f.write(f"Camera type: {camera_dict['type']}\n")
            f.write(f"Camera id: {serial_number}\n")
            f.write(f"Camera matrix:\n")
            f.write(np.array2string(camera_matrix))
            f.write(f"Distortion coefficients:\n")
            f.write(np.array2string(dist_coeffs))
            f.write("\n")
            f.write(f"Actual end effector poses used in estimation:\n")
            for pose in actual_ee_poses:
                f.write(np.array2string(pose))
                f.write("\n")
            f.write("\n")
            f.write("Output:\n")
            f.write("# 4x4 transformation matrix\n")
            f.write(np.array2string(estimated_base_pose))

        print(f"Base pose (4x4 transformation matrix):\n{estimated_base_pose}")
        print(f"Saved to {text_output_filepath}")
        print(f"Saved to {timed_output_filepath}")
    else:
        print("Did not find a sufficient number of observations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robot base calibration with ArUco markers."
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="Name of the robot under which the calibration should be saved.",
        required=True,
    )
    parser.add_argument(
        "-c", "--calibration_config_path",
        type=str,
        help="Path to the calibration config file.",
        required=True,
    )

    args = parser.parse_args()

    main(
        args.name,
        calibration_config_path=args.calibration_config_path
    )
