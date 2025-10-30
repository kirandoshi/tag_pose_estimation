import numpy as np

import cv2
import cv2.aruco as aruco

import pyrealsense2 as rs

import zmq
import copy

import json

import sys
import os
import time
import datetime

import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tag_pose_estimation.utils import get_larger_board
from tag_pose_estimation.transform_utils import (
    pose_to_homogeneous,
    seven_d_to_homogeneous,
    scalar_last_to_scalar_first,
)

from scipy.spatial.transform import Rotation as R


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


def load_ee_calibration_board(ee_calibration_board_path):
    def load_single_aruco_board(board_config):
        with open(board_config, "r") as f:
            board_data = json.load(f)

        # load which dict from config
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, board_data["dictionary"])
        )

        # load ids and corners from config
        marker_ids_list = []
        marker_corners_list = []

        for marker in board_data["markers"]:
            marker_ids_list.append(marker["id"])
            marker_corners_list.append(marker["corners"])

        board = cv2.aruco.Board(
            objPoints=np.array(marker_corners_list, np.float32),
            dictionary=aruco_dict,
            ids=np.array(marker_ids_list),
        )

        return board

    return load_single_aruco_board(ee_calibration_board_path)


def perturb_quaternion(q, angle_std_deg=5.0):
    """
    Apply a small random rotational perturbation to a quaternion.

    Parameters:
    - q: array-like of shape (4,) — original quaternion (x, y, z, w)
    - angle_std_deg: standard deviation of perturbation angle in degrees

    Returns:
    - perturbed quaternion as a numpy array (x, y, z, w)
    """
    # Generate small random rotation vector (axis-angle), with angle ~ N(0, angle_std_deg)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # normalize to get random direction
    angle_rad = np.random.normal(0, np.deg2rad(angle_std_deg))
    delta_rotvec = axis * angle_rad

    # Convert original quaternion to scipy Rotation
    r_orig = R.from_quat(q)  # scipy expects [x, y, z, w]
    r_delta = R.from_rotvec(delta_rotvec)

    # Apply perturbation
    r_perturbed = r_delta * r_orig
    return r_perturbed.as_quat()  # returns [x, y, z, w]


def main(
    name,
    zmq_ip="127.0.0.1",
    zmq_controller_port=5555,
    zmq_state_est_port=5556,
    camera_serial_number=None,
):
    print("setting up robot pose estimation socket")
    robot_pose_context = zmq.Context()
    robot_pose_socket = robot_pose_context.socket(zmq.SUB)
    robot_pose_socket.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
    robot_pose_socket.connect(f"tcp://{zmq_ip}:{zmq_state_est_port}")
    robot_pose_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("setting up robot control socket")
    controller_context = zmq.Context()
    controller_publisher = controller_context.socket(zmq.PUB)
    controller_publisher.bind(f"tcp://{zmq_ip}:{zmq_controller_port}")

    # board that we are using for calibration
    # board, charuco_marker_dictionary = get_board()
    board, charuco_marker_dictionary = get_larger_board(False)

    ee_calibration_board_path = "./calibration/robot_calibration_board.json"
    ee_board = load_ee_calibration_board(ee_calibration_board_path)

    # Initialize webcam
    # cap = cv2.VideoCapture(0)

    pipeline = rs.pipeline()
    config = rs.config()

    if camera_serial_number:
        config.enable_device(camera_serial_number)

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # color_sensor = profile.get_device().query_sensors()[0]
    # color_sensor.set_option(rs.option.enable_auto_exposure, True)
    # Get the sensor once at the beginning. (Sensor index: 1)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]

    # Set the exposure anytime during the operation
    # sensor.set_option(rs.option.exposure, 10000.000)
    # sensor.set_option(rs.option.enable_auto_exposure, True)
    # sensor.set_option(rs.option.enable_auto_white_balance, True)
    # sensor.set_option(rs.option.sharpness, 100)

    # Get camera intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # Create camera matrix from intrinsics
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Get distortion coefficients
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)

    # move robot around a bit
    # take pictures and save robot ee-pose along with it
    # compute basepose from it, and export
    robot_start_state = None

    while True:
        # update real state
        try:
            robot_start_state = robot_pose_socket.recv_json(flags=zmq.NOBLOCK)
            print("received robot state")
        except zmq.Again:
            pass

        if robot_start_state is not None:
            break

    robot_ee_start_pose = robot_start_state["pos"]

    # fill list with poses that we are going to do
    desired_ee_poses = []

    offset = 0.1
    dirs = [
        (offset / 2, offset / 2),
        (offset / 2, -offset / 2),
        (-offset / 2, -offset / 2),
        (-offset / 2, offset / 2),
    ]

    # square in xy
    for i in range(4):
        pose = copy.deepcopy(robot_ee_start_pose)
        pose[0] += dirs[i][0]
        pose[1] += dirs[i][1]

        perturbed_quat = perturb_quaternion(pose[3:])
        pose[3:] = perturbed_quat

        desired_ee_poses.append(pose)

    # square in z
    for i in range(4):
        pose = copy.deepcopy(robot_ee_start_pose)
        pose[0] += dirs[i][0]
        pose[2] += dirs[i][1]

        perturbed_quat = perturb_quaternion(pose[3:])
        pose[3:] = perturbed_quat

        desired_ee_poses.append(pose)

    robot_base_poses = []

    curr_pose_idx = 0
    while True:
        if curr_pose_idx >= len(desired_ee_poses):
            break

        robot_state = None
        while True:
            command = {"target_ee_pose": list(desired_ee_poses[curr_pose_idx])}
            # print(f"Sending target pose: {command}")
            controller_publisher.send_json(command)

            try:
                robot_state = robot_pose_socket.recv_json(flags=zmq.NOBLOCK)
                # print("received robot state")
            except zmq.Again:
                pass

            if robot_state is None:
                continue

            position_error = np.linalg.norm(
                np.array(robot_state["pos"][:3])
                - np.array(desired_ee_poses[curr_pose_idx][:3])
            )
            q1_inv = R.from_quat(robot_state["pos"][3:]).inv()
            q_rel = q1_inv * R.from_quat(
                desired_ee_poses[curr_pose_idx][3:]
            )  # relative rotation from q1 to q2

            # Get angle difference in radians:
            angle_diff = q_rel.magnitude()

            if position_error < 1e-2 and angle_diff < 1e-2:
                curr_pose_idx += 1

                time.sleep(0.5)
                break
            # else:
            #     print(f"Pose diff too big: {position_error}")

            time.sleep(0.01)

        # take picture, estimate board pose, and ee-pose
        max_img_taking_attempts = 10
        for _ in range(max_img_taking_attempts):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())

            # Capture frame
            # ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, charuco_marker_dictionary)

            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)

                # cv2.imshow("Camera image", frame)

                # while True:
                #     key = cv2.waitKey(1) & 0xFF
                #     if key == ord("c"):
                #         break

                # world frame calibration
                retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
                if retval > 0:
                    cv2.drawChessboardCorners(frame, (5, 4), charuco_corners, True)

                if retval > 4:  # Need at least 4 corners
                    rvec = np.zeros((3, 1), dtype=np.float32)
                    tvec = np.zeros((3, 1), dtype=np.float32)

                    success = aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids,
                        board=board,
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
                retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    corners,
                    ids,
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

                    def rot_x(theta):
                        """Rotation around x-axis by theta radians"""
                        c, s = np.cos(theta), np.sin(theta)
                        return np.array(
                            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
                        )

                    # First rotation: 90° around x-axis
                    T1 = rot_x(np.pi / 2)

                    offset = np.eye(4)
                    # offset[2, 3] = 0.145
                    # offset[2, 3] = 0.115
                    # offset[1, 3] = -0.05
                    offset[2, 3] = 0.112
                    offset[1, 3] = -0.055
                    # print(offset)

                    # Combined transformation (first rotate around x, then around z)
                    adapter_to_ee_pose = T1 @ offset

                    ee_pose_world_frame = (
                        ee_adapter_pose_world_frame @ adapter_to_ee_pose
                    )

                    # robot_state[pos] is [pos][quat], where quat is xyzw
                    tmp = scalar_last_to_scalar_first(robot_state["pos"])

                    # tmp = copy.deepcopy(robot_state["pos"])
                    # tmp[3] = robot_state["pos"][6]
                    # tmp[4] = robot_state["pos"][3]
                    # tmp[5] = robot_state["pos"][4]
                    # tmp[6] = robot_state["pos"][5]
                    ee_pose_robot_frame = seven_d_to_homogeneous(np.array(tmp))

                    robot_base_pose = ee_pose_world_frame @ np.linalg.inv(
                        ee_pose_robot_frame
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

                    break
                else:
                    print("Unable to locate end effector.")

            else:
                print("Was not able to identify corners.")

    command = {"target_ee_pose": list(robot_ee_start_pose)}
    # print(f"Sending target pose: {command}")
    controller_publisher.send_json(command)

    print("Computed base pose:")
    if len(robot_base_poses) > 1:
        # average robot_base_poses:
        estimated_base_pose = average_pose_estimates(robot_base_poses)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # we export this twice, once as a timed version to have access to a specific version,
        # and secondly to a generic version, which we generally use in our files.
        output_filepath = f"./calibration/base_pose_robot_{name}.npy"
        timed_output_filepath = f"./calibration/{timestamp}_base_pose_robot_{name}.npy"
        
        np.save(output_filepath, estimated_base_pose)
        np.save(timed_output_filepath, estimated_base_pose)
        
        print(f"Saved to {output_filepath} and {timed_output_filepath}")
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
        "-s", "--serial_number",
        type=str,
        help="Serial number of the camera that should be used for calibration.",
        required=True,
    )
    # parser.add_argument(
    #     "--export_camera_pose",
    #     type=bool,
    #     default=False,
    #     help="Robot config path.",
    #     required=True,
    # )
    parser.add_argument(
        "-r", "--robot_config_path",
        type=str,
        help="Robot config path.",
        required=True,
    )
    args = parser.parse_args()

    try:
        with open(args.robot_config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: Config file not found.", file=sys.stderr)
        sys.exit(1)

    camera_serial_number = args.serial_number

    main(
        args.name,
        zmq_controller_port=config["socket_port"],
        zmq_state_est_port=config["publisher_port"],
        camera_serial_number=camera_serial_number,
    )
