import sys
import argparse
import os
import time
import datetime

import cv2
import pyrealsense2 as rs
import numpy as np
import zmq
import copy
import json

from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

from tag_pose_estimation.utils import get_larger_board
from tag_pose_estimation.robot_base_calibration import (
    load_ee_calibration_board,
    perturb_quaternion
)


np.set_printoptions(suppress=True, precision=5)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def vecs2T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rvec.flatten()).as_matrix()
    T[:3, 3] = tvec.flatten()
    return T

def pose2T(pose):
    pose = np.asarray(pose)
    assert pose.shape == (7,)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T

def T2pose(T):
    T = np.asarray(T)
    assert T.shape == (4, 4)
    pose = np.zeros(7)
    pose[:3] = T[:3, 3].flatten()
    pose[3:] = R.from_matrix(T[:3, :3]).as_quat()
    return pose


def main(
    zmq_ip="127.0.0.1",
    zmq_controller_port=5555,
    zmq_state_est_port=5556,
    camera_serial_number=None,
    name=None
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
    board, charuco_marker_dictionary = get_larger_board(False)

    ee_calibration_board_path = "./calibration/robot_calibration_board.json"
    ee_board = load_ee_calibration_board(ee_calibration_board_path)

    pipeline = rs.pipeline()
    config = rs.config()

    if camera_serial_number:
        config.enable_device(camera_serial_number)

    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 6)

    # Start streaming
    profile = pipeline.start(config)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
    sensor.set_option(rs.option.exposure, 1000.000)

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

    robot_start_state = None
    while True:
        try:
            robot_start_state = robot_pose_socket.recv_json(flags=zmq.NOBLOCK)
            print("received robot state")
        except zmq.Again:
            pass

        if robot_start_state is not None:
            break

    robot_ee_start_pose = robot_start_state["pos"]

    T_base2gripper_arr, T_cam2marker_arr = [], []
    N_poses = 24
    for _ in range(N_poses):
        # random ee pose
        ee_pose = copy.deepcopy(robot_ee_start_pose)
        rand_range = np.ones(3) * 0.05        # 5cm in every direction
        ee_pose[:3] += np.random.uniform(low=-rand_range, high=rand_range)
        ee_pose[3:] = perturb_quaternion(ee_pose[3:], 17)

        robot_state = None
        command = {"target_pose": list(ee_pose)}
        controller_publisher.send_json(command)
        while True:
            time.sleep(0.2)
            robot_state = robot_pose_socket.recv_json(flags=zmq.NOBLOCK)
            if np.linalg.norm(robot_state['vel']) < 0.0005:
                time.sleep(0.2)
                break
            
        # get the frame and robot pose at the same time
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        robot_state = robot_pose_socket.recv_json(flags=zmq.NOBLOCK)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, charuco_marker_dictionary)

        if ids is None:
            print("no markers detected")
            continue

        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners,
            ids,
            ee_board,
            camera_matrix,
            dist_coeffs,
            None,
            None,
        )

        if not retval:
            print("unable to locate EE")
            continue
        
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
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
        if not use_this_estimate:
            continue

        T_cam2marker_ = vecs2T(rvec=rvec, tvec=tvec)
        T_base2gripper_ = pose2T(robot_state['pos'])

        T_cam2marker_arr.append(T_cam2marker_)
        T_base2gripper_arr.append(T_base2gripper_)
        print("--> successfully appended transformations")

    command = {"target_pose": list(robot_ee_start_pose)}
    controller_publisher.send_json(command)
    time.sleep(1)

    assert len(T_cam2marker_arr) == len(T_base2gripper_arr)
    print(f"\nGathered {len(T_cam2marker_arr)} poses")
    R_base2gripper_arr = [T[:3, :3] for T in T_base2gripper_arr]
    t_base2gripper_arr = [T[:3, 3] for T in T_base2gripper_arr]
    R_marker2cam_arr = [np.linalg.inv(T)[:3, :3] for T in T_cam2marker_arr]
    t_marker2cam_arr = [np.linalg.inv(T)[:3, 3] for T in T_cam2marker_arr]

    # hand 2 eye calibration here, we solve for X in the equation AX = XB
    # where A is the robot motion (pose changes of ee in base frame) (Tb_1⁻¹ @ Tb_2)
    # and B is the observed motion of the marker in camera frame (Tc_1⁻¹ @ Tc_2)
    R_ee2marker, t_ee2marker = cv2.calibrateHandEye(
        R_base2gripper_arr, t_base2gripper_arr,
        R_marker2cam_arr, t_marker2cam_arr,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_ee2marker = np.eye(4)
    T_ee2marker[:3, :3] = R_ee2marker
    T_ee2marker[:3, 3] = t_ee2marker.flatten()
    print("\nee2marker pose:  ", T2pose(T_ee2marker))
    print("should be around -11cm in Z, and 6cm in Y")

    T_marker2ee = np.linalg.inv(T_ee2marker)

    # loop through every pose and calculate T_camera2robotBase
    T_camera2robotBase = []
    for T_base2gripper, T_cam2marker in zip(T_base2gripper_arr, T_cam2marker_arr):
        T_ee2robotBase = np.linalg.inv(T_base2gripper)
        T_c2rb = T_cam2marker @ T_marker2ee @ T_ee2robotBase
        T_camera2robotBase.append(T_c2rb)

    T_camera2robotBase_pose = np.asarray([T2pose(T) for T in T_camera2robotBase])
    print("\ncam 2 robot base pose")
    print(T_camera2robotBase_pose.shape)
    print("mean:   ", np.mean(T_camera2robotBase_pose, axis=(0)))
    print("std:   ", np.std(T_camera2robotBase_pose, axis=(0)))

    # do some outlier detection and remove them
    mean = np.mean(T_camera2robotBase_pose, axis=0)
    cov = np.cov(T_camera2robotBase_pose, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    dists = np.array([mahalanobis(f, mean, inv_cov) for f in T_camera2robotBase_pose])
    threshold = np.sqrt(chi2.ppf(0.95, df=6))  # 95% confidence
    outliers = dists > threshold
    print(f"{sum(outliers)} of the {T_camera2robotBase_pose.shape[0]} camera to robot base transformations were rejected")

    T_camera2robotBase_pose_filtered = T_camera2robotBase_pose[~outliers, ...]
    T_camera2robotBase_mean_pose = np.zeros(7)
    T_camera2robotBase_mean_pose[:3] = np.mean(T_camera2robotBase_pose_filtered[:, :3], axis=0)
    rotation = R.from_quat(T_camera2robotBase_pose_filtered[:, 3:])
    T_camera2robotBase_mean_pose[3:] = rotation.mean().as_quat()

    # back to transformation matrix
    T_camera2robotBase_mean = pose2T(T_camera2robotBase_mean_pose)
    print("\ncam 2 robot base pose filtered  ", T_camera2robotBase_pose_filtered.shape)
    print("mean:   ", T_camera2robotBase_mean_pose)
    print("std:   ", np.std(T_camera2robotBase_pose_filtered, axis=(0)))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = f"./calibration/T_cam_to_robot_base_{name+'_' if name else''}{timestamp}.npy"
    np.save(output_filepath, T_camera2robotBase_mean)
    print(f"\ncalibrated transformation saved to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robot hand to eye with ArUco markers."
    )
    parser.add_argument(
        "-s", "--serial_number",
        type=str,
        help="Serial number of the camera that should be used for calibration.",
        required=True,
    )
    parser.add_argument(
        "-r", "--robot_config_path",
        type=str,
        help="Robot config path.",
        required=True,
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="Name of the robot under which the calibration should be saved.",
        required=False,
        default=None
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
        zmq_controller_port=config["socket_port"],
        zmq_state_est_port=config["publisher_port"],
        camera_serial_number=camera_serial_number,
        name=args.name
    )
