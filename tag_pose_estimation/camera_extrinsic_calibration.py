import numpy as np

import cv2
import cv2.aruco as aruco

import pyrealsense2 as rs

import time
import datetime

import json
import argparse

import sys
import os

from tag_pose_estimation.transform_utils import (
    pose_to_homogeneous,
)
from tag_pose_estimation.camera_wrappers import (
    RealSenseCamera,
    WebcamCamera,
)
from tag_pose_estimation.utils import get_larger_board

def main():
    parser = argparse.ArgumentParser(description="Camera calibration with ArUco markers.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="calibration/camera_config.json",
        help="Path to camera configuration file.",
    )
        
    args = parser.parse_args()
    config_path = args.config_path

    # Open config file
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: Config file not found.", file=sys.stderr)
        sys.exit(1)

    # board that we are using for calibration
    # board, charuco_marker_dictionary = get_board(True)
    board, charuco_marker_dictionary = get_larger_board(False)
    # board, charuco_marker_dictionary = get_even_larger_board(False)

    # Get camera: either a webcam by cam name or a realsense by serial number
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

    camera_id = camera_dict["name"] if "name" in camera_dict else serial_number
    print(f"Starting calibration for camera {camera_id}")

    cv2.namedWindow(
        f"Camera {camera_id}", cv2.WINDOW_NORMAL
    )  # explicit window creation

    camera_matrix = camera.camera_matrix
    dist_coeffs = camera.dist_coeffs
    save_id = camera_id
    

    # capture a couple of images, and get the position of a
    # fixed board.
    estimate_pose = False

    print("Press 'c' to use the current frame for calibration.")

    while True:
        frame = camera.get_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, charuco_marker_dictionary)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
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

        cv2.imshow("ArUco Markers", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and ids is not None and retval > 0:
            estimate_pose = True
            break

        # Break loop with 'q' key
        if key == ord("q"):
            break

        # Rate limiting
        time.sleep(0.01)

    # compute the relative pose of the camera to the board.
    if estimate_pose:
        cv2.drawFrameAxes(
            frame,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            0.1,
        )

        # Aruco boards have the z-axis pointing in a weird direction.
        # We rotate it to make the z positive.
        R_x_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Rotation matrix for 90 degrees around the z-axis
        R_z_90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Combined transformation (first rotate around x, then around z)
        T_board_to_world = R_z_90 @ R_x_180

        orig_rvec = rvec
        orig_tvec = tvec

        hom_board_in_camera_frame = pose_to_homogeneous(orig_rvec, orig_tvec)
        hom_cam_pose_in_world_frame = T_board_to_world @ np.linalg.inv(
            hom_board_in_camera_frame
        )

        print("Board pose in camera frame.")
        print(hom_board_in_camera_frame)

        print("Camera pose in world frame.")
        print(hom_cam_pose_in_world_frame)

        print("Press 'enter' to close and export the calibration.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure the calibration folder exists
        if not os.path.exists("calibration"):
            os.makedirs("calibration")

        np.save(f"calibration/{save_id}_{timestamp}_homogenous_transform.npy", hom_cam_pose_in_world_frame)
        np.save(f"calibration/{save_id}_homogenous_transform.npy", hom_cam_pose_in_world_frame)

        print("Calibration successful. Camera parameters saved.")

        cv2.imshow("ArUco Markers", frame)
        cv2.waitKey(0)  # Ensure the window stays open
        cv2.destroyAllWindows()

    # we assume knowledge of the board pose wrt. robot base


if __name__ == "__main__":
    main()
