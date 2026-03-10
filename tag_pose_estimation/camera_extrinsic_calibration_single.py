import numpy as np

import cv2
import cv2.aruco as aruco

import time
import datetime

import json

import sys
import os
from pathlib import Path
import logging

from tag_pose_estimation.transform_utils import (
    pose_to_homogeneous,
)
from tag_pose_estimation.camera_wrappers import (
    RealSenseCamera,
    T265RealSenseCamera,
    WebcamCamera,
)
from tag_pose_estimation.utils import (
    handle_config_path, 
    load_charuco_board_from_json,
    get_extrinsic_calibration_save_folder,
)

# Module logger
logger = logging.getLogger(__name__)

def single_camera_extrinsic_calibration(
        camera_config_path: str,
        board_config_path: str,
) -> None:
    
    # Check and handle camera config path
    camera_config_path = handle_config_path(
        camera_config_path,
        Path("config") / "camera_config",
        logger=logger
    )

    # The config file should exist now
    # Load config
    with open(camera_config_path) as f:
        config = json.load(f)

    # Check and handle board config path
    board_config_path = handle_config_path(
        board_config_path,
        Path("config") / "calibration_boards",
        logger=logger
    )

    # Board path is a directory. All boards are stored in a file called charuco_board.json
    if Path(board_config_path).is_dir():
        board_config_path = str(
            Path(board_config_path) / "charuco_board.json"
        )
    if not Path(board_config_path).exists():
        raise FileNotFoundError(
            f"Board config file {board_config_path} does not exist. "
            "The provided path must be a directory containing a file called "
            "'charuco_board.json' "
        )
    
    
    # Load board
    board, charuco_marker_dictionary = load_charuco_board_from_json(
        board_config_path
    )

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
    elif camera_dict["type"] == "T265":
        camera = T265RealSenseCamera(
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
        raise ValueError(f"Unsupported camera type {camera_dict['type']}")

    camera_id = camera_dict["name"] if "name" in camera_dict else serial_number
    print(f"Starting extrinsic calibration for camera {camera_id}")

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

        if frame is None:   
            continue

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

        cv2.imshow("ChArUco Markers", frame)

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

        # ChAruco boards have the z-axis pointing in a weird direction.
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

        # Get save folder
        save_folder = get_extrinsic_calibration_save_folder()

        np.save(
            save_folder / f"{save_id}_{timestamp}_extrinsic_calib_hom_transform.npy", 
            hom_cam_pose_in_world_frame
        )
        convenience = False
        if convenience:
            # Also save with a generic name for convenience
            np.save(save_folder / f"{save_id}_extrinsic_calib_hom_transform.npy", 
                    hom_cam_pose_in_world_frame)
            

        # Save calibration infos to a text file as well
        txt_file_path = save_folder / f"{save_id}_{timestamp}_extrinsic_calib_description.txt"
        with open(txt_file_path, "w") as f:
            f.write("Extrinsic Calibration Results\n")
            f.write(f"Camera Name/ID: {camera_id}\n")
            f.write(f"Date and Time: {timestamp}\n\n")
            f.write("\nCamera Pose (Homogeneous Transformation) in World Frame:\n")
            f.write(np.array2string(hom_cam_pose_in_world_frame))

        print("Calibration successful. Camera parameters saved.")

        cv2.imshow("ChArUco Markers", frame)
        cv2.waitKey(0)  # Ensure the window stays open
        cv2.destroyAllWindows()

        return None