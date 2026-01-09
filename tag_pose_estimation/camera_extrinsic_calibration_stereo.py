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
    T265RealSenseCamera,
    WebcamCamera,
)
from tag_pose_estimation.utils import (
    load_charuco_board_from_json,
    get_larger_board,
)

def main(config_path, display_image=True):
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: Config file not found.", file=sys.stderr)
        sys.exit(1)

    # Get charuco board path from config
    charuco_board_path = config.get("charuco_board_path", None)

    cameras = []

    serial_numbers = []
    camera_names = []

    for i, camera_dict in enumerate(config["cameras"]):
        serial_number = None
        if "serial_number" in camera_dict:
            serial_number = camera_dict["serial_number"]

        serial_numbers.append(serial_number)

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

        cameras.append(camera)
        camera_id = camera_dict["name"] if "name" in camera_dict else serial_number
        camera_names.append(camera_id)

    if len(cameras) > 2:
        raise NotImplementedError

    if charuco_board_path is not None and os.path.exists(charuco_board_path):
        board, charuco_marker_dictionary = load_charuco_board_from_json(
            charuco_board_path)
        print(f"Loaded Charuco board from {charuco_board_path}")
    else:
        board, charuco_marker_dictionary = get_larger_board(False)
        print("Using default larger Charuco board with 4x4 markers on 5x4 grid,"
              " marker length 0.04216 m.")

    print("Press 'c' to use the current frame for calibration.")

    frames_0 = []
    frames_1 = []

    while True:
        frame_0 = cameras[0].get_frame()
        frame_1 = cameras[1].get_frame()

        frame_0_draw = frame_0.copy()
        frame_1_draw = frame_1.copy()

        # Detect aruco markers
        gray_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        corners_0, ids_0, _ = cv2.aruco.detectMarkers(gray_0, charuco_marker_dictionary)
        corners_1, ids_1, _ = cv2.aruco.detectMarkers(gray_1, charuco_marker_dictionary)
        if ids_0 is not None:
            cv2.aruco.drawDetectedMarkers(frame_0_draw, corners_0, ids_0)
        if ids_1 is not None:
            cv2.aruco.drawDetectedMarkers(frame_1_draw, corners_1, ids_1)

        if display_image:
            # Make windows resizable
            cv2.namedWindow("ArUco Markers 0", cv2.WINDOW_NORMAL)
            cv2.namedWindow("ArUco Markers 1", cv2.WINDOW_NORMAL)

            cv2.imshow("ArUco Markers 0", frame_0_draw)
            cv2.imshow("ArUco Markers 1", frame_1_draw)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            frames_0.append(frame_0)
            frames_1.append(frame_1)
            print(f"Captured frame pair {len(frames_0)}")

        if key == ord("q"):
            break

        # Rate limiting
        time.sleep(0.01)

    # estimate relative pose
    all_corners_left = []
    all_ids_left = []
    all_corners_right = []
    all_ids_right = []

    all_obj_pts = []
    counter = 0

    # 3. Load stereo image pairs
    for img_left, img_right in zip(frames_0, frames_1):
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 4. Detect ArUco markers first
        corners_left, ids_left, _ = cv2.aruco.detectMarkers(
            gray_left, charuco_marker_dictionary
        )
        corners_right, ids_right, _ = cv2.aruco.detectMarkers(
            gray_right, charuco_marker_dictionary
        )

        print(f"Detected {len(corners_left)} markers in left image."
              f" Detected {len(corners_right)} markers in right image.")
        print(f"IDs left: {ids_left.flatten() if ids_left is not None else ids_left}")
        print(f"IDs right: {ids_right.flatten() if ids_right is not None else ids_right}")

        if ids_left is not None and ids_right is not None:
            # 5. Interpolate ChArUco corners
            retval_left, charuco_corners_left, charuco_ids_left = (
                cv2.aruco.interpolateCornersCharuco(
                    corners_left, ids_left, gray_left, board
                )
            )
            retval_right, charuco_corners_right, charuco_ids_right = (
                cv2.aruco.interpolateCornersCharuco(
                    corners_right, ids_right, gray_right, board
                )
            )

            # Find common markers between left and right images
            if charuco_ids_left is None or charuco_ids_right is None:
                continue
            common_ids = np.intersect1d(charuco_ids_left, charuco_ids_right)

            # Filter corners and ids to only include common markers
            if len(common_ids) >= 6:
                print(f"Using image pair {counter}, {len(common_ids)} common IDs found.")
                mask_left = np.isin(charuco_ids_left, common_ids)
                mask_right = np.isin(charuco_ids_right, common_ids)
                
                charuco_corners_left = charuco_corners_left[mask_left]
                charuco_corners_right = charuco_corners_right[mask_right]
                charuco_ids_left = charuco_ids_left[mask_left]
                charuco_ids_right = charuco_ids_right[mask_right]

                all_corners_left.append(charuco_corners_left)
                all_ids_left.append(charuco_ids_left)
                all_corners_right.append(charuco_corners_right)
                all_ids_right.append(charuco_ids_right)
                all_obj_pts.append(board.getChessboardCorners()[common_ids,:])
            else:
                print(f"Not enough common IDs found: {len(common_ids)} "
                      f"(need at least 4). Skipping image pair {counter}.")
        counter += 1

        print(f"All obj pts: {len(all_obj_pts)}")
        print(f"All corners left: {len(all_corners_left)}")
        print(f"All corners right: {len(all_corners_right)}")
            

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=all_obj_pts,
        imagePoints1=all_corners_left,
        imagePoints2=all_corners_right,
        cameraMatrix1=cameras[0].camera_matrix,
        distCoeffs1=cameras[0].dist_coeffs,
        cameraMatrix2=cameras[1].camera_matrix,
        distCoeffs2=cameras[1].dist_coeffs,
        imageSize=gray_left.shape[::-1],
        criteria=criteria,
        flags=flags,
    )

    reprojection_error = retval
    print(f"Reprojection error: {reprojection_error}")

    # compute the pose of the first camera to the board using the last img.
    frame = frames_0[-1]
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
                cameraMatrix=cameras[0].camera_matrix,
                distCoeffs=cameras[0].dist_coeffs,
                rvec=rvec,
                tvec=tvec,
            )

    cv2.drawFrameAxes(
        frames_0[-1],
        cameras[0].camera_matrix,
        cameras[0].dist_coeffs,
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

    other_camera_relative_pose = np.eye(4)
    other_camera_relative_pose[:3, :3] = R
    other_camera_relative_pose[:3, 3] = T.flatten()

    other_camera_pose_world_frame = hom_cam_pose_in_world_frame @ np.linalg.inv(other_camera_relative_pose)

    print("Camera2 pose in world frame.")
    print(other_camera_pose_world_frame)

    print("Press 'enter' to close and export the calibration.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure the calibration folder exists
    if not os.path.exists("calibration"):
        os.makedirs("calibration")

    np.save(
        f"calibration/{camera_names[0]}_{timestamp}_homogenous_transform.npy",
        hom_cam_pose_in_world_frame,
    )
    np.save(
        f"calibration/{camera_names[0]}_homogenous_transform.npy",
        hom_cam_pose_in_world_frame,
    )

    np.save(
        f"calibration/{camera_names[1]}_{timestamp}_homogenous_transform.npy",
        other_camera_pose_world_frame,
    )
    np.save(
        f"calibration/{camera_names[1]}_homogenous_transform.npy",
        other_camera_pose_world_frame,
    )

    # Save calibration transforms to a text file
    with open(f"calibration/calibration_{timestamp}_results.txt", "w") as f:
        f.write("Multi-Camera Calibration Results\n")
        f.write(f"Camera Names/IDs: {camera_names[0]}, {camera_names[1]}\n")
        f.write(f"Date and Time: {timestamp}\n\n")
        f.write("Board Pose in Camera Frame (Homogeneous Transformation):\n")
        f.write(np.array2string(hom_board_in_camera_frame))
        f.write("\n")
        f.write("Camera 1 Pose in World Frame (Homogeneous Transformation):\n")
        f.write(np.array2string(hom_cam_pose_in_world_frame))
        f.write("\n\n")
        f.write("Camera 2 Pose in World Frame (Homogeneous Transformation):\n")
        f.write(np.array2string(other_camera_pose_world_frame))
        f.write("\n\n")
        f.write("Stereo Calibration Rotation Matrix:\n")
        f.write(np.array2string(R))
        f.write("\n\n")
        f.write("Stereo Calibration Translation Vector:\n")
        f.write(np.array2string(T))
        f.write("\n")
        f.write("Reprojection Error:\n")
        f.write(str(reprojection_error))
        f.write("\n")

    print("Calibration successful. Camera parameters saved.")

    if display_image:
        cv2.imshow("ArUco Markers", frame)
        cv2.waitKey(0)  # Ensure the window stays open
        cv2.destroyAllWindows()

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate multiple cameras.")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="./pose_estimation/pose_estimation_config.json",
        help="Path to the configuration file.",
    )
    
    args = parser.parse_args()
    main(args.config_path)
