import cv2
import numpy as np

import sys
import os
from pathlib import Path
import datetime

from tag_pose_estimation.utils import (
    get_larger_board,
    load_charuco_board_from_json
)
from tag_pose_estimation.camera_wrappers import (
    WebcamCamera,
)

def main(args):
    # Create Charuco board and dictionary
    if args.charuco_board_path and os.path.exists(args.charuco_board_path):
        board, aruco_dict = load_charuco_board_from_json(
            args.charuco_board_path)
        print(f"Loaded Charuco board from {args.charuco_board_path}")
    else:
        board, aruco_dict = get_larger_board(False)
        print("Using default larger Charuco board with 4x4 markers on 5x4 grid,"
              " marker length 0.04216 m.")

    # Setup parameters
    all_corners = []
    all_ids = []
    img_size = None

    # Capture frames
    cam_name = args.cam_name if args.cam_name is not None else 0
    cam = WebcamCamera(camera_id=cam_name)

    # Set focus
    cam.set_focus(args.focus)

    captured_frames = []

    while True:
        frame = cam.get_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        # Draw and display the corners
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and ids is not None:

            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if charuco_corners is not None and len(charuco_corners) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                if img_size is None:
                    img_size = gray.shape[::-1]
                print(f"Saved frame {len(all_corners)}")
            else:
                print("Not enough corners, frame not saved.")
            print(f"Frames collected: {len(all_corners)}")

        if key == ord('q'):
            break

    if len(all_corners) < 15:
        print(f"Not enough valid frames collected: {len(all_corners)}")
        exit(1)

    # Calibrate
    ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    print(f"Camera Matrix after calibration:\n{camera_matrix}")
    print(f"Distortion Coefficients after calibration:\n{dist_coeffs}")
    print(f"Reprojection error after calibration: {ret}")

    # Unique time id
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all files to folder
    Path(f"intrinsic_calibration_{timestamp}").mkdir()

    focus_value = cam.cap.get(cv2.CAP_PROP_FOCUS)
    print(f"Read focus value at end: {focus_value}")

    # Save to text file
    with open(f"intrinsic_calibration_{timestamp}/description.txt", "w") as f:
        f.write("Intrinsic Calibration Results\n")
        f.write(f"Camera Name/ID: {cam_name}\n")
        f.write(f"Date and Time: {timestamp}\n\n")
        f.write(f"Number of frames used: {len(all_corners)}\n\n")
        f.write("Camera Matrix:\n")
        f.write(np.array2string(camera_matrix))
        f.write("\nDistortion Coefficients:\n")
        f.write(np.array2string(dist_coeffs))
        f.write("\nFocus Value:\n")
        f.write(str(focus_value))
        f.write(f"\nReprojection error: {ret}\n")
    print(f"Saved intrinsic_calibration_{timestamp}/results.txt")
    # Save focus value to npy
    
    np.save(f"intrinsic_calibration_{timestamp}/focus_value.npy", focus_value)
    print(f"Saved focus_value_{timestamp}.npy")
    # Save to .npy
    np.save(f"intrinsic_calibration_{timestamp}/camera_matrix.npy", camera_matrix)
    np.save(f"intrinsic_calibration_{timestamp}/dist_coeffs.npy", dist_coeffs)
    print(f"Saved camera_matrix_{timestamp}.npy and dists_coeffs_{timestamp}.npy")

    cam.release()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Intrinsic camera calibration using a Charuco board."
    )
    parser.add_argument(
        "--cam_name",
        type=str,
        help="Camera name or ID",
    )
    parser.add_argument(
        "-cb", "--charuco_board_path",
        type=str,
        default="charuco_board.json",
        help="Path to the Charuco board JSON file"
    )
    parser.add_argument(
        "-f", "--focus",
        type=int,
        default=0,
        help="Focus value to set between 0 and 250 or  0 and 1 depending on "
            " camera. Default is 0. Autofocus is turned off in this script.",
    )
    args = parser.parse_args()
    main(args)
