import cv2
import numpy as np

from pathlib import Path
import datetime
import logging
import sys

from tag_pose_estimation.utils import (
    get_larger_board,
    load_charuco_board_from_json,
    get_project_root,
)
from tag_pose_estimation.camera_wrappers import (
    WebcamCamera,
)

# Module logger
logger = logging.getLogger(__name__)

def intrinsic_camera_calibration(args):
    # Create Charuco board and dictionary
    if args.charuco_board_path and Path(args.charuco_board_path).exists():
        board, aruco_dict = load_charuco_board_from_json(
            args.charuco_board_path)
        logger.info(f"Loaded Charuco board from {args.charuco_board_path}")
    else:
        if args.charuco_board_path:
            logger.warning(f"Charuco board path {args.charuco_board_path} does not exist.")
            logger.info("Using default intrinsic calibration Charuco board instead.")
            logger.info("The default board can be found at 'config/calibration_boards/default_intrinsic_calibration_board.json'")
        else:
            logger.info("No Charuco board path provided.")
            logger.info("Using default intrinsic calibration Charuco board instead.")
            logger.info("The default board can be found at 'config/calibration_boards/default_intrinsic_calibration_board.json'")
            logger.info("The default board has 3x5 squares with 0.058m square length and 0.045m marker length.")
            logger.info("Ensure that the board is printed at the correct scale for accurate calibration.")
        board, aruco_dict = load_charuco_board_from_json(
            str(get_project_root() /
                "config" /
                "calibration_boards" /
                "default_intrinsic_calibration_board" /
                "charuco_board.json"
            )
        )

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
    logger.info("Press s to use frame for calibration. Press q when done collecting "
                "frames. Ensure image viewer is highlighted when pressing.")

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
                logger.info(f"Saved frame {len(all_corners)}")
            else:
                logger.warning("Not enough corners, frame not saved.")
            logger.info(f"Frames collected: {len(all_corners)}")

        if key == ord('q'):
            break

    if len(all_corners) < 15:
        logger.error(f"Not enough valid frames collected: {len(all_corners)}")
        logger.error("Need at least 15 frames with detected corners for calibration.")
        sys.exit(1)
    else:
        logger.info(f"Collected {len(all_corners)} valid frames for calibration.")
        logger.info("Calibrating...")

    try:
        # Calibrate
        ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
    except cv2.error as e:
        logger.error(f"Calibration failed: {e}")
        logger.error("Ensure that enough valid frames with detected corners were collected.")
        logger.error("Likely causes: insufficient number of corners or poor corner detection.")
        logger.info("Redo the calibration and try collecting more frames")
        sys.exit(1)

    logger.debug(f"Camera Matrix after calibration:\n{camera_matrix}")
    logger.debug(f"Distortion Coefficients after calibration:\n{dist_coeffs}")
    logger.info(f"Reprojection error after calibration: {ret}")

    # Unique time id
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all files to folder
    calibration_folder = get_intrinsic_calibration_save_folder()
    calibration_folder = calibration_folder / f"cal_{timestamp}"
    calibration_folder.mkdir()

    focus_value = cam.cap.get(cv2.CAP_PROP_FOCUS)
    logger.info(f"Read focus value at end: {focus_value}")

    # Save to text file
    txt_file_path = calibration_folder / "description.txt"
    with open(txt_file_path, "w") as f:
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
    logger.info("Saved %s", txt_file_path)
    # Save focus value to npy
    focus_value_path = calibration_folder / "focus_value.npy"
    np.save(focus_value_path, focus_value)
    logger.info(f"Saved {focus_value_path}")
    # Save to .npy
    camera_matrix_path = calibration_folder / "camera_matrix.npy"
    dist_coeffs_path = calibration_folder / "dist_coeffs.npy"
    np.save(camera_matrix_path, camera_matrix)
    np.save(dist_coeffs_path, dist_coeffs)
    logger.info(f"Saved {camera_matrix_path} and {dist_coeffs_path}")
    cam.release()

    return None

def get_intrinsic_calibration_save_folder() -> Path:
    root = get_project_root()
    save_folder = root / "config" / "intrinsic_calibration"
    # Create the directory if it doesn't exist
    if not save_folder.parent.exists():
        save_folder.parent.mkdir()
    if not save_folder.exists():
        save_folder.mkdir()
    return save_folder