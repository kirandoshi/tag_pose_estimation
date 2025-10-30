#!/usr/bin/env python3
import zmq
import time
import numpy as np
import json
import base64

import argparse

import warnings
import sys

import cv2
import cv2.aruco as aruco

import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

from tag_pose_estimation.transform_utils import (
    pose_to_homogeneous,
    translation_from_homogenous,
    rotation_from_homogenous,
)
from tag_pose_estimation.camera_wrappers import (
    RealSenseCamera,
    T265RealSenseCamera,
    WebcamCamera,
)
from tag_pose_estimation.utils import load_boards

# Define camera type
CameraType = WebcamCamera | RealSenseCamera | T265RealSenseCamera

parameters = cv2.aruco.DetectorParameters()
# parameters.adaptiveThreshWinSizeMin = 5
# parameters.adaptiveThreshWinSizeMax = 35
# parameters.adaptiveThreshWinSizeStep = 10

# parameters.polygonalApproxAccuracyRate = 0.06
# parameters.relativeCornerRefinmentWinSize = 0.01

# parameters.minCornerDistanceRate = 0.02
# parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

# parameters.cornerRefinementWinSize = 6
# parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.relativeCornerRefinmentWinSize = 0.15
parameters.cornerRefinementMaxIterations = 70

def camera_frame_to_world_frame(obj_in_camera_frame, camera_transform):
    pose_in_world_frame = camera_transform @ obj_in_camera_frame
    return pose_in_world_frame

def compute_marker_pose_in_board(marker_corners):
    # Get center of marker
    center = np.mean(marker_corners, axis=0)

    # Define x-axis as vector from corner 0 to corner 1
    x_axis = marker_corners[1] - marker_corners[0]
    x_axis /= np.linalg.norm(x_axis)

    # Define y-axis as vector from corner 0 to corner 3
    y_axis = marker_corners[3] - marker_corners[0]
    y_axis /= np.linalg.norm(y_axis)

    # Z is x cross y
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))
    t = center.reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H


def estimate_pose_by_averaging(gray, board, aruco_dict, camera):
    """
    Estimate board pose by averaging individual marker poses.
    """
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    # frame_marked = frame.copy()
    transform = None

    if ids is not None and len(ids) > 0:
        # cv2.aruco.drawDetectedMarkers(frame_marked, corners, ids)

        # Get all marker poses in camera frame
        marker_poses = []
        marker_weights = []  # Weight based on marker size in image
        board_points = board.getObjPoints()
        board_ids = board.getIds()

        for marker_corners, marker_id in zip(corners, ids):
            # Find this marker in the board configuration
            board_idx = None
            for i, board_id in enumerate(board_ids):
                if marker_id == board_id:
                    board_idx = i
                    break

            if board_idx is not None:
                # Get marker 3D points in board frame
                # obj_points = board_points[board_idx]
                obj_points = board_points[i]

                # Estimate pose for this marker
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    marker_corners[0],
                    camera.camera_matrix,
                    camera.dist_coeffs,
                    # flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                marker_pose = compute_marker_pose_in_board(board_points[board_idx])
                rel_transform = np.linalg.inv(marker_pose)
                # print(rel_transform)

                if success:
                    # cv2.drawFrameAxes(frame_marked, self.camera_matrix, self.dist_coeffs,
                    #         rvec, tvec, self.marker_size / 3, thickness=1)     # Convert to transformation matrix

                    rmat, _ = cv2.Rodrigues(rvec)
                    marker_transform = np.eye(4)
                    marker_transform[:3, :3] = rmat
                    marker_transform[:3, 3] = tvec.flatten()

                    marker_transform = marker_transform @ rel_transform

                    # Calculate weight based on marker size in image
                    # Larger apparent size = more reliable measurement
                    corners_array = marker_corners[0]
                    diag1 = corners_array[0] - corners_array[2]
                    diag2 = corners_array[1] - corners_array[3]
                    marker_area = np.linalg.norm(np.cross(diag1, diag2)) / 2

                    marker_poses.append(marker_transform)
                    marker_weights.append(marker_area)

                    rvec, _ = cv2.Rodrigues(marker_transform[:3, :3])
                    tvec = marker_transform[:3, 3]

                    # cv2.drawFrameAxes(frame_marked, self.camera_matrix, self.dist_coeffs,
                    #         rvec, tvec, self.marker_size / 3, thickness=1)

        if marker_poses:
            # Normalize weights
            weights = np.array(marker_weights)
            weights = weights / np.sum(weights)

            # Average translations with weights
            translations = np.array([pose[:3, 3] for pose in marker_poses])
            avg_translation = np.sum(translations * weights[:, np.newaxis], axis=0)

            # Average rotations using quaternions with weights
            rotations = [Rotation.from_matrix(pose[:3, :3]) for pose in marker_poses]
            quats = np.array([rot.as_quat() for rot in rotations])

            # Handle antipodal quaternions
            ref_quat = quats[0]
            for i in range(1, len(quats)):
                if np.dot(ref_quat, quats[i]) < 0:
                    quats[i] = -quats[i]

            avg_quat = np.sum(quats * weights[:, np.newaxis], axis=0)
            avg_quat = avg_quat / np.linalg.norm(avg_quat)
            avg_rotation = Rotation.from_quat(avg_quat)

            # Combine into transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = avg_rotation.as_matrix()
            transform[:3, 3] = avg_translation

            # Convert for visualization
            rvec, _ = cv2.Rodrigues(transform[:3, :3])
            tvec = transform[:3, 3]

            # Draw axes and pose information
            # cv2.drawFrameAxes(frame_marked, self.camera_matrix, self.dist_coeffs,
            #                 rvec, tvec, self.marker_size)

            position = tvec
            rotation_euler = cv2.RQDecomp3x3(transform[:3, :3])[0]

    #         cv2.putText(frame_marked, f"Markers used: {len(marker_poses)}",
    #                 (1000, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #         cv2.putText(frame_marked, "Tracking: GOOD",
    #                 (1000, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #         cv2.putText(frame_marked, f"Position: {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}m",
    #                 (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #         cv2.putText(frame_marked, f"Rotation: {rotation_euler[0]:.1f}, {rotation_euler[1]:.1f}, {rotation_euler[2]:.1f}deg",
    #                 (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #     else:
    #         cv2.putText(frame_marked, "Tracking: LOST",
    #                 (1000, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # else:
    #     cv2.putText(frame_marked, "Tracking: NO MARKERS",
    #             (1000, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return transform


DICT_NAME_MAP = {
    cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)): name
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def compute_reprojection_error(
    obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs
):
    # obj_pts_inliers = obj_points[inliers[:, 0]]
    # img_pts_inliers = img_points[inliers[:, 0]]

    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)

    error = np.linalg.norm(proj_pts.squeeze() - img_pts, axis=1)
    reproj_error = np.mean(error)

    return reproj_error


def compute_confidence(n_inliers, reproj_error, max_inliers=40, max_error=5.0):
    c_inliers = min(n_inliers / max_inliers, 1.0)
    c_error = max(0.0, 1.0 - reproj_error / max_error)
    return c_inliers * c_error

def apply_transformation(
       poses, transform
):
    """
    Apply a transformation to a set of poses.

    Parameters
    ----------
    poses : dict
        A dictionary of poses to transform.
    transform : np.ndarray
        A 4x4 transformation matrix to apply to the poses.

    Returns
    -------
    dict
        A dictionary of transformed poses.
    """
    transformed_poses = {}
    for i, pose in poses.items():
        # Apply the transformation to each pose
        position = (transform @ np.array(pose["position"] + [1]))[:3]
        rotation_matrix = transform[:3, :3] @ np.array(pose["rotation_matrix"])
        # Convert all data to lists
        transformed_poses[i] = {
            "position": position.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "confidence": pose["confidence"],
        }
    return transformed_poses

def detect_boards_in_camera_frame(
    gray_frame: np.ndarray,
    boards: list[cv2.aruco.Board],
    camera: CameraType,
    board_types: list[str],
    use_detection_type: str = "ransac",
    prev_guess: dict = {},
    aruco_dict: cv2.aruco.Dictionary = None,
):
    poses = {}

    id_cache = {}
    corner_cache = {}

    guess = {}

    for i, board in enumerate(boards):
        # TODO: FIX
        if aruco_dict is not None:
            used_dict = aruco_dict
        else:
            used_dict = board.getDictionary()
        dict_name = 5

        if dict_name not in id_cache:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray_frame, used_dict, parameters=parameters
            )

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)

            # Window size and zero zone (recommended typical values)
            win_size = (5, 5)
            zero_zone = (-1, -1)

            # Refine corners for aruco boards only
            if board_types[i] == "aruco" and ids is not None and corners:
                for marker_corners in corners:
                    cv2.cornerSubPix(
                        gray_frame,
                        marker_corners,  # This should be float32
                        win_size,
                        zero_zone,
                        criteria
                    )

            id_cache[dict_name] = ids
            corner_cache[dict_name] = corners

        else:
            ids = id_cache[dict_name]
            corners = corner_cache[dict_name]

        if ids is not None:
            if use_detection_type == "standard":
                if board_types[i] == "charuco":
                    # interpolate charuco corners
                    (retval, 
                     charuco_corners, 
                     charuco_ids) = cv2.aruco.interpolateCornersCharuco(
                        corners, 
                        ids, 
                        gray_frame, 
                        board,
                        cameraMatrix=camera.camera_matrix,
                        distCoeffs=camera.dist_coeffs,
                    )
                    rvec = np.zeros((3, 1), dtype=np.float32)
                    tvec = np.zeros((3, 1), dtype=np.float32)

                    retval = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners,
                        charuco_ids,
                        board,
                        camera.camera_matrix,
                        camera.dist_coeffs,
                        rvec,
                        tvec,
                    )
                else:
                    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                        corners,
                        ids,
                        board,
                        camera.camera_matrix,
                        camera.dist_coeffs,
                        None,
                        None,
                    )

                if retval:
                    pose_world_frame = camera_frame_to_world_frame(
                        pose_to_homogeneous(rvec, tvec), camera.homogeneous_transform
                    )

                    poses[i] = {
                        "position": translation_from_homogenous(
                            pose_world_frame
                        ).tolist(),
                        "rotation_matrix": rotation_from_homogenous(
                            pose_world_frame
                        ).tolist(),
                        "confidence": 1,
                        "rvec": rvec,
                        "tvec": tvec,
                    }

            elif use_detection_type == "standard_with_initial_guess":
                if board_types[i] == "charuco":
                    warnings.warn("Charuco board detection not implemented for "
                                  "'standard_with_initial_guess' detection "
                                  "type. Detecting as Aruco board instead.", 
                                  UserWarning)
                    
                obj_points = []  # 3D points
                img_points = []  # 2D detected corners

                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    if marker_id in board.getIds():
                        idx = np.where(board.getIds() == marker_id)[0][0]
                        obj_pts_marker = board.getObjPoints()[idx]  # (4, 3)

                        obj_points.append(obj_pts_marker)  # (4, 3)
                        img_points.append(marker_corners[0])  # (4, 2)

                if len(obj_points) * 4 > 4:
                    obj_points = np.vstack(obj_points).astype(np.float32)
                    img_points = np.vstack(img_points).astype(np.float32)

                    # print(f"obj_points.shape: {obj_points.shape}")
                    # print(f"img_points.shape: {img_points.shape}")

                    if i in prev_guess:
                        prev_rvec_guess = prev_guess[i][0]
                        prev_tvec_guess = prev_guess[i][1]

                        success, rvec, tvec = cv2.solvePnP(
                            obj_points,
                            img_points,
                            camera.camera_matrix,
                            camera.dist_coeffs,
                            rvec=prev_rvec_guess,
                            tvec=prev_tvec_guess,
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                    else:
                        success, rvec, tvec = cv2.solvePnP(
                            obj_points,
                            img_points,
                            camera.camera_matrix,
                            camera.dist_coeffs,
                            useExtrinsicGuess=False,
                        )

                    if success:
                        pose_world_frame = camera_frame_to_world_frame(
                            pose_to_homogeneous(rvec, tvec),
                            camera.homogeneous_transform,
                        )

                        poses[i] = {
                            "position": translation_from_homogenous(
                                pose_world_frame
                            ).tolist(),
                            "rotation_matrix": rotation_from_homogenous(
                                pose_world_frame
                            ).tolist(),
                            "confidence": 1,
                        }

                        guess[i] = [rvec, tvec]

            elif use_detection_type == "ransac":
                if board_types[i] == "charuco":
                    warnings.warn("Charuco board detection not implemented for "
                                  "'ransac' detection type. Detecting as "
                                  "Aruco board instead.", UserWarning)
                obj_points = []  # 3D points
                img_points = []  # 2D detected corners

                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    if marker_id in board.getIds():
                        idx = np.where(board.getIds() == marker_id)[0][0]
                        obj_pts_marker = board.getObjPoints()[idx]  # (4, 3)

                        obj_points.append(obj_pts_marker)  # (4, 3)
                        img_points.append(marker_corners[0])  # (4, 2)

                if len(obj_points) > 0:
                    obj_points = np.vstack(obj_points).astype(np.float32)
                    img_points = np.vstack(img_points).astype(np.float32)

                    # SolvePnPRansac
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj_points,
                        img_points,
                        camera.camera_matrix,
                        camera.dist_coeffs,
                        reprojectionError=3.0,  # tighten this if needed
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )

                    if success:
                        pose_world_frame = camera_frame_to_world_frame(
                            pose_to_homogeneous(rvec, tvec),
                            camera.homogeneous_transform,
                        )

                        obj_pts_inliers = obj_points[inliers[:, 0]]
                        img_pts_inliers = img_points[inliers[:, 0]]
                        reproj_error = compute_reprojection_error(obj_pts_inliers, img_pts_inliers, rvec, tvec, camera.camera_matrix, camera.dist_coeffs)

                        print(reproj_error)

                        poses[i] = {
                            "position": translation_from_homogenous(
                                pose_world_frame
                            ).tolist(),
                            "rotation_matrix": rotation_from_homogenous(
                                pose_world_frame
                            ).tolist(),
                            "confidence": len(obj_pts_inliers),
                        }

            elif use_detection_type == "ransac_with_refinement":
                if board_types[i] == "charuco":
                    warnings.warn("Charuco board detection not implemented for "
                                  "'ransac_with_refinement' detection type. "
                                  "Detecting as Aruco board instead.", 
                                  UserWarning)
                obj_points = []  # 3D points
                img_points = []  # 2D detected corners

                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    if marker_id in board.getIds():
                        idx = np.where(board.getIds() == marker_id)[0][0]
                        obj_pts_marker = board.getObjPoints()[idx]  # (4, 3)

                        obj_points.append(obj_pts_marker)  # (4, 3)
                        img_points.append(marker_corners[0])  # (4, 2)

                if len(obj_points) > 0:
                    obj_points = np.vstack(obj_points).astype(np.float32)
                    img_points = np.vstack(img_points).astype(np.float32)

                    # SolvePnPRansac
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj_points,
                        img_points,
                        camera.camera_matrix,
                        camera.dist_coeffs,
                        reprojectionError=1.5,  # tighten this if needed
                        # flags=cv2.SOLVEPNP_AP3P,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        iterationsCount=200
                    )

                    if not success or inliers is None or (len(inliers) < 4 and len(board.getIds())) > 1:
                        print("ransac rejected too much")

                    else:
                        obj_inliers = obj_points[inliers[:, 0]]
                        img_inliers = img_points[inliers[:, 0]]

                        success, rvec, tvec = cv2.solvePnP(
                            obj_inliers,
                            img_inliers,
                            camera.camera_matrix,
                            camera.dist_coeffs,
                            rvec=rvec,
                            tvec=tvec,
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )

                        if success:
                            pose_world_frame = camera_frame_to_world_frame(
                                pose_to_homogeneous(rvec, tvec),
                                camera.homogeneous_transform,
                            )

                            obj_pts_inliers = obj_points[inliers[:, 0]]
                            img_pts_inliers = img_points[inliers[:, 0]]
                            reproj_error = compute_reprojection_error(obj_pts_inliers, img_pts_inliers, rvec, tvec, camera.camera_matrix, camera.dist_coeffs)

                            confidence = compute_confidence(len(img_pts_inliers), reproj_error, len(img_points), 5)

                            poses[i] = {
                                "position": translation_from_homogenous(
                                    pose_world_frame
                                ).tolist(),
                                "rotation_matrix": rotation_from_homogenous(
                                    pose_world_frame
                                ).tolist(),
                                "confidence": confidence,
                                "rvec": rvec,
                                "tvec": tvec
                            }

            elif use_detection_type == "average":
                if board_types[i] == "charuco":
                    warnings.warn("Charuco board detection not implemented for "
                                  "'average' detection type. Detecting as "
                                  "Aruco board instead.", UserWarning)
                pose = estimate_pose_by_averaging(gray_frame, board, used_dict, camera)
                pose_world_frame = camera_frame_to_world_frame(
                    pose, camera.homogeneous_transform
                )
                poses[i] = {
                    "position": translation_from_homogenous(pose_world_frame).tolist(),
                    "rotation_matrix": rotation_from_homogenous(
                        pose_world_frame
                    ).tolist(),
                    "confidence": 1,
                }
            
            else:
                raise ValueError(f"Unknown detection type: "
                                 f"{use_detection_type}")

    return poses, guess


def update_board_poses(board_pose_measurements):
    return board_pose_measurements
    
def main(config_path, 
         goal_frequency=10, 
         detection_type="standard", 
         publish_image=False,
         use_additional_transform=False):
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: Config file not found.", file=sys.stderr)
        sys.exit(1)

    # Initialize ZMQ context and socket
    pose_publisher_context = zmq.Context()
    box_pose_socket = pose_publisher_context.socket(zmq.PUB)
    box_pose_socket.bind(f"tcp://*:{config['port']}")

    # make cameras and load their respective calibrations

    # Initialize webcam
    # cap = cv2.VideoCapture(0)

    # rvec_path = "./calibration/20250404_103226_rvecs.npy"
    # tvec_path = "./calibration/20250404_103226_tvecs.npy"

    cameras = []
    camera_ids = []

    for camera_dict in config["cameras"]:
        camera_calibration_file = camera_dict["calibration_file"]
        serial_number = None
        if "serial_number" in camera_dict:
            serial_number = camera_dict["serial_number"]

        if camera_dict["type"] == "D405":
            camera = RealSenseCamera(
                calibration_path=camera_calibration_file, 
                serial_number=serial_number
            )
        elif camera_dict["type"] == "T265":
            camera = T265RealSenseCamera(
                calibration_path=camera_calibration_file, 
                serial_number=serial_number
            )
        elif camera_dict["type"] == "Webcam":
            camera = WebcamCamera(
                calibration_path=camera_calibration_file,
                camera_id=serial_number,
                camera_matrix_path=camera_dict["camera_matrix"],
                camera_dist_path=camera_dict["dist_coeff"],
                focus_path=camera_dict.get("focus_setting"),
            )

        cameras.append(camera)
        camera_ids.append(serial_number)

        cv2.namedWindow(
            f"Camera {len(cameras)}", cv2.WINDOW_NORMAL
        )  # explicit window creation

    board_configs = config["boards"]

    # Load board type
    # If not specified, assume all are aruco boards
    board_types = config.get("board_types", ["aruco"] * len(board_configs))
    assert len(board_types) == len(board_configs), (
        "board_types and board_configs must have the same length"
    )
    boards = load_boards(board_configs, board_types)

    if use_additional_transform:
        transform_file = config["pose_estimation_frame_transform"]
        # This loaded transformation will be applied as a transformation from
        # the world frame to the specified frame
        T_W_SF = np.load(transform_file)
        # Invert homogeneous pose matrix explicitly for numerical stability
        R = T_W_SF[:3, :3]
        t = T_W_SF[:3, 3]
        T_SF_W = np.eye(4)
        T_SF_W[:3, :3] = R.T
        T_SF_W[:3, 3] = -R.T @ t
        print("Using additional transformation from config file:")
        print("World frame to specified frame:")
        print(T_W_SF)
        print("Specified frame to world frame:")
        print(T_SF_W)

    print(f"Pose estimation server started. Publishing to tcp://*:{config['port']}")
    print(f"Tracking {len(board_configs)} boards")

    guess = {}

    try:
        while True:
            start_time = time.time()

            if publish_image:
                frame_dict = {}

            for i, camera in enumerate(cameras):
                # Capture frame
                frame = camera.get_frame()

                if frame is None:
                    continue

                if publish_image:
                    _, buffer = cv2.imencode(".jpg", frame)
                    camera_id = str(camera_ids[i])
                    frame_dict[camera_id] = base64.b64encode(buffer).decode("utf-8")

                # Detect markers and estimate poses
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = frame

                # Poses are in world frame
                board_pose_measurements, guess = detect_boards_in_camera_frame(
                    gray,
                    boards,
                    camera,
                    board_types,
                    use_detection_type=detection_type,
                    prev_guess=guess,
                    aruco_dict=None, # Use board-specific dictionaries
                )

                # currently the identity function -> simple pass-through
                # should be some filtering and safety-mechanism
                board_pose_measurements = update_board_poses(board_pose_measurements)

                if use_additional_transform:
                    board_poses_pub = apply_transformation(board_pose_measurements, T_SF_W)
                else:
                    board_poses_pub = board_pose_measurements
                # print(f"Camera {i+1}: Detected {len(board_poses)} boards")
                # print(board_poses)

                # Publish poses
                
                # this should not be happening here
                # Use dict of the first board 
                current_aruco_dict = boards[0].getDictionary()
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, current_aruco_dict, parameters=parameters
                )

                # Optional: Display the frame with detected markers
                if ids is not None:
                    aruco.drawDetectedMarkers(frame, corners, ids)

                    for _,v in board_pose_measurements.items():
                        if "tvec" in v and "rvec" in v:
                            cv2.drawFrameAxes(
                                frame,
                                camera.camera_matrix,
                                camera.dist_coeffs,
                                v["rvec"],
                                v["tvec"],
                                0.1,
                            )
                            v["rvec"] = []
                            v["tvec"] = []

                if board_poses_pub:
                    # Put board pose into message
                    msg_dict = {
                        "timestamp": time.time(),
                        "poses": board_poses_pub
                    }
                    # print(f"Positions: {[v['position'] for _,v in board_poses.items()]}")
                    box_pose_socket.send_json(msg_dict)
                    print(f"Published {len(board_poses_pub)} board poses")

                

                cv2.imshow(f"Camera {i + 1}", frame)

                # Break loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Publish image frames
            if publish_image:
                img_msg = {
                    "timestamp": time.time(),
                    "images": frame_dict
                }
                box_pose_socket.send_json(img_msg)

            # Rate limiting
            end_time = time.time()
            delta = end_time - start_time
            if delta < 1 / goal_frequency:
                required_sleep = 1 / goal_frequency - delta
                time.sleep(required_sleep)

            end_time = time.time()
            print(f"Running at {1 / (end_time - start_time)} hz")

    finally:
        # Clean up
        cv2.destroyAllWindows()
        box_pose_socket.close()
        pose_publisher_context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track pose with a robot.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="./configs/pose_estimation_config.json",
        help="Path to the configuration file. (Absolute Path)",
    )
    parser.add_argument(
        "-d", "--detection_type",
        type=str,
        default="standard_with_initial_guess",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-f", "--frequency",
        type=int,
        default=10,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-ut", "--use_transform",
        action="store_true",
        help=("Whether to use the additional transform from world frame to a "
              "given frame, specified in the config with key "
              "'pose_estimation_frame_transform'.")
    )
    parser.add_argument(
        "-i", "--publish_image",
        action="store_true",
        default=True,
        help="Whether to publish the image."
    )
    args = parser.parse_args()

    config_path = args.config
    main(
        config_path, 
        args.frequency, 
        detection_type=args.detection_type,
        publish_image=args.publish_image,
        use_additional_transform=args.use_transform
    )
