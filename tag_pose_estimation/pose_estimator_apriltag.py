#!/usr/bin/env python3
import zmq
import time
import numpy as np
import json
import base64
import os

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
from tag_pose_estimation.apriltag_board import AprilTagBoard
from tag_pose_estimation.apriltag_utils import detections_to_corners_ids

# Add path to apriltag package
# Apritag package is located in root/python_apriltag
# This is a bit hacky but works for now
# This file is located in root/robot_ipc_control/pose_estimation
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from python_apriltag.python_apriltag.apriltag import apriltag

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
    boards: list[AprilTagBoard],
    camera: CameraType,
    board_types: list[str],
    use_detection_type: str = "ransac",
    prev_guess: dict = {},
    aruco_dict: cv2.aruco.Dictionary = None,
):  
    # Get apriltag dictionary
    # For now assume all boards use the same apriltag dictionary
    apriltag_dict = None
    for board in boards:
        if apriltag_dict is None:
            apriltag_dict = board.getDictionary()
        else:
            assert apriltag_dict == board.getDictionary(), (
                "All boards must use the same apriltag dictionary")
    
    apriltag_detector = apriltag(
        apriltag_dict,
        decimate=1.0,
        maxhamming=0,
        threads=4,
        refine_edges=True
    )
    
    poses = {}

    guess = {}

    for i, board in enumerate(boards):
        # corners, ids, rejected = cv2.aruco.detectMarkers(
        #     gray_frame, used_dict, parameters=parameters
        # )

        # Output of apriltag detection
        # {
        #     'hamming': 0,
        #     'margin': 45.2,
        #     'id': 17,
        #     'center': np.array([320.5, 240.5]),
        #     'lb-rb-rt-lt': np.array([[300, 250], [340, 250], [340, 230], [300, 230]])
        # },

        # Detect AprilTags
        detections = apriltag_detector.detect(gray_frame)

        corners, ids = detections_to_corners_ids(detections)

        detected_markers = {
            'corners': corners,
            'ids': ids
        }

        # Corner refinement not used for apriltag because apriltag has edge
        # refinement built-in

        if ids is not None:
            if use_detection_type == "ransac_with_refinement":
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

            else:
                raise ValueError(f"Unknown detection type: "
                                 f"{use_detection_type}")

    return poses, guess, detected_markers


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
    # If not specified, assume all are april tag boards
    board_types = config.get("board_types", ["apriltag"] * len(board_configs))
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
                    # Make a copy for publishing
                    frame_pub = frame.copy()
                    h, w = frame_pub.shape[:2]
                    # Ensure that the encoded image is maximally in 640x480 
                    # resolution. If the frame is larger, resize it.
                    if h > 480 or w > 640:
                        frame_pub = cv2.resize(frame_pub, (640, 480), 
                                               interpolation=cv2.INTER_AREA)
                    
                    _, buffer = cv2.imencode(".jpg", frame_pub)
                    camera_id = str(camera_ids[i])
                    frame_dict[camera_id] = base64.b64encode(buffer).decode("utf-8")

                # Detect markers and estimate poses
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = frame

                # Poses are in world frame
                board_pose_measurements, guess, detected_markers = detect_boards_in_camera_frame(
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

                
                # Optional: Display the frame with detected markers
                corners = detected_markers['corners']
                ids = detected_markers['ids']
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

                # Publish poses
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