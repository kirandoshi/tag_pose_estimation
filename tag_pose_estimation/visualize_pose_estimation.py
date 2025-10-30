import zmq
import time
import json
import argparse

import sys
import os

import numpy as np
import robotic as ry
import cv2
# from scipy.spatial.transform import Rotation

import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tag_pose_estimation.transform_utils import (
    rotation_matrix_to_quaternion,
    translation_from_homogenous,
    rotation_from_homogenous,
    seven_d_to_homogeneous,
)

from planning_utils.pose_opt import (
    compute_grasping_pose,
    handover,
    compute_reorientation_path,
    compute_reorientation_poses,
    compute_pick_and_place_trajectory_with_given_poses
)

from tag_pose_estimation.board_pose_estimator import BoardPoseEstimator

from scene_utils import make_scene


def visualize_camera_only(config):
    C, _, _, _ = make_scene(config, add_boxes=False, add_robots=False)
    C.view(True)


def visualize_robots_only(config):
    C, _, _, _ = make_scene(config, add_boxes=False, add_robots=True, add_cameras=False)
    C.view(True)


def visualize_scene(config):
    C, _, _, _ = make_scene(config)
    C.view(True)


from planning_utils.keyboard_input import KeyboardThread, get_keyboard_flag, reset_keyboard_flag


def track_box_poses(config):
    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
    socket.connect(f"tcp://localhost:{config['port']}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set target marker ID to track
    C, box_names, _, _ = make_scene(config)

    print(
        f"Pose estimation visualizer started. Subscribed to tcp://localhost:{config['port']}"
    )
    # print(f"Tracking ArUco marker ID: {target_board_id}")

    try:
        while True:
            # Receive pose data
            try:
                data = socket.recv_json(flags=zmq.NOBLOCK)
                timestamp = data["timestamp"]
                poses = data["poses"]

                # print(poses)
                print(f"Received {len(poses)} poses at time {timestamp}")

                for box_id, pose in poses.items():
                    t = np.array(pose["position"])
                    R = np.array(pose["rotation_matrix"])

                    box_name = box_names[int(box_id)]
                    C.getFrame(box_name).setRelativePosition(t)
                    C.getFrame(box_name).setRelativeQuaternion(
                        rotation_matrix_to_quaternion(R)
                    )

                C.view(False)

            except zmq.Again:
                # No message received yet, continue
                pass

            # Rate limiting
            time.sleep(0.1)

    except zmq.ZMQError as e:
        print(f"ZeroMQ error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        exit(0)


def threaded_box_pose_estimation(config):
    box_pose_estimator = BoardPoseEstimator(f"tcp://localhost:{config['port']}")
    box_pose_estimator.start()

    C, box_names, _, _ = make_scene(config)

    while True:
        # Receive pose data
        ids = box_pose_estimator.get_tracked_board_ids()

        for box_id in ids:
            box_pose = box_pose_estimator.get_pose(box_id)

            is_stable = box_pose_estimator.is_stable(box_id)
            box_name = box_names[int(box_id)]

            if not is_stable:
                print(box_id, "NOT STABLE")

            if is_stable and box_pose is not None:
                t = np.array(box_pose[:3])
                q = np.array(box_pose[3:])

                C.getFrame(box_name).setRelativePosition(t)
                C.getFrame(box_name).setRelativeQuaternion(q)

                C.getFrame(box_name).setColor([0.5, 0.5, 0.5, 1])

            else:
                C.getFrame(box_name).setColor([0.5, 0.5, 0.5, 0.2])

        C.view(False)

        # Rate limiting
        time.sleep(0.1)


def compute_and_show_grasping_poses(config):
    KeyboardThread()

    use_pose_estimation = False
    if use_pose_estimation:
        box_pose_estimator = BoardPoseEstimator(f"tcp://localhost:{config['port']}")
        box_pose_estimator.start()
    else:
        box_poses_path = "./configs/example_scene_001.json"
        try:
            with open(box_poses_path) as f:
                box_config = json.load(f)
        except FileNotFoundError:
            print("Error: Config file not found.", file=sys.stderr)
            sys.exit(1)

    C, box_names, robot_names, _ = make_scene(config)

    while True:
        # Receive pose data
        if use_pose_estimation:
            ids = box_pose_estimator.get_tracked_board_ids()

            for box_id in ids:
                box_pose = box_pose_estimator.get_pose(box_id)

                if box_pose is not None:
                    t = np.array(box_pose[:3])
                    q = np.array(box_pose[3:])

                    box_name = box_names[int(box_id)]
                    C.getFrame(box_name).setRelativePosition(t)
                    C.getFrame(box_name).setRelativeQuaternion(q)

        else:
            for p in box_config["poses"]:
                box_id = p["id"]
                box_pose = p["pose"]
                print(box_id)
                if box_pose is not None:
                    t = np.array(box_pose[:3])
                    q = np.array(box_pose[3:])

                    box_name = box_names[int(box_id)]
                    C.getFrame(box_name).setRelativePosition(t)
                    C.getFrame(box_name).setRelativeQuaternion(q)

        C.view(False)

        if get_keyboard_flag():
            reset_keyboard_flag()
            for robot_name in robot_names:
                for box in box_names:
                    pose = compute_grasping_pose(C, robot_name, box, view=False)
                    if pose is not None:
                        pass
                    else:
                        print(
                            f"Was not able to find a valid grasping pose for box {box}"
                        )

        # Rate limiting
        time.sleep(0.1)


def compute_and_show_reorientation_poses(config):
    KeyboardThread()

    use_pose_estimation = False
    if use_pose_estimation:
        box_pose_estimator = BoardPoseEstimator(f"tcp://localhost:{config['port']}")
        box_pose_estimator.start()
    else:
        box_poses_path = "./configs/example_scene_002_cubes.json"
        try:
            with open(box_poses_path) as f:
                box_config = json.load(f)
        except FileNotFoundError:
            print("Error: Config file not found.", file=sys.stderr)
            sys.exit(1)

    C, box_names, robot_names, _ = make_scene(config, use_inflated_robot_model=True)

    home_pose = C.getJointState()

    while True:
        # Receive pose data
        if use_pose_estimation:
            ids = box_pose_estimator.get_tracked_board_ids()

            for box_id in ids:
                box_pose = box_pose_estimator.get_pose(box_id)

                if box_pose is not None:
                    t = np.array(box_pose[:3])
                    q = np.array(box_pose[3:])

                    box_name = box_names[int(box_id)]
                    C.getFrame(box_name).setRelativePosition(t)
                    C.getFrame(box_name).setRelativeQuaternion(q)

        else:
            for p in box_config["poses"]:
                box_id = p["id"]
                box_pose = p["pose"]
                print(box_id)
                if box_pose is not None:
                    t = np.array(box_pose[:3])
                    q = np.array(box_pose[3:])

                    box_name = box_names[int(box_id)]
                    C.getFrame(box_name).setRelativePosition(t)
                    C.getFrame(box_name).setRelativeQuaternion(q)

        C.view(False)

        if get_keyboard_flag():
            reset_keyboard_flag()
            for robot_name in robot_names:
                for box in box_names:
                    # pose = compute_reorientation_path(C, robot_name, box, view=False)
                    obj_pose = C.getFrame(box).getPosition()
                    pick_pose, place_pose = compute_reorientation_poses(C, robot_name, box, view=False)
                    if pick_pose is not None:
                        pick_path, place_path = compute_pick_and_place_trajectory_with_given_poses(C, home_pose[:6], robot_name, box, pick_pose, place_pose, obj_pose, view=True, intermediate_height=0.1)
        # Rate limiting
        time.sleep(0.1)


def compute_and_show_handover(config, robot_configs):
    KeyboardThread()

    use_pose_estimation = False
    if use_pose_estimation:
        box_pose_estimator = BoardPoseEstimator(f"tcp://localhost:{config['port']}")
        box_pose_estimator.start()
    else:
        box_poses_path = "./configs/example_scene_001.json"
        try:
            with open(box_poses_path) as f:
                box_config = json.load(f)
        except FileNotFoundError:
            print("Error: Config file not found.", file=sys.stderr)
            sys.exit(1)

    C, box_names, robot_names, _ = make_scene(config, use_inflated_robot_model=True)

    while True:
        # Receive pose data
        if use_pose_estimation:
            ids = box_pose_estimator.get_tracked_board_ids()

            for box_id in ids:
                box_pose = box_pose_estimator.get_pose(box_id)

                if box_pose is not None:
                    t = np.array(box_pose[:3])
                    q = np.array(box_pose[3:])

                    box_name = box_names[int(box_id)]
                    C.getFrame(box_name).setRelativePosition(t)
                    C.getFrame(box_name).setRelativeQuaternion(q)

        else:
            for p in box_config["poses"]:
                box_id = p["id"]
                box_pose = p["pose"]
                print(box_id)
                if box_pose is not None:
                    t = np.array(box_pose[:3])
                    q = np.array(box_pose[3:])

                    box_name = box_names[int(box_id)]
                    C.getFrame(box_name).setRelativePosition(t)
                    C.getFrame(box_name).setRelativeQuaternion(q)

        C.view(False)

        if get_keyboard_flag():
            reset_keyboard_flag()
            for id0, id1 in [(0, 1), (1, 0)]:
                for box in box_names:
                    pose = handover(
                        C, robot_names[id0], robot_names[id1], box, view=False
                    )
                    if pose is not None:
                        pass
                    else:
                        print(
                            f"Was not able to find a valid grasping pose for box {box}"
                        )


def export_scene(config):
    KeyboardThread()

    box_pose_estimator = BoardPoseEstimator(f"tcp://localhost:{config['port']}")
    box_pose_estimator.start()

    C, box_names, robot_names, _ = make_scene(config)

    while True:
        # Receive pose data
        ids = box_pose_estimator.get_tracked_board_ids()

        for box_id in ids:
            box_pose = box_pose_estimator.get_pose(box_id)

            if box_pose is not None:
                t = np.array(box_pose[:3])
                q = np.array(box_pose[3:])

                box_name = box_names[int(box_id)]
                C.getFrame(box_name).setRelativePosition(t)
                C.getFrame(box_name).setRelativeQuaternion(q)

        C.view(False)

        if get_keyboard_flag():
            reset_keyboard_flag()

            poses_json = {"poses": []}
            for box_id in ids:
                box_pose = box_pose_estimator.get_pose(box_id)
                if box_pose is not None:
                    poses_json["poses"].append({
                        "id": int(box_id),
                        "pose": list(box_pose)
                    })
            print(json.dumps(poses_json, indent=2))

            break

        # Rate limiting
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the scene or track box poses."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "visualize",
            "track",
            "threaded_tracking",
            "show_grasping_poses",
            "show_handover",
            "export",
            "show_reorientation",
        ],
        default="visualize",
        help="Choose 'visualize' to visualize the scene or 'track' to track box poses.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./pose_estimation/pose_estimation_config.json",
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    config_path = args.config

    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: Config file not found.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "visualize":
        visualize_scene(config)
    elif args.mode == "track":
        track_box_poses(config)
    elif args.mode == "threaded_tracking":
        threaded_box_pose_estimation(config)
    elif args.mode == "show_grasping_poses":
        compute_and_show_grasping_poses(config)
    elif args.mode == "show_reorientation":
        compute_and_show_reorientation_poses(config)
    elif args.mode == "show_handover":
        compute_and_show_handover(config, None)
    elif args.mode == "export":
        export_scene(config)
    else:
        print("Invalid mode selected.", file=sys.stderr)
        sys.exit(1)
