import json

import numpy as np
import robotic as ry

from typing import List

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tag_pose_estimation.transform_utils import (
    load_camera_calibration,
    translation_from_homogenous,
    rotation_from_homogenous,
    rotation_matrix_to_quaternion,
)


def get_robot_joints(C: ry.Config, prefix: str) -> List[str]:
    links = []

    for name in C.getJointNames():
        if prefix in name:
            name = name.split(":")[0]

            if name not in links:
                links.append(name)

    return links


def make_scene(
    config,
    add_cameras=True,
    add_robots=True,
    add_boxes=True,
    use_inflated_robot_model=False,
):
    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 1, 0.01, 0.005])
        .setColor([0.6, 0.6, 0.6, 0.5])
        .setContact(1)
    )

    C.addFrame("origin").setParent(C.getFrame("table")).setRelativePosition(
        [0, 0, 0.0]
    ).setShape(ry.ST.marker, size=[0.2]).setColor([1, 0, 0.9]).setContact(0)

    robot_names = []
    robot_transformations = []

    if add_robots:
        robot_path = "./pose_estimation/ur5/ur5_vacuum.g"
        # TODO
        if use_inflated_robot_model:
            robot_path = "./pose_estimation/ur5/ur5_vacuum_inflated.g"


        for i, robot_transformation_path in enumerate(config["robots"]):
            print(robot_transformation_path)
            robot_name = f"a{i}_"
            robot_names.append(robot_name)

            transform = np.load(robot_transformation_path)
            print(transform)

            robot_transformations.append(transform)

            pos = transform[:3, 3]
            print(pos)
            R = transform[:3, :3]
            quat = rotation_matrix_to_quaternion(R)

            C.addFile(robot_path, namePrefix=f"a{i}_").setParent(
                C.getFrame("table")
            ).setRelativePosition(pos).setRelativeQuaternion(quat).setJoint(ry.JT.rigid)

    q0 = C.getJointState()
    actual_q0 = np.array(
        [-1.5708, -1.5708 + np.pi / 2, 1.5708, -1.5708 + np.pi / 2, -1.5708, 0.0]
    )
    for i in range(len(robot_names)):
        q0[i * 6 : (i + 1) * 6] = actual_q0
    C.setJointState(q0)

    box_names = []

    if add_boxes:
        for i, board_config in enumerate(config["boards"]):
            contact = -2
            if "ignore_collision" in config and board_config in config["ignore_collision"]:
                print(f"ignoring collison for {board_config}")
                contact = 0

            box_name = f"box_{i}"
            box_names.append(box_name)

            # make box of the appropriate dimensions
            # first, get the dimensions from the aruco board descritption
            min_pt = np.array([0.0, 0.0, 0.0])
            max_pt = np.array([0.0, 0.0, 0.0])
            with open(board_config, "r") as f:
                board_data = json.load(f)

                for marker in board_data["markers"]:
                    for corner in marker["corners"]:
                        for k in range(3):
                            min_pt[k] = min(min_pt[k], corner[k])
                            max_pt[k] = max(max_pt[k], corner[k])

            diff = max_pt - min_pt
            for j in range(3):
                diff[j] = max(0.01, diff[j])

            C.addFrame(f"{box_name}").setParent(C.getFrame("table")).setPosition(
                [0, 0, 0.2]
            ).setShape(ry.ST.box, size=[diff[0], diff[1], diff[2], 0.0005]).setColor(
                [0.9, 0.9, 0.9]
            ).setContact(contact).setJoint(ry.JT.rigid)

            C.addFrame(f"{box_name}_marker").setParent(
                C.getFrame(f"{box_name}")
            ).setPosition([0, 0, 0.0]).setShape(ry.ST.marker, size=[0.2]).setColor(
                [0.9, 0.9, 0.9]
            ).setContact(0)

    if add_cameras:
        for i, camera_config in enumerate(config["cameras"]):
            camera_pose = load_camera_calibration(camera_config["calibration_file"])

            print(camera_pose)

            # rot = rotation_from_homogenous(camera_pose)
            # rot_as_quat = Rotation.from_matrix(rot).as_quat()
            rot_as_quat = rotation_matrix_to_quaternion(
                rotation_from_homogenous(camera_pose)
            )

            camera_name = f"camera_{i}"
            C.addFrame(camera_name).setParent(C.getFrame("table")).setShape(
                ry.ST.marker, size=[0.5]
            ).setColor([1, 0, 0.9]).setContact(0).setRelativePosition(
                translation_from_homogenous(camera_pose)
            ).setRelativeQuaternion(rot_as_quat)

    return C, box_names, robot_names, robot_transformations
