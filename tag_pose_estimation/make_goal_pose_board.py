import numpy as np

import cv2
import cv2.aruco as aruco

import pyrealsense2 as rs

import time
import datetime

import json

def board_to_json(board, dict_name):
    marker_data = {"dictionary": dict_name, "markers": []}

    for i, corner in enumerate(board.getObjPoints()):
        corner[:, 2] += np.random.normal(0, 0.01, size=4)
        print(corner)

        marker_data["markers"].append(
            {
                "id": int(board.getIds()[i]),  # Marker ID
                "corners": corner.tolist(),  # Convert NumPy array to list
            }
        )

    return marker_data

def get_larger_board(export_image=False):
    charuco_marker_dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_1000
    )
    # Generate the ChArUco board
    SQUARE_LENGTH = 0.0332
    MARKER_LENGHT = 0.02266
    NUMBER_OF_SQUARES_VERTICALLY = 10
    NUMBER_OF_SQUARES_HORIZONTALLY = 8

    board = cv2.aruco.CharucoBoard(
        size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGHT,
        dictionary=charuco_marker_dictionary,
    )

    if export_image:
        image_name = f"ChArUco_Marker_{NUMBER_OF_SQUARES_HORIZONTALLY}x{NUMBER_OF_SQUARES_VERTICALLY}.png"
        charuco_board_image = board.generateImage(
            [
                i * 100
                for i in (NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY)
            ]
        )

        cv2.imwrite(image_name, charuco_board_image)

        serialized_board = board_to_json(board, "DICT_4X4_1000")

        # Save to JSON
        with open("calibration/goal_pose_board.json", "w") as f:
            json.dump(serialized_board, f, indent=4)

    return board, charuco_marker_dictionary

get_larger_board(True)