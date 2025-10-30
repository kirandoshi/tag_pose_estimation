import cv2
import json

import numpy as np

from tag_pose_estimation.apriltag_board import AprilTagBoard


def board_to_json(board, dict_name):
    marker_data = {"dictionary": dict_name, "markers": []}

    for i, corner in enumerate(board.getObjPoints()):
        marker_data["markers"].append(
            {
                "id": int(board.getIds()[i]),  # Marker ID
                "corners": corner.tolist(),  # Convert NumPy array to list
            }
        )

    return marker_data

def charuco_board_to_json(
    board: cv2.aruco.CharucoBoard,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    aruco_dict_name: str,
    save_file: str = "charuco_board.json",
) -> None:
    # Export the board and dictionary for later use as json
    serialized_board = {
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_length_m": square_length_m,
        "marker_length_m": marker_length_m,
        "dictionary": aruco_dict_name,
        "markers": []
    }
    for i, corner in enumerate(board.getObjPoints()):
        serialized_board["markers"].append(
            {
                "id": int(board.getIds()[i]),  # Marker ID
                "corners": corner.tolist(),  # Convert NumPy array to list
            }
        )

    with open(save_file, "w") as f:
        json.dump(serialized_board, f, indent=4)
    
    return None

def load_charuco_board_from_json(
        json_file: str
) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    with open(json_file, "r") as f:
        data = json.load(f)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, data["dictionary"]))

    # Read marker ids
    marker_ids = [marker["id"] for marker in data["markers"]]
    # Validate marker ids

    # Create the Charuco board
    board = cv2.aruco.CharucoBoard(
        size=(data["squares_x"], data["squares_y"]),
        squareLength=data["square_length_m"],
        markerLength=data["marker_length_m"],
        dictionary=aruco_dict,
        ids=np.array(marker_ids)
    )

    return board, aruco_dict

def get_larger_board(export_image=False):
    charuco_marker_dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_250
    )
    # Generate the ChArUco board
    SQUARE_LENGTH = 0.0617
    MARKER_LENGHT = 0.04216
    NUMBER_OF_SQUARES_VERTICALLY = 5
    NUMBER_OF_SQUARES_HORIZONTALLY = 4

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

        serialized_board = board_to_json(board, "DICT_4X4_250")

        # Save to JSON
        with open("calibration/calibration_board_large.json", "w") as f:
            json.dump(serialized_board, f, indent=4)

    return board, charuco_marker_dictionary

def load_single_aruco_board(board_config):
    with open(board_config, "r") as f:
        board_data = json.load(f)

    # load which dict from config
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, board_data["dictionary"])
    )
    print(f"Loaded board with dictionary {board_data['dictionary']}")

    # load ids and corners from config
    marker_ids_list = []
    marker_corners_list = []

    for marker in board_data["markers"]:
        marker_ids_list.append(marker["id"])
        marker_corners_list.append(marker["corners"])

    board = cv2.aruco.Board(
        objPoints=np.array(marker_corners_list, np.float32),
        dictionary=aruco_dict,
        ids=np.array(marker_ids_list),
    )

    return board

def load_single_apriltag_board(board_config):
    with open(board_config, "r") as f:
        board_data = json.load(f)

    # load which dict from config
    apriltag_dict_str = board_data["dictionary"]

    # load ids and corners from config
    marker_ids_list = []
    marker_corners_list = []

    for marker in board_data["markers"]:
        marker_ids_list.append(marker["id"])
        marker_corners_list.append(marker["corners"])

    board = AprilTagBoard(
        objPoints=np.array(marker_corners_list, np.float32),
        ids=np.array(marker_ids_list),
        dictionary=apriltag_dict_str
    )

    return board

def load_aruco_boards(board_configs):
    boards = []

    for board_config in board_configs:
        board = load_single_aruco_board(board_config)
        boards.append(board)

    return boards

def load_boards(
        board_configs: list[str],
        board_types: list[str]
) -> list[cv2.aruco.Board | cv2.aruco.CharucoBoard | AprilTagBoard]:
    """
    Load multiple boards of any of the three supported types:
    - Aruco boards
    - Charuco boards
    - AprilTag boards

    Parameters
    ----------
    board_configs : list of str
        List of paths to the board configuration JSON files.
    board_types : list of str
        List of board types corresponding to each configuration file.
        Supported types are "aruco", "charuco", and "apriltag".
    """
    boards = []
    for board_config, board_type in zip(board_configs, board_types):
        if board_type == "charuco":
            board, _ = load_charuco_board_from_json(board_config)
        elif board_type == "aruco":
            board = load_single_aruco_board(board_config)
        elif board_type == "apriltag":
            board = load_single_apriltag_board(board_config)
        else:
            raise ValueError(f"Unknown board type: {board_type}")
        boards.append(board)

    return boards