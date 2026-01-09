import argparse
import logging

from tag_pose_estimation.camera_extrinsic_calibration_single import (
    single_camera_extrinsic_calibration,
)

def main():
    parser = argparse.ArgumentParser(
        description="Extrinsic camera calibration with ChArUco board.")
    
    parser.add_argument(
        "--camera_config_path",
        type=str,
        help=(
            "Path to camera configuration file."
            "Three options are supported: "
            "(1) Absolute path to the JSON camera configuration file, "
            "(2) Path relative to project root directory, "
            "(3) Single filename, which is then assumed to be in the "
            "config/camera_config/ directory."
        ),
    )

    parser.add_argument(
        "--board_config_path",
        type=str,
        default=None,
        help=(
            "Path to JSON file defining the Charuco board. "
            "Three options are supported: "
            "(1) Absolute path to the JSON file, "
            "(2) Path relative to project root directory, "
            "(3) Single folder name, which is then assumed to be in the "
            "config/calibration_boards/ directory and includes a file called "
            "'charuco_board.json'."
        )
    )
        
    args = parser.parse_args()

    single_camera_extrinsic_calibration(
        args.camera_config_path,
        args.board_config_path
    )

if __name__ == "__main__":
    logging.basicConfig(
        format=('%(asctime)s | %(name)s | %(levelname)s | %(message)s'), 
        datefmt='%m/%d/%Y %H:%M:%S', 
        level=logging.INFO
    )
    main()