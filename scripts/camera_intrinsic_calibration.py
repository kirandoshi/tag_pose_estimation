import argparse
import logging

from tag_pose_estimation.intrinsic_calibration import (
    intrinsic_camera_calibration,
)


def main():
    parser = argparse.ArgumentParser(
        description="Intrinsic camera calibration using a Charuco board."
    )
    parser.add_argument(
        '--charuco_board_path',
        type=str,
        default=None,
        help='Path to JSON file defining the Charuco board. If not provided, '
             'a default larger Charuco board will be used.'
    )
    parser.add_argument(
        '--cam_name',
        type=str,
        default=None,
        help='Camera name or ID to use for capturing frames. '
             'If not provided, defaults to the first camera (ID 0).'
    )
    parser.add_argument(
        '-f', '--focus',
        type=int,
        default=0,
        help="Focus value to set between 0 and 250 or  0 and 1 depending on "
            "camera. Default is 0. Autofocus is turned off in this script. "
            "The focus value used during calibration then must be used for " 
            "subsequent use. Changinging focus changes the camera intrinsics.",
    )

    args = parser.parse_args()

    intrinsic_camera_calibration(args)

if __name__ == "__main__":
    logging.basicConfig(
        format=('%(asctime)s | %(name)s | %(levelname)s | %(message)s'), 
        datefmt='%m/%d/%Y %H:%M:%S', 
        level=logging.INFO
    )
    main()