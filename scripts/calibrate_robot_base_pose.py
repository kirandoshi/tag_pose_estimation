import argparse

from tag_pose_estimation.robot_base_calibration import calibrate_robot_base

def main():
    parser = argparse.ArgumentParser(
        description="Robot base calibration with markers."
    )
    parser.add_argument(
        "-r", "--robot type",
        type=str,
        help="The type of robot that is being calibrated.",
        required=True,
    )
    parser.add_argument(
        "--ee_board_path",
        type=str,
        help="Path to the calibration config file.",
        required=True,
    )
    parser.add_argument(
        "--ee_board_tag_type",
        type=str,
        help=("Type of tag on the ee board (charuco, aruco or apriltag). "
              "Charuco is recommended."
        ),
        required=True
    )
    parser.add_argument(
        "--world_board_path",
        type=str,
        help="Path to the world calibration board config file.",
        required=True,
    )
    parser.add_argument(
        "--camera_config_path",
        type=str,
        help="Path to the camera configuration file.",
        required=True,
    )

    args = parser.parse_args()

    calibrate_robot_base(
        camera_config_path=args.camera_config_path,
        ee_board_path=args.ee_board_path,
        ee_board_type=args.ee_board_tag_type,
        world_board_path=args.world_board_path,
        robot_type=args.robot_type,
    )

if __name__ == "__main__":
    main()