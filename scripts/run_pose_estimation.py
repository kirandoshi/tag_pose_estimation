import argparse

from tag_pose_estimation.pose_estimator import pose_estimator_runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track pose with a robot.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/pose_estimation_configs/default_pose_estimation_config.json",
        help=(
                "Path to the configuration file."
                "Three options are supported: "
                "(1) Absolute path to the JSON configuration file, "
                "(2) Path relative to project root directory, "
                "(3) Single filename, which is then assumed to be in the "
                "config/pose_estimation_configs/ directory."
        ),
    )
    parser.add_argument(
        "--tag_type",
        type=str,
        help="The tag type that should be used. Can be 'apriltag' or 'aruco'.",
    )
    parser.add_argument(
        "--tag_family",
        type=str,
        help="The tag family to use. These families are defined by the tag type.",
    )
    parser.add_argument(
        "-d", "--detection_type",
        type=str,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-f", "--frequency",
        type=int,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-ut", "--use_transform",
        type=bool,
        help=("Whether to use the additional transform from world frame to a "
              "given frame, specified in the config with key "
              "'pose_estimation_frame_transform'.")
    )
    parser.add_argument(
        "-i", "--publish_image",
        type=bool,
        help="Whether to publish the image."
    )
    
    args = parser.parse_args()

    config_path = args.config
    pose_estimator_runner(
        config_path,
        tag_type=args.tag_type,
        tag_family=args.tag_family,
        goal_frequency=args.frequency,
        detection_type=args.detection_type,
        publish_image=args.publish_image,
        use_additional_transform=args.use_transform
    )
