import argparse

from tag_pose_estimation.board_builder import build_object_board

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a board for an Object with tags arranged in a 3D " 
            "configuration from detected markers and save to JSON."
        )
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=0.04,
        help=("Indicate the size of the markers in meters. Ensure that you are "
             "using an accurate size and ensure that the side length that you" 
             " are using is the correct one for the marker type. Refer to "
             " the corresponding tag documentation for more details."
        ),
    )
    parser.add_argument(
        "--reference_marker",
        type=int,
        default=None,
        help=("Allow specifying a reference marker ID to define the origin of "
              "the board coordinate frame (optional). If not provided, the "
              "board pose will be centered between all markers. The origin "
              "of the reference marker will be at its center."
        ),
    )
    parser.add_argument(
        "--reference_marker_offset",
        type=str,
        default=None,
        help=(
            "Specify the offset of the board origin from the reference marker "
            "in the reference marker's coordinate frame as x,y,z in meters "
            "(e.g., '0.0,0.0,0.0')."
        ), 
    )
    parser.add_argument(
        "--center_pose",
        type=bool,
        default=True,
        help=(
            "Specify if the pose should be at the marker with the specified "
            "ID, or centered between all markers (default True)."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        nargs="?",
        const="",
        default="",
        help=(
            "Name of the board (optional). If omitted or provided without a "
            "value, will be empty."
        ),
    )
    parser.add_argument(
        "--tag_type",
        type=str,
        default="apriltag",
        help="The tag type that should be used. Can be 'apriltag' or 'aruco'.",
    )
    parser.add_argument(
        "--tag_family",
        type=str,
        help="The tag family to use. These families are defined by the tag type.",
    )
    parser.add_argument(
        "-c",
        "--camera_config_path",
        type=str,
        help="Path to the camera configuration file.",
    )
    parser.add_argument(
        "--max_id",
        type=int,
        help=(
            "Maximum marker ID to consider, if set any id higher than given "
            "value will be ignored (optional). Can be helpful when detection "
            "is noisy and spurious detections with high IDs are present."
        ),
    )

    args = parser.parse_args()
    
    # Use real CLI args by default, but allow an easy drop-in debug config via
    # the DEBUG_ARGS env var (set to "1" to enable).
    # args = argparse.Namespace(
    #     camera_config_path="config/camera_config/default_webcam_config.json",
    #     marker_size=0.032625,
    #     # tag_type="apriltag",
    #     tag_type="aruco",
    #     # tag_family="tag16h5",
    #     tag_family="DICT_4X4_50",
    #     name="",
    #     center_pose=False,
    #     reference_marker=7,
    #     # reference_marker=None,
    #     reference_marker_offset="0.0,0.02175,0.0",
    #     # reference_marker_offset=None,
    #     max_id=17,
    # )

    build_object_board(args)

if __name__ == "__main__":
    main()