import argparse

from tag_pose_estimation.charuco_board_making import create_charuco_board

def main():
    parser = argparse.ArgumentParser(
        description="Create a customizable Charuco board and save as PDF.")
    parser.add_argument(
        "--squares_x", type=int, required=True, 
        help="Number of squares in X direction"
    )
    parser.add_argument(
        "--squares_y", type=int, required=True, 
        help="Number of squares in Y direction"
    )
    parser.add_argument(
        "--square_length_m", type=float, required=True, 
        help="Square length in m"
    )
    parser.add_argument(
        "--marker_length_m", type=float, required=True, 
        help="Marker length in m"
    )
    parser.add_argument(
        "--aruco_dict", type=str, default="DICT_4X4_50", 
        help="Aruco dictionary name (e.g., DICT_4X4_50)"
    )
    parser.add_argument(
        "--start_id", type=int, default=0, 
        help="Starting marker ID (default: 0)"
    )

    args = parser.parse_args()

    create_charuco_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_m,
        marker_length_m=args.marker_length_m,
        aruco_dict_name=args.aruco_dict,
        start_id=args.start_id
    )

if __name__ == "__main__":
    main()