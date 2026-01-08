import argparse
import logging

from tag_pose_estimation.board_visualiser import visualise_board_from_config

def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--filepath", 
        help=(
            "Path to JSON file defining the board configuration. "
            "Three options are supported: "
            "(1) Absolute path to the JSON file, "
            "(2) Path relative to project root directory, "
            "(3) Single filename, which is then assumed to be in the "
            "config/object_boards/ directory."
        )
    )
    
    args = parser.parse_args()

    visualise_board_from_config(args.filepath)

    return None

if __name__ == "__main__":
    logging.basicConfig(
        format=('%(asctime)s | %(name)s | %(levelname)s | %(message)s'), 
        datefmt='%m/%d/%Y %H:%M:%S', 
        level=logging.INFO
    )
    main()