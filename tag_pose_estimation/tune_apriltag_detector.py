import os
import sys
from pathlib import Path
import cv2

# Add path to apriltag package
# Apritag package is located in root/python_apriltag
# This is a bit hacky but works for now
# This file is located in root/robot_ipc_control/pose_estimation
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from python_apriltag.python_apriltag.apriltag import apriltag

from tag_pose_estimation.apriltag_utils import (
    detections_to_corners_ids)

def main():
    # Defaults

    # apriltag(
    #     family: str,
    #     threads: int = 1,
    #     maxhamming: int = 1,
    #     decimate: float = 2.0,
    #     blur: float = 0.0,
    #     refine_edges: bool = True,
    #     debug: bool = False
    # )

    quad_decimate = 1.0
    quad_sigma = 0.0
    nthreads = int(4)
    debug = True
    refine_edges = False
    apriltag_family = 'tag16h5'

    detector = apriltag(
        apriltag_family,
        decimate=quad_decimate,
        maxhamming=0,
        blur=quad_sigma,
        threads=nthreads,
        debug=debug,
        refine_edges=refine_edges,
    )

    # Get test image path
    test_images_folder = Path(__file__).parent.parent.parent / "saved_frames"
    specific_folder = "20250930_140855" # Change this to your specific folder
    test_image_path = test_images_folder / specific_folder / "CAM_L_frame_0000.png"

    if not test_image_path.exists():
        print(f"Test image {test_image_path} does not exist. Please run show_cam_feeds.py with --save_frames first.")
        return
    print(f"Using test image: {test_image_path}")
    img = cv2.imread(str(test_image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    corners, ids = detections_to_corners_ids(detections)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
    cv2.imshow("Detections", img)
    print(f"Detected {len(detections)} markers.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
