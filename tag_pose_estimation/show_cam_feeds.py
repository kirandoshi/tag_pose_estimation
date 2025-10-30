import cv2
from tag_pose_estimation.camera_wrappers import WebcamCamera

cam_name_to_cam_id = {
      "/dev/LOGI_CAM_LOW": "CAM_R",
      "/dev/LOGI_CAM_LEFT_ARM": "CAM_L",
}

def main(args):
# Capture frames
      cam_names = args.cam_names.split(",") if args.cam_names else [0]
      cams = []
      for cam_name in cam_names:
            print(f"Starting camera {cam_name}. 'q' to quit.")
            cam = WebcamCamera(camera_id=cam_name)
            cam.set_focus(args.focus)
            cams.append(cam)

      aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

      if args.save_frames:
                  frames = {}
                  for cam_name in cam_names:
                        frames[cam_name] = []

      while True:
            for i, cam in enumerate(cams):
                  frame = cam.get_frame()
                  if frame is not None:
                        ret = True
                  if not ret:
                        raise RuntimeError(f"Failed to read from camera with name {cam_names[i]}")

                  if args.save_frames:
                        frames[cam_names[i]].append(frame)

                  # Detect aruco markers
                  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
                  if ids is not None:
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                  cv2.imshow(f'Camera {cam_names[i]}', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                  break

      if args.save_frames:
            from pathlib import Path
            from datetime import datetime
            folder = Path("saved_frames")
            folder.mkdir(parents=True, exist_ok=True)
            # time/date stamped subfolder
            sub_folder = folder / (datetime.now().strftime("%Y%m%d_%H%M%S"))
            sub_folder.mkdir(parents=True, exist_ok=True)
            cam_counter = 0
            for cam_name in cam_names:
                  cam_id = cam_name_to_cam_id.get(cam_name, None)
                  if cam_id is None:
                        cam_id = f'cam_{cam_counter}'
                  for idx, frame in enumerate(frames[cam_name]):
                        filename = sub_folder / f"{cam_id}_frame_{idx:04d}.png"
                        cv2.imwrite(str(filename), frame)
            print(f"Saved {len(frames[cam_name])} frames for each camera to {sub_folder}")

      for cam in cams:
            cam.release()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show camera feed")
    parser.add_argument("--cam_names", type=str, help="Camera names or indices")

    parser.add_argument("--save_frames", action="store_true", help="Save frames to disk")
    parser.add_argument(
        "-f", "--focus",
        type=int,
        default=0,
        help="Focus value to set between 0 and 250 or  0 and 1 depending on "
            " camera. Default is 0. Autofocus is turned off in this script.",
    )
    args = parser.parse_args()

    main(args)