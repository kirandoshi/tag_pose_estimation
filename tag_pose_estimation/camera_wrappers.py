import time
import numpy as np
import pyrealsense2 as rs
import cv2

from tag_pose_estimation.transform_utils import (
    load_camera_calibration,
)


class CameraWrapperBase:
    def __init__(self):
        self.camera = None
        self.dist_coeffs = None
        self.camera_matrix = None

        self.homogeneous_transform = None

    def get_frame(self):
        pass

class WebcamCamera:
    def __init__(
        self,
        calibration_path=None,
        camera_id=None,
        camera_matrix_path=None,
        camera_dist_path=None,
        focus_path=None,
    ):
        if calibration_path is not None:
            self.homogeneous_transform = load_camera_calibration(
                calibration_path)
        else:
            self.homogeneous_transform = None

        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        # request MJPEG codec
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # request 1080p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Turn off autofocus
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        if camera_matrix_path is not None:
            self.camera_matrix = np.load(camera_matrix_path)
        else:
            self.camera_matrix = None
        
        if camera_dist_path is not None:
            self.dist_coeffs = np.load(camera_dist_path)
        else:
            self.dist_coeffs = None

        if focus_path is not None:
            self.cam_calibration_focus = np.load(focus_path).item()
        else:
            self.cam_calibration_focus = None

        # Set focus value to that used during calibration
        if self.cam_calibration_focus is not None:
            print(f"Setting focus to {self.cam_calibration_focus}")
            self.cap.set(cv2.CAP_PROP_FOCUS, self.cam_calibration_focus)
            print("Current focus value:", self.cap.get(cv2.CAP_PROP_FOCUS))
            self.focus_set = True
        else:
            # Set to zero if no focus was saved
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)
            print("Current focus value:", self.cap.get(cv2.CAP_PROP_FOCUS))
            self.focus_set = False
        
        # Define how many frames to flush when requested
        # This number here was determined experimentally, i don't know how to
        # find it otherwise
        self._flush_buffer_count = 5
        
        # Wait a moment for camera to adjust with time package
        time.sleep(2)
    
    def _set_focus(
            self, 
            focus_value: int) -> None:
        print(f"Setting focus to {focus_value}")
        self.cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        print("Current focus value:", self.cap.get(cv2.CAP_PROP_FOCUS))
        time.sleep(2)

    def set_focus(
            self, 
            focus_value: int) -> None:
        if not self.focus_set:
            print("Focus was not set during initialization. Setting now.")
            self.focus_set = True
            self._set_focus(focus_value)
        else:
            print("Focus already set. Not changing.")

        return None

    def set_focus_override(
            self, 
            focus_value: int) -> None:
        print(f"Overriding focus")
        self._set_focus(focus_value)
        return None

    def get_frame(self, *, flush_buffer=False):
        if flush_buffer:
            # Flush the buffer by reading several frames
            for _ in range(self._flush_buffer_count):
                self.cap.read()

        ret, frame = self.cap.read()
        if ret:
            return frame

        return None
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        return None

class T265RealSenseCamera:
    def __init__(self, 
                 calibration_path=None, 
                 serial_number=None):
        """
        Returns a camera matrix K from librealsense intrinsics
        """

        def camera_matrix(intrinsics):
            return np.array(
                [
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1],
                ]
            )

        """
        Returns the fisheye distortion from librealsense intrinsics
        """

        def fisheye_distortion(intrinsics):
            return np.array(intrinsics.coeffs[:4])

        self.pipeline = rs.pipeline()
        config = rs.config()

        # if serial_number:
        #     config.enable_device(serial_number)

        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        streams = {
            "left": profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
            "right": profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile(),
        }
        intrinsics = {
            "left": streams["left"].get_intrinsics(),
            "right": streams["right"].get_intrinsics(),
        }

        # Print information about both cameras
        print("Left camera:", intrinsics["left"])
        print("Right camera:", intrinsics["right"])

        # Translate the intrinsics from librealsense into OpenCV
        self.camera_matrix = camera_matrix(intrinsics["left"])
        self.dist_coeffs = fisheye_distortion(intrinsics["left"])

        # K_left  = camera_matrix(intrinsics["left"])
        # D_left  = fisheye_distortion(intrinsics["left"])
        # K_right = camera_matrix(intrinsics["right"])
        # D_right = fisheye_distortion(intrinsics["right"])

        # color_sensor = profile.get_device().query_sensors()[0]
        # color_sensor.set_option(rs.option.enable_auto_exposure, True)
        # Get the sensor once at the beginning. (Sensor index: 1)

        # sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]

        # Set the exposure anytime during the operation
        # sensor.set_option(rs.option.exposure, 7000.000)
        # sensor.set_option(rs.option.enable_auto_exposure, True)
        # sensor.set_option(rs.option.enable_auto_white_balance, True)
        # sensor.set_option(rs.option.sharpness, 100)
        if calibration_path is not None:
            self.homogeneous_transform = load_camera_calibration(
                calibration_path)
        else:
            self.homogeneous_transform = None

    def get_frame(self, **kwargs):
        # frames = self.pipeline.wait_for_frames()
        frames = self.pipeline.poll_for_frames()
        if frames:
            f1 = frames.get_fisheye_frame(1).as_video_frame()
            frame = np.asanyarray(f1.get_data())
            # color_frame = frames.get_color_frame()
            # frame = np.asanyarray(color_frame.get_data())

            return frame

        return None

class RealSenseCamera:
    def __init__(self, 
                 calibration_path=None, 
                 serial_number=None):
        self.pipeline = rs.pipeline()
        config = rs.config()

        if serial_number:
            config.enable_device(serial_number)

        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        # color_sensor = profile.get_device().query_sensors()[0]
        # color_sensor.set_option(rs.option.enable_auto_exposure, True)
        # Get the sensor once at the beginning. (Sensor index: 1)

        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]

        # Set the exposure anytime during the operation
        sensor.set_option(rs.option.exposure, 7000.000)
        # sensor.set_option(rs.option.enable_auto_exposure, True)
        # sensor.set_option(rs.option.enable_auto_white_balance, True)
        # sensor.set_option(rs.option.sharpness, 100)

        # Get camera intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Create camera matrix from intrinsics
        self.camera_matrix = np.array(
            [
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Get distortion coefficients
        self.dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)

        if calibration_path is not None:
            self.homogeneous_transform = load_camera_calibration(
                calibration_path)
        else:
            self.homogeneous_transform = None

    def get_frame(self, **kwargs):
        # frames = self.pipeline.wait_for_frames()
        frames = self.pipeline.poll_for_frames()
        if frames:
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())

            return frame

        return None
