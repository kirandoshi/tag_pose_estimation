import threading
import time
import base64
import cv2
import zmq

from typing import Dict, Optional, Any
import numpy as np
from numpy.typing import NDArray

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tag_pose_estimation.board_pose_filter import BoardPoseFilter


class BoardPoseListner:
    box_filters: Dict[str, BoardPoseFilter]

    def __init__(self, box_pose_socket_address: str, update_rate: float = 0.01):
        """
        Initialize the board pose estimator.

        Args:
            box_pose_socket_address: The ZMQ socket address for receiving pose data
            update_rate: How frequently to check for updates (in seconds)
        """
        self.box_pose_socket_address = box_pose_socket_address

        self.context = None
        self.box_pose_socket = None

        self.update_rate = update_rate
        self.box_filters = {}
        self.frames = {}

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        """Start the board pose estimation thread"""
        try:
            # Initialize ZMQ context and socket
            self.context = zmq.Context()
            self.box_pose_socket = self.context.socket(zmq.SUB)
            self.box_pose_socket.setsockopt(
                zmq.CONFLATE, 1
            )  # Keep only the latest message
            self.box_pose_socket.connect(self.box_pose_socket_address)
            self.box_pose_socket.setsockopt(zmq.SUBSCRIBE, b"")

            self._running = True

            # Create thread and start it
            self._thread = threading.Thread(target=self._update_loop)
            self._thread.daemon = True
            self._thread.start()

            return True
        except Exception as e:
            print(f"Failed to start board pose estimator: {e}")
            if hasattr(self, "box_pose_socket"):
                self.box_pose_socket.close()
            if hasattr(self, "context"):
                self.context.term()
            return False

    def stop(self):
        """Stop the board pose estimation thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        # Clean up ZMQ resources
        if hasattr(self, "box_pose_socket"):
            self.box_pose_socket.close()
        if hasattr(self, "context"):
            self.context.term()

    def get_pose(self, board_id: str) -> Optional[NDArray]:
        """
        Get the latest pose of a specific board.

        Args:
            board_id: The ID of the board to get the pose for

        Returns:
            The board pose data if available, None otherwise
        """
        with self._lock:
            if board_id in self.box_filters:
                return self.box_filters[board_id].get_pose()

        return None

    def get_frame(self, camera_id: str) -> Optional[NDArray]:
        """
        Get the latest image frame of a specific camera.

        Args:
            camera_id: The ID of the camera to get the frame for

        Returns:
            The camera image frame if available, None otherwise
        """
        with self._lock:
            if camera_id in self.frames:
                return self.frames[camera_id]

        return None

    def is_stable(self, board_id: str) -> Optional[bool]:
        """
        Figure out if the box pose has converged.

        Args:
            board_id: The ID of the board to get the pose for

        Returns:
            Boolen if we think it has converged.
        """
        with self._lock:
            if board_id in self.box_filters:
                return self.box_filters[board_id].is_stable()

        return None

    def get_confidence(self, board_id: str) -> float:
        with self._lock:
            if board_id in self.box_filters:
                return self.box_filters[board_id].get_confidence()

        return None

    def get_tracked_board_ids(self):
        """Return a list of all board IDs currently being tracked"""
        with self._lock:
            return list(self.box_filters.keys())

    def _update_loop(self):
        """The main update loop that runs in a separate thread"""
        try:
            while self._running:
                # Get box pose estimation
                try:
                    data = self.box_pose_socket.recv_json(flags=zmq.NOBLOCK)
                    box_pose_measurement = data.get("poses", None)
                    # print(f"received {len(box_pose_measurement)} box pose measurements")
                    # print(box_pose_measurement)

                    with self._lock:
                        if box_pose_measurement is not None:
                            for box, measurement in box_pose_measurement.items():
                                box = str(box)
                                if box not in self.box_filters:
                                    self.box_filters[box] = BoardPoseFilter(measurement)

                            self.box_filters[box].update(measurement)
                    
                    # Check if images are being published
                    frame_dict = data.get("images", None)
                    if frame_dict is not None:
                        for camera_id, frame_b64 in frame_dict.items():
                            frame_data = base64.b64decode(frame_b64)
                            np_arr = np.frombuffer(frame_data, np.uint8)
                            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            with self._lock:
                                self.frames[camera_id] = frame
                except zmq.Again:
                    pass

                # Sleep for a bit to avoid hogging CPU
                time.sleep(self.update_rate)
        except Exception as e:
            print(f"Error in board pose estimator thread: {e}")

        finally:
            # Clean up ZMQ socket
            self.box_pose_socket.close()
