"""
Detector wrappers for AprilTag and Aruco detectors.

Provides a common interface for both detectors. The output of the detect()
method is consistent across both wrappers.

Typical usage example:

    # Using either detector wrapper
    detector = AprilTagDetectorWrapper() 
    # or
    detector = ArucoTagDetectorWrapper()
    # then to detect tags in an image:
    detections = detector.detect(image)
"""
import os
import sys

import numpy as np
import cv2

# This import should work if the apriltag repo was installed using the
# install_apriltag.sh script provided in this package.
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from apriltag.build.apriltag import apriltag


class AprilTagDetectorWrapper:
    def __init__(
            self, 
            tag_family_name: str, 
            nthreads: int = 1, 
            quad_decimate: float = 1.0, 
            quad_sigma: float = 0.0,
            refine_edges: bool = True, 
            maxhamming: int = 0, 
            debug: bool = False) -> None:
        """
        Initializes the AprilTag detector with specified parameters.
        
        Args:
            tag_family_name (str): 
                The family of AprilTags to detect.
            nthreads (int): 
                Number of threads to use for detection.
            quad_decimate (float): 
                Decimation factor for the input image.
            quad_sigma (float): 
                Sigma for Gaussian blur applied to the image.
            refine_edges (bool): 
                Whether to refine the edges of detected tags.
            decode_sharpening (float): 
                Sharpening factor for decoding tags.
            debug (bool): 
                Whether to enable debug mode.
        """
        # Package Defaults

        # apriltag(
        #     family: str,
        #     threads: int = 1,
        #     maxhamming: int = 1,
        #     decimate: float = 2.0,
        #     blur: float = 0.0,
        #     refine_edges: bool = True,
        #     debug: bool = False
        # )
        
        # Read in parameters, defaults are based on good choices from experiments
        self.detector = apriltag(
            tag_family_name,
            threads=nthreads,
            maxhamming=maxhamming,
            decimate=quad_decimate,
            blur=quad_sigma,
            refine_edges=refine_edges,
            debug=debug
        )

        # Save the tag family name
        self._tag_family_name = tag_family_name
        
        return None

    def detect(
            self, 
            image: np.ndarray) -> tuple[list, list]:
        """
        Detects AprilTags in the given image.
        
        Args:
            image (numpy.ndarray):
                The input image in which to detect AprilTags.
        
        Returns:
            tuple: A tuple containing corners and ids of detected AprilTags.
        """
        detections = self.detector.detect(image)

        # Convert detections to corners and ids in OpenCV format
        corners, ids = self._detections_to_corners_ids(detections)

        return corners, ids
    
    def _detections_to_corners_ids(
            self,
            detections: dict
    ) -> tuple[list, list]:
        if detections:
            # Corners should be Nx4x2
            corners = [np.expand_dims(d["lb-rb-rt-lt"].astype(np.float32),0) for d in detections]
            # Rearrange corners to lt-rt-rb-lb order for OpenCV
            corners = self._rearrange_corners(np.vstack(corners))
            # Make list again
            corners = [np.expand_dims(c.astype(np.float32), 0) for c in corners]

            # Get ids as Nx1 array of int32
            ids = np.array([[d['id']] for d in detections], dtype=np.int32)
        else:
            corners = []
            ids = None
        
        return corners, ids
    
    @property
    def tag_family(self) -> str:
        """
        Returns the tag family used by the detector.
        """
        return self._tag_family_name

    def _rearrange_corners(
            self,
            corners: np.ndarray
        ) -> np.ndarray:
        """
        Rearrange corners from C1: [lb, rb, rt, lt] to C2: [lt, rt, rb, lb]. 

        Apriltag uses C1 convention when it returns detected corners, while 
        OpenCV Aruco functions use C2 convention.

        Args:
            corners (numpy.ndarray):
                Array of shape Nx4x2 representing N detected markers with 4 corners
                in order [lb, rb, rt, lt].

        
        Returns:
            np.ndarray
                Array of shape Nx4x2 representing N detected markers with 4 corners
                in order [lt, rt, rb, lb].
        """
        if corners.ndim != 3 or corners.shape[1:] != (4, 2):
            raise ValueError("Input corners must be of shape Nx4x2.")
        
        # Define the new order of indices
        new_order = [3, 2, 1, 0]
        
        # Rearrange the corners using advanced indexing
        rearranged_corners = corners[:, new_order, :]
        
        return rearranged_corners

class ArucoTagDetectorWrapper:
    def __init__(
            self, 
            tag_family_name: str,
        ) -> None:
        """
        Initializes the Aruco marker detector with specified dictionary.
        
        Args:
            tag_family_name (str): 
                The OpenCV predefined dictionary name for the desired Aruco 
                marker family.
        """
        aruco_dict = getattr(cv2.aruco, tag_family_name)
        self._dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self._parameters = cv2.aruco.DetectorParameters()

        return None

    def set_parameters_for_pose_estimation(self) -> None:
        """
        Sets detector parameters optimized for pose estimation.
        """
        self._parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._parameters.relativeCornerRefinmentWinSize = 0.15
        self._parameters.cornerRefinementMaxIterations = 70
        return None
    
    def update_parameters(self, parameters: cv2.aruco.DetectorParameters) -> None:
        """
        Updates the detector parameters.
        
        Args:
            parameters (cv2.aruco.DetectorParameters): 
                The new detector parameters to set.
        """
        self._parameters = parameters
        return None

    def detect(
            self, image: np.ndarray) -> tuple[list, list]:
        """
        Detects Aruco AprilTags in the given image.
        
        Args:
            image (numpy.ndarray): 
                The input image in which to detect Aruco AprilTags.
        
        Returns:
            tuple: A tuple containing detected corners and ids.
        """
        corners, ids, _ = cv2.aruco.detectMarkers(
            image, self._dictionary, parameters=self._parameters)
        return corners, ids
    
    @property
    def tag_family(self) -> str:
        """
        Returns the tag family used by the detector.
        """
        return self._dictionary