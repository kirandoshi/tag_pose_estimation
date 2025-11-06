import sys
import os
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation
import json
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam import Pose3, Rot3, Point3, BetweenFactorPose3, noiseModel
from gtsam import (
    NonlinearFactorGraph,
    Values,
    LevenbergMarquardtOptimizer,
    PriorFactorPose3,
    LevenbergMarquardtParams,
    PriorFactorDouble
)

from tag_pose_estimation.camera_wrappers import WebcamCamera, RealSenseCamera
from tag_pose_estimation.utils import board_to_json
from tag_pose_estimation.apriltag_utils import (
    detections_to_corners_ids)
from tag_pose_estimation.apriltag_board import AprilTagBoard
from tag_pose_estimation.detector_wrappers import (
    AprilTagDetectorWrapper,
    ArucoTagDetectorWrapper
)

class BoardBuilder:
    """
    Class to build a board from detected markers with a connected live camera.

    Builds a board, a 3D configuration of markers with known relative positions,
    by detecting markers in the captured frames of a connected camera. Board can
    then be saved to a JSON file for later use in pose estimation.
    """
    def __init__(
            self,
            camera_config_path: str, 
            marker_size: int, 
            tag_type: str,
            tag_family: str,
            filter_max_id: int = None):
        """
        Initialize BoardBuilder with camera and detector.

        Args:
            camera_config_path (str): 
                Path to camera configuration file.
            marker_size (int): 
                Size of each marker in meters.
            tag_type (str): 
                Type of tag ("apriltag" or "aruco").
            tag_family (str): 
                Tag family for either of the tag types.
            filter_max_id (int, optional): 
                Maximum marker ID to consider. If set, any id higher than this 
                value will be ignored.
        """
        # Set up the detector
        self._tag_family = tag_family
        self._tag_type = tag_type
        if tag_type == "apriltag":
            self._detector = AprilTagDetectorWrapper(
                tag_family_name=tag_family,
                refine_edges=True
            )
        elif tag_type == "aruco":
            self._detector = ArucoTagDetectorWrapper(
                tag_family_name=tag_family
            )
        else:
            raise ValueError(f"Unsupported tag type: {tag_type}")

        # Set the maximum marker ID to consider
        self._filter_max_id = filter_max_id

        # Use Camera class from camera_wrappers.py
        try:
            with open(camera_config_path) as f:
                config = json.load(f)
        except FileNotFoundError:
            print("Error: Config file not found.", file=sys.stderr)
            sys.exit(1)
        
        camera_config = config["cameras"][0]

        if camera_config["type"] == "D405":
            serial_number = camera_config.get("serial_number")
            self.camera = RealSenseCamera(serial_number=serial_number)
        elif camera_config["type"] == "Webcam":
            serial_number = camera_config.get("serial_number")
            self.camera = WebcamCamera(
                camera_id=serial_number,
                camera_matrix_path=camera_config["camera_matrix"],
                camera_dist_path=camera_config["dist_coeff"],
                focus_path=camera_config.get("focus_setting"),
            )
        else:
            raise ValueError(f"Unsupported camera type: {camera_config['type']}"
                              " Add CameraWrapper support for this camera.")


        # Create camera matrix from intrinsics
        self.camera_matrix = self.camera.camera_matrix
        # Get distortion coefficients
        self.dist_coeffs = self.camera.dist_coeffs

        # Store parameters
        self.marker_size = marker_size
        self.last_rvec = None
        self.last_tvec = None
        self.tracking_lost_frames = 0
        self.max_lost_frames = 30  # Reset initial guess after these many frames

    def pose_from_matrix(self, T):
        R = Rot3(T[:3, :3])
        t = Point3(*T[:3, 3])
        return Pose3(R, t)

    def initialize_poses_from_spanning_tree(self, rel_poses, keys):
        """Initialize poses by traversing a spanning tree from the first pose."""
        poses = {}
        # Start with identity for the first pose
        first_key = list(keys.keys())[0]
        poses[first_key] = np.eye(4)

        # Build a queue of poses to process
        queue = [first_key]
        processed = set([first_key])

        while queue:
            current = queue.pop(0)

            # Process all connections from this pose
            if current in rel_poses:
                for next_pose, rel_pose in rel_poses[current].items():
                    if next_pose not in processed:
                        # Calculate absolute pose of next_pose
                        poses[next_pose] = poses[current] @ rel_pose
                        queue.append(next_pose)
                        processed.add(next_pose)

        # If any poses weren't reached, initialize them as identity
        for key in keys:
            if key not in poses:
                poses[key] = np.eye(4)
                print(f"Warning: Pose {key} not connected to main component")

        return poses

    def filter_rotation_outliers(self, rel_poses, max_error=0.5):
        """Identify and remove rotation outliers based on cycle consistency."""
        filtered_poses = {k: {} for k in rel_poses}
        outliers = []

        # Check all cycles of length 2 (A->B->A)
        for i in rel_poses:
            for j in rel_poses.get(i, {}):
                if j in rel_poses and i in rel_poses[j]:
                    # We have a cycle i->j->i
                    T_i_j = rel_poses[i][j]
                    T_j_i = rel_poses[j][i]

                    # Check cycle consistency
                    T_cycle = T_i_j @ T_j_i
                    rotation_error = np.linalg.norm(T_cycle[:3, :3] - np.eye(3), "fro")

                    if rotation_error < max_error:
                        # Keep this constraint
                        filtered_poses[i][j] = rel_poses[i][j]
                        filtered_poses[j][i] = rel_poses[j][i]
                    else:
                        outliers.append((i, j, rotation_error))
                        print(
                            f"Removing rotation outlier between {i}-{j}: error={rotation_error:.3f}"
                        )

        return filtered_poses, outliers

    def optimize_pose_graph_gtsam(self, rel_poses, reference_marker):
        graph = NonlinearFactorGraph()
        initial = Values()

        # Use a tighter noise model
        model = noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        )  # rotation, translation

        # Actually use the robust model
        huber = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(k=1.),  # lower k = more robust
            model,
        )

        rel_poses, filtered = self.filter_rotation_outliers(rel_poses)

        keys = {}
        for i, key in enumerate(rel_poses.keys()):
            keys[key] = i

        # Better initialization - use spanning tree or odometry chain
        poses_init = self.initialize_poses_from_spanning_tree(rel_poses, keys)

        # Add nodes with better initial guesses
        for idx, key in enumerate(keys):
            initial.insert(idx, self.pose_from_matrix(poses_init[key]))

        # Add relative pose constraints
        for i, js in rel_poses.items():
            for j, T_i_j in js.items():
                idx_i = keys[i]
                idx_j = keys[j]
                Tij = self.pose_from_matrix(T_i_j)
                graph.add(BetweenFactorPose3(idx_i, idx_j, Tij, huber))

        # Add prior to anchor pose
        anchor_key = list(keys.values())[reference_marker]
        prior_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        )
        graph.add(PriorFactorPose3(anchor_key, Pose3(), prior_noise))

        # Optimize with more iterations
        print(f"Initial error = {graph.error(initial)}")

        params = LevenbergMarquardtParams()
        params.setMaxIterations(100)
        params.setRelativeErrorTol(1e-8)
        optimizer = LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

        print(f"Final error = {graph.error(result)}")

        # Extract results
        optimized = {k: result.atPose3(i).matrix() for k, i in keys.items()}

        # Ensure output is in frame of reference marker
        T_ref = optimized[reference_marker]
        T_ref_inv = np.linalg.inv(T_ref)

        reference_frame_optimized = {}
        for k, T in optimized.items():
            reference_frame_optimized[k] = T_ref_inv @ T
        
        return reference_frame_optimized

    def optimize_with_switchable_constraints(self, rel_poses):
        """Use switchable constraints to automatically identify outliers."""
        graph = NonlinearFactorGraph()
        initial = Values()
        
        # Regular noise model
        model = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        
        keys = {}
        for i, key in enumerate(rel_poses.keys()):
            keys[key] = i

        # Add nodes
        for idx, key in enumerate(keys):
            initial.insert(idx, Pose3())  # Identity as initial guess
        
        # Add switchable variables (one for each constraint)
        switch_idx = len(keys)  # Start index for switch variables
        switch_map = {}  # Maps (i,j) -> switch_idx
        
        for i, js in rel_poses.items():
            for j, T_i_j in js.items():
                idx_i = keys[i]
                idx_j = keys[j]
                
                # Add a switch variable (initialized to 1.0 = fully trusted)
                switch_key = switch_idx
                initial.insert(switch_key, 1.0)
                switch_map[(i,j)] = switch_key
                switch_idx += 1
                
                # Add switchable constraint
                Tij = self.pose_from_matrix(T_i_j)
                graph.add(BetweenFactorPose3(idx_i, idx_j, Tij, model))
                
                # Add prior on switch variable (soft push toward 1.0)
                switch_prior = noiseModel.Diagonal.Sigmas(np.array([0.5]))
                graph.add(PriorFactorDouble(switch_key, 1.0, switch_prior))
        
        # Add prior to anchor pose
        anchor_key = list(keys.values())[0]
        graph.add(PriorFactorPose3(anchor_key, Pose3(), model))
        
        # Optimize
        optimizer = LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
        
        # Identify outliers based on switch values
        outliers = []
        for (i,j), switch_key in switch_map.items():
            switch_value = result.atDouble(switch_key)
            if switch_value < 0.5:  # Threshold for outlier classification
                outliers.append((i, j, switch_value))
        
        print(f"Identified {len(outliers)} outliers:")
        for i, j, val in outliers:
            print(f"  Outlier {i}-{j}: switch value = {val:.3f}")
        
        # Extract results
        optimized = {k: result.atPose3(i).matrix() for k, i in keys.items()}
        return optimized

    def compute_marker_positions_gtsam(self, observations, reference_marker=None):
        """
        Compute relative marker positions using multiple observations.

        Args:
            observations: List of tuples (ids, poses), where:
                - ids is a numpy array of marker IDs
                - poses is a dict mapping marker IDs to their 4x4 transformation matrices

        Returns:
            dict: Mapping of marker IDs to their positions relative to the reference marker
        """
        # First, find the marker that appears most frequently to use as reference
        marker_counts = defaultdict(int)
        for ids, poses in observations:
            for id in ids.flatten():
                marker_counts[id] += 1

        if not marker_counts:
            raise ValueError("No markers detected in observations")

        if reference_marker is None or reference_marker not in marker_counts:
            ref_id = max(marker_counts.items(), key=lambda x: x[1])[0]
        else:
            ref_id = reference_marker

        print(f"Using marker {ref_id} as reference")

        # Create a graph of marker connections
        # For each observation, create edges between all visible markers
        marker_connections = defaultdict(dict)
        for ids, poses in observations:
            visible_markers = ids.flatten()
            for id1 in visible_markers:
                for id2 in visible_markers:
                    if id1 != id2:
                        relative_pose = np.linalg.inv(poses[id1]) @ poses[id2]
                        marker_connections[id1][id2] = relative_pose

        print("starting gtsam")
        res = self.optimize_pose_graph_gtsam(marker_connections, ref_id)
        print(res)

        return res

    def compute_marker_positions(self, observations, reference_marker=None):
        """
        Compute relative marker positions using multiple observations.

        Args:
            observations: List of tuples (ids, poses), where:
                - ids is a numpy array of marker IDs
                - poses is a dict mapping marker IDs to their 4x4 transformation matrices

        Returns:
            dict: Mapping of marker IDs to their positions relative to the reference marker
        """
        # First, find the marker that appears most frequently to use as reference
        marker_counts = defaultdict(int)
        for ids, poses in observations:
            for id in ids.flatten():
                marker_counts[id] += 1

        if not marker_counts:
            raise ValueError("No markers detected in observations")

        if reference_marker is None or reference_marker not in marker_counts:
            ref_id = max(marker_counts.items(), key=lambda x: x[1])[0]
        else:
            ref_id = reference_marker

        print(f"Using marker {ref_id} as reference")

        # Create a graph of marker connections
        # For each observation, create edges between all visible markers
        marker_connections = defaultdict(list)
        for ids, poses in observations:
            visible_markers = ids.flatten()
            for i, id1 in enumerate(visible_markers):
                for id2 in visible_markers[i + 1 :]:
                    if id1 != id2:
                        relative_pose = np.linalg.inv(poses[id1]) @ poses[id2]
                        marker_connections[(id1, id2)].append(relative_pose)

        # Compute shortest paths from reference marker to all other markers
        marker_positions = {ref_id: np.eye(4)}  # Reference marker at origin
        markers_to_process = set(marker_counts.keys()) - {ref_id}

        while markers_to_process:
            best_marker = None
            best_transform = None
            best_error = None

            for target_marker in markers_to_process:
                transforms = []
                for known_marker in marker_positions.keys():
                    if (known_marker, target_marker) in marker_connections:
                        print(len(marker_connections[(known_marker, target_marker)]))
                        # for t in marker_connections[(known_marker, target_marker)]:
                        #     print(t)
                        # print()
                        avg_transform = self.average_transforms(
                            marker_connections[(known_marker, target_marker)]
                        )
                        total_transform = marker_positions[known_marker] @ avg_transform
                        transforms.append(total_transform)

                if transforms:
                    combined_transform = self.average_transforms(transforms)

                    to_ignore = False
                    print(combined_transform)
                    for t in transforms:
                        print(t)
                        if np.linalg.norm(combined_transform[:3, 3] - t[:3, 3]) > 1e-2:
                            print("diff large")
                            to_ignore = True

                    if to_ignore:
                        continue

                    current_error = 0

                    if best_marker is None or best_error > current_error:
                        best_marker = target_marker
                        best_transform = combined_transform
                        best_error = current_error

            if best_marker is None:
                print(f"Warning: Could not find path to markers: {markers_to_process}")
                break

            marker_positions[best_marker] = best_transform
            markers_to_process.remove(best_marker)

        return marker_positions

    def average_transforms(self, transforms):
        """
        Average multiple 4x4 transformation matrices.

        Args:
            transforms: List of 4x4 transformation matrices

        Returns:
            numpy.ndarray: Average 4x4 transformation matrix
        """
        # Separate rotations and translations
        rotations = [t[:3, :3] for t in transforms]
        translations = [t[:3, 3] for t in transforms]

        # Average translations
        avg_translation = np.mean(translations, axis=0)

        # Average rotations using quaternions
        quats = [Rotation.from_matrix(R).as_quat() for R in rotations]
        # Handle antipodal quaternions
        ref_quat = quats[0]
        for i in range(1, len(quats)):
            if np.dot(ref_quat, quats[i]) < 0:
                quats[i] = -quats[i]
        avg_quat = np.mean(quats, axis=0)
        # avg_quat = quats[0]
        avg_quat = avg_quat / np.linalg.norm(avg_quat)  # Normalize
        avg_rotation = Rotation.from_quat(avg_quat).as_matrix()

        # Combine into transformation matrix
        result = np.eye(4)
        result[:3, :3] = avg_rotation
        result[:3, 3] = avg_translation

        return result

    def estimate_marker_poses(self, corners, ids, frame):
        """
        Estimate 3D positions of markers using solvePnP for each marker.
        """
        marker_poses = {}

        # Define 3D coordinates of marker corners in marker's local coordinate system
        obj_points = np.array(
            [
                [-self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, -self.marker_size / 2, 0],
                [-self.marker_size / 2, -self.marker_size / 2, 0],
            ],
            dtype=np.float32,
        )

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            # Estimate pose for this marker
            success, rvec, tvec = cv2.solvePnP(
                obj_points, marker_corners[0], self.camera_matrix, self.dist_coeffs
            )

            if success:
                cv2.drawFrameAxes(
                    frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05
                )

                # Convert rotation vector to matrix
                rmat, _ = cv2.Rodrigues(rvec)

                # Create transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rmat
                transform[:3, 3] = tvec.flatten()

                marker_poses[marker_id] = transform

        return marker_poses
    
    def filter_detections(self, corners, ids):
        if ids is None:
            return corners, ids

        if self._filter_max_id is not None:
            mask = ids.flatten() <= self._filter_max_id
            filtered_corners = [corners[i] for i in range(len(corners)) if mask[i]]
            filtered_ids = ids[mask]
            return filtered_corners, filtered_ids

        return corners, ids

    def create_board_from_detections(
            self, 
            center_pose=False,
            reference_marker=None,
            reference_marker_offset=None, 
        ):
        """
        Create an ArUco board by detecting markers and their relative positions.
        Returns the board object and marker positions.
        """
        all_marker_poses = []
        print("Detecting marker positions... Press SPACE to capture, ESC when done.")

        try:
            while True:
                # Get frame from RealSense
                frame = self.camera.get_frame()

                # Detect markers
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                detections = self._detector.detect(gray)
                corners, ids = detections_to_corners_ids(detections)

                # Filter detections if needed
                corners, ids = self.filter_detections(corners, ids)

                # Draw detections
                frame_marked = frame.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame_marked, corners, ids)

                # Show frame
                cv2.putText(
                    frame_marked,
                    f"Captured frames: {len(all_marker_poses)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Marker Detection", frame_marked)

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == 32 and ids is not None:  # SPACE
                    # Estimate poses for this frame
                    marker_poses = self.estimate_marker_poses(
                        corners, ids, frame_marked
                    )

                    # Show frame
                    cv2.imshow("Marker t", frame_marked)

                    # Accept or reject
                    use_this_estimate = False
                    print("Press 'y' if this is acceptable, 'n' otherwise.")
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("y"):
                            use_this_estimate = True
                            break

                        if key == ord("n"):
                            break

                    if use_this_estimate:
                        all_marker_poses.append((ids, marker_poses))
                        print(f"Captured frame with {len(ids)} markers")
                    else:
                        print("Rejecting captured frame")

                    

            if not all_marker_poses:
                raise ValueError("No markers detected")

            # Process all captured poses to get consistent marker positions
            marker_positions = self.compute_marker_positions_gtsam(
                all_marker_poses, reference_marker
            )

            # Convert marker_positions to the format needed for board creation
            marker_corners_list = []
            marker_ids_list = []

            mins = [10, 10, 10]
            maxs = [0, 0, 0]

            ref_marker_corners = None

            for marker_id, transform in marker_positions.items():
                # Transform marker corners by pose
                corners_3d = np.array(
                    [
                        [-self.marker_size / 2, self.marker_size / 2, 0, 1],
                        [self.marker_size / 2, self.marker_size / 2, 0, 1],
                        [self.marker_size / 2, -self.marker_size / 2, 0, 1],
                        [-self.marker_size / 2, -self.marker_size / 2, 0, 1],
                    ],
                    dtype=np.float32,
                )

                transformed_corners = (transform @ corners_3d.T).T[:, :3]
                marker_corners_list.append(transformed_corners)
                marker_ids_list.append(marker_id)

                for i in range(3):
                    mins[i] = min(np.min(transformed_corners[:, i]), mins[i])
                    maxs[i] = max(np.max(transformed_corners[:, i]), maxs[i])

                if ref_marker_corners is None:
                    ref_marker_corners = transformed_corners

            if center_pose:
                center = (np.array(maxs) + np.array(mins)) / 2
                offset = -center

                for i in range(len(marker_corners_list)):
                    marker_corners_list[i] += offset
            elif reference_marker is not None and reference_marker_offset is not None:
                # Get the reference marker corners
                ref_transform = marker_positions[reference_marker]
                
                if isinstance(reference_marker_offset, str):
                    raise NotImplementedError(
                        "String parsing for reference_marker_offset not "
                        "implemented yet."
                    )
                # Given offset is in the reference marker frame
                offset_global = ref_transform[:3, :3] @ reference_marker_offset
                offset = -offset_global

                for i in range(len(marker_corners_list)):
                    marker_corners_list[i] += offset

            with open("marker_output.txt", "w") as f:
                for marker_id, corners in zip(marker_ids_list, marker_corners_list):
                    f.write(f"{marker_id}\n")
                    f.write(f"{corners}\n")

            # Create board object
            if self._tag_type == "apriltag":
                board = AprilTagBoard(
                    objPoints=np.array(marker_corners_list, np.float32),
                    dictionary=self._detector._tag_family,
                    ids=np.array(marker_ids_list),
                )
            elif self._tag_type == "aruco":
                board = cv2.aruco.Board_create(
                    objPoints=np.array(marker_corners_list, np.float32),
                    dictionary=self._detector._dictionary,
                    ids=np.array(marker_ids_list),
                )
            else:
                raise ValueError(f"Unsupported tag type: {self._tag_type}")
            
            print(board)

            return board, marker_corners_list, marker_positions

        finally:
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the scene or track box poses."
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=0.04,
        help="Indicate the size of the markers in meters.",
    )
    parser.add_argument(
        "--reference_marker",
        type=int,
        default=None,
        help="Specify the ID of the reference marker.",
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
        default="board",
        help="Name of the board.",
    )
    parser.add_argument(
        "--apriltag_family",
        type=str,
        default="tagStandard41h12",
        help="The apriltag family that should be used.",
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
            "value will be ignored (optional)."
        ),
    )

    args = parser.parse_args()

    
    # Use real CLI args by default, but allow an easy drop-in debug config via
    # the DEBUG_ARGS env var (set to "1" to enable).
    # args = argparse.Namespace(
    #     camera_config_path="/home/kiran/pose_estimation_ws/robot_ipc_control/robot_ipc_control/configs/test_cam_config.json",
    #     marker_size=0.032625,
    #     apriltag_family="tag16h5",
    #     name="debug_board",
    #     center_pose=False,
    #     reference_marker=7,
    #     # reference_marker=None,
    #     reference_marker_offset="0.0,0.02175,0.0",
    #     # reference_marker_offset=None,
    #     max_id=17,
    # )

    # Initialize detector
    detector = BoardBuilder(
        marker_size=args.marker_size,
        apriltag_family=args.apriltag_family,
        camera_config_path=args.camera_config_path,
        filter_max_id=args.max_id
    )

    if args.reference_marker_offset is not None:
        reference_marker_offset = np.array(
            [float(x) for x in args.reference_marker_offset.split(",")]
        )
    else:
        reference_marker_offset = None

    # Create new board configuration
    print("Creating new board configuration...")
    board, marker_corners_list, marker_positions = (
        detector.create_board_from_detections(
            center_pose=args.center_pose, 
            reference_marker=args.reference_marker,
            reference_marker_offset=reference_marker_offset,
        )
    )

    serialized_board = board_to_json(board, args.apriltag_family)

    # Save to JSON
    with open(f"{args.name}.json", "w") as f:
        json.dump(serialized_board, f, indent=4)


if __name__ == "__main__":
    main()
