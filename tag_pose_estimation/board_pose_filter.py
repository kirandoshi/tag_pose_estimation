import numpy as np

import collections

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from tag_pose_estimation.transform_utils import (
    rotation_matrix_to_quaternion,
)


class BoardPoseFilter:
    def __init__(self, initial_pose):
        self.position = np.array(initial_pose["position"])
        self.orientation_as_rot_mat = np.array(initial_pose["rotation_matrix"])
        self.confidence = 0

        self.initialized = True

        self.max_buffer_len = 10

        self.previous_positions = collections.deque(maxlen=self.max_buffer_len)
        self.prev_orientations = collections.deque(maxlen=self.max_buffer_len)

        self.transformation_to_desired_pose = np.eye(4)

        self.threshold = 0.4
        self.bounds = None

        self.reject_count = 0
        self.max_rejections = 50

        self._alpha = 0.1

        print("Done initializing")

    def update(self, new_pose):
        new_pos = np.array(new_pose["position"])
        new_orientation = np.array(new_pose["rotation_matrix"])
        self.confidence = np.array(new_pose["confidence"])

        if self.reject_count > self.max_rejections:
            print("Resetting rejection count to 0")
            self.reject_count = 0

            self.position = new_pos
            self.orientation_as_rot_mat = new_orientation

            self.previous_positions.clear()
            self.prev_orientations.clear()

            self.previous_positions.append(self.position)
            self.prev_orientations.append(self.orientation_as_rot_mat)

            return self.get_pose()

        # if any(new_pos < self.bounds[0, :]) or any(new_pos > self.bounds[1, :]):
        #     print("Pose out of bounds")
        #     return self.get_pose()

        # # deal with super off measurements
        # if np.linalg.norm(self.position - new_pos) > self.threshold:
        #     self.reject_count += 1

        #     print(f"POSE estimation jumped, rejected the {self.reject_count}-th time")
        #     return self.get_pose()

        # self.reject_count = 0

        # deal with super off measurements
        if np.linalg.norm(self.position - new_pos) > self.threshold:
            self.reject_count += 1

            print(f"POSE estimation jumped, rejected the {self.reject_count}-th time")
            return self.get_pose()

        # Low-pass for position
        # print("filtering position")
        self.position = self._alpha * new_pos + (1 - self._alpha) * self.position
        
        self.reject_count = 0

        # Low-pass for orientation
        # print("filtering orientation")
        self.orientation_as_rot_mat = Slerp(
            [0, 1], R.from_matrix([self.orientation_as_rot_mat, new_orientation])
        )([self._alpha])[0].as_matrix()

        # print(self.orientation_as_rot_mat)

        self.previous_positions.append(self.position)
        self.prev_orientations.append(self.orientation_as_rot_mat)

        return self.get_pose()

    def get_pose(self):
        # TODO: transform with above transformation?

        return np.concatenate(
            [self.position, rotation_matrix_to_quaternion(self.orientation_as_rot_mat)]
        )

    def get_confidence(self):
        return self.confidence

    def is_stable(self):
        """
        This method only makes sens for the static case!
        Use get_confidence() otherwise!
        """
        if len(self.prev_orientations) < self.max_buffer_len:
            print("Not enough measuremnts yet")
            return False

        def compute_pose_variance(positions, rotation_matrices):
            N = len(positions)
            assert N == len(rotation_matrices)

            # --- Translation ---
            positions_np = np.stack(positions)  # Shape: (N, 3)
            mean_pos = np.mean(positions_np, axis=0)
            translational_var_scalar = np.mean(
                np.linalg.norm(positions_np - mean_pos, axis=1) ** 2
            )

            # --- Rotation ---
            R_stack = np.stack(rotation_matrices)  # Shape: (N, 3, 3)
            R_avg = np.mean(R_stack, axis=0)
            U, _, Vt = np.linalg.svd(R_avg)
            R_mean = U @ Vt
            if np.linalg.det(R_mean) < 0:
                U[:, -1] *= -1
                R_mean = U @ Vt

            # Compute log-map of relative rotations
            R_mean_inv = R_mean.T  # Since R is in SO(3), R^-1 = R^T
            relative_rotvecs = []
            for R_i in rotation_matrices:
                R_rel = R_mean_inv @ R_i
                rotvec = R.from_matrix(R_rel).as_rotvec()
                relative_rotvecs.append(rotvec)

            relative_rotvecs_np = np.stack(relative_rotvecs)  # (N, 3)
            rotational_var_scalar = np.mean(
                np.linalg.norm(relative_rotvecs_np, axis=1) ** 2
            )

            return translational_var_scalar, rotational_var_scalar

        translational_var, rotational_var = compute_pose_variance(
            self.previous_positions, self.prev_orientations
        )

        std_translation = np.sqrt(translational_var)
        std_rotation = np.sqrt(rotational_var)

        # we get roughly 30 updates/s
        # we can assume that the pose estimate is stable/converged if it soes not move too much
        # 10cm variance
        if std_translation > 0.05:
            return False

        if std_rotation > 15 * np.pi*2 / 360:
            return False

        return True
